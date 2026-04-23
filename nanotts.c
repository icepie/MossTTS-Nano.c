#include "nanotts.h"
#include "tensor.h"
#include "ops.h"
#include "tts.h"
#include "codec.h"
#include "prompt.h"
#include "audio.h"
#include "embedded.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Global singleton */
static WeightStore g_ws;
static PromptBuilder g_pb;
static int g_loaded = 0;

static int read_exact(FILE *f, void *dst, size_t size) {
    return fread(dst, 1, size, f) == size;
}

/* ================================================================ */

TTS_API int load_model(void) {
    if (g_loaded) return 0;

#ifdef EMBED_WEIGHTS
    if (weights_load_mem(&g_ws, weights_data, (size_t)weights_size) != 0) return -1;
#else
    return -1;
#endif

    const Tensor *tok = weights_get(&g_ws, "tokenizer.model");
    if (!tok || !tok->data_raw) {
        fprintf(stderr, "ERROR: tokenizer.model not found\n");
        weights_free(&g_ws); return -1;
    }
    g_pb.sp = sp_load_mem(tok->data_raw, (size_t)tok->size);
    if (!g_pb.sp) { weights_free(&g_ws); return -1; }

    g_loaded = 1;
    return 0;
}

TTS_API void free_model(void) {
    if (!g_loaded) return;
    prompt_builder_free(&g_pb);
    weights_free(&g_ws);
    g_loaded = 0;
}

/* ================================================================
 * Internal helpers
 * ================================================================ */

static int prepare_audio(const char *wav_path,
                         float **audio_out, int *ch_out,
                         int *samp_out, int *orig_out) {
    float *audio = NULL;
    int ch, samp, sr;
    if (wav_read(wav_path, &audio, &ch, &samp, &sr) != 0) return -1;

    if (sr != CODEC_SAMPLE_RATE) {
        float *res = NULL;
        int ns = audio_resample(audio, ch, samp, sr, CODEC_SAMPLE_RATE, &res);
        if (ns < 0) { free(audio); return -1; }
        free(audio); audio = res; samp = ns;
    }
    if (ch == 1) {
        float *s = (float *)malloc((size_t)2 * samp * sizeof(float));
        memcpy(s, audio, (size_t)samp * sizeof(float));
        memcpy(s + samp, audio, (size_t)samp * sizeof(float));
        free(audio); audio = s; ch = 2;
    }
    int orig = samp;
    int rem = samp % CODEC_DOWNSAMPLE;
    if (rem) {
        int p = samp + CODEC_DOWNSAMPLE - rem;
        float *pa = (float *)calloc((size_t)ch * p, sizeof(float));
        for (int c = 0; c < ch; c++)
            memcpy(pa + c * p, audio + c * samp, (size_t)samp * sizeof(float));
        free(audio); audio = pa; samp = p;
    }
    *audio_out = audio; *ch_out = ch; *samp_out = samp; *orig_out = orig;
    return 0;
}

static int do_encode(const float *audio, int ch, int samp, int orig,
                     int **codes_out, int *cf_out) {
    int *codes = (int *)calloc((size_t)CODEC_NUM_QUANTIZERS * (samp * ch / 7680 + 10), sizeof(int));
    int cf = codec_encode(&g_ws, audio, ch, samp, orig, codes);
    *codes_out = codes; *cf_out = cf;
    return 0;
}

static int do_synth(const int *codes_t, int cf, const char *text, int stereo,
                    float **wav_out, int *out_samples, int *out_channels, int *out_sr) {
    int *ids = (int *)calloc(2048 * 17, sizeof(int));
    int seq = prompt_build_voice_clone(&g_pb, text, codes_t, cf, ids, 2048);

    TTSModel tts;
    tts_init(&tts, &g_ws);
    tts_prefill(&tts, ids, seq);
    free(ids);

    SamplingParams sp;
    sampling_params_default(&sp);

    int gen[512 * TTS_N_VQ];
    int nf = 0;
    for (int f = 0; f < sp.max_new_frames; f++) {
        int tok[TTS_N_VQ];
        if (!tts_generate_frame(&tts, tok, &sp)) break;
        for (int c = 0; c < TTS_N_VQ; c++) gen[c * 512 + f] = tok[c];
        nf++;
    }
    tts_free(&tts);
    if (nf == 0) return -1;

    int *compact = (int *)malloc((size_t)TTS_N_VQ * nf * sizeof(int));
    for (int q = 0; q < TTS_N_VQ; q++)
        for (int f = 0; f < nf; f++)
            compact[q * nf + f] = gen[q * 512 + f];

    float *decoded = (float *)calloc((size_t)512 * CODEC_DOWNSAMPLE * CODEC_NUM_CHANNELS, sizeof(float));
    int dl = codec_decode(&g_ws, compact, TTS_N_VQ, nf, decoded);
    free(compact);

    int total = dl / CODEC_NUM_CHANNELS;
    if (stereo) {
        float *out = (float *)malloc((size_t)total * 2 * sizeof(float));
        memcpy(out, decoded, (size_t)total * 2 * sizeof(float));
        *wav_out = out; *out_samples = total; *out_channels = 2;
    } else {
        float *out = (float *)malloc((size_t)total * sizeof(float));
        for (int i = 0; i < total; i++)
            out[i] = (decoded[i * 2] + decoded[i * 2 + 1]) / 2.0f;
        *wav_out = out; *out_samples = total; *out_channels = 1;
    }
    free(decoded);
    *out_sr = CODEC_SAMPLE_RATE;
    return 0;
}

/* ================================================================
 * Public API
 * ================================================================ */

TTS_API int style_extract(const char *wav_path, const char *codes_path) {
    float *audio; int ch, samp, orig;
    if (prepare_audio(wav_path, &audio, &ch, &samp, &orig) != 0) return -1;

    int *codes = NULL; int cf;
    do_encode(audio, ch, samp, orig, &codes, &cf);
    free(audio);

    FILE *f = fopen(codes_path, "wb");
    if (!f) { free(codes); return -1; }
    uint32_t magic = 0x434F4445, nq = CODEC_NUM_QUANTIZERS, frames = (uint32_t)cf;
    fwrite(&magic, 4, 1, f);
    fwrite(&nq, 4, 1, f);
    fwrite(&frames, 4, 1, f);
    fwrite(codes, sizeof(int), (size_t)nq * cf, f);
    fclose(f);
    free(codes);
    return 0;
}

TTS_API int generate_wav(const char *codes_path, const char *text,
                         float **wav_out, int *out_samples,
                         int *out_channels, int *out_sr, int stereo) {
    FILE *f = fopen(codes_path, "rb");
    if (!f) return -1;
    uint32_t magic = 0, nq = 0, frames = 0;
    if (!read_exact(f, &magic, sizeof(magic)) ||
        !read_exact(f, &nq, sizeof(nq)) ||
        !read_exact(f, &frames, sizeof(frames))) {
        fclose(f);
        return -1;
    }
    if (magic != 0x434F4445) { fclose(f); return -1; }
    int *codes = (int *)malloc((size_t)nq * frames * sizeof(int));
    if (!codes || fread(codes, sizeof(int), (size_t)nq * frames, f) != (size_t)nq * frames) {
        free(codes);
        fclose(f);
        return -1;
    }
    fclose(f);

    int *codes_t = (int *)malloc((size_t)frames * nq * sizeof(int));
    for (uint32_t q = 0; q < nq; q++)
        for (uint32_t t = 0; t < frames; t++)
            codes_t[t * nq + q] = codes[q * frames + t];
    free(codes);

    int ret = do_synth(codes_t, (int)frames, text, stereo,
                       wav_out, out_samples, out_channels, out_sr);
    free(codes_t);
    return ret;
}

TTS_API int generate_wav_from_ref(const char *ref_wav_path, const char *text,
                                   float **wav_out, int *out_samples,
                                   int *out_channels, int *out_sr, int stereo) {
    float *audio; int ch, samp, orig;
    if (prepare_audio(ref_wav_path, &audio, &ch, &samp, &orig) != 0) return -1;

    int *codes = NULL; int cf;
    do_encode(audio, ch, samp, orig, &codes, &cf);
    free(audio);

    int *codes_t = (int *)malloc((size_t)cf * TTS_N_VQ * sizeof(int));
    for (int q = 0; q < TTS_N_VQ; q++)
        for (int t = 0; t < cf; t++)
            codes_t[t * TTS_N_VQ + q] = codes[q * cf + t];
    free(codes);

    int ret = do_synth(codes_t, cf, text, stereo,
                       wav_out, out_samples, out_channels, out_sr);
    free(codes_t);
    return ret;
}

TTS_API int save_wav(const char *path, const float *wav,
                     int samples, int channels, int sr) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    uint32_t data_size = (uint32_t)(samples * channels * 2);
    uint32_t file_size = 36 + data_size;
    uint16_t num_ch = (uint16_t)channels, bits = 16, fmt = 1;
    uint32_t byte_rate = (uint32_t)(sr * channels * 2);
    uint16_t block_align = (uint16_t)(channels * 2);
    uint32_t sr32 = (uint32_t)sr, fmt_size = 16;

    fwrite("RIFF", 1, 4, f); fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f); fwrite("fmt ", 1, 4, f);
    fwrite(&fmt_size, 4, 1, f); fwrite(&fmt, 2, 1, f);
    fwrite(&num_ch, 2, 1, f); fwrite(&sr32, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f); fwrite(&block_align, 2, 1, f);
    fwrite(&bits, 2, 1, f); fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);

    for (int i = 0; i < samples * channels; i++) {
        float v = wav[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        int16_t s = (int16_t)(v * 32767.0f);
        fwrite(&s, 2, 1, f);
    }
    fclose(f);
    return 0;
}
