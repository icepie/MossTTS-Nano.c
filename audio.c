#include "audio.h"
#include "ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static int read_exact(FILE *f, void *dst, size_t size) {
    return fread(dst, 1, size, f) == size;
}

int wav_read(const char *path, float **data, int *channels, int *samples, int *sample_rate) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open WAV: %s\n", path); return -1; }

    /* Read RIFF header */
    char riff[4], wave[4];
    uint32_t file_size;
    if (!read_exact(f, riff, sizeof(riff)) ||
        !read_exact(f, &file_size, sizeof(file_size)) ||
        !read_exact(f, wave, sizeof(wave))) {
        fprintf(stderr, "Truncated WAV header: %s\n", path);
        fclose(f);
        return -1;
    }
    if (memcmp(riff, "RIFF", 4) || memcmp(wave, "WAVE", 4)) {
        fprintf(stderr, "Not a WAV file: %s\n", path); fclose(f); return -1;
    }

    /* Find fmt and data chunks */
    uint16_t audio_fmt = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sr = 0, data_size = 0;
    int found_fmt = 0, found_data = 0;

    while (!found_data) {
        char chunk_id[4];
        uint32_t chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4) break;
        if (fread(&chunk_size, 4, 1, f) != 1) break;

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            if (!read_exact(f, &audio_fmt, sizeof(audio_fmt)) ||
                !read_exact(f, &num_channels, sizeof(num_channels)) ||
                !read_exact(f, &sr, sizeof(sr))) {
                fprintf(stderr, "Truncated WAV fmt chunk: %s\n", path);
                fclose(f);
                return -1;
            }
            fseek(f, 6, SEEK_CUR); /* skip byte_rate, block_align */
            if (!read_exact(f, &bits_per_sample, sizeof(bits_per_sample))) {
                fprintf(stderr, "Truncated WAV fmt chunk: %s\n", path);
                fclose(f);
                return -1;
            }
            if (chunk_size > 16) fseek(f, chunk_size - 16, SEEK_CUR);
            found_fmt = 1;
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            found_data = 1;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }

    if (!found_fmt || !found_data) {
        fprintf(stderr, "Malformed WAV: %s\n", path); fclose(f); return -1;
    }

    int total_samples = data_size / (bits_per_sample / 8) / num_channels;
    *channels = num_channels;
    *samples = total_samples;
    *sample_rate = (int)sr;

    float *buf = (float *)malloc((size_t)num_channels * total_samples * sizeof(float));

    if (audio_fmt == 1 && bits_per_sample == 16) {
        /* PCM 16-bit */
        int16_t *raw = (int16_t *)malloc(data_size);
        if (!raw || !read_exact(f, raw, data_size)) {
            fprintf(stderr, "Truncated WAV PCM data: %s\n", path);
            free(raw); free(buf); fclose(f); return -1;
        }
        /* Deinterleave: WAV is [L,R,L,R,...] -> (channels, samples) */
        for (int s = 0; s < total_samples; s++)
            for (int c = 0; c < num_channels; c++)
                buf[c * total_samples + s] = (float)raw[s * num_channels + c] / 32768.0f;
        free(raw);
    } else if (audio_fmt == 3 && bits_per_sample == 32) {
        /* Float 32-bit */
        float *raw = (float *)malloc(data_size);
        if (!raw || !read_exact(f, raw, data_size)) {
            fprintf(stderr, "Truncated WAV float data: %s\n", path);
            free(raw); free(buf); fclose(f); return -1;
        }
        for (int s = 0; s < total_samples; s++)
            for (int c = 0; c < num_channels; c++)
                buf[c * total_samples + s] = raw[s * num_channels + c];
        free(raw);
    } else {
        fprintf(stderr, "Unsupported WAV format: fmt=%d bits=%d\n", audio_fmt, bits_per_sample);
        fclose(f); free(buf); return -1;
    }

    fclose(f);
    *data = buf;
    return 0;
}

int wav_write(const char *path, const float *data, int samples, int sample_rate) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot create WAV: %s\n", path); return -1; }

    uint16_t num_channels = 1, bits = 16;
    uint32_t data_size = (uint32_t)(samples * bits / 8);
    uint32_t file_size = 36 + data_size;
    uint32_t byte_rate = (uint32_t)(sample_rate * num_channels * bits / 8);
    uint16_t block_align = (uint16_t)(num_channels * bits / 8);

    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    uint16_t audio_fmt = 1;
    fwrite(&fmt_size, 4, 1, f);
    fwrite(&audio_fmt, 2, 1, f);
    fwrite(&num_channels, 2, 1, f);
    uint32_t sr = (uint32_t)sample_rate;
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);

    for (int i = 0; i < samples; i++) {
        float v = data[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        int16_t s = (int16_t)(v * 32767.0f);
        fwrite(&s, 2, 1, f);
    }

    fclose(f);
    return 0;
}

/* ================================================================
 * Audio resampling - matches torchaudio.functional.resample()
 *
 * Default parameters:
 *   lowpass_filter_width = 6
 *   rolloff = 0.99
 *   resampling_method = "sinc_interp_hann"
 * ================================================================ */

static int gcd(int a, int b) {
    while (b) { int t = b; b = a % b; a = t; }
    return a;
}

int audio_resample(const float *input, int num_channels, int in_samples,
                   int orig_freq, int new_freq,
                   float **output) {
    if (orig_freq <= 0 || new_freq <= 0) return -1;
    if (orig_freq == new_freq) {
        int total = num_channels * in_samples;
        *output = (float *)malloc((size_t)total * sizeof(float));
        memcpy(*output, input, (size_t)total * sizeof(float));
        return in_samples;
    }

    const int lowpass_filter_width = 6;
    const float rolloff = 0.99f;

    int g = gcd(orig_freq, new_freq);
    int orig = orig_freq / g;
    int newf = new_freq / g;

    float base_freq = (float)(orig < newf ? orig : newf) * rolloff;
    int width = (int)ceilf((float)(lowpass_filter_width * orig) / base_freq);

    /* Build sinc kernel: (newf, 1, 2*width + orig) */
    int kernel_len = 2 * width + orig;
    float *kernel = (float *)calloc((size_t)newf * kernel_len, sizeof(float));

    float scale = base_freq / (float)orig;
    float pi = 3.14159265358979323846f;

    for (int i = 0; i < newf; i++) {
        for (int j = 0; j < kernel_len; j++) {
            float idx = (float)(j - width) / (float)orig;
            float t = (-(float)i / (float)newf + idx) * base_freq;

            /* Clamp */
            if (t < -(float)lowpass_filter_width) t = -(float)lowpass_filter_width;
            if (t > (float)lowpass_filter_width) t = (float)lowpass_filter_width;

            /* Hann window */
            float window = cosf(t * pi / (float)lowpass_filter_width / 2.0f);
            window = window * window;

            /* Sinc */
            float sinc_val;
            float t_pi = t * pi;
            if (fabsf(t_pi) < 1e-7f)
                sinc_val = 1.0f;
            else
                sinc_val = sinf(t_pi) / t_pi;

            kernel[i * kernel_len + j] = sinc_val * window * scale;
        }
    }

    /* Apply: conv1d with stride=orig, after padding */
    int padded_len = in_samples + width + width + orig;
    int out_samples_raw = (padded_len - kernel_len) / orig + 1;
    /* Target length = ceil(newf * in_samples / orig) */
    long long target_ll = ((long long)newf * in_samples + orig - 1) / orig;
    int target_len = (int)target_ll;
    if (target_len > out_samples_raw * newf) target_len = out_samples_raw * newf;

    /* The conv1d output has shape (num_channels, newf, out_samples_raw)
     * then transposed and reshaped to (num_channels, out_samples_raw * newf)
     * then trimmed to target_len */
    int out_total = target_len;
    *output = (float *)malloc((size_t)num_channels * out_total * sizeof(float));

    for (int ch = 0; ch < num_channels; ch++) {
        /* Pad input */
        float *padded = (float *)calloc((size_t)padded_len, sizeof(float));
        memcpy(padded + width, input + ch * in_samples, (size_t)in_samples * sizeof(float));

        /* Conv1d: for each kernel filter i (0..newf-1), stride=orig */
        /* Output layout after transpose+reshape:
         * resampled[n] = conv_output[n % newf][n / newf]
         * where conv_output[i][s] = sum_k padded[s*orig + k] * kernel[i][k] */
        for (int n = 0; n < out_total; n++) {
            int i = n % newf;
            int s = n / newf;
            (*output)[ch * out_total + n] = dot_f32_vec(padded + s * orig, kernel + i * kernel_len, kernel_len);
        }
        free(padded);
    }

    free(kernel);
    return out_total;
}
