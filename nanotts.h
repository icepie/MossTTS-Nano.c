#ifndef NANOTTS_H
#define NANOTTS_H

#ifdef _WIN32
  #define TTS_API __declspec(dllexport)
#else
  #define TTS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Load / free model (global, single instance) */
TTS_API int  load_model(void);
TTS_API void free_model(void);

/* Extract speaker style → codes file */
TTS_API int style_extract(const char *wav_path, const char *codes_path);

/* Generate wav from codes file + text
 * stereo: 1=stereo 48kHz, 0=mono 48kHz
 * wav_out: allocated internally, caller must free() */
TTS_API int generate_wav(const char *codes_path, const char *text,
                         float **wav_out, int *out_samples,
                         int *out_channels, int *out_sr, int stereo);

/* Generate wav from ref audio + text (encode + generate + decode) */
TTS_API int generate_wav_from_ref(const char *ref_wav_path, const char *text,
                                   float **wav_out, int *out_samples,
                                   int *out_channels, int *out_sr, int stereo);

/* Save wav to file */
TTS_API int save_wav(const char *path, const float *wav,
                     int samples, int channels, int sr);

#ifdef __cplusplus
}
#endif

#endif
