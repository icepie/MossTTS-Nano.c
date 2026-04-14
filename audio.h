#ifndef WAV_H
#define WAV_H

/* Read WAV file. Returns float samples in [-1, 1].
 * Layout: (channels, samples) row-major. Allocates *data. Caller must free.
 * Returns 0 on success. */
int wav_read(const char *path, float **data, int *channels, int *samples, int *sample_rate);

/* Write WAV file (mono, float32 -> 16-bit PCM). Returns 0 on success. */
int wav_write(const char *path, const float *data, int samples, int sample_rate);

/* Resample audio using sinc interpolation with Hann window.
 * Matches torchaudio.functional.resample() with default parameters.
 * input:  (num_channels, in_samples) row-major
 * output: allocated by function, (num_channels, out_samples) row-major
 * Returns number of output samples per channel, or -1 on error. */
int audio_resample(const float *input, int num_channels, int in_samples,
                   int orig_freq, int new_freq,
                   float **output);

#endif /* WAV_H */
