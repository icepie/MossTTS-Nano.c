#ifndef CODEC_H
#define CODEC_H

#include "tensor.h"

#define CODEC_SAMPLE_RATE   48000
#define CODEC_NUM_CHANNELS  2
#define CODEC_DOWNSAMPLE    3840
#define CODEC_D_MODEL       256
#define CODEC_NUM_HEADS     4
#define CODEC_HEAD_DIM      64
#define CODEC_FFN_DIM       1024
#define CODEC_NUM_QUANTIZERS 16
#define CODEC_CODEBOOK_SIZE 1024
#define CODEC_CODEBOOK_DIM  8
#define CODEC_RVQ_DIM       512
#define CODEC_MAX_PERIOD    10000.0f

/* Encoder stage info */
typedef struct {
    int type;           /* 0 = patch, 1 = transformer */
    int patch_size;     /* for type 0 */
    const char *prefix; /* for type 1 */
    int num_layers;     /* for type 1 */
    float context_dur;  /* for type 1 */
} CodecStage;

/* Encode waveform to audio codes.
 * waveform: (channels, samples) interleaved float32
 * original_samples: number of samples per channel BEFORE padding
 * codes_out: (num_quantizers, code_frames) pre-allocated
 * Returns: number of code frames */
int codec_encode(const WeightStore *ws,
                 const float *waveform, int channels, int samples,
                 int original_samples,
                 int *codes_out);

/* Decode audio codes to waveform.
 * codes: (num_quantizers, code_frames) int
 * waveform_out: pre-allocated, (interleaved_samples,)
 * Returns: number of interleaved samples */
int codec_decode(const WeightStore *ws,
                 const int *codes, int num_quantizers, int code_frames,
                 float *waveform_out);

#endif /* CODEC_H */
