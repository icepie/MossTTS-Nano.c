#ifndef TTS_H
#define TTS_H

#include "tensor.h"

/* Model constants */
#define TTS_HIDDEN      768
#define TTS_NUM_LAYERS  12
#define TTS_NUM_HEADS   12
#define TTS_HEAD_DIM    64
#define TTS_MLP_DIM     3072
#define TTS_VOCAB       16384
#define TTS_N_VQ        16
#define TTS_AUDIO_CODEBOOK_SIZE 1024
#define TTS_AUDIO_PAD   1024
#define TTS_LOCAL_LAYERS 1

/* Token IDs (from config.json) */
#define TOK_PAD         3
#define TOK_IM_START    4
#define TOK_IM_END      5
#define TOK_AUDIO_START 6
#define TOK_AUDIO_END   7
#define TOK_AUDIO_USER  8
#define TOK_AUDIO_ASST  9

/* Sampling parameters (matching PyTorch defaults) */
typedef struct {
    int do_sample;           /* 1=sampling, 0=argmax */
    int max_new_frames;      /* 375 */
    float text_temperature;  /* 1.0 */
    float text_top_p;        /* 1.0 */
    int   text_top_k;        /* 50 */
    float audio_temperature; /* 0.8 */
    float audio_top_p;       /* 0.95 */
    int   audio_top_k;       /* 25 */
    float audio_rep_penalty; /* 1.2 */
    unsigned int seed;
} SamplingParams;

/* Set default PyTorch-matching parameters */
void sampling_params_default(SamplingParams *p);

/* KV cache for one layer */
typedef struct {
    float *key;    /* (seq_len, num_heads, head_dim) */
    float *value;  /* (seq_len, num_heads, head_dim) */
    int seq_len;
    int capacity;
} KVCache;

/* Cached weight pointers for one transformer block.
 * Matmul weights may be bf16 (on-the-fly conversion) or fp32. */
typedef struct {
    const float *ln1_w, *ln1_b;
    const float *attn_w, *attn_b;   /* c_attn (QKV) - fp32 */
    const float *proj_w, *proj_b;   /* c_proj - fp32 */
    const float *ln2_w, *ln2_b;
    const float *fc_in_w, *fc_in_b;
    const float *fc_out_w, *fc_out_b;
    const float *inv_freq;          /* RoPE, may be NULL */
    /* bf16 alternatives (non-NULL if weight is bf16) */
    const uint16_t *attn_w_bf16;
    const uint16_t *proj_w_bf16;
    const uint16_t *fc_in_w_bf16;
    const uint16_t *fc_out_w_bf16;
} BlockWeights;

/* Full TTS model state */
typedef struct {
    const WeightStore *ws;
    KVCache kv[TTS_NUM_LAYERS];
    float *global_hidden;  /* (TTS_HIDDEN,) */
    int total_seq_len;
    /* Generation history for repetition penalty */
    int *audio_history;    /* (max_frames, N_VQ) */
    int history_len;
    int history_cap;
    /* Cached weight pointers */
    BlockWeights global_blocks[TTS_NUM_LAYERS];
    BlockWeights local_blocks[TTS_LOCAL_LAYERS];
    const float *global_ln_w, *global_ln_b;
    const float *local_ln_w, *local_ln_b;
    const float *wte;                          /* (vocab, 768) */
    const float *audio_embeds[TTS_N_VQ];       /* each (1024, 768) */
    /* Local transformer KV cache (reset per frame) */
    KVCache local_kv[TTS_LOCAL_LAYERS];
    int local_kv_initialized;
    /* Pre-allocated workspace for run_block (avoids malloc per call) */
    float *ws_normed;    /* max_seq * HIDDEN */
    float *ws_qkv;       /* max_seq * 3*HIDDEN */
    float *ws_Q, *ws_K, *ws_V;  /* max_seq * HIDDEN each */
    float *ws_attn_out;  /* max_seq * HIDDEN */
    float *ws_proj;      /* max_seq * HIDDEN */
    float *ws_fc;        /* max_seq * MLP_DIM */
    float *ws_mlp;       /* max_seq * HIDDEN */
    int ws_max_seq;
} TTSModel;

/* Initialize / free */
void tts_init(TTSModel *m, const WeightStore *ws);
void tts_free(TTSModel *m);

/* Build input embeddings: input_ids (seq_len, 17) -> embeds (seq_len, 768) */
void tts_build_embeds(const TTSModel *m, const int *input_ids, int seq_len, float *embeds);

/* Prefill: process full prompt, populate KV cache */
void tts_prefill(TTSModel *m, const int *input_ids, int seq_len);

/* Generate one frame of 16 audio tokens with full sampling.
 * Returns 1 if generated, 0 if model chose to stop. */
int tts_generate_frame(TTSModel *m, int *audio_tokens_out, const SamplingParams *sp);

/* Decode step: feed one (1, 17) token row through global transformer with KV cache */
void tts_decode_step(TTSModel *m, const int *row17);

#endif /* TTS_H */
