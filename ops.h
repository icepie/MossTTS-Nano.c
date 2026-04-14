#ifndef OPS_H
#define OPS_H

#include "tensor.h"

/* ---- Basic ops ---- */

/* C = A @ B^T + bias.  A:(M,K) B:(N,K) bias:(N) or NULL -> C:(M,N) */
void matmul_t(const float *A, const float *B, const float *bias,
              float *C, int M, int K, int N);

/* In-place layer normalization: x[len] with weight[len], bias[len] */
void layer_norm(float *x, const float *weight, const float *bias,
                int len, float eps);

/* Activation functions (in-place, n elements) */
void gelu_new_inplace(float *x, int n);       /* TTS model */
void gelu_standard_inplace(float *x, int n);  /* Codec model */

/* Softmax in-place over n elements */
void softmax_inplace(float *x, int n);

/* ---- TTS RoPE (interleaved style) ---- */
/* q, k: (seq_len, num_heads, head_dim), position_ids: (seq_len,) */
void tts_rope_apply(float *q, float *k, const int *position_ids,
                    const float *inv_freq, int seq_len, int num_heads,
                    int head_dim);

/* ---- Codec RoPE (complex multiplication style) ---- */
/* q, k: (num_heads, seq_len, head_dim), offset=0 for non-streaming */
void codec_rope_apply(float *q, float *k, int offset,
                      float max_period, int num_heads, int seq_len,
                      int head_dim);

/* ---- SIMD helpers (exposed for attention loops) ---- */
float dot_f32_vec(const float *a, const float *b, int n);
void vec_add_scaled_vec(float *dst, const float *src, float scale, int n);

/* ---- bf16 on-the-fly matmul ---- */
/* C = A(fp32, M×K) @ B(bf16, N×K)^T + bias(fp32 or NULL) → C(fp32, M×N) */
void matmul_t_bf16(const float *A, const uint16_t *B_bf16, const float *bias,
                   float *C, int M, int K, int N);

/* Get float pointer from tensor (converts bf16 to fp32 if needed).
 * For bf16 tensors used outside matmul (embeddings, layernorm weights etc.) */
const float *tensor_data_f32(const Tensor *t);
void tensor_data_f32_free(const Tensor *t, const float *ptr);

/* ---- Fused FFN: matmul + gelu + matmul in one pass ---- */
/* out = (x @ W1^T + b1 |> gelu) @ W2^T + b2
 * Avoids writing/reading the full intermediate buffer to main memory.
 * x: (M, K1), W1: (K2, K1), W2: (N, K2), out: (M, N) */
void fused_ffn_gelu_new(const float *x, const float *W1, const float *b1,
                        const float *W2, const float *b2,
                        float *out, float *fc_buf,
                        int M, int K1, int K2, int N);

void fused_ffn_gelu_standard(const float *x, const float *W1, const float *b1,
                             const float *W2, const float *b2,
                             float *out, float *fc_buf,
                             int M, int K1, int K2, int N);

/* ---- Argmax ---- */
int argmax_f(const float *x, int n);

/* ---- Sampling ---- */

/* Apply repetition penalty in-place to logits.
 * prev_ids: array of previously generated token IDs for this codebook
 * prev_len: number of previous tokens */
void apply_repetition_penalty(float *logits, int vocab_size,
                              const int *prev_ids, int prev_len,
                              float penalty);

/* Sample a token from logits with temperature, top-k, top-p.
 * Returns sampled token index. Uses rand_state for RNG. */
int sample_token(float *logits, int vocab_size,
                 float temperature, int top_k, float top_p,
                 unsigned int *rand_state);

/* Simple xorshift RNG returning float in [0, 1) */
float rand_uniform(unsigned int *state);

#endif /* OPS_H */
