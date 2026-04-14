#include "ops.h"
#include "platform.h"
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdlib.h>

/* ================================================================
 * SIMD detection
 * ================================================================ */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  #include <arm_neon.h>
  #define USE_NEON 1
#elif defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
  #include <xmmintrin.h>
  #include <emmintrin.h>
  #define USE_SSE 1
#endif

/* ================================================================
 * Dot product (SIMD)
 * ================================================================ */
static inline float dot_f32(const float *a, const float *b, int n) {
#if defined(USE_NEON)
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        sum0 = vmlaq_f32(sum0, vld1q_f32(a + i),     vld1q_f32(b + i));
        sum1 = vmlaq_f32(sum1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    sum0 = vaddq_f32(sum0, sum1);
    for (; i + 3 < n; i += 4)
        sum0 = vmlaq_f32(sum0, vld1q_f32(a + i), vld1q_f32(b + i));
    float sum = vaddvq_f32(sum0);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#elif defined(USE_SSE)
    __m128 s0 = _mm_setzero_ps();
    __m128 s1 = _mm_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_loadu_ps(a + i),     _mm_loadu_ps(b + i)));
        s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_loadu_ps(a + i + 4), _mm_loadu_ps(b + i + 4)));
    }
    s0 = _mm_add_ps(s0, s1);
    for (; i + 3 < n; i += 4)
        s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    /* Horizontal sum */
    __m128 t = _mm_add_ps(s0, _mm_movehl_ps(s0, s0));
    t = _mm_add_ss(t, _mm_shuffle_ps(t, t, 1));
    float sum;
    _mm_store_ss(&sum, t);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
#endif
}

/* ================================================================
 * Vector add scaled: dst[i] += scale * src[i]  (SIMD)
 * ================================================================ */
static inline void vec_add_scaled(float *dst, const float *src, float scale, int n) {
#if defined(USE_NEON)
    float32x4_t vs = vdupq_n_f32(scale);
    int i = 0;
    for (; i + 3 < n; i += 4)
        vst1q_f32(dst + i, vmlaq_f32(vld1q_f32(dst + i), vld1q_f32(src + i), vs));
    for (; i < n; i++) dst[i] += scale * src[i];
#elif defined(USE_SSE)
    __m128 vs = _mm_set1_ps(scale);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        __m128 d = _mm_loadu_ps(dst + i);
        __m128 s = _mm_loadu_ps(src + i);
        _mm_storeu_ps(dst + i, _mm_add_ps(d, _mm_mul_ps(s, vs)));
    }
    for (; i < n; i++) dst[i] += scale * src[i];
#else
    for (int i = 0; i < n; i++) dst[i] += scale * src[i];
#endif
}

/* ================================================================
 * bf16 → f32 conversion helper
 * ================================================================ */
static inline float bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

/* Dot product: a(fp32) · b(bf16), on-the-fly conversion */
static inline float dot_f32_bf16(const float *a, const uint16_t *b, int n) {
#if defined(USE_NEON)
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        /* Convert 4 bf16 to f32: shift each uint16 left by 16 */
        uint16x4_t b16_0 = vld1_u16(b + i);
        uint16x4_t b16_1 = vld1_u16(b + i + 4);
        uint32x4_t b32_0 = vshll_n_u16(b16_0, 16);
        uint32x4_t b32_1 = vshll_n_u16(b16_1, 16);
        float32x4_t bf0 = vreinterpretq_f32_u32(b32_0);
        float32x4_t bf1 = vreinterpretq_f32_u32(b32_1);
        sum0 = vmlaq_f32(sum0, vld1q_f32(a + i), bf0);
        sum1 = vmlaq_f32(sum1, vld1q_f32(a + i + 4), bf1);
    }
    sum0 = vaddq_f32(sum0, sum1);
    for (; i + 3 < n; i += 4) {
        uint16x4_t b16 = vld1_u16(b + i);
        float32x4_t bf = vreinterpretq_f32_u32(vshll_n_u16(b16, 16));
        sum0 = vmlaq_f32(sum0, vld1q_f32(a + i), bf);
    }
    float sum = vaddvq_f32(sum0);
    for (; i < n; i++) sum += a[i] * bf16_to_f32(b[i]);
    return sum;
#elif defined(USE_SSE)
    /* SSE doesn't have bf16 support, scalar conversion */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * bf16_to_f32(b[i]);
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * bf16_to_f32(b[i]);
    return sum;
#endif
}

/* Public wrappers for SIMD helpers */
float dot_f32_vec(const float *a, const float *b, int n) { return dot_f32(a, b, n); }
void vec_add_scaled_vec(float *dst, const float *src, float scale, int n) { vec_add_scaled(dst, src, scale, n); }

/* ================================================================
 * Matrix multiply: C = A @ B^T + bias  (SIMD dot product)
 * B is (N, K) row-major, so each B row is contiguous — optimal for dot.
 *
 * For large M (>64), parallelize across rows using pthreads.
 * ================================================================ */

#ifndef USE_OPENBLAS
typedef struct {
    const float *A, *B, *bias;
    float *C;
    int row_start, row_end, K, N;
} MatmulChunk;

static void *matmul_worker(void *arg) {
    MatmulChunk *c = (MatmulChunk *)arg;
    for (int i = c->row_start; i < c->row_end; i++) {
        const float *a_row = c->A + i * c->K;
        for (int j = 0; j < c->N; j++) {
            c->C[i * c->N + j] = dot_f32(a_row, c->B + j * c->K, c->K)
                                + (c->bias ? c->bias[j] : 0.0f);
        }
    }
    return NULL;
}
#endif
#define MATMUL_THREAD_THRESHOLD 128

#ifdef _WIN32
#include <windows.h>
static int get_num_cores(void) {
    SYSTEM_INFO si; GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
}
#else
#include <unistd.h>
static int get_num_cores(void) {
    int n = (int)sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? n : 4;
}
#endif

static int num_threads(void) {
    static int cached = 0;
    if (!cached) {
        cached = get_num_cores();
        if (cached > 16) cached = 16;
        if (cached < 1) cached = 1;
    }
    return cached;
}

#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

void matmul_t(const float *A, const float *B, const float *bias,
              float *C, int M, int K, int N) {
#ifdef USE_OPENBLAS
    /* C = A @ B^T  =>  cblas_sgemm with transB=Trans
     * A: (M,K) row-major, B: (N,K) row-major
     * We want C[i][j] = sum_k A[i][k] * B[j][k]
     * cblas: C = alpha * A * B^T + beta * C  */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
    if (bias) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                C[i * N + j] += bias[j];
    }
#else
    if (M >= MATMUL_THREAD_THRESHOLD && M * N > 100000) {
        int NT = num_threads();
        thread_t threads[16];
        MatmulChunk chunks[16];
        int rows_per = (M + NT - 1) / NT;
        int nt = 0;
        for (int t = 0; t < NT; t++) {
            int start = t * rows_per;
            int end = start + rows_per;
            if (start >= M) break;
            if (end > M) end = M;
            chunks[t] = (MatmulChunk){A, B, bias, C, start, end, K, N};
            thread_create(&threads[t], matmul_worker, &chunks[t]);
            nt++;
        }
        for (int t = 0; t < nt; t++)
            thread_join(threads[t]);
    } else {
        for (int i = 0; i < M; i++) {
            const float *a_row = A + i * K;
            for (int j = 0; j < N; j++)
                C[i * N + j] = dot_f32(a_row, B + j * K, K) + (bias ? bias[j] : 0.0f);
        }
    }
#endif
}

/* ================================================================
 * bf16 matmul: C = A(fp32) @ B(bf16)^T + bias
 * ================================================================ */
typedef struct {
    const float *A; const uint16_t *B; const float *bias;
    float *C; int row_start, row_end, K, N;
} MatmulBF16Chunk;

static void *matmul_bf16_worker(void *arg) {
    MatmulBF16Chunk *c = (MatmulBF16Chunk *)arg;
    for (int i = c->row_start; i < c->row_end; i++) {
        const float *a_row = c->A + i * c->K;
        for (int j = 0; j < c->N; j++)
            c->C[i * c->N + j] = dot_f32_bf16(a_row, c->B + j * c->K, c->K)
                                + (c->bias ? c->bias[j] : 0.0f);
    }
    return NULL;
}

void matmul_t_bf16(const float *A, const uint16_t *B_bf16, const float *bias,
                   float *C, int M, int K, int N) {
    if (M >= MATMUL_THREAD_THRESHOLD && M * N > 100000) {
        int NT = num_threads();
        thread_t threads[16];
        MatmulBF16Chunk chunks[16];
        int rows_per = (M + NT - 1) / NT;
        int nt = 0;
        for (int t = 0; t < NT; t++) {
            int s = t * rows_per, e = s + rows_per;
            if (s >= M) break;
            if (e > M) e = M;
            chunks[t] = (MatmulBF16Chunk){A, B_bf16, bias, C, s, e, K, N};
            thread_create(&threads[t], matmul_bf16_worker, &chunks[t]);
            nt++;
        }
        for (int t = 0; t < nt; t++) thread_join(threads[t]);
    } else {
        for (int i = 0; i < M; i++) {
            const float *a_row = A + i * K;
            for (int j = 0; j < N; j++)
                C[i * N + j] = dot_f32_bf16(a_row, B_bf16 + j * K, K)
                              + (bias ? bias[j] : 0.0f);
        }
    }
}

/* ================================================================
 * Tensor data access helpers (bf16 → f32 on demand)
 * ================================================================ */
const float *tensor_data_f32(const Tensor *t) {
    if (t->data) return t->data;
    /* Convert bf16 to f32 */
    float *f = (float *)malloc((size_t)t->size * sizeof(float));
    for (int i = 0; i < t->size; i++)
        f[i] = bf16_to_f32(t->data_bf16[i]);
    return f;
}

void tensor_data_f32_free(const Tensor *t, const float *ptr) {
    /* Only free if we allocated (bf16 case) */
    if (t->data == NULL && ptr != NULL)
        free((void *)ptr);
}

/* ================================================================
 * Fused FFN: matmul + activation + matmul
 *
 * For M=1 (decode): compute fc_in row, apply gelu immediately,
 * then multiply with fc_out — the intermediate stays in cache.
 * ================================================================ */
void fused_ffn_gelu_new(const float *x, const float *W1, const float *b1,
                        const float *W2, const float *b2,
                        float *out, float *fc_buf,
                        int M, int K1, int K2, int N) {
    /* Step 1: fc_buf = x @ W1^T + b1 */
    matmul_t(x, W1, b1, fc_buf, M, K1, K2);
    /* Step 2: gelu in-place */
    gelu_new_inplace(fc_buf, M * K2);
    /* Step 3: out = fc_buf @ W2^T + b2 */
    matmul_t(fc_buf, W2, b2, out, M, K2, N);
}

void fused_ffn_gelu_standard(const float *x, const float *W1, const float *b1,
                             const float *W2, const float *b2,
                             float *out, float *fc_buf,
                             int M, int K1, int K2, int N) {
    matmul_t(x, W1, b1, fc_buf, M, K1, K2);
    gelu_standard_inplace(fc_buf, M * K2);
    matmul_t(fc_buf, W2, b2, out, M, K2, N);
}

/* ================================================================
 * Layer Normalization (in-place, SIMD)
 * ================================================================ */
void layer_norm(float *x, const float *weight, const float *bias,
                int len, float eps) {
    /* Mean */
    float mean = 0.0f;
#if defined(USE_NEON)
    float32x4_t vsum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < len; i += 4) vsum = vaddq_f32(vsum, vld1q_f32(x + i));
    mean = vaddvq_f32(vsum);
    for (; i < len; i++) mean += x[i];
#elif defined(USE_SSE)
    __m128 vsum = _mm_setzero_ps();
    int i = 0;
    for (; i + 3 < len; i += 4) vsum = _mm_add_ps(vsum, _mm_loadu_ps(x + i));
    __m128 t = _mm_add_ps(vsum, _mm_movehl_ps(vsum, vsum));
    t = _mm_add_ss(t, _mm_shuffle_ps(t, t, 1));
    _mm_store_ss(&mean, t);
    for (; i < len; i++) mean += x[i];
#else
    for (int i = 0; i < len; i++) mean += x[i];
#endif
    mean /= (float)len;

    /* Variance */
    float var = 0.0f;
    for (int i = 0; i < len; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)len;
    float inv_std = 1.0f / sqrtf(var + eps);

    /* Normalize */
#if defined(USE_NEON)
    float32x4_t vmean = vdupq_n_f32(mean);
    float32x4_t vistd = vdupq_n_f32(inv_std);
    i = 0;
    for (; i + 3 < len; i += 4) {
        float32x4_t v = vsubq_f32(vld1q_f32(x + i), vmean);
        v = vmulq_f32(v, vistd);
        v = vmlaq_f32(vld1q_f32(bias + i), v, vld1q_f32(weight + i));
        vst1q_f32(x + i, v);
    }
    for (; i < len; i++)
        x[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
#elif defined(USE_SSE)
    __m128 vmean = _mm_set1_ps(mean);
    __m128 vistd = _mm_set1_ps(inv_std);
    i = 0;
    for (; i + 3 < len; i += 4) {
        __m128 v = _mm_sub_ps(_mm_loadu_ps(x + i), vmean);
        v = _mm_mul_ps(v, vistd);
        v = _mm_add_ps(_mm_mul_ps(v, _mm_loadu_ps(weight + i)), _mm_loadu_ps(bias + i));
        _mm_storeu_ps(x + i, v);
    }
    for (; i < len; i++)
        x[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
#else
    for (int i = 0; i < len; i++)
        x[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
#endif
}

/* ================================================================
 * GELU activations (in-place)
 * ================================================================ */
void gelu_new_inplace(float *x, int n) {
    const float c = 0.7978845608f;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float inner = c * (v + 0.044715f * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

void gelu_standard_inplace(float *x, int n) {
    const float inv_sqrt2 = 0.7071067811865475f;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + erff(v * inv_sqrt2));
    }
}

/* ================================================================
 * Softmax (in-place)
 * ================================================================ */
void softmax_inplace(float *x, int n) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < n; i++)
        if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv_sum;
}

/* ================================================================
 * TTS RoPE (interleaved style)
 * ================================================================ */
void tts_rope_apply(float *q, float *k, const int *position_ids,
                    const float *inv_freq, int seq_len, int num_heads,
                    int head_dim) {
    int half_dim = head_dim / 2;
    for (int s = 0; s < seq_len; s++) {
        float pos = (float)position_ids[s];
        for (int h = 0; h < num_heads; h++) {
            float *qp = q + (s * num_heads + h) * head_dim;
            float *kp = k + (s * num_heads + h) * head_dim;
            for (int d = 0; d < half_dim; d++) {
                float freq = pos * inv_freq[d];
                float cos_val = cosf(freq);
                float sin_val = sinf(freq);

                float q0 = qp[2 * d];
                float q1 = qp[2 * d + 1];
                qp[2 * d]     = q0 * cos_val - q1 * sin_val;
                qp[2 * d + 1] = q1 * cos_val + q0 * sin_val;

                float k0 = kp[2 * d];
                float k1 = kp[2 * d + 1];
                kp[2 * d]     = k0 * cos_val - k1 * sin_val;
                kp[2 * d + 1] = k1 * cos_val + k0 * sin_val;
            }
        }
    }
}

/* ================================================================
 * Codec RoPE (complex multiplication style)
 * ================================================================ */
void codec_rope_apply(float *q, float *k, int offset,
                      float max_period, int num_heads, int seq_len,
                      int head_dim) {
    int half_d = head_dim / 2;
    float log_mp = logf(max_period);
    for (int t = 0; t < seq_len; t++) {
        float ts = (float)(offset + t);
        for (int h = 0; h < num_heads; h++) {
            float *qp = q + (h * seq_len + t) * head_dim;
            float *kp = k + (h * seq_len + t) * head_dim;
            for (int d = 0; d < half_d; d++) {
                float freq = expf((float)d * (-log_mp * 2.0f / (float)head_dim));
                float angle = freq * ts;
                float cos_val = cosf(angle);
                float sin_val = sinf(angle);

                float qr = qp[2 * d], qi = qp[2 * d + 1];
                qp[2 * d]     = qr * cos_val - qi * sin_val;
                qp[2 * d + 1] = qr * sin_val + qi * cos_val;

                float kr = kp[2 * d], ki = kp[2 * d + 1];
                kp[2 * d]     = kr * cos_val - ki * sin_val;
                kp[2 * d + 1] = kr * sin_val + ki * cos_val;
            }
        }
    }
}

/* ================================================================
 * Argmax
 * ================================================================ */
int argmax_f(const float *x, int n) {
    int best = 0;
    float best_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > best_val) { best_val = x[i]; best = i; }
    }
    return best;
}

/* ================================================================
 * RNG (xorshift32)
 * ================================================================ */
float rand_uniform(unsigned int *state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return (float)(*state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

/* ================================================================
 * Repetition penalty
 * ================================================================ */
void apply_repetition_penalty(float *logits, int vocab_size,
                              const int *prev_ids, int prev_len,
                              float penalty) {
    if (penalty == 1.0f || prev_len == 0) return;

    int *unique = (int *)malloc((size_t)prev_len * sizeof(int));
    int unique_len = 0;
    for (int i = 0; i < prev_len; i++) {
        int tid = prev_ids[i];
        if (tid < 0 || tid >= vocab_size) continue;
        int seen = 0;
        for (int j = 0; j < unique_len; j++) {
            if (unique[j] == tid) { seen = 1; break; }
        }
        if (!seen) unique[unique_len++] = tid;
    }

    for (int i = 0; i < unique_len; i++) {
        int tid = unique[i];
        if (logits[tid] > 0.0f)
            logits[tid] /= penalty;
        else
            logits[tid] *= penalty;
    }
    free(unique);
}

/* ================================================================
 * Token sampling with temperature, top-k, top-p
 * ================================================================ */

static void sort_indices_desc(int *indices, const float *scores, int n) {
    for (int i = 0; i < n; i++) indices[i] = i;
    for (int i = 1; i < n; i++) {
        int key = indices[i];
        float key_val = scores[key];
        int j = i - 1;
        while (j >= 0 && scores[indices[j]] < key_val) {
            indices[j + 1] = indices[j];
            j--;
        }
        indices[j + 1] = key;
    }
}

int sample_token(float *logits, int vocab_size,
                 float temperature, int top_k, float top_p,
                 unsigned int *rand_state) {
    if (temperature > 0.0f && temperature != 1.0f) {
        float inv_t = 1.0f / temperature;
        for (int i = 0; i < vocab_size; i++) logits[i] *= inv_t;
    }

    if (top_k > 0 && top_k < vocab_size) {
        float *tmp = (float *)malloc((size_t)vocab_size * sizeof(float));
        memcpy(tmp, logits, (size_t)vocab_size * sizeof(float));
        for (int i = 0; i < top_k; i++) {
            int max_idx = i;
            for (int j = i + 1; j < vocab_size; j++)
                if (tmp[j] > tmp[max_idx]) max_idx = j;
            float t = tmp[i]; tmp[i] = tmp[max_idx]; tmp[max_idx] = t;
        }
        float threshold = tmp[top_k - 1];
        free(tmp);
        for (int i = 0; i < vocab_size; i++)
            if (logits[i] < threshold) logits[i] = -FLT_MAX;
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        int *sorted = (int *)malloc((size_t)vocab_size * sizeof(int));
        float *sorted_scores = (float *)malloc((size_t)vocab_size * sizeof(float));
        sort_indices_desc(sorted, logits, vocab_size);
        for (int i = 0; i < vocab_size; i++)
            sorted_scores[i] = logits[sorted[i]];

        float max_val = sorted_scores[0];
        float sum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            sorted_scores[i] = expf(sorted_scores[i] - max_val);
            sum += sorted_scores[i];
        }
        float inv_sum = 1.0f / sum;

        float cumsum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += sorted_scores[i] * inv_sum;
            if (i > 0 && cumsum > top_p) {
                logits[sorted[i]] = -FLT_MAX;
            }
        }
        free(sorted);
        free(sorted_scores);
    }

    softmax_inplace(logits, vocab_size);

    float r = rand_uniform(rand_state);
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += logits[i];
        if (cumsum >= r) return i;
    }
    return vocab_size - 1;
}
