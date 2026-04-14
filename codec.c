#include "codec.h"
#include "ops.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "platform.h"


/* ================================================================
 * PatchedPretransform
 * ================================================================ */

/* Encode: (D, T) -> (D*ps, T/ps) */
static void patch_encode(const float *in, float *out, int D, int T, int ps) {
    int T2 = T / ps;
    /* in[d][t] -> out[d*ps + p][t/ps] where t = (t/ps)*ps + p */
    for (int d = 0; d < D; d++) {
        for (int t2 = 0; t2 < T2; t2++) {
            for (int p = 0; p < ps; p++) {
                out[(d * ps + p) * T2 + t2] = in[d * T + t2 * ps + p];
            }
        }
    }
}

/* Decode: (D*ps, T) -> (D, T*ps) */
static void patch_decode(const float *in, float *out, int DH, int T, int ps) {
    int D = DH / ps;
    int T2 = T * ps;
    memset(out, 0, (size_t)D * T2 * sizeof(float));
    for (int d = 0; d < D; d++) {
        for (int t = 0; t < T; t++) {
            for (int p = 0; p < ps; p++) {
                out[d * T2 + t * ps + p] = in[(d * ps + p) * T + t];
            }
        }
    }
}

/* ================================================================
 * Codec attention head worker (for pthread)
 * ================================================================ */

typedef struct {
    const float *Q, *K, *V;
    float *attn_out;
    int h, T, D, HD, valid_len, context;
    float scale;
} CodecHeadArgs;

static void *codec_compute_head(void *arg) {
    CodecHeadArgs *a = (CodecHeadArgs *)arg;
    int T = a->T, HD = a->HD, D = a->D, h = a->h;
    int ctx = a->context;
    int valid_len = a->valid_len;

    /* Context window optimization:
     * Each query qi only attends to keys in range [max(0, qi-ctx+1), qi] (causal + context).
     * Instead of computing full T×T and masking, only compute the valid window.
     * For ctx=1600, T=3040: saves ~47% of dot products. */
    if (ctx > 0 && ctx < T) {
        /* Windowed attention: allocate only (T, ctx) scores */
        float *scores = (float *)malloc((size_t)T * ctx * sizeof(float));

        for (int qi = 0; qi < T; qi++) {
            int ki_start = qi - ctx + 1;
            if (ki_start < 0) ki_start = 0;
            int ki_end = qi + 1; /* causal: can attend up to qi */
            if (ki_end > valid_len) ki_end = valid_len;

            const float *q_vec = a->Q + (h * T + qi) * HD;

            /* Fill scores for valid window */
            int window_size = ki_end - ki_start;
            for (int w = 0; w < ctx; w++) {
                int ki = ki_start + w;
                if (w < window_size && ki >= 0 && ki < T)
                    scores[qi * ctx + w] = dot_f32_vec(q_vec, a->K + (h * T + ki) * HD, HD) * a->scale;
                else
                    scores[qi * ctx + w] = -1e9f;
            }
            softmax_inplace(scores + qi * ctx, ctx);

            /* Weighted sum */
            if (qi < valid_len) {
                float *out_vec = a->attn_out + qi * D + h * HD;
                for (int w = 0; w < window_size; w++) {
                    int ki = ki_start + w;
                    vec_add_scaled_vec(out_vec, a->V + (h * T + ki) * HD, scores[qi * ctx + w], HD);
                }
            }
        }
        free(scores);
    } else {
        /* Full attention (no context window or context >= T) */
        float *scores = (float *)malloc((size_t)T * T * sizeof(float));
        for (int qi = 0; qi < T; qi++) {
            const float *q_vec = a->Q + (h * T + qi) * HD;
            for (int ki = 0; ki < T; ki++)
                scores[qi * T + ki] = dot_f32_vec(q_vec, a->K + (h * T + ki) * HD, HD) * a->scale;
        }
        for (int qi = 0; qi < T; qi++) {
            for (int ki = 0; ki < T; ki++) {
                int delta = qi - ki;
                if (!(ki < valid_len && delta >= 0))
                    scores[qi * T + ki] = -1e9f;
            }
        }
        for (int qi = 0; qi < T; qi++)
            softmax_inplace(scores + qi * T, T);
        for (int qi = 0; qi < T; qi++) {
            if (qi >= valid_len) continue;
            float *out_vec = a->attn_out + qi * D + h * HD;
            for (int ki = 0; ki < T; ki++)
                vec_add_scaled_vec(out_vec, a->V + (h * T + ki) * HD, scores[qi * T + ki], HD);
        }
        free(scores);
    }
    return NULL;
}

/* ================================================================
 * Codec transformer block (pre-norm + attention + FFN with layer scale)
 * ================================================================ */

typedef struct {
    float *residual, *normed, *qkv, *Q, *K, *V, *attn_out, *proj, *fc, *ffn_out;
} CodecBlockWS;

static void codec_block(const WeightStore *ws, const char *prefix,
                        float *x, int T, int D, int num_heads,
                        const int *input_lengths, int context,
                        CodecBlockWS *cws) {
    int HD = D / num_heads;
    char name[MAX_NAME_LEN];
    int use_heap = (cws == NULL);

    float *residual, *normed, *qkv, *Q, *K, *V, *attn_out, *proj_out, *fc, *ffn_out;
    if (use_heap) {
        residual = (float *)malloc((size_t)T * D * sizeof(float));
        normed   = (float *)malloc((size_t)T * D * sizeof(float));
    } else {
        residual = cws->residual;
        normed = cws->normed;
    }

    /* --- Attention --- */
    memcpy(residual, x, (size_t)T * D * sizeof(float));
    memcpy(normed, x, (size_t)T * D * sizeof(float));
    snprintf(name, sizeof(name), "%s.norm1.weight", prefix);
    const float *n1w = weights_must_get(ws, name)->data;
    snprintf(name, sizeof(name), "%s.norm1.bias", prefix);
    const float *n1b = weights_must_get(ws, name)->data;
    for (int t = 0; t < T; t++)
        layer_norm(normed + t * D, n1w, n1b, D, 1e-5f);

    /* QKV: (T, D) @ (3D, D)^T -> (T, 3D) */
    snprintf(name, sizeof(name), "%s.self_attn.in_proj.weight", prefix);
    const float *in_proj_w = weights_must_get(ws, name)->data; /* (3D, D) */
    if (use_heap) {
        qkv = (float *)malloc((size_t)T * 3 * D * sizeof(float));
        Q = (float *)malloc((size_t)num_heads * T * HD * sizeof(float));
        K = (float *)malloc((size_t)num_heads * T * HD * sizeof(float));
        V = (float *)malloc((size_t)num_heads * T * HD * sizeof(float));
        attn_out = (float *)calloc((size_t)T * D, sizeof(float));
        proj_out = (float *)malloc((size_t)T * D * sizeof(float));
        fc = (float *)malloc((size_t)T * CODEC_FFN_DIM * sizeof(float));
        ffn_out = (float *)malloc((size_t)T * D * sizeof(float));
    } else {
        qkv = cws->qkv; Q = cws->Q; K = cws->K; V = cws->V;
        attn_out = cws->attn_out; proj_out = cws->proj; fc = cws->fc; ffn_out = cws->ffn_out;
        memset(attn_out, 0, (size_t)T * D * sizeof(float));
    }
    matmul_t(normed, in_proj_w, NULL, qkv, T, D, 3 * D);

    /* Split and reshape to (num_heads, T, HD) */
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < num_heads; h++) {
            memcpy(Q + (h * T + t) * HD, qkv + t * 3 * D + h * HD,         (size_t)HD * sizeof(float));
            memcpy(K + (h * T + t) * HD, qkv + t * 3 * D + D + h * HD,     (size_t)HD * sizeof(float));
            memcpy(V + (h * T + t) * HD, qkv + t * 3 * D + 2 * D + h * HD, (size_t)HD * sizeof(float));
        }
    }

    /* Codec RoPE */
    codec_rope_apply(Q, K, 0, CODEC_MAX_PERIOD, num_heads, T, HD);

    /* Attention */
    float scale = 1.0f / sqrtf((float)HD);
    int valid_len = input_lengths[0];

    if (T >= 256 && num_heads >= 2) {
        thread_t threads[16];
        CodecHeadArgs args[16];
        for (int h = 0; h < num_heads; h++) {
            args[h] = (CodecHeadArgs){Q, K, V, attn_out, h, T, D, HD, valid_len, context, scale};
            thread_create(&threads[h], codec_compute_head, &args[h]);
        }
        for (int h = 0; h < num_heads; h++)
            thread_join(threads[h]);
    } else {
        for (int h = 0; h < num_heads; h++) {
            CodecHeadArgs a = {Q, K, V, attn_out, h, T, D, HD, valid_len, context, scale};
            codec_compute_head(&a);
        }
    }
    /* Output projection */
    snprintf(name, sizeof(name), "%s.self_attn.out_proj.weight", prefix);
    const float *out_proj_w = weights_must_get(ws, name)->data;
    matmul_t(attn_out, out_proj_w, NULL, proj_out, T, D, D);

    /* Layer scale 1 + residual */
    snprintf(name, sizeof(name), "%s.layer_scale_1.scale", prefix);
    const float *ls1 = weights_must_get(ws, name)->data;
    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++)
            x[t * D + d] = residual[t * D + d] + ls1[d] * proj_out[t * D + d];

    /* --- FFN --- */
    memcpy(residual, x, (size_t)T * D * sizeof(float));
    memcpy(normed, x, (size_t)T * D * sizeof(float));
    snprintf(name, sizeof(name), "%s.norm2.weight", prefix);
    const float *n2w = weights_must_get(ws, name)->data;
    snprintf(name, sizeof(name), "%s.norm2.bias", prefix);
    const float *n2b = weights_must_get(ws, name)->data;
    for (int t = 0; t < T; t++)
        layer_norm(normed + t * D, n2w, n2b, D, 1e-5f);

    snprintf(name, sizeof(name), "%s.ffn.0.weight", prefix);
    const float *ffn0_w = weights_must_get(ws, name)->data;
    snprintf(name, sizeof(name), "%s.ffn.2.weight", prefix);
    const float *ffn2_w = weights_must_get(ws, name)->data;
    fused_ffn_gelu_standard(normed, ffn0_w, NULL, ffn2_w, NULL,
                            ffn_out, fc, T, D, CODEC_FFN_DIM, D);

    /* Layer scale 2 + residual */
    snprintf(name, sizeof(name), "%s.layer_scale_2.scale", prefix);
    const float *ls2 = weights_must_get(ws, name)->data;
    for (int t = 0; t < T; t++)
        for (int d = 0; d < D; d++)
            x[t * D + d] = residual[t * D + d] + ls2[d] * ffn_out[t * D + d];

    if (use_heap) {
        free(residual); free(normed); free(qkv); free(Q); free(K); free(V);
        free(attn_out); free(proj_out); free(fc); free(ffn_out);
    }
}

/* ================================================================
 * Codec projected transformer
 * x: (D_in, T) conv layout
 * -> input_proj -> transformer layers -> output_proj
 * -> (D_out, T) conv layout
 * ================================================================ */

static void codec_projected_transformer(const WeightStore *ws, const char *prefix,
                                        float *x_in, int D_in, int T,
                                        float *x_out, int D_out,
                                        int num_layers, int num_heads,
                                        const int *input_lengths, int context) {
    char name[MAX_NAME_LEN];
    int D = CODEC_D_MODEL;

    /* input_proj: (T, D_in) @ (D, D_in)^T -> (T, D) */
    snprintf(name, sizeof(name), "%s.input_proj.weight", prefix);
    const float *ip_w = weights_must_get(ws, name)->data; /* (D, D_in) */

    /* Transpose x from (D_in, T) to (T, D_in) */
    float *x_t = (float *)malloc((size_t)T * D_in * sizeof(float));
    for (int d = 0; d < D_in; d++)
        for (int t = 0; t < T; t++)
            x_t[t * D_in + d] = x_in[d * T + t];

    float *h = (float *)malloc((size_t)T * D * sizeof(float));
    matmul_t(x_t, ip_w, NULL, h, T, D_in, D);
    free(x_t);

    /* Allocate workspace once, reuse across layers */
    CodecBlockWS cws;
    cws.residual = (float *)malloc((size_t)T * D * sizeof(float));
    cws.normed   = (float *)malloc((size_t)T * D * sizeof(float));
    cws.qkv      = (float *)malloc((size_t)T * 3 * D * sizeof(float));
    cws.Q        = (float *)malloc((size_t)num_heads * T * (D/num_heads) * sizeof(float));
    cws.K        = (float *)malloc((size_t)num_heads * T * (D/num_heads) * sizeof(float));
    cws.V        = (float *)malloc((size_t)num_heads * T * (D/num_heads) * sizeof(float));
    cws.attn_out = (float *)malloc((size_t)T * D * sizeof(float));
    cws.proj     = (float *)malloc((size_t)T * D * sizeof(float));
    cws.fc       = (float *)malloc((size_t)T * CODEC_FFN_DIM * sizeof(float));
    cws.ffn_out  = (float *)malloc((size_t)T * D * sizeof(float));

    for (int l = 0; l < num_layers; l++) {
        char layer_prefix[MAX_NAME_LEN];
        snprintf(layer_prefix, sizeof(layer_prefix), "%s.transformer.layers.%d", prefix, l);
        codec_block(ws, layer_prefix, h, T, D, num_heads, input_lengths, context, &cws);
    }

    free(cws.residual); free(cws.normed); free(cws.qkv);
    free(cws.Q); free(cws.K); free(cws.V);
    free(cws.attn_out); free(cws.proj); free(cws.fc); free(cws.ffn_out);

    /* output_proj: (T, D) @ (D_out, D)^T -> (T, D_out) */
    snprintf(name, sizeof(name), "%s.output_proj.weight", prefix);
    const float *op_w = weights_must_get(ws, name)->data; /* (D_out, D) */
    float *out_t = (float *)malloc((size_t)T * D_out * sizeof(float));
    matmul_t(h, op_w, NULL, out_t, T, D, D_out);
    free(h);

    /* Transpose back: (T, D_out) -> (D_out, T) */
    for (int d = 0; d < D_out; d++)
        for (int t = 0; t < T; t++)
            x_out[d * T + t] = out_t[t * D_out + d];
    free(out_t);
}

/* ================================================================
 * RLFQ Quantize (encode)
 * ================================================================ */

static void rlfq_quantize(const WeightStore *ws, const float *encoder_out,
                          int D, int T, const int *input_lengths,
                          int *codes_out) {
    char name[MAX_NAME_LEN];
    /* input_proj: Conv1d(D, 512, 1) -> z (512, T) */
    snprintf(name, sizeof(name), "codec.quantizer.input_proj.weight");
    const float *ip_w = weights_must_get(ws, name)->data; /* (512, D, 1) */
    snprintf(name, sizeof(name), "codec.quantizer.input_proj.bias");
    const float *ip_b = weights_must_get(ws, name)->data;

    /* Transpose encoder_out (D, T) -> (T, D) for cache-friendly access */
    float *enc_t = (float *)malloc((size_t)T * D * sizeof(float));
    for (int i = 0; i < D; i++)
        for (int t = 0; t < T; t++)
            enc_t[t * D + i] = encoder_out[i * T + t];

    float *z = (float *)malloc((size_t)CODEC_RVQ_DIM * T * sizeof(float));
    /* Conv1d kernel=1 using matmul: z_t(T,512) = enc_t(T,D) @ ip_w(512,D)^T + ip_b */
    float *z_t = (float *)malloc((size_t)T * CODEC_RVQ_DIM * sizeof(float));
    matmul_t(enc_t, ip_w, ip_b, z_t, T, D, CODEC_RVQ_DIM);
    free(enc_t);
    /* Transpose z_t (T, 512) -> z (512, T) */
    for (int o = 0; o < CODEC_RVQ_DIM; o++)
        for (int t = 0; t < T; t++)
            z[o * T + t] = z_t[t * CODEC_RVQ_DIM + o];
    free(z_t);

    int valid_len = input_lengths[0];

    /* Residual quantization loop */
    float *residual = (float *)malloc((size_t)CODEC_RVQ_DIM * T * sizeof(float));
    memcpy(residual, z, (size_t)CODEC_RVQ_DIM * T * sizeof(float));

    for (int q = 0; q < CODEC_NUM_QUANTIZERS; q++) {
        snprintf(name, sizeof(name), "codec.quantizer.quantizers.%d.in_proj.weight", q);
        const float *qip_w = weights_must_get(ws, name)->data; /* (8, 512, 1) */
        snprintf(name, sizeof(name), "codec.quantizer.quantizers.%d.in_proj.bias", q);
        const float *qip_b = weights_must_get(ws, name)->data;
        snprintf(name, sizeof(name), "codec.quantizer.quantizers.%d.codebook.weight", q);
        const float *cb = weights_must_get(ws, name)->data; /* (1024, 8) */
        snprintf(name, sizeof(name), "codec.quantizer.quantizers.%d.out_proj.weight", q);
        const float *qop_w = weights_must_get(ws, name)->data; /* (512, 8, 1) */
        snprintf(name, sizeof(name), "codec.quantizer.quantizers.%d.out_proj.bias", q);
        const float *qop_b = weights_must_get(ws, name)->data;

        /* Pre-compute normalized codebook entries (once per quantizer) */
        float cb_normed[CODEC_CODEBOOK_SIZE * CODEC_CODEBOOK_DIM];
        for (int c = 0; c < CODEC_CODEBOOK_SIZE; c++) {
            float norm_cb = 0;
            for (int d = 0; d < CODEC_CODEBOOK_DIM; d++)
                norm_cb += cb[c * CODEC_CODEBOOK_DIM + d] * cb[c * CODEC_CODEBOOK_DIM + d];
            float inv_norm = 1.0f / sqrtf(norm_cb + 1e-8f);
            for (int d = 0; d < CODEC_CODEBOOK_DIM; d++)
                cb_normed[c * CODEC_CODEBOOK_DIM + d] = cb[c * CODEC_CODEBOOK_DIM + d] * inv_norm;
        }

        for (int t = 0; t < T; t++) {
            /* in_proj: (8,) from residual at time t */
            float z_e[CODEC_CODEBOOK_DIM];
            for (int d = 0; d < CODEC_CODEBOOK_DIM; d++) {
                float sum = qip_b[d];
                int valid = t < valid_len;
                for (int i = 0; i < CODEC_RVQ_DIM; i++)
                    sum += qip_w[d * CODEC_RVQ_DIM + i] * (valid ? residual[i * T + t] : 0.0f);
                z_e[d] = sum;
            }

            /* L2 normalize z_e */
            float norm_ze = 0;
            for (int d = 0; d < CODEC_CODEBOOK_DIM; d++) norm_ze += z_e[d] * z_e[d];
            float inv_norm_ze = 1.0f / sqrtf(norm_ze + 1e-8f);
            float z_e_norm[CODEC_CODEBOOK_DIM];
            for (int d = 0; d < CODEC_CODEBOOK_DIM; d++) z_e_norm[d] = z_e[d] * inv_norm_ze;

            /* Find nearest codebook (L2 on pre-normalized) */
            int best = 0;
            float best_dist = 1e30f;
            for (int c = 0; c < CODEC_CODEBOOK_SIZE; c++) {
                float dist = 0;
                for (int d = 0; d < CODEC_CODEBOOK_DIM; d++) {
                    float diff = z_e_norm[d] - cb_normed[c * CODEC_CODEBOOK_DIM + d];
                    dist += diff * diff;
                }
                if (dist < best_dist) { best_dist = dist; best = c; }
            }
            codes_out[q * T + t] = best;

            /* Dequantize and update residual */
            float z_q[CODEC_CODEBOOK_DIM];
            for (int d = 0; d < CODEC_CODEBOOK_DIM; d++)
                z_q[d] = cb[best * CODEC_CODEBOOK_DIM + d];

            /* out_proj: (512,) from z_q (8,) */
            if (t < valid_len) {
                for (int o = 0; o < CODEC_RVQ_DIM; o++) {
                    float sum = qop_b[o];
                    for (int d = 0; d < CODEC_CODEBOOK_DIM; d++)
                        sum += qop_w[o * CODEC_CODEBOOK_DIM + d] * z_q[d];
                    residual[o * T + t] -= sum;
                }
            }
        }
    }
    free(residual);
    free(z);
}

/* ================================================================
 * RLFQ Dequantize (decode)
 * ================================================================ */

static void rlfq_dequantize(const WeightStore *ws, const int *codes,
                            int num_q, int T, float *out) {
    char name[MAX_NAME_LEN];
    /* out: (768, T) */
    int out_dim = 768;
    float *emb = (float *)calloc((size_t)CODEC_RVQ_DIM * T, sizeof(float));

    for (int q = 0; q < num_q; q++) {
        snprintf(name, sizeof(name), "codec.quantizer.quantizers.%d.codebook.weight", q);
        const float *cb = weights_must_get(ws, name)->data;
        snprintf(name, sizeof(name), "codec.quantizer.quantizers.%d.out_proj.weight", q);
        const float *qop_w = weights_must_get(ws, name)->data;
        snprintf(name, sizeof(name), "codec.quantizer.quantizers.%d.out_proj.bias", q);
        const float *qop_b = weights_must_get(ws, name)->data;

        for (int t = 0; t < T; t++) {
            int code = codes[q * T + t];
            float z_q[CODEC_CODEBOOK_DIM];
            for (int d = 0; d < CODEC_CODEBOOK_DIM; d++)
                z_q[d] = cb[code * CODEC_CODEBOOK_DIM + d];
            for (int o = 0; o < CODEC_RVQ_DIM; o++) {
                float sum = qop_b[o];
                for (int d = 0; d < CODEC_CODEBOOK_DIM; d++)
                    sum += qop_w[o * CODEC_CODEBOOK_DIM + d] * z_q[d];
                emb[o * T + t] += sum;
            }
        }
    }

    /* output_proj: Conv1d(512, 768, 1) - transpose for cache-friendly matmul */
    snprintf(name, sizeof(name), "codec.quantizer.output_proj.weight");
    const float *op_w = weights_must_get(ws, name)->data;
    snprintf(name, sizeof(name), "codec.quantizer.output_proj.bias");
    const float *op_b = weights_must_get(ws, name)->data;

    float *emb_t = (float *)malloc((size_t)T * CODEC_RVQ_DIM * sizeof(float));
    for (int i = 0; i < CODEC_RVQ_DIM; i++)
        for (int t = 0; t < T; t++)
            emb_t[t * CODEC_RVQ_DIM + i] = emb[i * T + t];
    free(emb);

    float *out_t = (float *)malloc((size_t)T * out_dim * sizeof(float));
    matmul_t(emb_t, op_w, op_b, out_t, T, CODEC_RVQ_DIM, out_dim);
    free(emb_t);

    for (int o = 0; o < out_dim; o++)
        for (int t = 0; t < T; t++)
            out[o * T + t] = out_t[t * out_dim + o];
    free(out_t);
}

/* ================================================================
 * Encoder stages configuration
 * ================================================================ */

typedef struct { int type; int ps; const char *prefix; int nl; float cd; } Stage;

static const Stage ENC_STAGES[] = {
    {0, 240, NULL, 0, 0},
    {1, 0, "codec.encoder.1", 4, 4.0f},
    {0, 2, NULL, 0, 0},
    {1, 0, "codec.encoder.3", 2, 6.0f},
    {0, 2, NULL, 0, 0},
    {1, 0, "codec.encoder.5", 2, 8.0f},
    {0, 2, NULL, 0, 0},
    {1, 0, "codec.encoder.7", 4, 10.0f},
    {0, 4, NULL, 0, 0},
};
#define NUM_ENC_STAGES 9

static const Stage DEC_STAGES[] = {
    {2, 4, NULL, 0, 0},      /* patch_up */
    {1, 0, "codec.decoder.1", 4, 10.0f},
    {2, 2, NULL, 0, 0},
    {1, 0, "codec.decoder.3", 2, 8.0f},
    {2, 2, NULL, 0, 0},
    {1, 0, "codec.decoder.5", 2, 6.0f},
    {2, 2, NULL, 0, 0},
    {1, 0, "codec.decoder.7", 4, 4.0f},
    {2, 240, NULL, 0, 0},
};
#define NUM_DEC_STAGES 9

/* ================================================================
 * Codec Encode
 * ================================================================ */

int codec_encode(const WeightStore *ws,
                 const float *waveform, int channels, int samples,
                 int original_samples,
                 int *codes_out) {
    /* Channel interleave: (C, T) -> (1, C*T) */
    int interleaved_len = samples * channels;
    float *x = (float *)malloc((size_t)interleaved_len * sizeof(float));
    for (int t = 0; t < samples; t++)
        for (int c = 0; c < channels; c++)
            x[t * channels + c] = waveform[c * samples + t];

    int D = 1, T = interleaved_len;
    int input_lengths = original_samples * channels;

    float frame_rate = (float)(CODEC_SAMPLE_RATE * CODEC_NUM_CHANNELS);

    for (int s = 0; s < NUM_ENC_STAGES; s++) {
        if (ENC_STAGES[s].type == 0) {
            /* Patch encode */
            int ps = ENC_STAGES[s].ps;
            int T2 = T / ps;
            int D2 = D * ps;
            float *out = (float *)malloc((size_t)D2 * T2 * sizeof(float));
            patch_encode(x, out, D, T, ps);
            free(x); x = out;
            D = D2; T = T2;
            input_lengths = input_lengths / ps;
            frame_rate /= ps;
        } else {
            /* Transformer */
            char ip_name[MAX_NAME_LEN];
            snprintf(ip_name, sizeof(ip_name), "%s.output_proj.weight", ENC_STAGES[s].prefix);
            int D_out = weights_must_get(ws, ip_name)->shape[0];
            int context = (int)(frame_rate * ENC_STAGES[s].cd + 0.5f);
            float *out = (float *)malloc((size_t)D_out * T * sizeof(float));
            codec_projected_transformer(ws, ENC_STAGES[s].prefix,
                                        x, D, T, out, D_out,
                                        ENC_STAGES[s].nl, CODEC_NUM_HEADS,
                                        &input_lengths, context);
            free(x); x = out;
            D = D_out;
        }
    }

    /* Quantize: x is (D, T) where D=768 */
    rlfq_quantize(ws, x, D, T, &input_lengths, codes_out);
    free(x);
    return T; /* code_frames */
}

/* ================================================================
 * Codec Decode
 * ================================================================ */

int codec_decode(const WeightStore *ws,
                 const int *codes, int num_quantizers, int code_frames,
                 float *waveform_out) {
    int T = code_frames;

    /* Dequantize */
    float *x = (float *)malloc((size_t)768 * T * sizeof(float));
    rlfq_dequantize(ws, codes, num_quantizers, T, x);

    int D = 768;
    int input_lengths = T;
    float frame_rate = (float)(CODEC_SAMPLE_RATE * CODEC_NUM_CHANNELS);
    /* Compute decoder input frame rate */
    for (int s = 0; s < NUM_ENC_STAGES; s++) {
        if (ENC_STAGES[s].type == 0)
            frame_rate /= ENC_STAGES[s].ps;
    }

    for (int s = 0; s < NUM_DEC_STAGES; s++) {
        if (DEC_STAGES[s].type == 2) {
            /* Patch decode (upsample) */
            int ps = DEC_STAGES[s].ps;
            int D2 = D / ps;
            int T2 = T * ps;
            float *out = (float *)malloc((size_t)D2 * T2 * sizeof(float));
            patch_decode(x, out, D, T, ps);
            free(x); x = out;
            D = D2; T = T2;
            input_lengths = input_lengths * ps;
            frame_rate *= ps;
        } else {
            char op_name[MAX_NAME_LEN];
            snprintf(op_name, sizeof(op_name), "%s.output_proj.weight", DEC_STAGES[s].prefix);
            int D_out = weights_must_get(ws, op_name)->shape[0];
            int context = (int)(frame_rate * DEC_STAGES[s].cd + 0.5f);
            float *out = (float *)malloc((size_t)D_out * T * sizeof(float));
            codec_projected_transformer(ws, DEC_STAGES[s].prefix,
                                        x, D, T, out, D_out,
                                        DEC_STAGES[s].nl, CODEC_NUM_HEADS,
                                        &input_lengths, context);
            free(x); x = out;
            D = D_out;
        }
    }

    /* x is now (1, interleaved_T) -- copy to output */
    memcpy(waveform_out, x, (size_t)T * sizeof(float));
    free(x);
    return T;
}
