#include "tts.h"
#include "ops.h"
#include "platform.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

/* ================================================================
 * KV Cache management
 * ================================================================ */

static void kv_init(KVCache *kv, int capacity) {
    int elem = capacity * TTS_NUM_HEADS * TTS_HEAD_DIM;
    kv->key   = (float *)calloc((size_t)elem, sizeof(float));
    kv->value = (float *)calloc((size_t)elem, sizeof(float));
    kv->seq_len  = 0;
    kv->capacity = capacity;
}

static void kv_free(KVCache *kv) {
    free(kv->key);   kv->key = NULL;
    free(kv->value); kv->value = NULL;
}

static void kv_append(KVCache *kv, const float *new_k, const float *new_v) {
    if (kv->seq_len >= kv->capacity) {
        kv->capacity = kv->capacity * 2 + 64;
        int elem = kv->capacity * TTS_NUM_HEADS * TTS_HEAD_DIM;
        kv->key   = (float *)realloc(kv->key,   (size_t)elem * sizeof(float));
        kv->value = (float *)realloc(kv->value,  (size_t)elem * sizeof(float));
    }
    int stride = TTS_NUM_HEADS * TTS_HEAD_DIM;
    memcpy(kv->key   + kv->seq_len * stride, new_k, (size_t)stride * sizeof(float));
    memcpy(kv->value + kv->seq_len * stride, new_v, (size_t)stride * sizeof(float));
    kv->seq_len++;
}

/* ================================================================
 * Init / Free
 * ================================================================ */

void sampling_params_default(SamplingParams *p) {
    p->do_sample = 1;
    p->max_new_frames = 375;
    p->text_temperature = 1.0f;
    p->text_top_p = 1.0f;
    p->text_top_k = 50;
    p->audio_temperature = 0.8f;
    p->audio_top_p = 0.95f;
    p->audio_top_k = 25;
    p->audio_rep_penalty = 1.2f;
    p->seed = 42;
}

static void cache_block_weights(const WeightStore *ws, const char *prefix,
                                int layer_idx, BlockWeights *bw) {
    char n[MAX_NAME_LEN];
    /* All weights are now fp32 in memory (bf16 converted at load time) */
    snprintf(n,sizeof(n),"%s.h.%d.ln_1.weight",prefix,layer_idx);     bw->ln1_w = weights_must_get(ws,n)->data;
    snprintf(n,sizeof(n),"%s.h.%d.ln_1.bias",prefix,layer_idx);       bw->ln1_b = weights_must_get(ws,n)->data;
    snprintf(n,sizeof(n),"%s.h.%d.attn.c_attn.weight",prefix,layer_idx); bw->attn_w = weights_must_get(ws,n)->data;
    snprintf(n,sizeof(n),"%s.h.%d.attn.c_attn.bias",prefix,layer_idx);   bw->attn_b = weights_must_get(ws,n)->data;
    snprintf(n,sizeof(n),"%s.h.%d.attn.c_proj.weight",prefix,layer_idx); bw->proj_w = weights_must_get(ws,n)->data;
    snprintf(n,sizeof(n),"%s.h.%d.attn.c_proj.bias",prefix,layer_idx);   bw->proj_b = weights_must_get(ws,n)->data;
    snprintf(n,sizeof(n),"%s.h.%d.ln_2.weight",prefix,layer_idx);     bw->ln2_w = weights_must_get(ws,n)->data;
    snprintf(n,sizeof(n),"%s.h.%d.ln_2.bias",prefix,layer_idx);       bw->ln2_b = weights_must_get(ws,n)->data;
    snprintf(n,sizeof(n),"%s.h.%d.mlp.fc_in.weight",prefix,layer_idx);  bw->fc_in_w = weights_must_get(ws,n)->data;
    snprintf(n,sizeof(n),"%s.h.%d.mlp.fc_in.bias",prefix,layer_idx);    bw->fc_in_b = weights_must_get(ws,n)->data;
    snprintf(n,sizeof(n),"%s.h.%d.mlp.fc_out.weight",prefix,layer_idx); bw->fc_out_w = weights_must_get(ws,n)->data;
    snprintf(n,sizeof(n),"%s.h.%d.mlp.fc_out.bias",prefix,layer_idx);   bw->fc_out_b = weights_must_get(ws,n)->data;
    bw->attn_w_bf16 = NULL; bw->proj_w_bf16 = NULL;
    bw->fc_in_w_bf16 = NULL; bw->fc_out_w_bf16 = NULL;
    snprintf(n,sizeof(n),"%s.h.%d.attn.rotary_emb.inv_freq",prefix,layer_idx);
    const Tensor *t = weights_get(ws, n);
    bw->inv_freq = t ? t->data : NULL;
}

void tts_init(TTSModel *m, const WeightStore *ws) {
    m->ws = ws;
    m->total_seq_len = 0;
    m->global_hidden = (float *)calloc(TTS_HIDDEN, sizeof(float));
    for (int i = 0; i < TTS_NUM_LAYERS; i++)
        kv_init(&m->kv[i], 512);
    m->history_cap = 512;
    m->history_len = 0;
    m->audio_history = (int *)calloc((size_t)m->history_cap * TTS_N_VQ, sizeof(int));

    /* Cache all weight pointers once */
    for (int i = 0; i < TTS_NUM_LAYERS; i++)
        cache_block_weights(ws, "tts.transformer", i, &m->global_blocks[i]);
    for (int i = 0; i < TTS_LOCAL_LAYERS; i++)
        cache_block_weights(ws, "tts.local_transformer", i, &m->local_blocks[i]);
    m->global_ln_w = weights_must_get(ws, "tts.transformer.ln_f.weight")->data;
    m->global_ln_b = weights_must_get(ws, "tts.transformer.ln_f.bias")->data;
    m->local_ln_w = weights_must_get(ws, "tts.local_transformer.ln_f.weight")->data;
    m->local_ln_b = weights_must_get(ws, "tts.local_transformer.ln_f.bias")->data;
    m->wte = weights_must_get(ws, "tts.transformer.wte.weight")->data;
    for (int i = 0; i < TTS_N_VQ; i++) {
        char n[MAX_NAME_LEN];
        snprintf(n, sizeof(n), "tts.audio_embeddings.%d.weight", i);
        m->audio_embeds[i] = weights_must_get(ws, n)->data;
    }
    /* Local KV cache */
    for (int i = 0; i < TTS_LOCAL_LAYERS; i++)
        kv_init(&m->local_kv[i], 20);
    m->local_kv_initialized = 0;

    /* Pre-allocate workspace (sized for max expected seq_len) */
    int ws_max = 512; /* covers prefill up to 512 tokens */
    m->ws_max_seq = ws_max;
    m->ws_normed  = (float *)malloc((size_t)ws_max * TTS_HIDDEN * sizeof(float));
    m->ws_qkv     = (float *)malloc((size_t)ws_max * 3 * TTS_HIDDEN * sizeof(float));
    m->ws_Q       = (float *)malloc((size_t)ws_max * TTS_HIDDEN * sizeof(float));
    m->ws_K       = (float *)malloc((size_t)ws_max * TTS_HIDDEN * sizeof(float));
    m->ws_V       = (float *)malloc((size_t)ws_max * TTS_HIDDEN * sizeof(float));
    m->ws_attn_out= (float *)calloc((size_t)ws_max * TTS_HIDDEN, sizeof(float));
    m->ws_proj    = (float *)malloc((size_t)ws_max * TTS_HIDDEN * sizeof(float));
    m->ws_fc      = (float *)malloc((size_t)ws_max * TTS_MLP_DIM * sizeof(float));
    m->ws_mlp     = (float *)malloc((size_t)ws_max * TTS_HIDDEN * sizeof(float));
}

void tts_free(TTSModel *m) {
    free(m->global_hidden);
    free(m->audio_history);
    for (int i = 0; i < TTS_NUM_LAYERS; i++)
        kv_free(&m->kv[i]);
    for (int i = 0; i < TTS_LOCAL_LAYERS; i++)
        kv_free(&m->local_kv[i]);
    free(m->ws_normed); free(m->ws_qkv);
    free(m->ws_Q); free(m->ws_K); free(m->ws_V);
    free(m->ws_attn_out); free(m->ws_proj);
    free(m->ws_fc); free(m->ws_mlp);
}

/* ================================================================
 * Build input embeddings
 * ================================================================ */

void tts_build_embeds(const TTSModel *m, const int *input_ids, int seq_len, float *embeds) {
    const float *wte = m->wte;
    for (int s = 0; s < seq_len; s++) {
        int tid = input_ids[s * 17 + 0];
        memcpy(embeds + s * TTS_HIDDEN, wte + tid * TTS_HIDDEN, (size_t)TTS_HIDDEN * sizeof(float));
    }
    for (int ch = 0; ch < TTS_N_VQ; ch++) {
        const float *ae = m->audio_embeds[ch];
        for (int s = 0; s < seq_len; s++) {
            int aid = input_ids[s * 17 + ch + 1];
            if (aid == TTS_AUDIO_PAD) continue;
            vec_add_scaled_vec(embeds + s * TTS_HIDDEN, ae + aid * TTS_HIDDEN, 1.0f, TTS_HIDDEN);
        }
    }
}

/* ================================================================
 * Generic attention + MLP block using cached BlockWeights
 * ================================================================ */

/* Thread worker for parallel head computation */
typedef struct {
    const float *Q, *K_full, *V_full;
    float *attn_out;
    int h, q_len, kv_len, H, NH, HD;
    float scale;
} TTSHeadArgs;

static void *tts_compute_head(void *arg) {
    TTSHeadArgs *a = (TTSHeadArgs *)arg;
    int q_len = a->q_len, kv_len = a->kv_len, h = a->h;
    int NH = a->NH, HD = a->HD, H = a->H;
    float *scores = (float *)malloc((size_t)q_len * kv_len * sizeof(float));
    for (int qi = 0; qi < q_len; qi++) {
        const float *q_vec = a->Q + qi * NH * HD + h * HD;
        for (int ki = 0; ki < kv_len; ki++)
            scores[qi * kv_len + ki] = dot_f32_vec(q_vec, a->K_full + ki * NH * HD + h * HD, HD) * a->scale;
    }
    for (int qi = 0; qi < q_len; qi++) {
        int q_pos = qi + (kv_len - q_len);
        for (int ki = q_pos + 1; ki < kv_len; ki++)
            scores[qi * kv_len + ki] = -1e9f;
    }
    for (int qi = 0; qi < q_len; qi++)
        softmax_inplace(scores + qi * kv_len, kv_len);
    for (int qi = 0; qi < q_len; qi++) {
        float *out_vec = a->attn_out + qi * H + h * HD;
        for (int ki = 0; ki < kv_len; ki++)
            vec_add_scaled_vec(out_vec, a->V_full + ki * NH * HD + h * HD, scores[qi * kv_len + ki], HD);
    }
    free(scores);
    return NULL;
}

/* Workspace pointers for zero-alloc run_block */
typedef struct {
    float *normed, *qkv, *Q, *K, *V, *attn_out, *proj, *fc, *mlp;
} BlockWorkspace;

static void run_block(const BlockWeights *bw,
                      const float *hidden_in, float *hidden_out,
                      int q_len, const int *position_ids,
                      KVCache *kv, BlockWorkspace *ws) {
    const int H = TTS_HIDDEN, NH = TTS_NUM_HEADS, HD = TTS_HEAD_DIM;
    int use_heap = (ws == NULL);
    float *normed, *qkv, *Q, *K, *V, *attn_out, *proj, *fc, *mlp;

    if (use_heap) {
        normed   = (float *)malloc((size_t)q_len * H * sizeof(float));
        qkv      = (float *)malloc((size_t)q_len * 3 * H * sizeof(float));
        Q        = (float *)malloc((size_t)q_len * H * sizeof(float));
        K        = (float *)malloc((size_t)q_len * H * sizeof(float));
        V        = (float *)malloc((size_t)q_len * H * sizeof(float));
        attn_out = (float *)calloc((size_t)q_len * H, sizeof(float));
        proj     = (float *)malloc((size_t)q_len * H * sizeof(float));
        fc       = (float *)malloc((size_t)q_len * TTS_MLP_DIM * sizeof(float));
        mlp      = (float *)malloc((size_t)q_len * H * sizeof(float));
    } else {
        normed = ws->normed; qkv = ws->qkv; Q = ws->Q; K = ws->K; V = ws->V;
        attn_out = ws->attn_out; proj = ws->proj; fc = ws->fc; mlp = ws->mlp;
        memset(attn_out, 0, (size_t)q_len * H * sizeof(float));
    }

    memcpy(normed, hidden_in, (size_t)q_len * H * sizeof(float));
    for (int s = 0; s < q_len; s++)
        layer_norm(normed + s * H, bw->ln1_w, bw->ln1_b, H, 1e-5f);

    matmul_t(normed, bw->attn_w, bw->attn_b, qkv, q_len, H, 3 * H);

    for (int s = 0; s < q_len; s++) {
        memcpy(Q + s * NH * HD, qkv + s * 3 * H,         (size_t)H * sizeof(float));
        memcpy(K + s * NH * HD, qkv + s * 3 * H + H,     (size_t)H * sizeof(float));
        memcpy(V + s * NH * HD, qkv + s * 3 * H + 2 * H, (size_t)H * sizeof(float));
    }

    if (bw->inv_freq)
        tts_rope_apply(Q, K, position_ids, bw->inv_freq, q_len, NH, HD);

    const float *K_full, *V_full;
    int kv_len;
    if (kv) {
        for (int s = 0; s < q_len; s++)
            kv_append(kv, K + s * NH * HD, V + s * NH * HD);
        K_full = kv->key; V_full = kv->value; kv_len = kv->seq_len;
    } else {
        K_full = K; V_full = V; kv_len = q_len;
    }

    float scale = 1.0f / sqrtf((float)HD);
    if (q_len >= 16 && NH >= 2) {
        thread_t threads[16];
        TTSHeadArgs args[16];
        for (int h = 0; h < NH; h++) {
            args[h] = (TTSHeadArgs){Q, K_full, V_full, attn_out, h, q_len, kv_len, H, NH, HD, scale};
            thread_create(&threads[h], tts_compute_head, &args[h]);
        }
        for (int h = 0; h < NH; h++) thread_join(threads[h]);
    } else {
        for (int h = 0; h < NH; h++) {
            TTSHeadArgs a = {Q, K_full, V_full, attn_out, h, q_len, kv_len, H, NH, HD, scale};
            tts_compute_head(&a);
        }
    }

    matmul_t(attn_out, bw->proj_w, bw->proj_b, proj, q_len, H, H);
    for (int i = 0; i < q_len * H; i++)
        hidden_out[i] = hidden_in[i] + proj[i];

    /* MLP */
    memcpy(normed, hidden_out, (size_t)q_len * H * sizeof(float));
    for (int s = 0; s < q_len; s++)
        layer_norm(normed + s * H, bw->ln2_w, bw->ln2_b, H, 1e-5f);

    fused_ffn_gelu_new(normed, bw->fc_in_w, bw->fc_in_b,
                       bw->fc_out_w, bw->fc_out_b,
                       mlp, fc, q_len, H, TTS_MLP_DIM, H);

    for (int i = 0; i < q_len * H; i++)
        hidden_out[i] += mlp[i];

    if (use_heap) {
        free(normed); free(qkv); free(Q); free(K); free(V);
        free(attn_out); free(proj); free(fc); free(mlp);
    }
}

/* ================================================================
 * Query attention mask
 * ================================================================ */

static void apply_query_mask(float *hidden, const int *mask, int seq_len, int H) {
    if (!mask) return;
    for (int s = 0; s < seq_len; s++)
        if (!mask[s])
            memset(hidden + s * H, 0, (size_t)H * sizeof(float));
}

/* ================================================================
 * Global transformer forward
 * ================================================================ */

static void tts_transformer_forward(TTSModel *m, const float *embeds, int seq_len,
                                    const int *position_ids,
                                    const int *attention_mask) {
    const int H = TTS_HIDDEN;
    float *buf_a = (float *)malloc((size_t)seq_len * H * sizeof(float));
    float *buf_b = (float *)malloc((size_t)seq_len * H * sizeof(float));
    memcpy(buf_a, embeds, (size_t)seq_len * H * sizeof(float));
    apply_query_mask(buf_a, attention_mask, seq_len, H);

    BlockWorkspace bws = {m->ws_normed, m->ws_qkv, m->ws_Q, m->ws_K, m->ws_V,
                          m->ws_attn_out, m->ws_proj, m->ws_fc, m->ws_mlp};
    BlockWorkspace *bws_ptr = (seq_len <= m->ws_max_seq) ? &bws : NULL;

    for (int l = 0; l < TTS_NUM_LAYERS; l++) {
        run_block(&m->global_blocks[l], buf_a, buf_b, seq_len, position_ids, &m->kv[l], bws_ptr);
        float *tmp = buf_a; buf_a = buf_b; buf_b = tmp;
        apply_query_mask(buf_a, attention_mask, seq_len, H);
    }

    for (int s = 0; s < seq_len; s++)
        layer_norm(buf_a + s * H, m->global_ln_w, m->global_ln_b, H, 1e-5f);
    apply_query_mask(buf_a, attention_mask, seq_len, H);

    memcpy(m->global_hidden, buf_a + (seq_len - 1) * H, (size_t)H * sizeof(float));
    m->total_seq_len += seq_len;
    free(buf_a);
    free(buf_b);
}

/* ================================================================
 * Local transformer with KV cache (incremental decode)
 * Reset cache, then call with 1 token at a time.
 * ================================================================ */

static void tts_local_reset_kv(TTSModel *m) {
    for (int l = 0; l < TTS_LOCAL_LAYERS; l++)
        m->local_kv[l].seq_len = 0;
}

static void tts_local_forward_cached(TTSModel *m, const float *token_embed,
                                     int position, float *out) {
    const int H = TTS_HIDDEN;
    float buf_a[TTS_HIDDEN], buf_b[TTS_HIDDEN];
    memcpy(buf_a, token_embed, (size_t)H * sizeof(float));

    int pos = position;
    for (int l = 0; l < TTS_LOCAL_LAYERS; l++) {
        run_block(&m->local_blocks[l], buf_a, buf_b, 1, &pos, &m->local_kv[l], NULL);
        memcpy(buf_a, buf_b, (size_t)H * sizeof(float));
    }

    layer_norm(buf_a, m->local_ln_w, m->local_ln_b, H, 1e-5f);
    memcpy(out, buf_a, (size_t)H * sizeof(float));
}

/* ================================================================
 * Prefill / Decode step
 * ================================================================ */

void tts_prefill(TTSModel *m, const int *input_ids, int seq_len) {
    float *embeds = (float *)calloc((size_t)seq_len * TTS_HIDDEN, sizeof(float));
    tts_build_embeds(m, input_ids, seq_len, embeds);
    int *pos = (int *)malloc((size_t)seq_len * sizeof(int));
    int *mask = (int *)malloc((size_t)seq_len * sizeof(int));
    for (int i = 0; i < seq_len; i++) { pos[i] = i; mask[i] = 1; }
    tts_transformer_forward(m, embeds, seq_len, pos, mask);
    free(embeds); free(pos); free(mask);
}

void tts_decode_step(TTSModel *m, const int *row17) {
    float embeds[TTS_HIDDEN] = {0};
    tts_build_embeds(m, row17, 1, embeds);
    int pos = m->total_seq_len;
    int mask = 1;
    tts_transformer_forward(m, embeds, 1, &pos, &mask);
}

/* ================================================================
 * Generate one frame
 * ================================================================ */

int tts_generate_frame(TTSModel *m, int *audio_tokens_out, const SamplingParams *sp) {
    const int H = TTS_HIDDEN;
    const float *wte = m->wte;
    unsigned int rng = sp->seed + (unsigned int)m->history_len * 1337u;

    /* 1. Reset local KV cache for this frame */
    tts_local_reset_kv(m);

    /* Text token prediction: feed global_hidden as position 0 */
    float local_hidden[TTS_HIDDEN];
    tts_local_forward_cached(m, m->global_hidden, 0, local_hidden);

    float slot_score = dot_f32_vec(local_hidden, wte + TOK_AUDIO_ASST * H, H);
    float end_score  = dot_f32_vec(local_hidden, wte + TOK_AUDIO_END * H, H);
    float two[2] = {slot_score, end_score};
    int text_choice;
    if (sp->do_sample) {
        text_choice = sample_token(two, 2, sp->text_temperature,
                                   sp->text_top_k, sp->text_top_p, &rng);
    } else {
        text_choice = argmax_f(two, 2);
    }
    if (text_choice == 1) return 0;

    /* 2. Generate 16 codebook tokens using cached local transformer */
    /* Feed slot token embedding as position 1 */
    tts_local_forward_cached(m, wte + TOK_AUDIO_ASST * H, 1, local_hidden);

    for (int ch = 0; ch < TTS_N_VQ; ch++) {
        const float *ae = m->audio_embeds[ch];

        float *logits = (float *)malloc((size_t)TTS_AUDIO_CODEBOOK_SIZE * sizeof(float));
        matmul_t(local_hidden, ae, NULL, logits, 1, H, TTS_AUDIO_CODEBOOK_SIZE);

        int tok;
        if (sp->do_sample) {
            if (sp->audio_rep_penalty != 1.0f && m->history_len > 0) {
                int *prev = (int *)malloc((size_t)m->history_len * sizeof(int));
                for (int f = 0; f < m->history_len; f++)
                    prev[f] = m->audio_history[f * TTS_N_VQ + ch];
                apply_repetition_penalty(logits, TTS_AUDIO_CODEBOOK_SIZE,
                                         prev, m->history_len, sp->audio_rep_penalty);
                free(prev);
            }
            tok = sample_token(logits, TTS_AUDIO_CODEBOOK_SIZE,
                               sp->audio_temperature, sp->audio_top_k,
                               sp->audio_top_p, &rng);
        } else {
            tok = argmax_f(logits, TTS_AUDIO_CODEBOOK_SIZE);
        }
        free(logits);
        audio_tokens_out[ch] = tok;

        /* Feed this codebook's embedding for next position */
        if (ch < TTS_N_VQ - 1) {
            tts_local_forward_cached(m, ae + tok * H, 2 + ch, local_hidden);
        }
    }

    /* 3. Record history */
    if (m->history_len >= m->history_cap) {
        m->history_cap *= 2;
        m->audio_history = (int *)realloc(m->audio_history,
                                          (size_t)m->history_cap * TTS_N_VQ * sizeof(int));
    }
    for (int ch = 0; ch < TTS_N_VQ; ch++)
        m->audio_history[m->history_len * TTS_N_VQ + ch] = audio_tokens_out[ch];
    m->history_len++;

    /* 4. Global decode step */
    int row[17];
    row[0] = TOK_AUDIO_ASST;
    for (int ch = 0; ch < TTS_N_VQ; ch++) row[ch + 1] = audio_tokens_out[ch];
    tts_decode_step(m, row);

    return 1;
}
