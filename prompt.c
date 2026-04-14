#include "prompt.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* Template strings matching PyTorch prompting.py exactly */
#define USER_ROLE_PREFIX          "user\n"
#define USER_TMPL_REF_PREFIX      "<user_inst>\n- Reference(s):\n"
#define USER_TMPL_AFTER_REF       "\n- Instruction:\nNone\n- Tokens:\nNone\n- Quality:\nNone\n" \
                                  "- Sound Event:\nNone\n- Ambient Sound:\nNone\n- Language:\nNone\n- Text:\n"
#define USER_TMPL_SUFFIX          "\n</user_inst>"
#define ASSISTANT_TURN_PREFIX     "\n"
#define ASSISTANT_ROLE_PREFIX     "assistant\n"

#define MAX_TOKENS_BUF 512

int prompt_builder_init(PromptBuilder *pb, const char *sp_model_path) {
    pb->sp = sp_load(sp_model_path);
    return pb->sp ? 0 : -1;
}

void prompt_builder_free(PromptBuilder *pb) {
    if (pb->sp) { sp_free(pb->sp); pb->sp = NULL; }
}

/* Helper: encode text, append tokens to output */
static int encode_and_append(const SPTokenizer *sp, const char *text,
                             int *out, int pos, int max_seq) {
    int buf[MAX_TOKENS_BUF];
    int n = sp_encode(sp, text, buf, MAX_TOKENS_BUF);
    if (n < 0) return pos;
    for (int i = 0; i < n && pos < max_seq; i++, pos++) {
        out[pos * 17 + 0] = buf[i];
        for (int c = 1; c <= TTS_N_VQ; c++)
            out[pos * 17 + c] = TTS_AUDIO_PAD;
    }
    return pos;
}

/* Helper: add single token row */
static int add_token(int *out, int pos, int token_id, int max_seq) {
    if (pos >= max_seq) return pos;
    out[pos * 17 + 0] = token_id;
    for (int c = 1; c <= TTS_N_VQ; c++)
        out[pos * 17 + c] = TTS_AUDIO_PAD;
    return pos + 1;
}

/* Helper: add audio reference rows */
static int add_audio_rows(int *out, int pos, const int *codes, int frames, int max_seq) {
    for (int f = 0; f < frames && pos < max_seq; f++, pos++) {
        out[pos * 17 + 0] = TOK_AUDIO_USER;
        for (int c = 0; c < TTS_N_VQ; c++)
            out[pos * 17 + c + 1] = codes[f * TTS_N_VQ + c];
    }
    return pos;
}

int prompt_build_voice_clone(const PromptBuilder *pb,
                             const char *text,
                             const int *ref_codes, int code_frames,
                             int *input_ids_out, int max_seq) {
    int pos = 0;

    /* [IM_START] + "user\n" + "<user_inst>\n- Reference(s):\n" */
    pos = add_token(input_ids_out, pos, TOK_IM_START, max_seq);
    pos = encode_and_append(pb->sp, USER_ROLE_PREFIX, input_ids_out, pos, max_seq);
    pos = encode_and_append(pb->sp, USER_TMPL_REF_PREFIX, input_ids_out, pos, max_seq);

    /* [AUDIO_START] */
    pos = add_token(input_ids_out, pos, TOK_AUDIO_START, max_seq);

    /* Audio reference codes */
    pos = add_audio_rows(input_ids_out, pos, ref_codes, code_frames, max_seq);

    /* [AUDIO_END] */
    pos = add_token(input_ids_out, pos, TOK_AUDIO_END, max_seq);

    /* after_reference template */
    pos = encode_and_append(pb->sp, USER_TMPL_AFTER_REF, input_ids_out, pos, max_seq);

    /* user text */
    pos = encode_and_append(pb->sp, text, input_ids_out, pos, max_seq);

    /* suffix + assistant prefix */
    pos = encode_and_append(pb->sp, USER_TMPL_SUFFIX, input_ids_out, pos, max_seq);
    pos = add_token(input_ids_out, pos, TOK_IM_END, max_seq);
    pos = encode_and_append(pb->sp, ASSISTANT_TURN_PREFIX, input_ids_out, pos, max_seq);
    pos = add_token(input_ids_out, pos, TOK_IM_START, max_seq);
    pos = encode_and_append(pb->sp, ASSISTANT_ROLE_PREFIX, input_ids_out, pos, max_seq);

    /* [AUDIO_START] */
    pos = add_token(input_ids_out, pos, TOK_AUDIO_START, max_seq);

    return pos;
}
