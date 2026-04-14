#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "tts.h"
#include "sentencepiece.h"

/* Prompt builder: uses SentencePiece tokenizer to encode text and
 * build the full voice-clone input_ids array.
 *
 * No external template files needed - template strings are hardcoded
 * to match PyTorch's prompting.py exactly. */

typedef struct {
    SPTokenizer *sp;
} PromptBuilder;

/* Initialize with SentencePiece model path */
int prompt_builder_init(PromptBuilder *pb, const char *sp_model_path);
void prompt_builder_free(PromptBuilder *pb);

/* Build voice clone input_ids.
 * text: target text string (will be tokenized internally)
 * ref_codes: (code_frames, 16) audio codes, row-major
 * input_ids_out: (max_seq, 17) pre-allocated, row-major
 * Returns: actual sequence length */
int prompt_build_voice_clone(const PromptBuilder *pb,
                             const char *text,
                             const int *ref_codes, int code_frames,
                             int *input_ids_out, int max_seq);

#endif /* TOKENIZER_H */
