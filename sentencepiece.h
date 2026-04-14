#ifndef TOKENIZER_SP_H
#define TOKENIZER_SP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct SPTokenizer SPTokenizer;

/* Load SentencePiece .model file. Returns NULL on failure. */
SPTokenizer *sp_load(const char *model_path);

/* Load SentencePiece from memory buffer. */
SPTokenizer *sp_load_mem(const void *data, size_t size);

/* Encode text to token IDs. Returns number of tokens.
 * ids_out must be pre-allocated (max_tokens). */
int sp_encode(const SPTokenizer *sp, const char *text, int *ids_out, int max_tokens);

/* Free tokenizer */
void sp_free(SPTokenizer *sp);

#ifdef __cplusplus
}
#endif

#endif /* TOKENIZER_SP_H */
