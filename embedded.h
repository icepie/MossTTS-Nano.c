#ifndef EMBEDDED_H
#define EMBEDDED_H

#include <stdint.h>

#ifdef EMBED_WEIGHTS
/* Symbols defined by assembly .incbin */
extern const char weights_data[];
extern const uint64_t weights_size;
extern const char tokenizer_data[];
extern const uint64_t tokenizer_size;
#endif

#endif
