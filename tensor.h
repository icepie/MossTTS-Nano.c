#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <stdlib.h>

#define MAX_DIMS 4
#define MAX_TENSORS 600
#define MAX_NAME_LEN 128

typedef struct {
    float *data;         /* FP32 data (NULL if raw) */
    uint16_t *data_bf16; /* bf16 data (unused, kept for compat) */
    uint8_t *data_raw;   /* raw bytes (for tokenizer etc.) */
    int dtype;           /* 0=fp32, 1=bf16, 2=raw */
    int ndim;
    int shape[MAX_DIMS];
    int size; /* total number of elements */
} Tensor;

/* Weight storage: name -> Tensor mapping */
typedef struct {
    char name[MAX_NAME_LEN];
    Tensor t;
} NamedTensor;

typedef struct {
    NamedTensor entries[MAX_TENSORS];
    int count;
} WeightStore;

/* Create / free tensors */
Tensor tensor_alloc(int ndim, const int *shape);
Tensor tensor_zeros(int ndim, const int *shape);
void   tensor_free(Tensor *t);

/* Load all weights from binary file */
int weights_load(WeightStore *ws, const char *path);

/* Load all weights from memory buffer */
int weights_load_mem(WeightStore *ws, const void *data, size_t size);
void weights_free(WeightStore *ws);

/* Find a tensor by name (returns NULL if not found) */
const Tensor *weights_get(const WeightStore *ws, const char *name);

/* Convenience: get or abort */
const Tensor *weights_must_get(const WeightStore *ws, const char *name);

#endif /* TENSOR_H */
