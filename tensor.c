#include "tensor.h"
#include <stdio.h>
#include <string.h>


Tensor tensor_alloc(int ndim, const int *shape) {
    Tensor t;
    t.ndim = ndim;
    t.size = 1;
    for (int i = 0; i < ndim; i++) {
        t.shape[i] = shape[i];
        t.size *= shape[i];
    }
    for (int i = ndim; i < MAX_DIMS; i++) t.shape[i] = 0;
    t.data = (float *)malloc((size_t)t.size * sizeof(float));
    t.data_bf16 = NULL;
    t.data_raw = NULL;
    t.dtype = 0;
    return t;
}

Tensor tensor_zeros(int ndim, const int *shape) {
    Tensor t = tensor_alloc(ndim, shape);
    memset(t.data, 0, (size_t)t.size * sizeof(float));
    return t;
}

void tensor_free(Tensor *t) {
    if (t->data) { free(t->data); t->data = NULL; }
    if (t->data_bf16) { free(t->data_bf16); t->data_bf16 = NULL; }
    if (t->data_raw) { free(t->data_raw); t->data_raw = NULL; }
    t->size = 0;
}

/* ---------- Binary loader ---------- */

static int read_u32(FILE *f, uint32_t *out) {
    return fread(out, 4, 1, f) == 1;
}

int weights_load(WeightStore *ws, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }

    char magic[4];
    uint32_t version, num_tensors;
    fread(magic, 1, 4, f);
    if (memcmp(magic, "MTTS", 4) != 0) {
        fprintf(stderr, "Bad magic\n"); fclose(f); return -1;
    }
    read_u32(f, &version);
    read_u32(f, &num_tensors);
    if (version < 1 || version > 2) {
        fprintf(stderr, "Unsupported version %u\n", version); fclose(f); return -1;
    }

    ws->count = 0;
    for (uint32_t i = 0; i < num_tensors; i++) {
        if (ws->count >= MAX_TENSORS) {
            fprintf(stderr, "Too many tensors (max %d)\n", MAX_TENSORS);
            fclose(f); return -1;
        }
        NamedTensor *nt = &ws->entries[ws->count];

        uint32_t name_len;
        read_u32(f, &name_len);
        if (name_len >= MAX_NAME_LEN) {
            fprintf(stderr, "Name too long: %u\n", name_len);
            fclose(f); return -1;
        }
        fread(nt->name, 1, name_len, f);
        nt->name[name_len] = '\0';

        /* v2: read dtype before ndim */
        uint32_t dtype_id = 0;
        if (version >= 2)
            read_u32(f, &dtype_id);

        uint32_t ndim;
        read_u32(f, &ndim);
        int shape[MAX_DIMS] = {0};
        int size = 1;
        for (uint32_t d = 0; d < ndim; d++) {
            uint32_t dim;
            read_u32(f, &dim);
            shape[d] = (int)dim;
            size *= (int)dim;
        }

        nt->t.ndim = (int)ndim;
        nt->t.size = size;
        for (int d = 0; d < MAX_DIMS; d++) nt->t.shape[d] = shape[d];
        nt->t.dtype = (int)dtype_id;
        nt->t.data_bf16 = NULL;
        nt->t.data_raw = NULL;
        if (dtype_id == 2) {
            /* Raw bytes: store as-is */
            nt->t.data = NULL;
            nt->t.data_raw = (uint8_t *)malloc((size_t)size);
            fread(nt->t.data_raw, 1, (size_t)size, f);
        } else if (dtype_id == 1) {
            /* bfloat16: read as bf16, convert to fp32 for OpenBLAS compatibility */
            uint16_t *bf16 = (uint16_t *)malloc((size_t)size * sizeof(uint16_t));
            fread(bf16, sizeof(uint16_t), (size_t)size, f);
            nt->t.data = (float *)malloc((size_t)size * sizeof(float));
            for (int j = 0; j < size; j++) {
                uint32_t bits = (uint32_t)bf16[j] << 16;
                memcpy(&nt->t.data[j], &bits, sizeof(float));
            }
            free(bf16);
        } else {
            nt->t.data = (float *)malloc((size_t)size * sizeof(float));
            fread(nt->t.data, sizeof(float), (size_t)size, f);
        }

        ws->count++;
    }

    char footer[4];
    fread(footer, 1, 4, f);
    fclose(f);
    if (memcmp(footer, "DONE", 4) != 0) {
        fprintf(stderr, "Bad footer\n"); return -1;
    }

    printf("Loaded %d tensors from %s\n", ws->count, path);
    return 0;
}

void weights_free(WeightStore *ws) {
    for (int i = 0; i < ws->count; i++) {
        if (ws->entries[i].t.data) free(ws->entries[i].t.data);
        if (ws->entries[i].t.data_bf16) free(ws->entries[i].t.data_bf16);
        if (ws->entries[i].t.data_raw) free(ws->entries[i].t.data_raw);
        ws->entries[i].t.data = NULL;
        ws->entries[i].t.data_bf16 = NULL;
        ws->entries[i].t.data_raw = NULL;
    }
    ws->count = 0;
}

/* ---- Memory buffer reader (for embedded weights) ---- */

typedef struct { const uint8_t *p; size_t remaining; } MemReader;

static int mem_read(MemReader *r, void *dst, size_t n) {
    if (r->remaining < n) return -1;
    memcpy(dst, r->p, n); r->p += n; r->remaining -= n;
    return 0;
}
static uint32_t mem_u32(MemReader *r) {
    uint32_t v; mem_read(r, &v, 4); return v;
}

int weights_load_mem(WeightStore *ws, const void *data, size_t total_size) {
    MemReader r = { (const uint8_t *)data, total_size };
    char magic[4]; mem_read(&r, magic, 4);
    if (memcmp(magic, "MTTS", 4) != 0) { fprintf(stderr, "Bad magic\n"); return -1; }
    uint32_t version = mem_u32(&r);
    uint32_t num_tensors = mem_u32(&r);
    if (version < 1 || version > 2) { fprintf(stderr, "Bad version\n"); return -1; }

    ws->count = 0;
    for (uint32_t i = 0; i < num_tensors; i++) {
        if (ws->count >= MAX_TENSORS) { fprintf(stderr, "Too many tensors\n"); return -1; }
        NamedTensor *nt = &ws->entries[ws->count];
        uint32_t name_len = mem_u32(&r);
        if (name_len >= MAX_NAME_LEN) { fprintf(stderr, "Name too long\n"); return -1; }
        mem_read(&r, nt->name, name_len); nt->name[name_len] = '\0';

        uint32_t dtype_id = 0;
        if (version >= 2) dtype_id = mem_u32(&r);

        uint32_t ndim = mem_u32(&r);
        int shape[MAX_DIMS] = {0};
        int size = 1;
        for (uint32_t d = 0; d < ndim; d++) { shape[d] = (int)mem_u32(&r); size *= shape[d]; }

        nt->t.ndim = (int)ndim; nt->t.size = size;
        for (int d = 0; d < MAX_DIMS; d++) nt->t.shape[d] = shape[d];
        nt->t.dtype = (int)dtype_id; nt->t.data_bf16 = NULL; nt->t.data_raw = NULL;

        if (dtype_id == 2) {
            nt->t.data = NULL;
            nt->t.data_raw = (uint8_t *)malloc((size_t)size);
            mem_read(&r, nt->t.data_raw, (size_t)size);
        } else if (dtype_id == 1) {
            const uint16_t *bf16 = (const uint16_t *)r.p;
            r.p += (size_t)size * 2; r.remaining -= (size_t)size * 2;
            nt->t.data = (float *)malloc((size_t)size * sizeof(float));
            for (int j = 0; j < size; j++) {
                uint32_t bits = (uint32_t)bf16[j] << 16;
                memcpy(&nt->t.data[j], &bits, sizeof(float));
            }
        } else {
            nt->t.data = (float *)malloc((size_t)size * sizeof(float));
            mem_read(&r, nt->t.data, (size_t)size * sizeof(float));
        }
        ws->count++;
    }
    printf("Loaded %d tensors from embedded data\n", ws->count);
    return 0;
}

const Tensor *weights_get(const WeightStore *ws, const char *name) {
    for (int i = 0; i < ws->count; i++) {
        if (strcmp(ws->entries[i].name, name) == 0)
            return &ws->entries[i].t;
    }
    return NULL;
}

const Tensor *weights_must_get(const WeightStore *ws, const char *name) {
    const Tensor *t = weights_get(ws, name);
    if (!t) {
        fprintf(stderr, "FATAL: weight not found: %s\n", name);
        exit(1);
    }
    return t;
}
