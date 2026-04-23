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

static int read_exact(FILE *f, void *dst, size_t size) {
    return fread(dst, 1, size, f) == size;
}

int weights_load(WeightStore *ws, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }

    char magic[4];
    uint32_t version, num_tensors;
    if (!read_exact(f, magic, sizeof(magic))) {
        fprintf(stderr, "Truncated header\n"); fclose(f); return -1;
    }
    if (memcmp(magic, "MTTS", 4) != 0) {
        fprintf(stderr, "Bad magic\n"); fclose(f); return -1;
    }
    if (!read_u32(f, &version) || !read_u32(f, &num_tensors)) {
        fprintf(stderr, "Truncated header\n"); fclose(f); return -1;
    }
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
        if (!read_u32(f, &name_len)) {
            fprintf(stderr, "Truncated tensor name length\n");
            fclose(f); return -1;
        }
        if (name_len >= MAX_NAME_LEN) {
            fprintf(stderr, "Name too long: %u\n", name_len);
            fclose(f); return -1;
        }
        if (!read_exact(f, nt->name, name_len)) {
            fprintf(stderr, "Truncated tensor name\n");
            fclose(f); return -1;
        }
        nt->name[name_len] = '\0';

        /* v2: read dtype before ndim */
        uint32_t dtype_id = 0;
        if (version >= 2 && !read_u32(f, &dtype_id)) {
            fprintf(stderr, "Truncated tensor dtype\n");
            fclose(f); return -1;
        }

        uint32_t ndim;
        if (!read_u32(f, &ndim)) {
            fprintf(stderr, "Truncated tensor ndim\n");
            fclose(f); return -1;
        }
        if (ndim > MAX_DIMS) {
            fprintf(stderr, "Too many dims: %u\n", ndim);
            fclose(f); return -1;
        }
        int shape[MAX_DIMS] = {0};
        int size = 1;
        for (uint32_t d = 0; d < ndim; d++) {
            uint32_t dim;
            if (!read_u32(f, &dim)) {
                fprintf(stderr, "Truncated tensor shape\n");
                fclose(f); return -1;
            }
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
            if (!nt->t.data_raw || !read_exact(f, nt->t.data_raw, (size_t)size)) {
                fprintf(stderr, "Truncated raw tensor data: %s\n", nt->name);
                free(nt->t.data_raw);
                fclose(f); return -1;
            }
        } else if (dtype_id == 1) {
            /* bfloat16: read as bf16, convert to fp32 for OpenBLAS compatibility */
            uint16_t *bf16 = (uint16_t *)malloc((size_t)size * sizeof(uint16_t));
            if (!bf16 || fread(bf16, sizeof(uint16_t), (size_t)size, f) != (size_t)size) {
                fprintf(stderr, "Truncated bf16 tensor data: %s\n", nt->name);
                free(bf16);
                fclose(f); return -1;
            }
            nt->t.data = (float *)malloc((size_t)size * sizeof(float));
            if (!nt->t.data) {
                free(bf16);
                fclose(f); return -1;
            }
            for (int j = 0; j < size; j++) {
                uint32_t bits = (uint32_t)bf16[j] << 16;
                memcpy(&nt->t.data[j], &bits, sizeof(float));
            }
            free(bf16);
        } else {
            nt->t.data = (float *)malloc((size_t)size * sizeof(float));
            if (!nt->t.data || fread(nt->t.data, sizeof(float), (size_t)size, f) != (size_t)size) {
                fprintf(stderr, "Truncated fp32 tensor data: %s\n", nt->name);
                free(nt->t.data);
                fclose(f); return -1;
            }
        }

        ws->count++;
    }

    char footer[4];
    if (!read_exact(f, footer, sizeof(footer))) {
        fprintf(stderr, "Truncated footer\n"); fclose(f); return -1;
    }
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

static int mem_u32(MemReader *r, uint32_t *out) {
    return mem_read(r, out, sizeof(*out));
}

int weights_load_mem(WeightStore *ws, const void *data, size_t total_size) {
    MemReader r = { (const uint8_t *)data, total_size };
    char magic[4];
    if (mem_read(&r, magic, sizeof(magic)) != 0) {
        fprintf(stderr, "Truncated embedded header\n"); return -1;
    }
    if (memcmp(magic, "MTTS", 4) != 0) { fprintf(stderr, "Bad magic\n"); return -1; }
    uint32_t version = 0, num_tensors = 0;
    if (mem_u32(&r, &version) != 0 || mem_u32(&r, &num_tensors) != 0) {
        fprintf(stderr, "Truncated embedded header\n"); return -1;
    }
    if (version < 1 || version > 2) { fprintf(stderr, "Bad version\n"); return -1; }

    ws->count = 0;
    for (uint32_t i = 0; i < num_tensors; i++) {
        if (ws->count >= MAX_TENSORS) { fprintf(stderr, "Too many tensors\n"); return -1; }
        NamedTensor *nt = &ws->entries[ws->count];
        uint32_t name_len = 0;
        if (mem_u32(&r, &name_len) != 0) {
            fprintf(stderr, "Truncated embedded tensor name length\n"); return -1;
        }
        if (name_len >= MAX_NAME_LEN) { fprintf(stderr, "Name too long\n"); return -1; }
        if (mem_read(&r, nt->name, name_len) != 0) {
            fprintf(stderr, "Truncated embedded tensor name\n"); return -1;
        }
        nt->name[name_len] = '\0';

        uint32_t dtype_id = 0;
        if (version >= 2 && mem_u32(&r, &dtype_id) != 0) {
            fprintf(stderr, "Truncated embedded tensor dtype\n"); return -1;
        }

        uint32_t ndim = 0;
        if (mem_u32(&r, &ndim) != 0) {
            fprintf(stderr, "Truncated embedded tensor ndim\n"); return -1;
        }
        if (ndim > MAX_DIMS) { fprintf(stderr, "Too many dims\n"); return -1; }
        int shape[MAX_DIMS] = {0};
        int size = 1;
        for (uint32_t d = 0; d < ndim; d++) {
            uint32_t dim = 0;
            if (mem_u32(&r, &dim) != 0) {
                fprintf(stderr, "Truncated embedded tensor shape\n"); return -1;
            }
            shape[d] = (int)dim;
            size *= shape[d];
        }

        nt->t.ndim = (int)ndim; nt->t.size = size;
        for (int d = 0; d < MAX_DIMS; d++) nt->t.shape[d] = shape[d];
        nt->t.dtype = (int)dtype_id; nt->t.data_bf16 = NULL; nt->t.data_raw = NULL;

        if (dtype_id == 2) {
            nt->t.data = NULL;
            nt->t.data_raw = (uint8_t *)malloc((size_t)size);
            if (!nt->t.data_raw || mem_read(&r, nt->t.data_raw, (size_t)size) != 0) {
                fprintf(stderr, "Truncated embedded raw tensor data: %s\n", nt->name);
                free(nt->t.data_raw);
                return -1;
            }
        } else if (dtype_id == 1) {
            if (r.remaining < (size_t)size * sizeof(uint16_t)) {
                fprintf(stderr, "Truncated embedded bf16 tensor data: %s\n", nt->name);
                return -1;
            }
            const uint16_t *bf16 = (const uint16_t *)r.p;
            r.p += (size_t)size * 2; r.remaining -= (size_t)size * 2;
            nt->t.data = (float *)malloc((size_t)size * sizeof(float));
            if (!nt->t.data) return -1;
            for (int j = 0; j < size; j++) {
                uint32_t bits = (uint32_t)bf16[j] << 16;
                memcpy(&nt->t.data[j], &bits, sizeof(float));
            }
        } else {
            nt->t.data = (float *)malloc((size_t)size * sizeof(float));
            if (!nt->t.data || mem_read(&r, nt->t.data, (size_t)size * sizeof(float)) != 0) {
                fprintf(stderr, "Truncated embedded fp32 tensor data: %s\n", nt->name);
                free(nt->t.data);
                return -1;
            }
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
