/*
 * BucketMul for GGML - build buckets from dense weight, run effort-based sparse matmul.
 * CPU-only, GGUF format unchanged.
 */

#include "ggml.h"
#include "ggml-bucket-mul.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

#define BUCKET_MUL_CACHE_MAGIC 0x4C554D42
#define BUCKET_MUL_CACHE_VERSION 1

struct ggml_bucket_mul_cache {
    int n;
    char **names;
    struct ggml_bucket_weights **bws;
    int *taken;
};

static inline float f16_to_f32(uint16_t h) {
    return ggml_fp16_to_fp32((ggml_fp16_t)h);
}
static inline uint16_t f32_to_f16(float f) {
    return (uint16_t)ggml_fp32_to_fp16(f);
}

/* --------------------------------------------------------------------------
 * Effort and registry
 * -------------------------------------------------------------------------- */
#define GGML_BUCKET_MUL_MAX_REGISTRY 512

static float s_effort = 1.0f;

struct bucket_registry_entry {
    struct ggml_tensor *tensor;
    struct ggml_bucket_weights *bw;
};

static struct bucket_registry_entry s_registry[GGML_BUCKET_MUL_MAX_REGISTRY];
static int s_registry_count = 0;

void ggml_bucket_mul_set_effort(float effort) {
    s_effort = effort;
}

float ggml_bucket_mul_get_effort(void) {
    return s_effort;
}

void ggml_bucket_mul_register(struct ggml_tensor *weight, struct ggml_bucket_weights *bw) {
    if (!weight || !bw || s_registry_count >= GGML_BUCKET_MUL_MAX_REGISTRY) return;
    for (int i = 0; i < s_registry_count; i++) {
        if (s_registry[i].tensor == weight) {
            s_registry[i].bw = bw;
            return;
        }
    }
    s_registry[s_registry_count].tensor = weight;
    s_registry[s_registry_count].bw = bw;
    s_registry_count++;
}

void ggml_bucket_mul_unregister(struct ggml_tensor *weight) {
    for (int i = 0; i < s_registry_count; i++) {
        if (s_registry[i].tensor == weight) {
            ggml_bucket_mul_free_bucket_weights(s_registry[i].bw);
            s_registry_count--;
            if (i < s_registry_count) {
                s_registry[i] = s_registry[s_registry_count];
            }
            return;
        }
    }
}

struct ggml_bucket_weights * ggml_bucket_mul_get(struct ggml_tensor *weight) {
    return ggml_bucket_mul_get_const(weight);
}
struct ggml_bucket_weights * ggml_bucket_mul_get_const(const struct ggml_tensor *weight) {
    for (int i = 0; i < s_registry_count; i++) {
        if ((const void *)s_registry[i].tensor == (const void *)weight) return s_registry[i].bw;
    }
    return NULL;
}

/* --------------------------------------------------------------------------
 * Build buckets from dense weight
 * Weight layout: row-major [out_size, in_size] -> row r has in_size elements.
 * We transpose: for each input dim i we have out_size values; sort by abs, bucketize.
 * -------------------------------------------------------------------------- */
static int compare_abs_desc(const void *a, const void *b) {
    float va = fabsf(*(const float *)a);
    float vb = fabsf(*(const float *)b);
    return va > vb ? -1 : (va < vb ? 1 : 0);
}

bool ggml_bucket_mul_build_from_f16(struct ggml_bucket_weights *bw,
    const void *wdata, int64_t out_size, int64_t in_size,
    int64_t nb1, int64_t nb0) {

    if (!bw || !wdata || out_size <= 0 || in_size <= 0 ||
        (out_size % GGML_BUCKET_MUL_BUCKET_SIZE) != 0) return false;

    const int bucket_size = GGML_BUCKET_MUL_BUCKET_SIZE;
    const int bucket_cols = (int)(out_size / bucket_size);
    const int rows_full = (int)in_size * bucket_size;
    const size_t buckets_sz = (size_t)rows_full * bucket_cols * sizeof(uint16_t);
    const size_t stats_sz   = (size_t)rows_full * 4 * sizeof(float);
    const size_t probes_sz  = (size_t)GGML_BUCKET_MUL_PROBES_COUNT * sizeof(uint16_t);

    bw->in_size = (int)in_size;
    bw->out_size = (int)out_size;
    bw->bucket_size = bucket_size;
    bw->n_experts = 1;
    bw->rows_full = rows_full;
    bw->rows_available = rows_full;
    bw->buckets = (uint16_t *)malloc(buckets_sz);
    bw->stats   = (float *)malloc(stats_sz);
    bw->probes  = (uint16_t *)malloc(probes_sz);
    if (!bw->buckets || !bw->stats || !bw->probes) {
        free(bw->buckets);
        free(bw->stats);
        free(bw->probes);
        return false;
    }
    memset(bw->buckets, 0, buckets_sz);

    float *row_vals = (float *)malloc((size_t)out_size * sizeof(float));
    if (!row_vals) {
        ggml_bucket_mul_free_bucket_weights(bw);
        return false;
    }

    const uint16_t *w = (const uint16_t *)wdata;
    const int pos_mask = bucket_size - 1;

    for (int64_t i = 0; i < in_size; i++) {
        for (int64_t j = 0; j < out_size; j++) {
            size_t off = (size_t)(j * (nb1 / sizeof(uint16_t)) + i * (nb0 / sizeof(uint16_t)));
            row_vals[j] = f16_to_f32(w[off]);
        }
        qsort(row_vals, (size_t)out_size, sizeof(float), compare_abs_desc);
        for (int b = 0; b < bucket_cols; b++) {
            for (int p = 0; p < bucket_size; p++) {
                float val = row_vals[b * bucket_size + p];
                uint16_t raw = f32_to_f16(val);
                raw &= (uint16_t)(~pos_mask);
                raw |= (uint16_t)p;
                int br = (int)i * bucket_size + p;
                bw->buckets[br * bucket_cols + b] = raw;
            }
        }
    }

    for (int r = 0; r < rows_full; r++) {
        float sum = 0;
        for (int c = 0; c < bucket_cols; c++) {
            uint16_t raw = bw->buckets[r * bucket_cols + c];
            sum += fabsf(f16_to_f32((uint16_t)(raw & ~pos_mask)));
        }
        bw->stats[r * 4 + 0] = bw->stats[r * 4 + 1] = bw->stats[r * 4 + 2] = bw->stats[r * 4 + 3] = sum / (float)bucket_cols;
    }

    int n_probes = GGML_BUCKET_MUL_PROBES_COUNT;
    if ((int64_t)n_probes > in_size * out_size) n_probes = (int)(in_size * out_size);
    for (int k = 0; k < n_probes; k++) {
        int64_t idx = (k * 97 + 31) % (in_size * out_size);
        int64_t row = idx / in_size;
        int64_t col = idx % in_size;
        size_t off = (size_t)(row * (nb1 / sizeof(uint16_t)) + col * (nb0 / sizeof(uint16_t)));
        bw->probes[k] = w[off];
    }
    for (int k = n_probes; k < GGML_BUCKET_MUL_PROBES_COUNT; k++) {
        bw->probes[k] = bw->probes[k % n_probes];
    }

    free(row_vals);
    return true;
}

bool ggml_bucket_mul_build_from_f32(struct ggml_bucket_weights *bw,
    const float *wdata, int64_t out_size, int64_t in_size,
    int64_t nb1, int64_t nb0) {

    if (!bw || !wdata || out_size <= 0 || in_size <= 0 ||
        (out_size % GGML_BUCKET_MUL_BUCKET_SIZE) != 0) return false;

    const int bucket_size = GGML_BUCKET_MUL_BUCKET_SIZE;
    const int bucket_cols = (int)(out_size / bucket_size);
    const int rows_full = (int)in_size * bucket_size;
    const size_t buckets_sz = (size_t)rows_full * bucket_cols * sizeof(uint16_t);
    const size_t stats_sz   = (size_t)rows_full * 4 * sizeof(float);
    const size_t probes_sz  = (size_t)GGML_BUCKET_MUL_PROBES_COUNT * sizeof(uint16_t);

    bw->in_size = (int)in_size;
    bw->out_size = (int)out_size;
    bw->bucket_size = bucket_size;
    bw->n_experts = 1;
    bw->rows_full = rows_full;
    bw->rows_available = rows_full;
    bw->buckets = (uint16_t *)malloc(buckets_sz);
    bw->stats   = (float *)malloc(stats_sz);
    bw->probes  = (uint16_t *)malloc(probes_sz);
    if (!bw->buckets || !bw->stats || !bw->probes) {
        free(bw->buckets);
        free(bw->stats);
        free(bw->probes);
        return false;
    }
    memset(bw->buckets, 0, buckets_sz);

    float *row_vals = (float *)malloc((size_t)out_size * sizeof(float));
    if (!row_vals) {
        ggml_bucket_mul_free_bucket_weights(bw);
        return false;
    }

    const int pos_mask = bucket_size - 1;
    const size_t row_stride = (size_t)(nb1 / sizeof(float));
    const size_t col_stride = (size_t)(nb0 / sizeof(float));

    for (int64_t i = 0; i < in_size; i++) {
        for (int64_t j = 0; j < out_size; j++) {
            size_t off = (size_t)j * row_stride + (size_t)i * col_stride;
            row_vals[j] = wdata[off];
        }
        qsort(row_vals, (size_t)out_size, sizeof(float), compare_abs_desc);
        for (int b = 0; b < bucket_cols; b++) {
            for (int p = 0; p < bucket_size; p++) {
                float val = row_vals[b * bucket_size + p];
                uint16_t raw = f32_to_f16(val);
                raw &= (uint16_t)(~pos_mask);
                raw |= (uint16_t)p;
                int br = (int)i * bucket_size + p;
                bw->buckets[br * bucket_cols + b] = raw;
            }
        }
    }

    for (int r = 0; r < rows_full; r++) {
        float sum = 0;
        for (int c = 0; c < bucket_cols; c++) {
            uint16_t raw = bw->buckets[r * bucket_cols + c];
            sum += fabsf(f16_to_f32((uint16_t)(raw & ~pos_mask)));
        }
        bw->stats[r * 4 + 0] = bw->stats[r * 4 + 1] = bw->stats[r * 4 + 2] = bw->stats[r * 4 + 3] = sum / (float)bucket_cols;
    }

    int n_probes = GGML_BUCKET_MUL_PROBES_COUNT;
    if ((int64_t)n_probes > in_size * out_size) n_probes = (int)(in_size * out_size);
    for (int k = 0; k < n_probes; k++) {
        int64_t idx = (k * 97 + 31) % (in_size * out_size);
        int64_t row = idx / in_size;
        int64_t col = idx % in_size;
        size_t off = (size_t)row * row_stride + (size_t)col * col_stride;
        bw->probes[k] = f32_to_f16(wdata[off]);
    }
    for (int k = n_probes; k < GGML_BUCKET_MUL_PROBES_COUNT; k++) {
        bw->probes[k] = bw->probes[k % n_probes];
    }

    free(row_vals);
    return true;
}

void ggml_bucket_mul_free_bucket_weights(struct ggml_bucket_weights *bw) {
    if (!bw) return;
    free(bw->buckets);
    free(bw->stats);
    free(bw->probes);
    bw->buckets = NULL;
    bw->stats = NULL;
    bw->probes = NULL;
}

/* --------------------------------------------------------------------------
 * Cutoff and dispatch (single expert)
 * -------------------------------------------------------------------------- */
typedef struct { float input_val; uint32_t weight_offset; } dispatch_entry_t;

#define CUTOFF_PROBES 256
static float find_cutoff(const float *v, int in_size, const uint16_t *probes, int probes_count, float effort) {
    if (effort >= 1.0f) return 0.0f;
    if (effort <= 0.0f) return FLT_MAX;
    int n = probes_count < CUTOFF_PROBES ? probes_count : CUTOFF_PROBES;
    static float products[CUTOFF_PROBES];
    for (int i = 0; i < n; i++) {
        int vi = i % in_size;
        products[i] = fabsf(GGML_BUCKET_MUL_CUTOFF_SCALE * v[vi] * f16_to_f32(probes[i]));
    }
    float hi = 0;
    for (int i = 0; i < n; i++) if (products[i] > hi) hi = products[i];
    int target_above = (int)(effort * n);
    if (target_above >= n) return 0.0f;
    if (target_above <= 0) return hi + 1.0f;
    float lo = 0.0f;
    for (int iter = 0; iter < 32; iter++) {
        float mid = (lo + hi) * 0.5f;
        int count = 0;
        for (int i = 0; i < n; i++) if (products[i] > mid) count++;
        if (count > target_above) lo = mid; else hi = mid;
        if (hi - lo < 1e-5f) break;
    }
    return (lo + hi) * 0.5f;
}

#define DISPATCH_CAP 256*1024
static dispatch_entry_t s_dispatch[DISPATCH_CAP];
static int s_dispatch_size;

static void prepare_dispatch(const float *v, struct ggml_bucket_weights *w, float cutoff) {
    s_dispatch_size = 0;
    int total_rows = w->rows_available > 0 ? w->rows_available : w->rows_full;
    int bucket_cols = w->out_size / w->bucket_size;
    for (int r = 0; r < total_rows && s_dispatch_size < DISPATCH_CAP; r++) {
        int input_idx = r % w->in_size;
        float input_val = v[input_idx];
        float stat_mean = w->stats[r * 4 + 3];
        float product = GGML_BUCKET_MUL_CUTOFF_SCALE * stat_mean * fabsf(input_val);
        if (product > cutoff) {
            s_dispatch[s_dispatch_size].input_val = input_val;
            s_dispatch[s_dispatch_size].weight_offset = (uint32_t)(r * bucket_cols);
            s_dispatch_size++;
        }
    }
}

static void exec_dispatch(struct ggml_bucket_weights *w, float *out) {
    memset(out, 0, (size_t)w->out_size * sizeof(float));
    int bucket_cols = w->out_size / w->bucket_size;
    int pos_mask = w->bucket_size - 1;
    for (int d = 0; d < s_dispatch_size; d++) {
        float input_val = s_dispatch[d].input_val;
        uint32_t base = s_dispatch[d].weight_offset;
        int c = 0;
        for (; c + 3 < bucket_cols; c += 4) {
            uint16_t r0 = w->buckets[base + c + 0], r1 = w->buckets[base + c + 1];
            uint16_t r2 = w->buckets[base + c + 2], r3 = w->buckets[base + c + 3];
            float w0 = f16_to_f32(r0), w1 = f16_to_f32(r1), w2 = f16_to_f32(r2), w3 = f16_to_f32(r3);
            int p0 = r0 & pos_mask, p1 = r1 & pos_mask, p2 = r2 & pos_mask, p3 = r3 & pos_mask;
            out[c * w->bucket_size + p0] += input_val * w0;
            out[(c+1) * w->bucket_size + p1] += input_val * w1;
            out[(c+2) * w->bucket_size + p2] += input_val * w2;
            out[(c+3) * w->bucket_size + p3] += input_val * w3;
        }
        for (; c < bucket_cols; c++) {
            uint16_t raw = w->buckets[base + c];
            float weight_val = f16_to_f32(raw);
            int pos = raw & pos_mask;
            int out_idx = c * w->bucket_size + pos;
            if (out_idx < w->out_size)
                out[out_idx] += input_val * weight_val;
        }
    }
}

void ggml_bucket_mul_mul_vec(struct ggml_bucket_weights *bw,
    const float *v, float *out, int expert_no, float effort) {
    (void)expert_no;
    float cutoff = find_cutoff(v, bw->in_size, bw->probes, GGML_BUCKET_MUL_PROBES_COUNT, effort);
    prepare_dispatch(v, bw, cutoff);
    exec_dispatch(bw, out);
}

/* --------------------------------------------------------------------------
 * Cache file save/load
 * -------------------------------------------------------------------------- */
bool ggml_bucket_mul_save(const char *path, const char **names, struct ggml_bucket_weights **bws, int n) {
    if (!path || !names || !bws || n <= 0) return false;
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    uint32_t magic = BUCKET_MUL_CACHE_MAGIC;
    uint32_t version = BUCKET_MUL_CACHE_VERSION;
    if (fwrite(&magic, 4, 1, f) != 1 || fwrite(&version, 4, 1, f) != 1) { fclose(f); return false; }
    uint32_t un = (uint32_t)n;
    if (fwrite(&un, 4, 1, f) != 1) { fclose(f); return false; }
    for (int i = 0; i < n; i++) {
        struct ggml_bucket_weights *bw = bws[i];
        const char *name = names[i];
        if (!bw || !name) { fclose(f); return false; }
        uint32_t name_len = (uint32_t)strlen(name);
        if (fwrite(&name_len, 4, 1, f) != 1 || fwrite(name, 1, name_len, f) != name_len) { fclose(f); return false; }
        int32_t in_s = (int32_t)bw->in_size, out_s = (int32_t)bw->out_size, bs = (int32_t)bw->bucket_size;
        int32_t ne = (int32_t)bw->n_experts, rf = (int32_t)bw->rows_full, ra = (int32_t)bw->rows_available;
        if (fwrite(&in_s, 4, 1, f) != 1 || fwrite(&out_s, 4, 1, f) != 1 || fwrite(&bs, 4, 1, f) != 1 ||
            fwrite(&ne, 4, 1, f) != 1 || fwrite(&rf, 4, 1, f) != 1 || fwrite(&ra, 4, 1, f) != 1) { fclose(f); return false; }
        int bucket_cols = bw->out_size / bw->bucket_size;
        size_t buckets_sz = (size_t)bw->rows_full * bucket_cols * sizeof(uint16_t);
        size_t stats_sz   = (size_t)bw->rows_full * 4 * sizeof(float);
        size_t probes_sz  = (size_t)GGML_BUCKET_MUL_PROBES_COUNT * sizeof(uint16_t);
        if (fwrite(bw->buckets, 1, buckets_sz, f) != buckets_sz ||
            fwrite(bw->stats, 1, stats_sz, f) != stats_sz ||
            fwrite(bw->probes, 1, probes_sz, f) != probes_sz) { fclose(f); return false; }
    }
    fclose(f);
    return true;
}

struct ggml_bucket_mul_cache * ggml_bucket_mul_load(const char *path) {
    if (!path) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t magic, version, un;
    if (fread(&magic, 4, 1, f) != 1 || magic != BUCKET_MUL_CACHE_MAGIC ||
        fread(&version, 4, 1, f) != 1 || version != BUCKET_MUL_CACHE_VERSION ||
        fread(&un, 4, 1, f) != 1) { fclose(f); return NULL; }
    int n = (int)un;
    if (n <= 0 || n > 1024) { fclose(f); return NULL; }
    struct ggml_bucket_mul_cache *c = (struct ggml_bucket_mul_cache *)malloc(sizeof(*c));
    if (!c) { fclose(f); return NULL; }
    c->n = n;
    c->names = (char **)malloc((size_t)n * sizeof(char *));
    c->bws = (struct ggml_bucket_weights **)malloc((size_t)n * sizeof(struct ggml_bucket_weights *));
    c->taken = (int *)malloc((size_t)n * sizeof(int));
    if (!c->names || !c->bws || !c->taken) {
        if (c->names) free(c->names);
        if (c->bws) free(c->bws);
        if (c->taken) free(c->taken);
        free(c);
        fclose(f);
        return NULL;
    }
    memset(c->taken, 0, (size_t)n * sizeof(int));
    int i;
    for (i = 0; i < n; i++) {
        uint32_t name_len;
        if (fread(&name_len, 4, 1, f) != 1 || name_len > 256) goto fail_load;
        c->names[i] = (char *)malloc(name_len + 1);
        if (!c->names[i] || fread(c->names[i], 1, name_len, f) != name_len) goto fail_load;
        c->names[i][name_len] = '\0';
        int32_t in_s, out_s, bs, ne, rf, ra;
        if (fread(&in_s, 4, 1, f) != 1 || fread(&out_s, 4, 1, f) != 1 || fread(&bs, 4, 1, f) != 1 ||
            fread(&ne, 4, 1, f) != 1 || fread(&rf, 4, 1, f) != 1 || fread(&ra, 4, 1, f) != 1) goto fail_load;
        struct ggml_bucket_weights *bw = (struct ggml_bucket_weights *)malloc(sizeof(struct ggml_bucket_weights));
        if (!bw) goto fail_load;
        bw->in_size = in_s;
        bw->out_size = out_s;
        bw->bucket_size = bs;
        bw->n_experts = ne;
        bw->rows_full = rf;
        bw->rows_available = ra;
        int bucket_cols = out_s / bs;
        size_t buckets_sz = (size_t)rf * bucket_cols * sizeof(uint16_t);
        size_t stats_sz   = (size_t)rf * 4 * sizeof(float);
        size_t probes_sz  = (size_t)GGML_BUCKET_MUL_PROBES_COUNT * sizeof(uint16_t);
        bw->buckets = (uint16_t *)malloc(buckets_sz);
        bw->stats   = (float *)malloc(stats_sz);
        bw->probes  = (uint16_t *)malloc(probes_sz);
        if (!bw->buckets || !bw->stats || !bw->probes ||
            fread(bw->buckets, 1, buckets_sz, f) != buckets_sz ||
            fread(bw->stats, 1, stats_sz, f) != stats_sz ||
            fread(bw->probes, 1, probes_sz, f) != probes_sz) {
            if (bw->buckets) free(bw->buckets);
            if (bw->stats) free(bw->stats);
            if (bw->probes) free(bw->probes);
            free(bw);
            goto fail_load;
        }
        c->bws[i] = bw;
    }
    fclose(f);
    return c;
fail_load:
    for (int j = 0; j < i; j++) {
        free(c->names[j]);
        ggml_bucket_mul_free_bucket_weights(c->bws[j]);
        free(c->bws[j]);
    }
    free(c->names);
    free(c->bws);
    free(c->taken);
    free(c);
    fclose(f);
    return NULL;
}

struct ggml_bucket_weights * ggml_bucket_mul_cache_get(struct ggml_bucket_mul_cache *c, const char *name) {
    if (!c || !name) return NULL;
    for (int i = 0; i < c->n; i++) {
        if (c->taken[i]) continue;
        if (strcmp(c->names[i], name) == 0) {
            c->taken[i] = 1;
            return c->bws[i];
        }
    }
    return NULL;
}

void ggml_bucket_mul_cache_free(struct ggml_bucket_mul_cache *c) {
    if (!c) return;
    for (int i = 0; i < c->n; i++) {
        free(c->names[i]);
        if (!c->taken[i]) {
            ggml_bucket_mul_free_bucket_weights(c->bws[i]);
            free(c->bws[i]);
        }
    }
    free(c->names);
    free(c->bws);
    free(c->taken);
    free(c);
}
