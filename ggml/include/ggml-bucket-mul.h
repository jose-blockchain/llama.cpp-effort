/*
 * BucketMul for GGML - effort-based sparse matmul (CPU, GGUF-only).
 * Build buckets from dense weight at load time; no new file format.
 */

#ifndef GGML_BUCKET_MUL_H
#define GGML_BUCKET_MUL_H

#include "ggml.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_BUCKET_MUL_PROBES_COUNT 4096
#define GGML_BUCKET_MUL_BUCKET_SIZE  16
#define GGML_BUCKET_MUL_CUTOFF_SCALE 100000.0f

struct ggml_bucket_weights {
    int in_size;
    int out_size;
    int bucket_size;
    int n_experts;
    uint16_t *buckets;
    float    *stats;
    uint16_t *probes;
    int rows_available;
    int rows_full;
};

void ggml_bucket_mul_set_effort(float effort);
float ggml_bucket_mul_get_effort(void);
void ggml_bucket_mul_register(struct ggml_tensor *weight, struct ggml_bucket_weights *bw);
void ggml_bucket_mul_unregister(struct ggml_tensor *weight);
struct ggml_bucket_weights * ggml_bucket_mul_get(struct ggml_tensor *weight);
struct ggml_bucket_weights * ggml_bucket_mul_get_const(const struct ggml_tensor *weight);

bool ggml_bucket_mul_build_from_f16(struct ggml_bucket_weights *bw,
    const void *wdata, int64_t out_size, int64_t in_size,
    int64_t nb1, int64_t nb0);
bool ggml_bucket_mul_build_from_f32(struct ggml_bucket_weights *bw,
    const float *wdata, int64_t out_size, int64_t in_size,
    int64_t nb1, int64_t nb0);
void ggml_bucket_mul_free_bucket_weights(struct ggml_bucket_weights *bw);

void ggml_bucket_mul_mul_vec(struct ggml_bucket_weights *bw,
    const float *v, float *out, int expert_no, float effort);

/* Cache file: load pre-built buckets to avoid rebuild. path = model path + ".bucket_mul". */
struct ggml_bucket_mul_cache;
bool ggml_bucket_mul_save(const char *path, const char **names, struct ggml_bucket_weights **bws, int n);
struct ggml_bucket_mul_cache * ggml_bucket_mul_load(const char *path);
struct ggml_bucket_weights * ggml_bucket_mul_cache_get(struct ggml_bucket_mul_cache *c, const char *name);
void ggml_bucket_mul_cache_free(struct ggml_bucket_mul_cache *c);

#ifdef __cplusplus
}
#endif

#endif
