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

/* Opaque bucket weights (built from dense weight tensor) */
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

/* Set effort (0.0 = skip all, 1.0 = full compute). When < 1.0 and tensor has bucket data, use bucket path. */
void ggml_bucket_mul_set_effort(float effort);

float ggml_bucket_mul_get_effort(void);

/* Register bucket data for a weight tensor (keyed by tensor pointer). */
void ggml_bucket_mul_register(struct ggml_tensor *weight, struct ggml_bucket_weights *bw);

/* Unregister and free bucket data for a tensor. */
void ggml_bucket_mul_unregister(struct ggml_tensor *weight);

/* Get registered bucket data for a tensor, or NULL. */
struct ggml_bucket_weights * ggml_bucket_mul_get(struct ggml_tensor *weight);
struct ggml_bucket_weights * ggml_bucket_mul_get_const(const struct ggml_tensor *weight);

/* Build bucket_weights from a dense FP16 weight matrix W [out_size, in_size] (row-major).
 * Allocates bw->buckets, bw->stats, bw->probes. Caller must free with ggml_bucket_mul_free_bucket_weights. */
bool ggml_bucket_mul_build_from_f16(struct ggml_bucket_weights *bw,
    const void *wdata, int64_t out_size, int64_t in_size,
    int64_t nb1, int64_t nb0);

/* Build from F32 weight (converts to FP16 in buckets). */
bool ggml_bucket_mul_build_from_f32(struct ggml_bucket_weights *bw,
    const float *wdata, int64_t out_size, int64_t in_size,
    int64_t nb1, int64_t nb0);

void ggml_bucket_mul_free_bucket_weights(struct ggml_bucket_weights *bw);

/* Single mat-vec with bucket path: v [in_size], out [out_size], expert_no 0 for dense. */
void ggml_bucket_mul_mul_vec(struct ggml_bucket_weights *bw,
    const float *v, float *out, int expert_no, float effort);

#ifdef __cplusplus
}
#endif

#endif
