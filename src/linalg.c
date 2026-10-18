#include "linalg.h"

#include "random.h"
#include "simd.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if RL_SIMD_BACKEND == RL_SIMD_NEON
#include <arm_neon.h>
#elif RL_SIMD_BACKEND == RL_SIMD_AVX2
#include <immintrin.h>
#endif

// Row-major storage: element (r, c) lives at data[r * cols + c]. All indices
// are widened to u64 before multiplying so large matrices cannot overflow.

static u64 mat_count(const Mat *m) { return (u64)m->rows * (u64)m->cols; }

static u64 count_of(u32 rows, u32 cols) {
  return (u64)rows * (u64)cols;
}

Mat *mat_new(Arena *arena, u32 rows, u32 cols) {
  Mat *m = ARENA_OBJ(arena, Mat);
  RL_LIB_ASSERT(m != NULL, "mat_new: arena exhausted");

  m->rows = rows;
  m->cols = cols;
  m->data = ARENA_ARR(arena, real, count_of(rows, cols));
  RL_LIB_ASSERT(m->data != NULL, "mat_new: arena exhausted allocating data");

  return m;
}

void mat_zero(Mat *mat) { memset(mat->data, 0, sizeof(real) * mat_count(mat)); }

void mat_fill(Mat *mat, real x) {
  const u64 n = mat_count(mat);
  for (u64 i = 0; i < n; i++) {
    mat->data[i] = x;
  }
}

void mat_rand(Mat *mat, real lower, real upper) {
  const real span = upper - lower;
  const u64 n = mat_count(mat);
  for (u64 i = 0; i < n; i++) {
    mat->data[i] = rng_unit_g() * span + lower;
  }
}

void mat_rand_r(Mat *mat, Rng *rng, real lower, real upper) {
  const real span = upper - lower;
  const u64 n = mat_count(mat);
  for (u64 i = 0; i < n; i++) {
    mat->data[i] = rng_unit(rng) * span + lower;
  }
}

void mat_scale(Mat *mat, real scale_factor) {
  const u64 n = mat_count(mat);
  for (u64 i = 0; i < n; i++) {
    mat->data[i] *= scale_factor;
  }
}

// ---------------------------------------------------------------------------
// Elementwise add/sub with column broadcasting: one operand may have a single
// column, in which case it is reused for every column of the result.
// ---------------------------------------------------------------------------

static b32 can_broadcast_cols(u32 ca, u32 cb) {
  return ca == cb || ca == 1u || cb == 1u;
}

b32 mat_add(Mat *out, const Mat *a, const Mat *b) {
  const u32 rows = a->rows;
  const u32 ca = a->cols;
  const u32 cb = b->cols;
  const u32 width = (ca > cb) ? ca : cb;

  if (a->rows != b->rows || !can_broadcast_cols(ca, cb)) {
    return false;
  }
  if (out->rows != rows || out->cols != width) {
    return false;
  }

  for (u32 r = 0; r < rows; r++) {
    const real *ra = a->data + (u64)r * ca;
    const real *rb = b->data + (u64)r * cb;
    real *ro = out->data + (u64)r * width;
    for (u32 c = 0; c < width; c++) {
      ro[c] = ra[ca == 1u ? 0u : c] + rb[cb == 1u ? 0u : c];
    }
  }
  return true;
}

b32 mat_sub(Mat *out, const Mat *a, const Mat *b) {
  const u32 rows = a->rows;
  const u32 ca = a->cols;
  const u32 cb = b->cols;
  const u32 width = (ca > cb) ? ca : cb;

  if (a->rows != b->rows || !can_broadcast_cols(ca, cb)) {
    return false;
  }
  if (out->rows != rows || out->cols != width) {
    return false;
  }

  for (u32 r = 0; r < rows; r++) {
    const real *ra = a->data + (u64)r * ca;
    const real *rb = b->data + (u64)r * cb;
    real *ro = out->data + (u64)r * width;
    for (u32 c = 0; c < width; c++) {
      ro[c] = ra[ca == 1u ? 0u : c] - rb[cb == 1u ? 0u : c];
    }
  }
  return true;
}

b32 mat_mul(Mat *out, const Mat *a, const Mat *b) {
  if (a->rows != b->rows || a->cols != b->cols) {
    return false;
  }
  if (out->rows != a->rows || out->cols != a->cols) {
    return false;
  }
  const u64 n = mat_count(out);
  for (u64 i = 0; i < n; i++) {
    out->data[i] = a->data[i] * b->data[i];
  }
  return true;
}

// ---------------------------------------------------------------------------
// Reverse-mode helpers for broadcasting and elementwise products.
// ---------------------------------------------------------------------------

// Accumulates `grad` into a target, collapsing a single-column target by
// summing across the columns it was broadcast over. A NULL target is skipped.
static b32 accumulate_broadcast(Mat *dst, const Mat *grad, real sign) {
  if (dst == NULL) {
    return true;
  }
  const u32 rows = grad->rows;
  const u32 cols = grad->cols;
  if (dst->rows != rows) {
    return false;
  }
  if (dst->cols == cols) {
    const u64 n = count_of(rows, cols);
    for (u64 i = 0; i < n; i++) {
      dst->data[i] += sign * grad->data[i];
    }
    return true;
  }
  if (dst->cols == 1u) {
    for (u32 r = 0; r < rows; r++) {
      real acc = 0.0f;
      for (u32 c = 0; c < cols; c++) {
        acc += grad->data[(u64)r * cols + c];
      }
      dst->data[r] += sign * acc;
    }
    return true;
  }
  return false;
}

b32 mat_add_broadcast_grad(Mat *a_grad, Mat *b_grad, const Mat *grad) {
  return accumulate_broadcast(a_grad, grad, (real)1.0) &&
         accumulate_broadcast(b_grad, grad, (real)1.0);
}

b32 mat_sub_broadcast_grad(Mat *a_grad, Mat *b_grad, const Mat *grad) {
  return accumulate_broadcast(a_grad, grad, (real)1.0) &&
         accumulate_broadcast(b_grad, grad, (real)-1.0);
}

b32 mat_mul_add_grad(Mat *a_grad, Mat *b_grad, const Mat *a, const Mat *b,
                     const Mat *grad) {
  if (a->rows != b->rows || a->cols != b->cols) {
    return false;
  }
  if (grad->rows != a->rows || grad->cols != a->cols) {
    return false;
  }

  const u64 n = mat_count(a);
  if (a_grad != NULL) {
    for (u64 i = 0; i < n; i++) {
      a_grad->data[i] += grad->data[i] * b->data[i];
    }
  }
  if (b_grad != NULL) {
    for (u64 i = 0; i < n; i++) {
      b_grad->data[i] += grad->data[i] * a->data[i];
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Matrix multiply. Four accumulation kernels selected by the
// (transpose_a, transpose_b) flags; each adds into `out`, so callers that want
// a fresh result zero it first. The scalar kernels below are the portable
// reference; the SIMD kernels further down vectorise the same loops and agree
// with the reference to floating-point tolerance.
// ---------------------------------------------------------------------------

static void mm_nn(Mat *out, const Mat *a, const Mat *b) {
  const u32 M = out->rows;
  const u32 N = out->cols;
  const u32 K = a->cols;
  for (u32 i = 0; i < M; i++) {
    real *orow = out->data + (u64)i * N;
    const real *arow = a->data + (u64)i * K;
    for (u32 k = 0; k < K; k++) {
      const real aik = arow[k];
      const real *brow = b->data + (u64)k * N;
      for (u32 j = 0; j < N; j++) {
        orow[j] += aik * brow[j];
      }
    }
  }
}

static void mm_nt(Mat *out, const Mat *a, const Mat *b) {
  // b is stored transposed: b[j][k] for j < N, k < K.
  const u32 M = out->rows;
  const u32 N = out->cols;
  const u32 K = a->cols;
  for (u32 i = 0; i < M; i++) {
    real *orow = out->data + (u64)i * N;
    const real *arow = a->data + (u64)i * K;
    for (u32 j = 0; j < N; j++) {
      const real *brow = b->data + (u64)j * K;
      for (u32 k = 0; k < K; k++) {
        orow[j] += arow[k] * brow[k];
      }
    }
  }
}

static void mm_tn(Mat *out, const Mat *a, const Mat *b) {
  // a is stored transposed: a[k][i] for k < K, i < M.
  const u32 M = out->rows;
  const u32 N = out->cols;
  const u32 K = a->rows;
  for (u32 i = 0; i < M; i++) {
    real *orow = out->data + (u64)i * N;
    for (u32 k = 0; k < K; k++) {
      const real aik = a->data[(u64)k * M + i];
      const real *brow = b->data + (u64)k * N;
      for (u32 j = 0; j < N; j++) {
        orow[j] += aik * brow[j];
      }
    }
  }
}

static void mm_tt(Mat *out, const Mat *a, const Mat *b) {
  // a stored transposed (a[k][i]); b stored transposed (b[j][k]).
  const u32 M = out->rows;
  const u32 N = out->cols;
  const u32 K = a->rows;
  for (u32 i = 0; i < M; i++) {
    real *orow = out->data + (u64)i * N;
    for (u32 k = 0; k < K; k++) {
      const real aik = a->data[(u64)k * M + i];
      for (u32 j = 0; j < N; j++) {
        orow[j] += aik * b->data[(u64)j * K + k];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// SIMD kernels. Selected by the same (transpose_a, transpose_b) flags as the
// scalar reference and compiled for both precisions: 4-wide f32 / 2-wide f64 on
// NEON, 8-wide f32 / 4-wide f64 on AVX2. NN/TN now process two output rows at
// once (every B element is loaded once for two rows) and keep the reduction
// order of the scalar reference, so their results are element-wise identical to
// the single-row kernels; NT vectorises the K reduction and adds the lane
// partials horizontally, so it may differ by a few ULPs; TT transposes a tile of
// B and reuses the TN path.
// ---------------------------------------------------------------------------

#if RL_SIMD_BACKEND != RL_SIMD_NONE

#if RL_SIMD_BACKEND == RL_SIMD_NEON

#if defined(RL_LIB_USE_DOUBLE)

// Double precision is AArch64-only: ARMv7 NEON loads 64-bit lanes but has no
// double-precision arithmetic, so simd.h forces the scalar path there.
typedef float64x2_t RlVec;
#define RL_VEC_W 2u
static inline RlVec rl_vload(const real *p) { return vld1q_f64(p); }
static inline void rl_vstore(real *p, RlVec v) { vst1q_f64(p, v); }
static inline RlVec rl_vset1(real x) { return vdupq_n_f64(x); }
static inline RlVec rl_vfma(RlVec acc, RlVec a, RlVec b) {
  return vfmaq_f64(acc, a, b);
}
static inline RlVec rl_vzero(void) { return vdupq_n_f64(0.0); }
static inline real rl_vhsum(RlVec v) {
  return vgetq_lane_f64(v, 0) + vgetq_lane_f64(v, 1);
}

#else // f32

typedef float32x4_t RlVec;
#define RL_VEC_W 4u
static inline RlVec rl_vload(const real *p) { return vld1q_f32(p); }
static inline void rl_vstore(real *p, RlVec v) { vst1q_f32(p, v); }
static inline RlVec rl_vset1(real x) { return vdupq_n_f32(x); }
static inline RlVec rl_vfma(RlVec acc, RlVec a, RlVec b) {
  return vfmaq_f32(acc, a, b);
}
static inline RlVec rl_vzero(void) { return vdupq_n_f32(0.0f); }
static inline real rl_vhsum(RlVec v) {
  const float32x2_t lo = vget_low_f32(v);
  const float32x2_t hi = vget_high_f32(v);
  const float32x2_t s = vadd_f32(lo, hi);
  return vget_lane_f32(vpadd_f32(s, s), 0);
}

#endif // RL_LIB_USE_DOUBLE

#else // RL_SIMD_BACKEND == RL_SIMD_AVX2

// Per-function target attribute: these use AVX2/FMA even though the translation
// unit targets the baseline x86-64 ISA, so the binary still runs on CPUs
// without AVX2 (mat_mm() probes at runtime).
#define RL_AVX2_TARGET __attribute__((target("avx2,fma")))

#if defined(RL_LIB_USE_DOUBLE)

typedef __m256d RlVec;
#define RL_VEC_W 4u
RL_AVX2_TARGET static inline __attribute__((always_inline)) RlVec
rl_vload(const real *p) {
  return _mm256_loadu_pd(p);
}
RL_AVX2_TARGET static inline __attribute__((always_inline)) void
rl_vstore(real *p, RlVec v) {
  _mm256_storeu_pd(p, v);
}
RL_AVX2_TARGET static inline __attribute__((always_inline)) RlVec
rl_vset1(real x) {
  return _mm256_set1_pd(x);
}
RL_AVX2_TARGET static inline __attribute__((always_inline)) RlVec
rl_vfma(RlVec acc, RlVec a, RlVec b) {
  return _mm256_fmadd_pd(a, b, acc);
}
RL_AVX2_TARGET static inline __attribute__((always_inline)) RlVec
rl_vzero(void) {
  return _mm256_setzero_pd();
}
RL_AVX2_TARGET static inline __attribute__((always_inline)) real
rl_vhsum(RlVec v) {
  const __m128d lo = _mm256_castpd256_pd128(v);
  const __m128d hi = _mm256_extractf128_pd(v, 1);
  const __m128d s = _mm_add_pd(lo, hi);
  return _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)));
}

#else // f32

typedef __m256 RlVec;
#define RL_VEC_W 8u
RL_AVX2_TARGET static inline __attribute__((always_inline)) RlVec
rl_vload(const real *p) {
  return _mm256_loadu_ps(p);
}
RL_AVX2_TARGET static inline __attribute__((always_inline)) void
rl_vstore(real *p, RlVec v) {
  _mm256_storeu_ps(p, v);
}
RL_AVX2_TARGET static inline __attribute__((always_inline)) RlVec
rl_vset1(real x) {
  return _mm256_set1_ps(x);
}
RL_AVX2_TARGET static inline __attribute__((always_inline)) RlVec
rl_vfma(RlVec acc, RlVec a, RlVec b) {
  return _mm256_fmadd_ps(a, b, acc);
}
RL_AVX2_TARGET static inline __attribute__((always_inline)) RlVec
rl_vzero(void) {
  return _mm256_setzero_ps();
}
RL_AVX2_TARGET static inline __attribute__((always_inline)) real
rl_vhsum(RlVec v) {
  const __m128 lo = _mm256_castps256_ps128(v);
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}

#endif // RL_LIB_USE_DOUBLE

// The kernels themselves are compiled with the AVX2/FMA target so the helpers
// inline; mat_mm() only calls them after the runtime probe succeeds.
#define RL_SIMD_KERNEL RL_AVX2_TARGET

#endif

#if RL_SIMD_BACKEND == RL_SIMD_NEON
#define RL_SIMD_KERNEL
#endif

// Shared NN/TN micro-kernel over a rectangle of the output:
//
//   out[i][j] += sum(k0 <= k < k1) a_eff[k][i] * b[k - k0][j - j0]
//
// for j0 <= j < j1, where a_eff is A (M x K, row-major) when `transpose_a` is
// false and A^T (A stored K x M) when true. `b` supplies its own row stride
// (`b->cols`), which is what lets the TT kernel below feed in a transposed
// scratch tile. The plain NN/TN multiplies pass the whole matrix (k0 = 0,
// k1 = K, j0 = 0, j1 = out->cols).
//
// Two output rows are processed at once and four column vectors are held in
// registers per row across the whole K run: every element of B is loaded once
// for two rows instead of once per row, and `out` is loaded and stored once per
// row rather than once per (k, row). Each output element still accumulates over
// k in ascending order, so the result is element-wise identical to a single-row
// kernel and agrees with the scalar reference to floating-point tolerance.
RL_SIMD_KERNEL static void simd_mm_columns_range(Mat *out, const Mat *a,
                                                 const Mat *b, b32 transpose_a,
                                                 u32 k0, u32 k1, u32 j0,
                                                 u32 j1) {
  const u32 M = out->rows;
  const u32 out_stride = out->cols;
  const u32 b_stride = b->cols;
  const u32 W = RL_VEC_W;
  const u32 BLK = 4u * W;
  const u32 width = j1 - j0;
  // Stride from row i to row i + 1 in `a`: the K axis is contiguous for the
  // row-major case and the M axis for the transposed one.
  const u64 a_row_step = transpose_a ? 1u : (u64)a->cols;

  u32 i = 0;
  for (; i + 2 <= M; i += 2) {
    real *orow0 = out->data + (u64)i * out_stride + j0;
    real *orow1 = orow0 + out_stride;
    u32 j = 0;

    for (; j + BLK <= width; j += BLK) {
      RlVec v00 = rl_vload(orow0 + j);
      RlVec v01 = rl_vload(orow0 + j + W);
      RlVec v02 = rl_vload(orow0 + j + 2u * W);
      RlVec v03 = rl_vload(orow0 + j + 3u * W);
      RlVec v10 = rl_vload(orow1 + j);
      RlVec v11 = rl_vload(orow1 + j + W);
      RlVec v12 = rl_vload(orow1 + j + 2u * W);
      RlVec v13 = rl_vload(orow1 + j + 3u * W);
      for (u32 k = k0; k < k1; k++) {
        const u64 ai0 = transpose_a ? (u64)k * M + i : (u64)i * a->cols + k;
        const RlVec va0 = rl_vset1(a->data[ai0]);
        const RlVec va1 = rl_vset1(a->data[ai0 + a_row_step]);
        const real *brow = b->data + (u64)(k - k0) * b_stride + j;
        const RlVec b0 = rl_vload(brow);
        const RlVec b1 = rl_vload(brow + W);
        const RlVec b2 = rl_vload(brow + 2u * W);
        const RlVec b3 = rl_vload(brow + 3u * W);
        v00 = rl_vfma(v00, va0, b0);
        v01 = rl_vfma(v01, va0, b1);
        v02 = rl_vfma(v02, va0, b2);
        v03 = rl_vfma(v03, va0, b3);
        v10 = rl_vfma(v10, va1, b0);
        v11 = rl_vfma(v11, va1, b1);
        v12 = rl_vfma(v12, va1, b2);
        v13 = rl_vfma(v13, va1, b3);
      }
      rl_vstore(orow0 + j, v00);
      rl_vstore(orow0 + j + W, v01);
      rl_vstore(orow0 + j + 2u * W, v02);
      rl_vstore(orow0 + j + 3u * W, v03);
      rl_vstore(orow1 + j, v10);
      rl_vstore(orow1 + j + W, v11);
      rl_vstore(orow1 + j + 2u * W, v12);
      rl_vstore(orow1 + j + 3u * W, v13);
    }

    for (; j + W <= width; j += W) {
      RlVec v0 = rl_vload(orow0 + j);
      RlVec v1 = rl_vload(orow1 + j);
      for (u32 k = k0; k < k1; k++) {
        const u64 ai0 = transpose_a ? (u64)k * M + i : (u64)i * a->cols + k;
        const RlVec bv = rl_vload(b->data + (u64)(k - k0) * b_stride + j);
        v0 = rl_vfma(v0, rl_vset1(a->data[ai0]), bv);
        v1 = rl_vfma(v1, rl_vset1(a->data[ai0 + a_row_step]), bv);
      }
      rl_vstore(orow0 + j, v0);
      rl_vstore(orow1 + j, v1);
    }

    for (; j < width; j++) {
      real s0 = orow0[j];
      real s1 = orow1[j];
      for (u32 k = k0; k < k1; k++) {
        const u64 ai0 = transpose_a ? (u64)k * M + i : (u64)i * a->cols + k;
        const real bv = b->data[(u64)(k - k0) * b_stride + j];
        s0 += a->data[ai0] * bv;
        s1 += a->data[ai0 + a_row_step] * bv;
      }
      orow0[j] = s0;
      orow1[j] = s1;
    }
  }

  // Odd row out: the same three loops for a single row.
  if (i < M) {
    real *orow = out->data + (u64)i * out_stride + j0;
    u32 j = 0;

    for (; j + BLK <= width; j += BLK) {
      RlVec v0 = rl_vload(orow + j);
      RlVec v1 = rl_vload(orow + j + W);
      RlVec v2 = rl_vload(orow + j + 2u * W);
      RlVec v3 = rl_vload(orow + j + 3u * W);
      for (u32 k = k0; k < k1; k++) {
        const u64 ai = transpose_a ? (u64)k * M + i : (u64)i * a->cols + k;
        const RlVec va = rl_vset1(a->data[ai]);
        const real *brow = b->data + (u64)(k - k0) * b_stride + j;
        v0 = rl_vfma(v0, va, rl_vload(brow));
        v1 = rl_vfma(v1, va, rl_vload(brow + W));
        v2 = rl_vfma(v2, va, rl_vload(brow + 2u * W));
        v3 = rl_vfma(v3, va, rl_vload(brow + 3u * W));
      }
      rl_vstore(orow + j, v0);
      rl_vstore(orow + j + W, v1);
      rl_vstore(orow + j + 2u * W, v2);
      rl_vstore(orow + j + 3u * W, v3);
    }

    for (; j + W <= width; j += W) {
      RlVec acc = rl_vload(orow + j);
      for (u32 k = k0; k < k1; k++) {
        const u64 ai = transpose_a ? (u64)k * M + i : (u64)i * a->cols + k;
        acc = rl_vfma(acc, rl_vset1(a->data[ai]),
                      rl_vload(b->data + (u64)(k - k0) * b_stride + j));
      }
      rl_vstore(orow + j, acc);
    }

    for (; j < width; j++) {
      real acc = orow[j];
      for (u32 k = k0; k < k1; k++) {
        const u64 ai = transpose_a ? (u64)k * M + i : (u64)i * a->cols + k;
        acc += a->data[ai] * b->data[(u64)(k - k0) * b_stride + j];
      }
      orow[j] = acc;
    }
  }
}

RL_SIMD_KERNEL static void simd_mm_nn(Mat *out, const Mat *a, const Mat *b) {
  simd_mm_columns_range(out, a, b, false, 0, a->cols, 0, out->cols);
}

RL_SIMD_KERNEL static void simd_mm_tn(Mat *out, const Mat *a, const Mat *b) {
  simd_mm_columns_range(out, a, b, true, 0, a->rows, 0, out->cols);
}

// out += A @ B^T, where B is stored transposed (b[j][k]). The K reduction is
// vectorised and the lane partials are summed horizontally.
RL_SIMD_KERNEL static void simd_mm_nt(Mat *out, const Mat *a, const Mat *b) {
  const u32 M = out->rows;
  const u32 N = out->cols;
  const u32 K = a->cols;
  for (u32 i = 0; i < M; i++) {
    const real *arow = a->data + (u64)i * K;
    real *orow = out->data + (u64)i * N;
    for (u32 j = 0; j < N; j++) {
      const real *brow = b->data + (u64)j * K;
      RlVec acc = rl_vzero();
      u32 k = 0;
      for (; k + RL_VEC_W <= K; k += RL_VEC_W) {
        acc = rl_vfma(acc, rl_vload(arow + k), rl_vload(brow + k));
      }
      real s = rl_vhsum(acc);
      for (; k < K; k++) {
        s += arow[k] * brow[k];
      }
      orow[j] += s;
    }
  }
}

// out += A^T @ B^T, where A is stored K x M and B stored N x K (both
// transposed). Transposing a tile of B turns this into the TN form the kernel
// above already handles, which is the standard treatment (BLAS does the same
// thing): the multiply stays the vectorised one, and the transpose costs one
// O(N*K) pass over B against an O(M*N*K) multiply. The tile keeps the scratch
// block in L1 instead of materialising a whole K x N copy.
#define RL_TT_TILE 32u

RL_SIMD_KERNEL static void simd_mm_tt(Mat *out, const Mat *a, const Mat *b) {
  const u32 N = out->cols;
  const u32 K = a->rows;
  real scratch[RL_TT_TILE * RL_TT_TILE];
  Mat tile;
  tile.rows = 0;
  tile.cols = 0;
  tile.data = scratch;

  for (u32 kt = 0; kt < K; kt += RL_TT_TILE) {
    const u32 kmax = (kt + RL_TT_TILE < K) ? kt + RL_TT_TILE : K;
    for (u32 jt = 0; jt < N; jt += RL_TT_TILE) {
      const u32 jmax = (jt + RL_TT_TILE < N) ? jt + RL_TT_TILE : N;
      const u32 tw = jmax - jt;

      // Transpose the tile: scratch[(k - kt) * tw + (j - jt)] = b[j][k].
      for (u32 j = jt; j < jmax; j++) {
        const real *brow = b->data + (u64)j * K;
        real *dst = scratch + (j - jt);
        for (u32 k = kt; k < kmax; k++) {
          dst[(u64)(k - kt) * tw] = brow[k];
        }
      }

      // out[:, jt:jmax] += A^T[:, kt:kmax] @ scratch.
      tile.rows = kmax - kt;
      tile.cols = tw;
      simd_mm_columns_range(out, a, &tile, true, kt, kmax, jt, jmax);
    }
  }
}

#undef RL_TT_TILE

#endif // RL_SIMD_BACKEND != RL_SIMD_NONE

// Cached SIMD availability: a runtime AVX2/FMA probe on x86-64, a compile-time
// constant on NEON, and false when no backend is compiled in.
#if RL_SIMD_BACKEND != RL_SIMD_NONE
static b32 use_simd(void) {
  static int cached = -1;
  if (cached < 0) {
    cached = simd_accelerated() ? 1 : 0;
  }
  return cached != 0;
}
#endif

// Shared operand-shape check for both the scalar and the dispatching entry
// points. Returns false on any mismatch, so callers can reject the call.
static b32 mm_shape_ok(const Mat *out, const Mat *a, const Mat *b, b8 transpose_a,
                       b8 transpose_b) {
  const u32 a_rows = transpose_a ? a->cols : a->rows;
  const u32 a_cols = transpose_a ? a->rows : a->cols;
  const u32 b_rows = transpose_b ? b->cols : b->rows;
  const u32 b_cols = transpose_b ? b->rows : b->cols;

  if (a_cols != b_rows) {
    return false;
  }
  return out->rows == a_rows && out->cols == b_cols;
}

b32 mat_mm_scalar(Mat *out, const Mat *a, const Mat *b, b8 zero_out,
                  b8 transpose_a, b8 transpose_b) {
  if (!mm_shape_ok(out, a, b, transpose_a, transpose_b)) {
    return false;
  }
  if (zero_out) {
    mat_zero(out);
  }

  if (!transpose_a && !transpose_b) {
    mm_nn(out, a, b);
  } else if (!transpose_a && transpose_b) {
    mm_nt(out, a, b);
  } else if (transpose_a && !transpose_b) {
    mm_tn(out, a, b);
  } else {
    mm_tt(out, a, b);
  }
  return true;
}

b32 mat_mm(Mat *out, const Mat *a, const Mat *b, b8 zero_out, b8 transpose_a,
           b8 transpose_b) {
  if (!mm_shape_ok(out, a, b, transpose_a, transpose_b)) {
    return false;
  }
  if (zero_out) {
    mat_zero(out);
  }

#if RL_SIMD_BACKEND != RL_SIMD_NONE
  if (use_simd()) {
    if (!transpose_a && !transpose_b) {
      simd_mm_nn(out, a, b);
      return true;
    }
    if (!transpose_a && transpose_b) {
      simd_mm_nt(out, a, b);
      return true;
    }
    if (transpose_a && !transpose_b) {
      simd_mm_tn(out, a, b);
      return true;
    }
    if (transpose_a && transpose_b) {
      simd_mm_tt(out, a, b);
      return true;
    }
  }
#endif

  if (!transpose_a && !transpose_b) {
    mm_nn(out, a, b);
  } else if (!transpose_a && transpose_b) {
    mm_nt(out, a, b);
  } else if (transpose_a && !transpose_b) {
    mm_tn(out, a, b);
  } else {
    mm_tt(out, a, b);
  }
  return true;
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

b32 mat_relu(Mat *out, const Mat *in) {
  if (out->rows != in->rows || out->cols != in->cols) {
    return false;
  }
  const u64 n = mat_count(out);
  for (u64 i = 0; i < n; i++) {
    const real x = in->data[i];
    out->data[i] = x > 0.0f ? x : 0.0f;
  }
  return true;
}

b32 mat_relu_add_grad(Mat *out, const Mat *in, const Mat *grad) {
  if (out->rows != in->rows || out->cols != in->cols) {
    return false;
  }
  if (out->rows != grad->rows || out->cols != grad->cols) {
    return false;
  }
  const u64 n = mat_count(out);
  for (u64 i = 0; i < n; i++) {
    if (in->data[i] > 0.0f) {
      out->data[i] += grad->data[i];
    }
  }
  return true;
}

b32 mat_tanh(Mat *out, const Mat *in) {
  if (out->rows != in->rows || out->cols != in->cols) {
    return false;
  }
  const u64 n = mat_count(out);
  for (u64 i = 0; i < n; i++) {
    out->data[i] = real_tanh(in->data[i]);
  }
  return true;
}

b32 mat_tanh_add_grad(Mat *out, const Mat *tanh_out, const Mat *grad) {
  if (out->rows != tanh_out->rows || out->cols != tanh_out->cols) {
    return false;
  }
  if (out->rows != grad->rows || out->cols != grad->cols) {
    return false;
  }
  const u64 n = mat_count(out);
  for (u64 i = 0; i < n; i++) {
    const real y = tanh_out->data[i];
    out->data[i] += grad->data[i] * (1.0f - y * y);
  }
  return true;
}

// Per-column softmax: each column c is an independent distribution over rows.
b32 mat_softmax(Mat *out, const Mat *in) {
  if (out->rows != in->rows || out->cols != in->cols) {
    return false;
  }
  const u32 rows = in->rows;
  const u32 cols = in->cols;

  for (u32 c = 0; c < cols; c++) {
    real hi = in->data[c];
    for (u32 r = 1; r < rows; r++) {
      const real v = in->data[(u64)r * cols + c];
      if (v > hi) {
        hi = v;
      }
    }

    real sum = 0.0f;
    for (u32 r = 0; r < rows; r++) {
      const real e = real_exp(in->data[(u64)r * cols + c] - hi);
      out->data[(u64)r * cols + c] = e;
      sum += e;
    }

    const real inv = 1.0f / sum;
    for (u32 r = 0; r < rows; r++) {
      out->data[(u64)r * cols + c] *= inv;
    }
  }
  return true;
}

b32 mat_softmax_add_grad(Mat *out, const Mat *softmax_out, const Mat *grad) {
  if (out->rows != softmax_out->rows || out->cols != softmax_out->cols) {
    return false;
  }
  if (grad->rows != softmax_out->rows || grad->cols != softmax_out->cols) {
    return false;
  }
  const u32 rows = softmax_out->rows;
  const u32 cols = softmax_out->cols;

  for (u32 c = 0; c < cols; c++) {
    real dot = 0.0f;
    for (u32 r = 0; r < rows; r++) {
      const u64 idx = (u64)r * cols + c;
      dot += grad->data[idx] * softmax_out->data[idx];
    }
    for (u32 r = 0; r < rows; r++) {
      const u64 idx = (u64)r * cols + c;
      out->data[idx] += softmax_out->data[idx] * (grad->data[idx] - dot);
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Losses
// ---------------------------------------------------------------------------

b32 mat_mse_loss(Mat *out, const Mat *pred, const Mat *target) {
  if (pred->rows != target->rows || pred->cols != target->cols) {
    return false;
  }
  if (out->rows != 1u || out->cols != 1u) {
    return false;
  }

  const u64 n = mat_count(pred);
  real sum_sq = 0.0f;
  for (u64 i = 0; i < n; i++) {
    const real d = pred->data[i] - target->data[i];
    sum_sq += d * d;
  }
  out->data[0] = sum_sq / (real)n;
  return true;
}

b32 mat_mse_add_grad(Mat *pred_grad, const Mat *pred, const Mat *target,
                     const Mat *grad) {
  if (pred->rows != target->rows || pred->cols != target->cols) {
    return false;
  }
  if (pred_grad->rows != pred->rows || pred_grad->cols != pred->cols) {
    return false;
  }
  const u64 n = mat_count(pred);
  const real scale = (2.0f / (real)n) * grad->data[0];
  for (u64 i = 0; i < n; i++) {
    pred_grad->data[i] += scale * (pred->data[i] - target->data[i]);
  }
  return true;
}

b32 mat_reinforce_loss(Mat *out, const Mat *probs, const Mat *advantages) {
  if (probs->rows != advantages->rows || probs->cols != advantages->cols) {
    return false;
  }
  if (out->rows != advantages->rows || out->cols != advantages->cols) {
    return false;
  }
  const u64 n = mat_count(out);
  for (u64 i = 0; i < n; i++) {
    const real p = MAX(probs->data[i], 1e-8f);
    out->data[i] = -real_log(p) * advantages->data[i];
  }
  return true;
}

b32 mat_reinforce_add_grad(Mat *prob_grads, const Mat *probs, const Mat *rt,
                           const Mat *grad) {
  if (probs->rows != rt->rows || probs->cols != rt->cols) {
    return false;
  }
  if (grad->rows != probs->rows || grad->cols != probs->cols) {
    return false;
  }
  if (prob_grads->rows != probs->rows || prob_grads->cols != probs->cols) {
    return false;
  }
  const u64 n = mat_count(probs);
  for (u64 i = 0; i < n; i++) {
    const real p = MAX(probs->data[i], 1e-8f);
    prob_grads->data[i] += grad->data[i] * (-rt->data[i] / p);
  }
  return true;
}

// ---------------------------------------------------------------------------
// Fused batched policy loss over columns (act_dim rows x batch_size cols).
// ---------------------------------------------------------------------------

// Writes softmax(logits[:, b]) into probs (length act_dim) and returns the
// column entropy H_b = -sum p log p, stabilised by the column max.
static real column_softmax_entropy(const Mat *logits, u32 b, real *probs) {
  const u32 act_dim = logits->rows;
  const u32 batch = logits->cols;

  real hi = logits->data[b];
  for (u32 a = 1; a < act_dim; a++) {
    const real z = logits->data[(u64)a * batch + b];
    if (z > hi) {
      hi = z;
    }
  }

  real sum = 0.0f;
  for (u32 a = 0; a < act_dim; a++) {
    const real e = real_exp(logits->data[(u64)a * batch + b] - hi);
    probs[a] = e;
    sum += e;
  }

  const real inv = 1.0f / sum;
  real entropy = 0.0f;
  for (u32 a = 0; a < act_dim; a++) {
    const real p = probs[a] * inv;
    probs[a] = p;
    if (p > 0.0f) {
      entropy -= p * real_log(p);
    }
  }
  return entropy;
}

b32 mat_fused_policy_loss(Mat *out, const Mat *logits, const u32 *actions,
                          const real *advantages, u32 batch_size) {
  if (out == NULL || logits == NULL || actions == NULL ||
      advantages == NULL) {
    return false;
  }
  if (out->rows != 1u || out->cols != 1u || logits->cols != batch_size) {
    return false;
  }

  const u32 act_dim = logits->rows;
  const u32 batch = batch_size;
  real total = 0.0f;

  for (u32 b = 0; b < batch; b++) {
    const u32 chosen = actions[b];
    if (chosen >= act_dim) {
      return false;
    }

    real hi = logits->data[b];
    for (u32 a = 1; a < act_dim; a++) {
      const real z = logits->data[(u64)a * batch + b];
      if (z > hi) {
        hi = z;
      }
    }
    real sum = 0.0f;
    for (u32 a = 0; a < act_dim; a++) {
      sum += real_exp(logits->data[(u64)a * batch + b] - hi);
    }

    const real log_prob = logits->data[(u64)chosen * batch + b] -
                          (hi + real_log(sum));
    total += -advantages[b] * log_prob;
  }

  out->data[0] = total / (real)batch;
  return true;
}

b32 mat_fused_policy_add_grad(Mat *logits_grad, const Mat *logits,
                              const u32 *actions, const real *advantages,
                              u32 batch_size, const Mat *grad) {
  if (logits == NULL || logits_grad == NULL || actions == NULL ||
      advantages == NULL || grad == NULL) {
    return false;
  }
  if (logits->cols != batch_size || logits_grad->cols != batch_size) {
    return false;
  }

  const u32 act_dim = logits->rows;
  const u32 batch = batch_size;
  const real g_scale = grad->data[0] / (real)batch;

  for (u32 b = 0; b < batch; b++) {
    const u32 chosen = actions[b];

    real hi = logits->data[b];
    for (u32 a = 1; a < act_dim; a++) {
      const real z = logits->data[(u64)a * batch + b];
      if (z > hi) {
        hi = z;
      }
    }
    real sum = 0.0f;
    for (u32 a = 0; a < act_dim; a++) {
      sum += real_exp(logits->data[(u64)a * batch + b] - hi);
    }
    const real inv = 1.0f / sum;
    const real adv_scale = advantages[b] * g_scale;

    for (u32 a = 0; a < act_dim; a++) {
      const real p =
          real_exp(logits->data[(u64)a * batch + b] - hi) * inv;
      const real indicator = (a == chosen) ? 1.0f : 0.0f;
      // d/da of -adv * log p_a is adv * (p_a - 1[a == chosen]).
      logits_grad->data[(u64)a * batch + b] += adv_scale * (p - indicator);
    }
  }
  return true;
}

b32 mat_entropy_loss(Mat *out, const Mat *logits, u32 batch_size) {
  if (out == NULL || logits == NULL) {
    return false;
  }
  if (out->rows != 1u || out->cols != 1u || batch_size == 0u) {
    return false;
  }
  if (logits->cols != batch_size) {
    return false;
  }

  real *probs = malloc(sizeof(real) * logits->rows);
  if (probs == NULL) {
    return false;
  }

  real total = 0.0f;
  for (u32 b = 0; b < batch_size; b++) {
    total += column_softmax_entropy(logits, b, probs);
  }
  free(probs);

  out->data[0] = total / (real)batch_size;
  return true;
}

b32 mat_entropy_add_grad(Mat *logits_grad, const Mat *logits, u32 batch_size,
                         const Mat *grad) {
  if (logits_grad == NULL || logits == NULL || grad == NULL) {
    return false;
  }
  if (logits->cols != batch_size || logits_grad->cols != batch_size) {
    return false;
  }
  if (logits->rows != logits_grad->rows) {
    return false;
  }

  const u32 act_dim = logits->rows;
  real *probs = malloc(sizeof(real) * act_dim);
  if (probs == NULL) {
    return false;
  }

  const real g_scale = grad->data[0] / (real)batch_size;
  for (u32 b = 0; b < batch_size; b++) {
    const real h = column_softmax_entropy(logits, b, probs);
    for (u32 a = 0; a < act_dim; a++) {
      const real p = probs[a];
      const real dH = -p * (real_log(p + 1e-30f) + h);
      logits_grad->data[(u64)a * batch_size + b] += g_scale * dH;
    }
  }
  free(probs);
  return true;
}

// ---------------------------------------------------------------------------
// PPO clipped surrogate policy loss over columns (act_dim rows x batch cols).
// ---------------------------------------------------------------------------

// Column log-sum-exp: max_z + log(sum_z exp(z - max)). `logits->cols` is the
// batch size and `b` selects the column.
static real column_logsumexp(const Mat *logits, u32 b) {
  const u32 act_dim = logits->rows;
  const u32 batch = logits->cols;

  real hi = logits->data[b];
  for (u32 a = 1; a < act_dim; a++) {
    const real z = logits->data[(u64)a * batch + b];
    if (z > hi) {
      hi = z;
    }
  }

  real sum = 0.0f;
  for (u32 a = 0; a < act_dim; a++) {
    sum += real_exp(logits->data[(u64)a * batch + b] - hi);
  }
  return hi + real_log(sum);
}

static real ppo_clamp(real x, real lo, real hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// Numerical guard on the importance ratio. exp() overflows f32 at ~88 nats, and
// a single overflowing sample would poison the whole minibatch's gradient with
// Inf, so the log-ratio is clamped before the exponential. A ratio beyond
// e^+-20 is far outside PPO's trust region (the clip range is +-0.2), so the
// clamp only ever engages on samples whose gradient the surrogate already
// discounts.
#define PPO_LOG_RATIO_MAX ((real)20)

// exp(new_logp - old_logp) with the overflow guard applied.
static real ppo_ratio(real new_logp, real old_logp) {
  return real_exp(ppo_clamp(new_logp - old_logp, -PPO_LOG_RATIO_MAX,
                            PPO_LOG_RATIO_MAX));
}

b32 mat_ppo_policy_loss(Mat *out, const Mat *logits, const u32 *actions,
                        const real *advantages, const real *old_logp,
                        real clip_eps, u32 batch_size) {
  if (out == NULL || logits == NULL || actions == NULL ||
      advantages == NULL || old_logp == NULL) {
    return false;
  }
  if (out->rows != 1u || out->cols != 1u || logits->cols != batch_size ||
      clip_eps <= 0.0f) {
    return false;
  }

  const u32 act_dim = logits->rows;
  const real lo = 1.0f - clip_eps;
  const real hi = 1.0f + clip_eps;
  real total = 0.0f;

  for (u32 b = 0; b < batch_size; b++) {
    const u32 chosen = actions[b];
    if (chosen >= act_dim) {
      return false;
    }
    const real lse = column_logsumexp(logits, b);
    const real new_logp = logits->data[(u64)chosen * batch_size + b] - lse;
    const real ratio = ppo_ratio(new_logp, old_logp[b]);
    const real adv = advantages[b];
    const real surr1 = ratio * adv;
    const real surr2 = ppo_clamp(ratio, lo, hi) * adv;
    total += (surr1 < surr2) ? surr1 : surr2;
  }

  out->data[0] = -total / (real)batch_size;
  return true;
}

b32 mat_ppo_policy_add_grad(Mat *logits_grad, const Mat *logits,
                            const u32 *actions, const real *advantages,
                            const real *old_logp, real clip_eps,
                            u32 batch_size, const Mat *grad) {
  if (logits_grad == NULL || logits == NULL || actions == NULL ||
      advantages == NULL || old_logp == NULL || grad == NULL) {
    return false;
  }
  if (logits->cols != batch_size || logits_grad->cols != batch_size ||
      logits->rows != logits_grad->rows || clip_eps <= 0.0f) {
    return false;
  }

  const u32 act_dim = logits->rows;
  const real lo = 1.0f - clip_eps;
  const real hi = 1.0f + clip_eps;
  const real g_scale = grad->data[0] / (real)batch_size;

  for (u32 b = 0; b < batch_size; b++) {
    const u32 chosen = actions[b];
    if (chosen >= act_dim) {
      return false;
    }
    const real lse = column_logsumexp(logits, b);
    const real new_logp = logits->data[(u64)chosen * batch_size + b] - lse;
    const real ratio = ppo_ratio(new_logp, old_logp[b]);
    const real adv = advantages[b];
    const real surr1 = ratio * adv;
    const real surr2 = ppo_clamp(ratio, lo, hi) * adv;

    // d(min)/d(ratio) is `adv` when the unclipped branch is the minimum (which
    // includes the in-range case) and 0 when the clip is binding.
    const real d_ratio = (surr1 <= surr2) ? adv : 0.0f;
    // loss = -mean(min), so d(loss)/d(new_logp) = -d_ratio * ratio, and
    // d(new_logp)/d(logit_a) = 1[a == chosen] - softmax_a.
    const real dnew = -d_ratio * ratio;
    for (u32 a = 0; a < act_dim; a++) {
      const real p = real_exp(logits->data[(u64)a * batch_size + b] - lse);
      const real indicator = (a == chosen) ? 1.0f : 0.0f;
      logits_grad->data[(u64)a * batch_size + b] +=
          g_scale * dnew * (indicator - p);
    }
  }
  return true;
}

// ── Squashed diagonal-Gaussian policy (continuous actions) ────────────
// Actions live in a box, so the policy is a diagonal Gaussian over a pre-squash
// coordinate u, squashed into that box with tanh:
//
//   u_d = mu_d + exp(log_std_d) * eps,   eps ~ N(0, 1)
//   a_d = center_d + half_d * tanh(u_d)
//
// with center_d = (min_d + max_d) / 2 and half_d = (max_d - min_d) / 2. The
// density is the Gaussian density on u, corrected by the log-determinant of the
// squash:
//
//   log pi(a) = log N(u | mu, sigma) - log(half_d) - log(sech^2(u_d))
//
// Four numerical decisions make this safe on a bounded action space; each one
// was paid for by a failure of a simpler design.
//
//   * The density is evaluated at the *sampled* u, which the sampler hands back
//     next to the action. Recovering u from the action instead needs atanh, which
//     is infinite at a = +-half, and once the mean walks outside the box the
//     recovered u no longer describes the sample that was drawn. That mismatch
//     produced log-probabilities off by hundreds of nats and diverged training.
//   * log(sech^2(u)) = 2 * (log 2 - |u| - log1p(exp(-2|u|))), which stays finite
//     for any u -- unlike log(1 - tanh(u)^2) evaluated through tanh, which is
//     -inf as soon as tanh saturates in f32 (|u| > ~9).
//   * The mean is clamped to +-RL_GAUSSIAN_MEAN_BOUND. The squash does not bound the
//     mean, and the density's peak height grows like exp(2|mu|), so an unclamped
//     mean lets the policy chase a spiking density far outside the box.
//     Gradients pass through the clamp unchanged (straight-through) because an
//     exact zero gradient outside the bound would park the mean head forever.
//   * log_std is clamped on read to [RL_GAUSSIAN_LOG_STD_MIN, RL_GAUSSIAN_LOG_STD_MAX], which
//     floors sigma (no division by zero, no collapsed exploration) and keeps the
//     noise from swamping the box. The clamp zeroes the gradient when active, so
//     the parameter is not driven further into the wall.
//
// The score gradients are the plain Gaussian ones, because the squash's
// Jacobian depends on the sample but not on mu or log_std:
//
//   d(log pi)/d(mu)      = (u - mu) / sigma^2
//   d(log pi)/d(log_std) = (u - mu)^2 / sigma^2 - 1
//
// `min_action` and `max_action` each hold act_dim entries, matching
// Env.action_min / Env.action_max.

// Mean head output for action dimension d, column b, clamped into
// [-RL_GAUSSIAN_MEAN_BOUND, RL_GAUSSIAN_MEAN_BOUND]. A single-column mean is broadcast.
static real gaussian_mean_at(const Mat *raw_mean, u32 d, u32 b) {
  real m =
      raw_mean->data[(u64)d * raw_mean->cols + (raw_mean->cols == 1u ? 0u : b)];
  if (m > RL_GAUSSIAN_MEAN_BOUND) {
    m = RL_GAUSSIAN_MEAN_BOUND;
  }
  if (m < -RL_GAUSSIAN_MEAN_BOUND) {
    m = -RL_GAUSSIAN_MEAN_BOUND;
  }
  return m;
}

// log_std for dimension d after the read-time clamp. `deriv_out` (optional)
// receives 1 while the gradient should reach the parameter and 0 when the clamp
// is active.
static real gaussian_log_std_at(const Mat *log_std, u32 d, real *deriv_out) {
  real l = log_std->data[(u64)d * log_std->cols];
  b32 inside = true;
  if (l > RL_GAUSSIAN_LOG_STD_MAX) {
    l = RL_GAUSSIAN_LOG_STD_MAX;
    inside = false;
  } else if (l < RL_GAUSSIAN_LOG_STD_MIN) {
    l = RL_GAUSSIAN_LOG_STD_MIN;
    inside = false;
  }
  if (deriv_out != NULL) {
    *deriv_out = inside ? (real)1 : (real)0;
  }
  return l;
}

// log(sech^2(u)) = log(1 - tanh(u)^2), computed without ever forming tanh(u)^2
// so it stays finite and accurate for arbitrarily large |u|.
static real gaussian_log_sech2(real u) {
  const real a = real_fabs(u);
  const real log_2 = (real)0.69314718055994530942;
  return (real)2 *
         ((log_2 - a) - real_log1p(real_exp((real)-2 * a)));
}

// log pi(a) summed over the action dimensions for column b of an
// (act_dim x batch_size) pre-squash sample block.
static real gaussian_logp_at(const Mat *raw_mean, const Mat *log_std,
                             const Mat *u, const real *min_action,
                             const real *max_action, u32 batch_size, u32 b) {
  const real log_2pi = (real)1.8378770664093454835606594728112;
  real acc = (real)0;
  for (u32 d = 0; d < raw_mean->rows; d++) {
    const real half = ((real)0.5) * (max_action[d] - min_action[d]);
    const real uu = u->data[(u64)d * batch_size + b];
    const real log_sigma = gaussian_log_std_at(log_std, d, NULL);
    const real eps =
        (uu - gaussian_mean_at(raw_mean, d, b)) / real_exp(log_sigma);
    acc += (real)-0.5 * eps * eps - log_sigma - (real)0.5 * log_2pi -
           real_log(half) - gaussian_log_sech2(uu);
  }
  return acc;
}

b32 mat_gaussian_sample(Mat *action_out, Mat *u_out, const Mat *raw_mean,
                        const Mat *log_std, const real *min_action,
                        const real *max_action, Rng *rng) {
  if (action_out == NULL || u_out == NULL || raw_mean == NULL ||
      log_std == NULL || rng == NULL || min_action == NULL ||
      max_action == NULL) {
    return false;
  }
  if (action_out->rows != raw_mean->rows || action_out->cols != 1u ||
      u_out->rows != raw_mean->rows || u_out->cols != 1u ||
      raw_mean->cols != 1u || log_std->rows != raw_mean->rows ||
      log_std->cols != 1u) {
    return false;
  }

  for (u32 d = 0; d < raw_mean->rows; d++) {
    const real center = ((real)0.5) * (min_action[d] + max_action[d]);
    const real half = ((real)0.5) * (max_action[d] - min_action[d]);
    const real sigma = real_exp(gaussian_log_std_at(log_std, d, NULL));
    const real u = gaussian_mean_at(raw_mean, d, 0u) + sigma * rng_normal(rng);
    u_out->data[d] = u;
    action_out->data[d] = center + half * real_tanh(u);
  }
  return true;
}

// Inverse of the squash for an action that was not produced by
// mat_gaussian_sample (a stored replay, say). Near the bounds this is
// lossy -- atanh blows up at a = +-half, so the normalised action is clamped
// just inside the box -- which is exactly why the training path carries u
// instead of re-deriving it from the action.
b32 mat_gaussian_unsquash(Mat *u_out, const Mat *actions,
                          const real *min_action, const real *max_action) {
  if (u_out == NULL || actions == NULL || min_action == NULL ||
      max_action == NULL) {
    return false;
  }
  if (u_out->rows != actions->rows || u_out->cols != actions->cols) {
    return false;
  }

  const real limit = (real)1 - (real)1e-6;
  for (u32 d = 0; d < actions->rows; d++) {
    const real center = ((real)0.5) * (min_action[d] + max_action[d]);
    const real half = ((real)0.5) * (max_action[d] - min_action[d]);
    for (u32 b = 0; b < actions->cols; b++) {
      real x = (actions->data[(u64)d * actions->cols + b] - center) / half;
      if (x > limit) {
        x = limit;
      }
      if (x < -limit) {
        x = -limit;
      }
      u_out->data[(u64)d * actions->cols + b] = real_atanh(x);
    }
  }
  return true;
}

b32 mat_gaussian_logp(Mat *out, const Mat *raw_mean, const Mat *log_std,
                      const Mat *u, const real *min_action,
                      const real *max_action, u32 batch_size) {
  if (out == NULL || raw_mean == NULL || log_std == NULL || u == NULL ||
      min_action == NULL || max_action == NULL) {
    return false;
  }
  const u32 act_dim = raw_mean->rows;
  if (act_dim == 0u || batch_size == 0u || out->rows != 1u ||
      out->cols != batch_size || u->rows != act_dim || u->cols != batch_size ||
      log_std->rows != act_dim || log_std->cols != 1u ||
      (raw_mean->cols != 1u && raw_mean->cols != batch_size)) {
    return false;
  }

  for (u32 b = 0; b < batch_size; b++) {
    out->data[b] = gaussian_logp_at(raw_mean, log_std, u, min_action,
                                    max_action, batch_size, b);
  }
  return true;
}

b32 mat_gaussian_ppo_loss(Mat *out, const Mat *raw_mean, const Mat *log_std,
                          const Mat *u, const real *advantages,
                          const real *old_logp, const real *min_action,
                          const real *max_action, real clip_eps,
                          u32 batch_size) {
  if (out == NULL || raw_mean == NULL || log_std == NULL || u == NULL ||
      advantages == NULL || old_logp == NULL || min_action == NULL ||
      max_action == NULL) {
    return false;
  }
  const u32 act_dim = raw_mean->rows;
  if (act_dim == 0u || batch_size == 0u || out->rows != 1u ||
      out->cols != 1u || clip_eps <= (real)0 ||
      raw_mean->cols != batch_size || u->rows != act_dim ||
      u->cols != batch_size || log_std->rows != act_dim ||
      log_std->cols != 1u) {
    return false;
  }

  const real lo = (real)1 - clip_eps;
  const real hi = (real)1 + clip_eps;
  real acc = (real)0;
  for (u32 b = 0; b < batch_size; b++) {
    const real new_logp = gaussian_logp_at(raw_mean, log_std, u, min_action,
                                           max_action, batch_size, b);
    const real ratio = ppo_ratio(new_logp, old_logp[b]);
    const real adv = advantages[b];
    const real surr1 = ratio * adv;
    const real surr2 = ppo_clamp(ratio, lo, hi) * adv;
    acc += (surr1 < surr2) ? surr1 : surr2;
  }
  out->data[0] = -acc / (real)batch_size;
  return true;
}

b32 mat_gaussian_ppo_add_grad(Mat *mean_grad, Mat *log_std_grad,
                              const Mat *raw_mean, const Mat *log_std,
                              const Mat *u, const real *advantages,
                              const real *old_logp, const real *min_action,
                              const real *max_action, real clip_eps,
                              u32 batch_size, const Mat *grad) {
  if (mean_grad == NULL || log_std_grad == NULL || raw_mean == NULL ||
      log_std == NULL || u == NULL || advantages == NULL ||
      old_logp == NULL || min_action == NULL || max_action == NULL ||
      grad == NULL) {
    return false;
  }
  const u32 act_dim = raw_mean->rows;
  if (act_dim == 0u || batch_size == 0u || clip_eps <= (real)0 ||
      raw_mean->cols != batch_size || mean_grad->rows != act_dim ||
      mean_grad->cols != batch_size || u->rows != act_dim ||
      u->cols != batch_size || log_std->rows != act_dim ||
      log_std->cols != 1u || log_std_grad->rows != act_dim ||
      log_std_grad->cols != 1u) {
    return false;
  }

  const real lo = (real)1 - clip_eps;
  const real hi = (real)1 + clip_eps;
  const real g_scale = grad->data[0] / (real)batch_size;

  for (u32 b = 0; b < batch_size; b++) {
    const real new_logp = gaussian_logp_at(raw_mean, log_std, u, min_action,
                                           max_action, batch_size, b);
    const real ratio = ppo_ratio(new_logp, old_logp[b]);
    const real adv = advantages[b];
    const real surr1 = ratio * adv;
    const real surr2 = ppo_clamp(ratio, lo, hi) * adv;

    // Same rule as the discrete loss: the unclipped branch carries the gradient
    // whenever it is the minimum (which includes the in-range case).
    const real d_ratio = (surr1 <= surr2) ? adv : (real)0;
    // loss = -mean(min), so d(loss)/d(new_logp) = -d_ratio * ratio.
    const real dnew = -d_ratio * ratio;

    for (u32 d = 0; d < act_dim; d++) {
      real ld_deriv = (real)0;
      const real log_sigma = gaussian_log_std_at(log_std, d, &ld_deriv);
      const real inv_sigma = real_exp(-log_sigma);
      const real eps =
          (u->data[(u64)d * batch_size + b] - gaussian_mean_at(raw_mean, d, b)) *
          inv_sigma;
      // The squash's Jacobian depends on the sample, not on mu or log_std, so
      // d(new_logp)/d(mu) = eps / sigma and d(new_logp)/d(log_std) = eps^2 - 1.
      mean_grad->data[(u64)d * batch_size + b] +=
          g_scale * dnew * eps * inv_sigma;
      log_std_grad->data[(u64)d * log_std_grad->cols] +=
          g_scale * dnew * (eps * eps - (real)1) * ld_deriv;
    }
  }
  return true;
}

b32 mat_gaussian_entropy(Mat *out, const Mat *log_std) {
  if (out == NULL || log_std == NULL || out->rows != 1u || out->cols != 1u ||
      log_std->cols != 1u) {
    return false;
  }
  // H = sum_d log(sigma_d) + 0.5 * D * log(2*pi*e)
  real acc = (real)0;
  for (u32 d = 0; d < log_std->rows; d++) {
    acc += gaussian_log_std_at(log_std, d, NULL);
  }
  const real log_2pi_e = (real)2.8378770664093454835606594728112;
  out->data[0] = acc + (real)0.5 * (real)log_std->rows * log_2pi_e;
  return true;
}

b32 mat_gaussian_entropy_add_grad(Mat *log_std_grad, const Mat *log_std,
                                  const Mat *grad) {
  if (log_std_grad == NULL || log_std == NULL || grad == NULL) {
    return false;
  }
  if (log_std_grad->rows != log_std->rows || log_std_grad->cols != 1u ||
      log_std->cols != 1u) {
    return false;
  }
  // d(mean H)/d(log_std_d) = 1, since H does not depend on the mean and the
  // batch mean of a constant-per-sample quantity is that constant. The
  // read-time clamp zeroes the gradient while it is active.
  const real scale = grad->data[0];
  for (u32 d = 0; d < log_std->rows; d++) {
    real ld_deriv = (real)0;
    (void)gaussian_log_std_at(log_std, d, &ld_deriv);
    log_std_grad->data[(u64)d * log_std_grad->cols] += scale * ld_deriv;
  }
  return true;
}
