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
// scalar reference. NN/TN keep the scalar reduction order (the output columns
// are vectorised); NT vectorises the K reduction and adds the lane partials
// horizontally, so it may differ from the scalar result by a few ULPs. TT is
// cold (the autograd engine never emits it) and stays on the scalar path.
// ---------------------------------------------------------------------------

#if RL_SIMD_BACKEND != RL_SIMD_NONE

#if RL_SIMD_BACKEND == RL_SIMD_NEON

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

#else // RL_SIMD_BACKEND == RL_SIMD_AVX2

// Per-function target attribute: these use AVX2/FMA even though the translation
// unit targets the baseline x86-64 ISA, so the binary still runs on CPUs
// without AVX2 (mat_mm() probes at runtime).
#define RL_AVX2_TARGET __attribute__((target("avx2,fma")))

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

// The kernels themselves are compiled with the AVX2/FMA target so the helpers
// inline; mat_mm() only calls them after the runtime probe succeeds.
#define RL_SIMD_KERNEL RL_AVX2_TARGET

#endif

#if RL_SIMD_BACKEND == RL_SIMD_NEON
#define RL_SIMD_KERNEL
#endif

// Shared NN/TN micro-kernel: out += A_eff @ B, where A_eff is A (M x K,
// row-major) when `transpose_a` is false and A^T (A stored K x M) when true.
// Four column vectors are held in registers across the whole K reduction, so
// `out` is loaded and stored once per row instead of once per (k, row). Each
// output element still accumulates over k in the same order as the scalar
// reference, so results match to floating-point tolerance.
RL_SIMD_KERNEL static void simd_mm_columns(Mat *out, const Mat *a, const Mat *b,
                                          b32 transpose_a) {
  const u32 M = out->rows;
  const u32 N = out->cols;
  const u32 K = transpose_a ? a->rows : a->cols;
  const u32 W = RL_VEC_W;
  const u32 BLK = 4u * W;

  for (u32 i = 0; i < M; i++) {
    real *orow = out->data + (u64)i * N;
    const real *arow = a->data + (u64)i * K;

    u32 j = 0;
    for (; j + BLK <= N; j += BLK) {
      RlVec v0 = rl_vload(orow + j);
      RlVec v1 = rl_vload(orow + j + W);
      RlVec v2 = rl_vload(orow + j + 2u * W);
      RlVec v3 = rl_vload(orow + j + 3u * W);
      for (u32 k = 0; k < K; k++) {
        const real aik = transpose_a ? a->data[(u64)k * M + i] : arow[k];
        const RlVec va = rl_vset1(aik);
        const real *brow = b->data + (u64)k * N + j;
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

    for (; j + W <= N; j += W) {
      RlVec acc = rl_vload(orow + j);
      for (u32 k = 0; k < K; k++) {
        const real aik = transpose_a ? a->data[(u64)k * M + i] : arow[k];
        acc = rl_vfma(acc, rl_vset1(aik), rl_vload(b->data + (u64)k * N + j));
      }
      rl_vstore(orow + j, acc);
    }

    for (; j < N; j++) {
      real acc = orow[j];
      for (u32 k = 0; k < K; k++) {
        const real aik = transpose_a ? a->data[(u64)k * M + i] : arow[k];
        acc += aik * b->data[(u64)k * N + j];
      }
      orow[j] = acc;
    }
  }
}

RL_SIMD_KERNEL static void simd_mm_nn(Mat *out, const Mat *a, const Mat *b) {
  simd_mm_columns(out, a, b, false);
}

RL_SIMD_KERNEL static void simd_mm_tn(Mat *out, const Mat *a, const Mat *b) {
  simd_mm_columns(out, a, b, true);
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
    // TT has no vector kernel; fall through to the scalar reference.
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
    const real ratio = real_exp(new_logp - old_logp[b]);
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
    const real ratio = real_exp(new_logp - old_logp[b]);
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
