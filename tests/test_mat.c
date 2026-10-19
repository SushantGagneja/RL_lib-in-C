// test_mat.c — Unit tests for the matrix operations in linalg.c
// Phase 0.2: Prove the math layer is correct.
#include "memory.h"
#include "base.h"
#include "linalg.h"
#include "random.h"
#include "simd.h"

#include <math.h>

// ── Helpers ──────────────────────────────────────────────────────────

static u32 tests_passed = 0;
static u32 tests_failed = 0;

#define ASSERT_TRUE(cond, msg)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: %s (line %d)\n", msg, __LINE__);                         \
      tests_failed++;                                                          \
    } else {                                                                   \
      tests_passed++;                                                          \
    }                                                                          \
  } while (0)

#define ASSERT_NEAR(a, b, tol, msg)                                            \
  ASSERT_TRUE(real_fabs((a) - (b)) < (tol), msg)

// Naive triple-loop mat_mm for reference (no transpose, always zeroes out).
static void naive_matmul(real *out, const real *a, const real *b, u32 M, u32 K,
                         u32 N) {
  for (u32 i = 0; i < M; i++) {
    for (u32 j = 0; j < N; j++) {
      real s = 0.0f;
      for (u32 k = 0; k < K; k++) {
        s += a[i * K + k] * b[k * N + j];
      }
      out[i * N + j] = s;
    }
  }
}

// ── Matmul tests ─────────────────────────────────────────────────────

static void test_matmul_nn(Arena *arena) {
  printf("  mat_mm NN... ");
  // Non-square: (3x5) * (5x4) = (3x4)
  Mat *a = mat_new(arena, 3, 5);
  Mat *b = mat_new(arena, 5, 4);
  Mat *out = mat_new(arena, 3, 4);

  rng_seed_g(42, 1);
  mat_rand(a, -1.0f, 1.0f);
  mat_rand(b, -1.0f, 1.0f);

  real ref[12];
  naive_matmul(ref, a->data, b->data, 3, 5, 4);

  mat_mm(out, a, b, 1, 0, 0);

  b32 ok = true;
  for (u32 i = 0; i < 12; i++) {
    if (real_fabs(out->data[i] - ref[i]) > 1e-5f) {
      ok = false;
      break;
    }
  }
  ASSERT_TRUE(ok, "matmul_nn (3x5)*(5x4)");
  printf("%s\n", ok ? "OK" : "FAIL");
}

static void test_matmul_nt(Arena *arena) {
  printf("  mat_mm NT... ");
  // A(3x5) * B^T where B is (4x5) → out(3x4)
  Mat *a = mat_new(arena, 3, 5);
  Mat *b = mat_new(arena, 4, 5); // will be transposed
  Mat *out = mat_new(arena, 3, 4);

  rng_seed_g(100, 2);
  mat_rand(a, -1.0f, 1.0f);
  mat_rand(b, -1.0f, 1.0f);

  // Build explicit transpose of b for reference
  real bt[20]; // 5x4
  for (u32 i = 0; i < 4; i++)
    for (u32 j = 0; j < 5; j++)
      bt[j * 4 + i] = b->data[i * 5 + j];

  real ref[12];
  naive_matmul(ref, a->data, bt, 3, 5, 4);

  mat_mm(out, a, b, 1, 0, 1);

  b32 ok = true;
  for (u32 i = 0; i < 12; i++) {
    if (real_fabs(out->data[i] - ref[i]) > 1e-5f) {
      ok = false;
      break;
    }
  }
  ASSERT_TRUE(ok, "matmul_nt A(3x5)*B^T(4x5)");
  printf("%s\n", ok ? "OK" : "FAIL");
}

static void test_matmul_tn(Arena *arena) {
  printf("  mat_mm TN... ");
  // A^T where A is (5x3) → effective A^T(3x5) * B(5x4) = out(3x4)
  Mat *a = mat_new(arena, 5, 3); // will be transposed
  Mat *b = mat_new(arena, 5, 4);
  Mat *out = mat_new(arena, 3, 4);

  rng_seed_g(200, 3);
  mat_rand(a, -1.0f, 1.0f);
  mat_rand(b, -1.0f, 1.0f);

  // Build explicit transpose of a for reference
  real at[15]; // 3x5
  for (u32 i = 0; i < 5; i++)
    for (u32 j = 0; j < 3; j++)
      at[j * 5 + i] = a->data[i * 3 + j];

  real ref[12];
  naive_matmul(ref, at, b->data, 3, 5, 4);

  mat_mm(out, a, b, 1, 1, 0);

  b32 ok = true;
  for (u32 i = 0; i < 12; i++) {
    if (real_fabs(out->data[i] - ref[i]) > 1e-5f) {
      ok = false;
      break;
    }
  }
  ASSERT_TRUE(ok, "matmul_tn A^T(5x3)*B(5x4)");
  printf("%s\n", ok ? "OK" : "FAIL");
}

static void test_matmul_tt(Arena *arena) {
  printf("  mat_mm TT... ");
  // A^T where A is (5x3) and B^T where B is (4x5) → out(3x4)
  Mat *a = mat_new(arena, 5, 3);
  Mat *b = mat_new(arena, 4, 5);
  Mat *out = mat_new(arena, 3, 4);

  rng_seed_g(300, 4);
  mat_rand(a, -1.0f, 1.0f);
  mat_rand(b, -1.0f, 1.0f);

  real at[15]; // 3x5
  for (u32 i = 0; i < 5; i++)
    for (u32 j = 0; j < 3; j++)
      at[j * 5 + i] = a->data[i * 3 + j];
  real bt[20]; // 5x4
  for (u32 i = 0; i < 4; i++)
    for (u32 j = 0; j < 5; j++)
      bt[j * 4 + i] = b->data[i * 5 + j];

  real ref[12];
  naive_matmul(ref, at, bt, 3, 5, 4);

  mat_mm(out, a, b, 1, 1, 1);

  b32 ok = true;
  for (u32 i = 0; i < 12; i++) {
    if (real_fabs(out->data[i] - ref[i]) > 1e-5f) {
      ok = false;
      break;
    }
  }
  ASSERT_TRUE(ok, "matmul_tt A^T(5x3)*B^T(4x5)");
  printf("%s\n", ok ? "OK" : "FAIL");
}

// The vectorised path must agree with the portable scalar reference across all
// transpose combinations. The shapes exercise the vector tails and the odd-row
// path, and the second one crosses the TT kernel's 32x32 transpose tile
// boundaries in both directions.
static void test_simd_matches_scalar(Arena *arena) {
  printf("  SIMD path == scalar reference... ");
  static const u32 shapes[2][3] = {{37, 41, 53}, {65, 70, 67}};

  b32 ok = true;
  real worst = 0.0f;
  for (u32 s = 0; s < 2; s++) {
    const u32 M = shapes[s][0];
    const u32 K = shapes[s][1];
    const u32 N = shapes[s][2];
    Mat *a = mat_new(arena, M, K);
    Mat *b = mat_new(arena, K, N);
    Mat *at = mat_new(arena, K, M);
    Mat *bt = mat_new(arena, N, K);
    Mat *ref = mat_new(arena, M, N);
    Mat *got = mat_new(arena, M, N);

    rng_seed_g(31337, 9);
    mat_rand(a, -1.0f, 1.0f);
    mat_rand(b, -1.0f, 1.0f);
    mat_rand(at, -1.0f, 1.0f);
    mat_rand(bt, -1.0f, 1.0f);

    for (u32 combo = 0; combo < 4; combo++) {
      b8 ta = (b8)(combo & 1u);
      b8 tb = (b8)((combo >> 1) & 1u);
      const Mat *lhs = ta ? at : a;
      const Mat *rhs = tb ? bt : b;
      RL_LIB_CHECK(mat_mm_scalar(ref, lhs, rhs, 1, ta, tb));
      RL_LIB_CHECK(mat_mm(got, lhs, rhs, 1, ta, tb));
      const u64 n = (u64)M * N;
      for (u64 i = 0; i < n; i++) {
        real d = real_fabs(got->data[i] - ref->data[i]);
        if (d > worst) worst = d;
        if (d > 1e-4f) ok = false;
      }
    }
  }

  // Shapes shorter than the vector width exercise the scalar tail loops.
  Mat *s1 = mat_new(arena, 1, 3);
  Mat *s2 = mat_new(arena, 3, 1);
  Mat *sr = mat_new(arena, 1, 1);
  Mat *sg = mat_new(arena, 1, 1);
  mat_rand(s1, -1.0f, 1.0f);
  mat_rand(s2, -1.0f, 1.0f);
  RL_LIB_CHECK(mat_mm_scalar(sr, s1, s2, 1, 0, 0));
  RL_LIB_CHECK(mat_mm(sg, s1, s2, 1, 0, 0));
  if (real_fabs(sg->data[0] - sr->data[0]) > 1e-5f) ok = false;

  printf("(backend %s, max diff %.2e) %s\n", simd_backend_name(), (double)worst,
         ok ? "OK" : "FAIL");
  ASSERT_TRUE(ok, "SIMD matmul matches the scalar reference");
}

// ── Softmax tests ────────────────────────────────────────────────────
static void test_softmax_sums_to_one(Arena *arena) {
  printf("  mat_softmax sums to 1... ");
  Mat *in = mat_new(arena, 5, 1);
  Mat *out = mat_new(arena, 5, 1);

  rng_seed_g(500, 5);
  mat_rand(in, -3.0f, 3.0f);
  mat_softmax(out, in);

  real sum = 0.0f;
  for (u32 i = 0; i < 5; i++)
    sum += out->data[i];

  ASSERT_NEAR(sum, 1.0f, 1e-6f, "mat_softmax sum == 1");
  printf("%s\n", real_fabs(sum - 1.0f) < 1e-6f ? "OK" : "FAIL");
}

static void test_softmax_positive(Arena *arena) {
  printf("  mat_softmax all positive... ");
  Mat *in = mat_new(arena, 5, 1);
  Mat *out = mat_new(arena, 5, 1);

  rng_seed_g(600, 6);
  mat_rand(in, -10.0f, 10.0f);
  mat_softmax(out, in);

  b32 ok = true;
  for (u32 i = 0; i < 5; i++) {
    if (out->data[i] <= 0.0f) {
      ok = false;
      break;
    }
  }
  ASSERT_TRUE(ok, "mat_softmax all values > 0");
  printf("%s\n", ok ? "OK" : "FAIL");
}

static void test_softmax_large_values(Arena *arena) {
  printf("  mat_softmax large values (no overflow)... ");
  Mat *in = mat_new(arena, 3, 1);
  Mat *out = mat_new(arena, 3, 1);

  in->data[0] = 1000.0f;
  in->data[1] = 1001.0f;
  in->data[2] = 1002.0f;

  mat_softmax(out, in);

  real sum = out->data[0] + out->data[1] + out->data[2];
  b32 ok = !isnan(sum) && !isinf(sum) && real_fabs(sum - 1.0f) < 1e-5f;
  ASSERT_TRUE(ok, "mat_softmax with large inputs is stable");
  printf("%s\n", ok ? "OK" : "FAIL");
}

// ── ReLU tests ───────────────────────────────────────────────────────

static void test_relu(Arena *arena) {
  printf("  mat_relu... ");
  Mat *in = mat_new(arena, 4, 1);
  Mat *out = mat_new(arena, 4, 1);

  in->data[0] = -2.0f;
  in->data[1] = 0.0f;
  in->data[2] = 3.5f;
  in->data[3] = -0.001f;

  mat_relu(out, in);

  b32 ok = out->data[0] == 0.0f && out->data[1] == 0.0f &&
           out->data[2] == 3.5f && out->data[3] == 0.0f;
  ASSERT_TRUE(ok, "mat_relu edge cases");
  printf("%s\n", ok ? "OK" : "FAIL");
}

// ── Add / Sub tests ──────────────────────────────────────────────────

static void test_add_sub(Arena *arena) {
  printf("  mat_add/mat_sub... ");
  Mat *a = mat_new(arena, 3, 2);
  Mat *b = mat_new(arena, 3, 2);
  Mat *sum_out = mat_new(arena, 3, 2);
  Mat *sub_out = mat_new(arena, 3, 2);

  for (u32 i = 0; i < 6; i++) {
    a->data[i] = (real)i;
    b->data[i] = (real)(i * 2);
  }

  mat_add(sum_out, a, b);
  mat_sub(sub_out, a, b);

  b32 ok = true;
  for (u32 i = 0; i < 6; i++) {
    if (real_fabs(sum_out->data[i] - ((real)i + (real)(i * 2))) > 1e-7f)
      ok = false;
    if (real_fabs(sub_out->data[i] - ((real)i - (real)(i * 2))) > 1e-7f)
      ok = false;
  }
  ASSERT_TRUE(ok, "mat_add and mat_sub element-wise");
  printf("%s\n", ok ? "OK" : "FAIL");
}

static void test_softmax_per_column(Arena *arena) {
  printf("  mat_softmax per-column (4x3 Mat)... ");
  Mat *in = mat_new(arena, 4, 3);
  Mat *out = mat_new(arena, 4, 3);

  rng_seed_g(777, 8);
  mat_rand(in, -5.0f, 5.0f);
  mat_softmax(out, in);

  b32 ok = true;
  for (u32 c = 0; c < 3; c++) {
    real col_sum = 0.0f;
    for (u32 r = 0; r < 4; r++) {
      col_sum += out->data[r * 3 + c];
    }
    if (real_fabs(col_sum - 1.0f) > 1e-6f) {
      ok = false;
      break;
    }
  }
  ASSERT_TRUE(ok, "each column of batched mat_softmax independently sums to 1.0");
  printf("%s\n", ok ? "OK" : "FAIL");
}

static void test_broadcast_add(Arena *arena) {
  printf("  broadcast mat_add A(3x4) + b(3x1)... ");
  Mat *a = mat_new(arena, 3, 4);
  Mat *b = mat_new(arena, 3, 1);
  Mat *out = mat_new(arena, 3, 4);

  for (u32 i = 0; i < 12; i++) a->data[i] = (real)i;
  b->data[0] = 10.0f;
  b->data[1] = 20.0f;
  b->data[2] = 30.0f;

  mat_add(out, a, b);

  b32 ok = true;
  for (u32 r = 0; r < 3; r++) {
    for (u32 c = 0; c < 4; c++) {
      real expected = a->data[r * 4 + c] + b->data[r];
      if (real_fabs(out->data[r * 4 + c] - expected) > 1e-6f) {
        ok = false;
        break;
      }
    }
  }
  ASSERT_TRUE(ok, "broadcast addition matches expectation");
  printf("%s\n", ok ? "OK" : "FAIL");
}

static void test_mul_tanh_mse(Arena *arena) {
  printf("  mat_mul, tanh, mse... ");
  Mat *a = mat_new(arena, 2, 2);
  Mat *b = mat_new(arena, 2, 2);
  Mat *mul_out = mat_new(arena, 2, 2);
  Mat *tanh_out = mat_new(arena, 2, 2);
  Mat *mse_out = mat_new(arena, 1, 1);

  a->data[0] = 2.0f; a->data[1] = -3.0f; a->data[2] = 0.5f; a->data[3] = 4.0f;
  b->data[0] = 3.0f; b->data[1] = 2.0f;  b->data[2] = 4.0f; b->data[3] = -1.0f;

  mat_mul(mul_out, a, b);
  b32 ok_mul = (mul_out->data[0] == 6.0f && mul_out->data[1] == -6.0f &&
                mul_out->data[2] == 2.0f && mul_out->data[3] == -4.0f);

  mat_tanh(tanh_out, a);
  b32 ok_tanh = (real_fabs(tanh_out->data[0] - real_tanh(2.0f)) < 1e-6f &&
                 real_fabs(tanh_out->data[1] - real_tanh(-3.0f)) < 1e-6f);

  mat_mse_loss(mse_out, a, b);
  // diffs: (2-3)^2=1, (-3-2)^2=25, (0.5-4)^2=12.25, (4-(-1))^2=25. sum = 63.25. mean = 15.8125
  b32 ok_mse = real_fabs(mse_out->data[0] - 15.8125f) < 1e-5f;

  ASSERT_TRUE(ok_mul && ok_tanh && ok_mse, "mat_mul, tanh, mse operations");
  printf("%s\n", (ok_mul && ok_tanh && ok_mse) ? "OK" : "FAIL");
}

static void test_shape_reduction_kernels(Arena *arena) {
  printf("  reshape, broadcast, reductions... ");
  Mat *m = mat_new(arena, 2, 3);
  Mat *r = mat_new(arena, 3, 2);
  Mat *bad = mat_new(arena, 2, 2);
  for (u32 i = 0; i < 6; i++) {
    m->data[i] = (real)(i + 1);
  }

  b32 ok = mat_reshape(r, m);
  ok = ok && r->data[0] == 1.0f && r->data[1] == 2.0f && r->data[2] == 3.0f &&
       r->data[3] == 4.0f && r->data[4] == 5.0f && r->data[5] == 6.0f;
  ok = ok && !mat_reshape(bad, m); // 4 != 6 elements
  ASSERT_TRUE(ok, "reshape copies row-major and rejects a count mismatch");

  Mat *row = mat_new(arena, 1, 2);
  Mat *bc = mat_new(arena, 3, 2);
  row->data[0] = 7.0f;
  row->data[1] = 8.0f;
  ok = mat_broadcast(bc, row);
  ok = ok && bc->data[0] == 7.0f && bc->data[1] == 8.0f &&
       bc->data[2] == 7.0f && bc->data[3] == 8.0f && bc->data[4] == 7.0f &&
       bc->data[5] == 8.0f;
  Mat *col = mat_new(arena, 2, 1);
  Mat *bc2 = mat_new(arena, 2, 3);
  col->data[0] = 1.0f;
  col->data[1] = 2.0f;
  ok = ok && mat_broadcast(bc2, col);
  ok = ok && bc2->data[0] == 1.0f && bc2->data[1] == 1.0f &&
       bc2->data[2] == 1.0f && bc2->data[3] == 2.0f && bc2->data[4] == 2.0f &&
       bc2->data[5] == 2.0f;
  ok = ok && !mat_broadcast(bad, m);
  ASSERT_TRUE(ok, "broadcast replicates rows and columns");

  Mat *sum_rows = mat_new(arena, 1, 3);
  Mat *sum_cols = mat_new(arena, 2, 1);
  Mat *sum_all = mat_new(arena, 1, 1);
  ok = mat_sum(sum_rows, m, REDUCE_ROWS);
  ok = ok && sum_rows->data[0] == 5.0f && sum_rows->data[1] == 7.0f &&
       sum_rows->data[2] == 9.0f;
  ok = ok && mat_sum(sum_cols, m, REDUCE_COLS) && sum_cols->data[0] == 6.0f &&
       sum_cols->data[1] == 15.0f;
  ok = ok && mat_sum(sum_all, m, REDUCE_ALL) && sum_all->data[0] == 21.0f;
  ok = ok && mat_mean(sum_all, m, REDUCE_ALL) && sum_all->data[0] == 3.5f;
  ASSERT_TRUE(ok, "sum/mean over rows, columns and all elements");

  Mat *mean_grad = mat_new(arena, 2, 3);
  Mat *mean_seed = mat_new(arena, 1, 1);
  mat_zero(mean_grad);
  mean_seed->data[0] = 12.0f;
  ok = mat_mean_add_grad(mean_grad, mean_seed, REDUCE_ALL);
  ok = ok && mean_grad->data[0] == 2.0f && mean_grad->data[5] == 2.0f;
  ok = ok && mat_mean_add_grad(mean_grad, mean_seed, REDUCE_ALL) &&
       mean_grad->data[0] == 4.0f; // accumulates, does not replace
  ASSERT_TRUE(ok, "mean backward scales its contribution and accumulates");

  Mat *values = mat_new(arena, 3, 2);
  for (u32 i = 0; i < 6; i++) {
    values->data[i] = (real)(i + 1);
  }
  Mat *idx = mat_new(arena, 1, 2);
  Mat *gathered = mat_new(arena, 1, 2);
  idx->data[0] = 2.0f;
  idx->data[1] = 0.0f;
  ok = mat_gather(gathered, values, idx) && gathered->data[0] == 5.0f &&
       gathered->data[1] == 2.0f;
  Mat *values_grad = mat_new(arena, 3, 2);
  Mat *gather_seed = mat_new(arena, 1, 2);
  mat_zero(values_grad);
  gather_seed->data[0] = 10.0f;
  gather_seed->data[1] = 20.0f;
  ok = ok && mat_gather_add_grad(values_grad, idx, gather_seed) &&
       values_grad->data[4] == 10.0f && values_grad->data[1] == 20.0f;
  idx->data[0] = 3.0f; // out of range
  ok = ok && !mat_gather(gathered, values, idx);
  idx->data[0] = 0.5f; // not integral
  ok = ok && !mat_gather(gathered, values, idx);
  idx->data[0] = -1.0f;
  ok = ok && !mat_gather(gathered, values, idx);
  ASSERT_TRUE(ok, "gather selects per column and rejects bad indices");

  Mat *a = mat_new(arena, 1, 2);
  Mat *b = mat_new(arena, 1, 2);
  Mat *mn = mat_new(arena, 1, 2);
  Mat *mx = mat_new(arena, 1, 2);
  a->data[0] = 1.0f;
  a->data[1] = 4.0f;
  b->data[0] = 2.0f;
  b->data[1] = 3.0f;
  ok = mat_min(mn, a, b) && mn->data[0] == 1.0f && mn->data[1] == 3.0f;
  ok = ok && mat_max(mx, a, b) && mx->data[0] == 2.0f && mx->data[1] == 4.0f;
  Mat *a_grad = mat_new(arena, 1, 2);
  Mat *b_grad = mat_new(arena, 1, 2);
  Mat *ones = mat_new(arena, 1, 2);
  mat_zero(a_grad);
  mat_zero(b_grad);
  mat_fill(ones, 1.0f);
  ok = ok && mat_min_add_grad(a_grad, b_grad, a, b, ones) &&
       a_grad->data[0] == 1.0f && a_grad->data[1] == 0.0f &&
       b_grad->data[0] == 0.0f && b_grad->data[1] == 1.0f;
  ASSERT_TRUE(ok, "min/max route gradients to the selected operand");

  Mat *clip_in = mat_new(arena, 1, 4);
  Mat *clip_out = mat_new(arena, 1, 4);
  Mat *clip_grad = mat_new(arena, 1, 4);
  Mat *clip_seed = mat_new(arena, 1, 4);
  clip_in->data[0] = -2.0f;
  clip_in->data[1] = -0.5f;
  clip_in->data[2] = 0.5f;
  clip_in->data[3] = 2.0f;
  mat_fill(clip_seed, 1.0f);
  mat_zero(clip_grad);
  ok = mat_clip(clip_out, clip_in, -1.0f, 1.0f) && clip_out->data[0] == -1.0f &&
       clip_out->data[1] == -0.5f && clip_out->data[2] == 0.5f &&
       clip_out->data[3] == 1.0f;
  ok = ok && mat_clip_add_grad(clip_grad, clip_in, -1.0f, 1.0f, clip_seed) &&
       clip_grad->data[0] == 0.0f && clip_grad->data[1] == 1.0f &&
       clip_grad->data[2] == 1.0f && clip_grad->data[3] == 0.0f;
  ok = ok && !mat_clip(clip_out, clip_in, 1.0f, -1.0f);
  ASSERT_TRUE(ok, "clip clamps forward and blocks gradient outside the bounds");
}

// ── Main ─────────────────────────────────────────────────────────────

int main(void) {
  Arena *arena = arena_new(MiB(16));
  RL_LIB_ASSERT(arena != NULL, "arena_new failed");

  printf("\n=== linalg.c unit tests ===\n\n");
  printf("[mat_mm variants]\n");
  test_matmul_nn(arena);
  test_matmul_nt(arena);
  test_matmul_tn(arena);
  test_matmul_tt(arena);
  test_simd_matches_scalar(arena);

  printf("\n[mat_softmax & batching]\n");
  test_softmax_sums_to_one(arena);
  test_softmax_positive(arena);
  test_softmax_large_values(arena);
  test_softmax_per_column(arena);

  printf("\n[activations & elementwise]\n");
  test_relu(arena);
  test_add_sub(arena);
  test_broadcast_add(arena);
  test_mul_tanh_mse(arena);

  printf("\n[shape & reduction kernels]\n");
  test_shape_reduction_kernels(arena);

  printf("\n---\nPassed: %u  Failed: %u\n\n", tests_passed, tests_failed);

  arena_dispose(arena);
  return tests_failed > 0 ? 1 : 0;
}
