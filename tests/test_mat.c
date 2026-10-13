// test_mat.c — Unit tests for mat.c operations
// Phase 0.2: Prove the math layer is correct.
#include "arena.h"
#include "base.h"
#include "mat.h"
#include "prng.h"

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
  ASSERT_TRUE(fabsf((a) - (b)) < (tol), msg)

// Naive triple-loop matmul for reference (no transpose, always zeroes out).
static void naive_matmul(f32 *out, const f32 *a, const f32 *b, u32 M, u32 K,
                         u32 N) {
  for (u32 i = 0; i < M; i++) {
    for (u32 j = 0; j < N; j++) {
      f32 s = 0.0f;
      for (u32 k = 0; k < K; k++) {
        s += a[i * K + k] * b[k * N + j];
      }
      out[i * N + j] = s;
    }
  }
}

// ── Matmul tests ─────────────────────────────────────────────────────

static void test_matmul_nn(mem_arena *arena) {
  printf("  matmul NN... ");
  // Non-square: (3x5) * (5x4) = (3x4)
  matrix *a = create_matrix(arena, 3, 5);
  matrix *b = create_matrix(arena, 5, 4);
  matrix *out = create_matrix(arena, 3, 4);

  prng_seed(42, 1);
  fill_rand(a, -1.0f, 1.0f);
  fill_rand(b, -1.0f, 1.0f);

  f32 ref[12];
  naive_matmul(ref, a->data, b->data, 3, 5, 4);

  matmul(out, a, b, 1, 0, 0);

  b32 ok = true;
  for (u32 i = 0; i < 12; i++) {
    if (fabsf(out->data[i] - ref[i]) > 1e-5f) {
      ok = false;
      break;
    }
  }
  ASSERT_TRUE(ok, "matmul_nn (3x5)*(5x4)");
  printf("%s\n", ok ? "OK" : "FAIL");
}

static void test_matmul_nt(mem_arena *arena) {
  printf("  matmul NT... ");
  // A(3x5) * B^T where B is (4x5) → out(3x4)
  matrix *a = create_matrix(arena, 3, 5);
  matrix *b = create_matrix(arena, 4, 5); // will be transposed
  matrix *out = create_matrix(arena, 3, 4);

  prng_seed(100, 2);
  fill_rand(a, -1.0f, 1.0f);
  fill_rand(b, -1.0f, 1.0f);

  // Build explicit transpose of b for reference
  f32 bt[20]; // 5x4
  for (u32 i = 0; i < 4; i++)
    for (u32 j = 0; j < 5; j++)
      bt[j * 4 + i] = b->data[i * 5 + j];

  f32 ref[12];
  naive_matmul(ref, a->data, bt, 3, 5, 4);

  matmul(out, a, b, 1, 0, 1);

  b32 ok = true;
  for (u32 i = 0; i < 12; i++) {
    if (fabsf(out->data[i] - ref[i]) > 1e-5f) {
      ok = false;
      break;
    }
  }
  ASSERT_TRUE(ok, "matmul_nt A(3x5)*B^T(4x5)");
  printf("%s\n", ok ? "OK" : "FAIL");
}

static void test_matmul_tn(mem_arena *arena) {
  printf("  matmul TN... ");
  // A^T where A is (5x3) → effective A^T(3x5) * B(5x4) = out(3x4)
  matrix *a = create_matrix(arena, 5, 3); // will be transposed
  matrix *b = create_matrix(arena, 5, 4);
  matrix *out = create_matrix(arena, 3, 4);

  prng_seed(200, 3);
  fill_rand(a, -1.0f, 1.0f);
  fill_rand(b, -1.0f, 1.0f);

  // Build explicit transpose of a for reference
  f32 at[15]; // 3x5
  for (u32 i = 0; i < 5; i++)
    for (u32 j = 0; j < 3; j++)
      at[j * 5 + i] = a->data[i * 3 + j];

  f32 ref[12];
  naive_matmul(ref, at, b->data, 3, 5, 4);

  matmul(out, a, b, 1, 1, 0);

  b32 ok = true;
  for (u32 i = 0; i < 12; i++) {
    if (fabsf(out->data[i] - ref[i]) > 1e-5f) {
      ok = false;
      break;
    }
  }
  ASSERT_TRUE(ok, "matmul_tn A^T(5x3)*B(5x4)");
  printf("%s\n", ok ? "OK" : "FAIL");
}

static void test_matmul_tt(mem_arena *arena) {
  printf("  matmul TT... ");
  // A^T where A is (5x3) and B^T where B is (4x5) → out(3x4)
  matrix *a = create_matrix(arena, 5, 3);
  matrix *b = create_matrix(arena, 4, 5);
  matrix *out = create_matrix(arena, 3, 4);

  prng_seed(300, 4);
  fill_rand(a, -1.0f, 1.0f);
  fill_rand(b, -1.0f, 1.0f);

  f32 at[15]; // 3x5
  for (u32 i = 0; i < 5; i++)
    for (u32 j = 0; j < 3; j++)
      at[j * 5 + i] = a->data[i * 3 + j];
  f32 bt[20]; // 5x4
  for (u32 i = 0; i < 4; i++)
    for (u32 j = 0; j < 5; j++)
      bt[j * 4 + i] = b->data[i * 5 + j];

  f32 ref[12];
  naive_matmul(ref, at, bt, 3, 5, 4);

  matmul(out, a, b, 1, 1, 1);

  b32 ok = true;
  for (u32 i = 0; i < 12; i++) {
    if (fabsf(out->data[i] - ref[i]) > 1e-5f) {
      ok = false;
      break;
    }
  }
  ASSERT_TRUE(ok, "matmul_tt A^T(5x3)*B^T(4x5)");
  printf("%s\n", ok ? "OK" : "FAIL");
}

// ── Softmax tests ────────────────────────────────────────────────────

static void test_softmax_sums_to_one(mem_arena *arena) {
  printf("  softmax sums to 1... ");
  matrix *in = create_matrix(arena, 5, 1);
  matrix *out = create_matrix(arena, 5, 1);

  prng_seed(500, 5);
  fill_rand(in, -3.0f, 3.0f);
  softmax(out, in);

  f32 sum = 0.0f;
  for (u32 i = 0; i < 5; i++)
    sum += out->data[i];

  ASSERT_NEAR(sum, 1.0f, 1e-6f, "softmax sum == 1");
  printf("%s\n", fabsf(sum - 1.0f) < 1e-6f ? "OK" : "FAIL");
}

static void test_softmax_positive(mem_arena *arena) {
  printf("  softmax all positive... ");
  matrix *in = create_matrix(arena, 5, 1);
  matrix *out = create_matrix(arena, 5, 1);

  prng_seed(600, 6);
  fill_rand(in, -10.0f, 10.0f);
  softmax(out, in);

  b32 ok = true;
  for (u32 i = 0; i < 5; i++) {
    if (out->data[i] <= 0.0f) {
      ok = false;
      break;
    }
  }
  ASSERT_TRUE(ok, "softmax all values > 0");
  printf("%s\n", ok ? "OK" : "FAIL");
}

static void test_softmax_large_values(mem_arena *arena) {
  printf("  softmax large values (no overflow)... ");
  matrix *in = create_matrix(arena, 3, 1);
  matrix *out = create_matrix(arena, 3, 1);

  in->data[0] = 1000.0f;
  in->data[1] = 1001.0f;
  in->data[2] = 1002.0f;

  softmax(out, in);

  f32 sum = out->data[0] + out->data[1] + out->data[2];
  b32 ok = !isnan(sum) && !isinf(sum) && fabsf(sum - 1.0f) < 1e-5f;
  ASSERT_TRUE(ok, "softmax with large inputs is stable");
  printf("%s\n", ok ? "OK" : "FAIL");
}

// ── ReLU tests ───────────────────────────────────────────────────────

static void test_relu(mem_arena *arena) {
  printf("  relu... ");
  matrix *in = create_matrix(arena, 4, 1);
  matrix *out = create_matrix(arena, 4, 1);

  in->data[0] = -2.0f;
  in->data[1] = 0.0f;
  in->data[2] = 3.5f;
  in->data[3] = -0.001f;

  relu(out, in);

  b32 ok = out->data[0] == 0.0f && out->data[1] == 0.0f &&
           out->data[2] == 3.5f && out->data[3] == 0.0f;
  ASSERT_TRUE(ok, "relu edge cases");
  printf("%s\n", ok ? "OK" : "FAIL");
}

// ── Add / Sub tests ──────────────────────────────────────────────────

static void test_add_sub(mem_arena *arena) {
  printf("  add/sub... ");
  matrix *a = create_matrix(arena, 3, 2);
  matrix *b = create_matrix(arena, 3, 2);
  matrix *sum_out = create_matrix(arena, 3, 2);
  matrix *sub_out = create_matrix(arena, 3, 2);

  for (u32 i = 0; i < 6; i++) {
    a->data[i] = (f32)i;
    b->data[i] = (f32)(i * 2);
  }

  add(sum_out, a, b);
  sub(sub_out, a, b);

  b32 ok = true;
  for (u32 i = 0; i < 6; i++) {
    if (fabsf(sum_out->data[i] - ((f32)i + (f32)(i * 2))) > 1e-7f)
      ok = false;
    if (fabsf(sub_out->data[i] - ((f32)i - (f32)(i * 2))) > 1e-7f)
      ok = false;
  }
  ASSERT_TRUE(ok, "add and sub element-wise");
  printf("%s\n", ok ? "OK" : "FAIL");
}

// ── Main ─────────────────────────────────────────────────────────────

int main(void) {
  mem_arena *arena = arena_create(MiB(16));
  assert(arena != NULL);

  printf("\n=== mat.c unit tests ===\n\n");
  printf("[matmul variants]\n");
  test_matmul_nn(arena);
  test_matmul_nt(arena);
  test_matmul_tn(arena);
  test_matmul_tt(arena);

  printf("\n[softmax]\n");
  test_softmax_sums_to_one(arena);
  test_softmax_positive(arena);
  test_softmax_large_values(arena);

  printf("\n[relu]\n");
  test_relu(arena);

  printf("\n[add/sub]\n");
  test_add_sub(arena);

  printf("\n---\nPassed: %u  Failed: %u\n\n", tests_passed, tests_failed);

  arena_destroy(arena);
  return tests_failed > 0 ? 1 : 0;
}
