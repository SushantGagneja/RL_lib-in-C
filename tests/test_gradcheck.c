// test_gradcheck.c — Numerical gradient check for the autograd engine
// Phase 0.1: The highest-value item. Compare analytic gradients from
// schedule_grad against central finite differences.
#include "memory.h"
#include "graph.h"
#include "base.h"
#include "linalg.h"
#include "random.h"

#include <math.h>
#include <string.h>

// Build the whole library with -DCRL_USE_DOUBLE (see `make gradcheck-double`)
// to run these checks in double precision with a much tighter tolerance.
// The library is single precision by default, so the default build keeps a
// looser tolerance and is only a rough sanity check.
#ifdef CRL_USE_DOUBLE
#define GRADCHECK_NAME "f64"
#define GRAD_H (real)(1e-5)
#define GRAD_TOL (real)(1e-6)
#else
#define GRADCHECK_NAME "f32"
#define GRAD_H (real)(1e-2)
#define GRAD_TOL (real)(5e-2)
#endif

// ── Helpers ──────────────────────────────────────────────────────────

static u32 tests_passed = 0;
static u32 tests_failed = 0;
static u32 total_params_checked = 0;
static u32 params_ok = 0;

// Sum all elements of a Mat (the scalar "loss" when backward seeds with 1).
static real sum_of(const Mat *m) {
  real s = 0.0f;
  u64 n = (u64)m->rows * m->cols;
  for (u64 i = 0; i < n; i++)
    s += m->data[i];
  return s;
}

// Clear all parameter gradients in the graph to zero.
static void clear_param_grads(Schedule *g) {
  for (u32 i = 0; i < g->len; i++) {
    Tensor *v = g->ops[i];
    if ((v->flags & TENSOR_PARAM) && v->grad != NULL)
      mat_zero(v->grad);
  }
}

// Clear all non-parameter activation gradients in the graph to zero.
static void clear_activation_grads(Schedule *g) {
  for (u32 i = 0; i < g->len; i++) {
    Tensor *v = g->ops[i];
    if (!(v->flags & TENSOR_PARAM) && v->grad != NULL)
      mat_zero(v->grad);
  }
}

// ── Gradient check core ──────────────────────────────────────────────

// Check gradients for a single parameter Tensor against central differences.
// Returns the number of elements that failed the tolerance check.
static u32 gradcheck_param(Schedule *graph, Tensor *output, Tensor *param, real h,
                           real tol) {
  u64 n = (u64)param->val->rows * param->val->cols;
  u32 failures = 0;

  // 1. Run forward + backward to get analytic gradients
  clear_param_grads(graph);
  clear_activation_grads(graph);
  schedule_run(graph);
  schedule_grad(graph);

  // Save analytic gradients (backward accumulates into param->grad)
  real *analytic = malloc(sizeof(real) * n);
  memcpy(analytic, param->grad->data, sizeof(real) * n);

  // 2. Central differences for each element
  for (u64 i = 0; i < n; i++) {
    real orig = param->val->data[i];

    param->val->data[i] = orig + h;
    schedule_run(graph);
    real up = sum_of(output->val);

    param->val->data[i] = orig - h;
    schedule_run(graph);
    real dn = sum_of(output->val);

    param->val->data[i] = orig; // restore

    real numeric = (up - dn) / (2.0f * h);
    // Relative error against the larger magnitude, with an absolute floor of 1
    // for near-zero gradients. Using max() instead of |n| + |a| avoids the
    // factor-of-two leniency of the old formula.
    real denom =
        real_fmax((real)1.0, real_fmax(real_fabs(numeric), real_fabs(analytic[i])));
    real rel = real_fabs(numeric - analytic[i]) / denom;

    total_params_checked++;
    if (rel < tol) {
      params_ok++;
    } else {
      failures++;
      if (failures <= 5) { // only print first 5 to avoid spam
        printf("    param[%llu]: analytic=%.6f numeric=%.6f rel=%.4f\n",
               (unsigned long long)i, analytic[i], numeric, rel);
      }
    }
  }

  free(analytic);
  return failures;
}

// ── Test: mat_mm → mat_relu → mat_mm → mat_add → output ─────────────────────

static void test_gradcheck_mlp(void) {
  printf("[gradcheck: mat_mm → mat_relu → mat_mm → mat_add]\n");

  Arena *arena = arena_new(MiB(32));
  Net *model = ARENA_OBJ(arena, Net);

  // Build a small 2-layer network: input(4x1) → W0(3x4) → mat_relu → W1(2x3) +
  // b1(2x1)
  Tensor *input = tensor_create(arena, model, 4, 1, TENSOR_PLAIN);
  Tensor *W0 = tensor_create(arena, model, 3, 4,
                       TENSOR_PARAM | TENSOR_GRAD);
  Tensor *W1 = tensor_create(arena, model, 2, 3,
                       TENSOR_PARAM | TENSOR_GRAD);
  Tensor *b1 = tensor_create(arena, model, 2, 1,
                       TENSOR_PARAM | TENSOR_GRAD);

  Tensor *z0 = tensor_mm(arena, model, W0, input);
  Tensor *a0 = tensor_relu(arena, model, z0);
  Tensor *z1 = tensor_mm(arena, model, W1, a0);
  Tensor *output = tensor_add(arena, model, z1, b1);

  Schedule graph = schedule_from(arena, model, output);
  CRL_ASSERT(graph.len > 0, "graph build failed");

  // Init parameters away from zero (some positive, some negative so both ReLU
  // branches run)
  rng_seed_g(12345, 1);
  mat_rand(W0->val, -1.0f, 1.0f);
  // Ensure at least one pre-activation is strictly negative and one strictly
  // positive
  W0->val->data[0] = -2.0f;
  W0->val->data[1] = -1.5f;
  W0->val->data[4] = 2.0f;
  mat_rand(W1->val, -0.5f, 0.5f);
  mat_rand(b1->val, -0.2f, 0.2f);
  mat_rand(input->val, 0.5f, 1.5f);

  real h = GRAD_H;
  real tol = GRAD_TOL;

  printf("  W0 (3x4, %d params, testing both ReLU branches)... ", 12);
  u32 f0 = gradcheck_param(&graph, output, W0, h, tol);
  printf("%s (%u failures)\n", f0 == 0 ? "OK" : "FAIL", f0);

  printf("  W1 (2x3, %d params)... ", 6);
  u32 f1 = gradcheck_param(&graph, output, W1, h, tol);
  printf("%s (%u failures)\n", f1 == 0 ? "OK" : "FAIL", f1);

  printf("  b1 (2x1, %d params)... ", 2);
  u32 fb = gradcheck_param(&graph, output, b1, h, tol);
  printf("%s (%u failures)\n", fb == 0 ? "OK" : "FAIL", fb);

  if (f0 + f1 + fb == 0)
    tests_passed++;
  else
    tests_failed++;

  arena_dispose(arena);
}

// ── Test: mat_mm → mat_softmax ───────────────────────────────────────────

static void test_gradcheck_softmax(void) {
  printf("\n[gradcheck: mat_mm → mat_softmax]\n");

  Arena *arena = arena_new(MiB(32));
  Net *model = ARENA_OBJ(arena, Net);

  Tensor *input = tensor_create(arena, model, 3, 1, TENSOR_PLAIN);
  Tensor *W = tensor_create(arena, model, 5, 3,
                      TENSOR_PARAM | TENSOR_GRAD);
  Tensor *b = tensor_create(arena, model, 5, 1,
                      TENSOR_PARAM | TENSOR_GRAD);

  Tensor *z = tensor_mm(arena, model, W, input);
  Tensor *zb = tensor_add(arena, model, z, b);
  Tensor *sm = tensor_softmax(arena, model, zb);

  // Project mat_softmax with non-uniform weights to test mat_softmax Jacobian
  // non-trivially (otherwise sum(mat_softmax) is identically 1 with zero derivative
  // everywhere).
  Tensor *target = tensor_create(arena, model, 1, 5, TENSOR_PLAIN);
  target->val->data[0] = 1.0f;
  target->val->data[1] = 2.0f;
  target->val->data[2] = 0.5f;
  target->val->data[3] = -1.0f;
  target->val->data[4] = 3.0f;
  Tensor *output = tensor_mm(arena, model, target, sm);

  Schedule graph = schedule_from(arena, model, output);
  CRL_ASSERT(graph.len > 0, "graph build failed");

  rng_seed_g(99999, 2);
  mat_rand(W->val, -0.5f, 0.5f);
  mat_rand(b->val, -0.2f, 0.2f);
  mat_rand(input->val, 0.5f, 1.5f);

  real h = GRAD_H;
  real tol = GRAD_TOL;

  printf("  W (5x3, %d params)... ", 15);
  u32 fw = gradcheck_param(&graph, output, W, h, tol);
  printf("%s (%u failures)\n", fw == 0 ? "OK" : "FAIL", fw);

  printf("  b (5x1, %d params)... ", 5);
  u32 fb = gradcheck_param(&graph, output, b, h, tol);
  printf("%s (%u failures)\n", fb == 0 ? "OK" : "FAIL", fb);

  if (fw + fb == 0)
    tests_passed++;
  else
    tests_failed++;

  arena_dispose(arena);
}

// ── Test: mat_sub backward ───────────────────────────────────────────────

static void test_gradcheck_sub(void) {
  printf("\n[gradcheck: mat_mm → mat_sub]\n");

  Arena *arena = arena_new(MiB(32));
  Net *model = ARENA_OBJ(arena, Net);

  Tensor *input = tensor_create(arena, model, 3, 1, TENSOR_PLAIN);
  Tensor *W = tensor_create(arena, model, 4, 3,
                      TENSOR_PARAM | TENSOR_GRAD);
  Tensor *b = tensor_create(arena, model, 4, 1,
                      TENSOR_PARAM | TENSOR_GRAD);

  Tensor *z = tensor_mm(arena, model, W, input);
  Tensor *output = tensor_sub(arena, model, z, b);

  Schedule graph = schedule_from(arena, model, output);
  CRL_ASSERT(graph.len > 0, "graph build failed");

  rng_seed_g(77777, 3);
  mat_rand(W->val, -1.0f, 1.0f);
  mat_rand(b->val, -0.5f, 0.5f);
  mat_rand(input->val, 0.5f, 1.5f);

  real h = GRAD_H;
  real tol = GRAD_TOL;

  printf("  W (4x3, %d params)... ", 12);
  u32 fw = gradcheck_param(&graph, output, W, h, tol);
  printf("%s (%u failures)\n", fw == 0 ? "OK" : "FAIL", fw);

  printf("  b (4x1, %d params)... ", 4);
  u32 fb = gradcheck_param(&graph, output, b, h, tol);
  printf("%s (%u failures)\n", fb == 0 ? "OK" : "FAIL", fb);

  if (fw + fb == 0)
    tests_passed++;
  else
    tests_failed++;

  arena_dispose(arena);
}

// ── Test: full actor network
// (mat_mm→mat_add→mat_relu→mat_mm→mat_add→mat_relu→mat_mm→mat_add→mat_softmax→reinforce) ──

static void test_gradcheck_reinforce(void) {
  printf("\n[gradcheck: full actor → reinforce loss]\n");

  Arena *arena = arena_new(MiB(64));
  Net *model = ARENA_OBJ(arena, Net);

  // Smaller version of the actual model: input(8x1) → 6 → 4 → mat_softmax(3)
  Tensor *input = tensor_create(arena, model, 8, 1, TENSOR_PLAIN);

  Tensor *W0 = tensor_create(arena, model, 6, 8,
                       TENSOR_PARAM | TENSOR_GRAD);
  Tensor *b0 = tensor_create(arena, model, 6, 1,
                       TENSOR_PARAM | TENSOR_GRAD);
  Tensor *W1 = tensor_create(arena, model, 4, 6,
                       TENSOR_PARAM | TENSOR_GRAD);
  Tensor *b1 = tensor_create(arena, model, 4, 1,
                       TENSOR_PARAM | TENSOR_GRAD);
  Tensor *W2 = tensor_create(arena, model, 3, 4,
                       TENSOR_PARAM | TENSOR_GRAD);
  Tensor *b2 = tensor_create(arena, model, 3, 1,
                       TENSOR_PARAM | TENSOR_GRAD);

  Tensor *z0 = tensor_mm(arena, model, W0, input);
  Tensor *z0b = tensor_add(arena, model, z0, b0);
  Tensor *a0 = tensor_relu(arena, model, z0b);

  Tensor *z1 = tensor_mm(arena, model, W1, a0);
  Tensor *z1b = tensor_add(arena, model, z1, b1);
  Tensor *a1 = tensor_relu(arena, model, z1b);

  Tensor *z2 = tensor_mm(arena, model, W2, a1);
  Tensor *z2b = tensor_add(arena, model, z2, b2);
  Tensor *probs = tensor_softmax(arena, model, z2b);

  Tensor *advantage = tensor_create(arena, model, 3, 1, TENSOR_PLAIN);
  Tensor *cost = tensor_reinforce_loss(arena, model, probs, advantage);

  Schedule graph = schedule_from(arena, model, cost);
  CRL_ASSERT(graph.len > 0, "graph build failed");

  // Init everything away from zero
  rng_seed_g(54321, 7);
  mat_rand(W0->val, 0.1f, 0.5f);
  mat_rand(b0->val, 0.01f, 0.1f);
  mat_rand(W1->val, 0.1f, 0.5f);
  mat_rand(b1->val, 0.01f, 0.1f);
  mat_rand(W2->val, 0.1f, 0.5f);
  mat_rand(b2->val, 0.01f, 0.1f);
  mat_rand(input->val, 0.5f, 2.0f);

  // Set advantage to a one-hot-like vector with a positive value at action 1
  mat_zero(advantage->val);
  advantage->val->data[1] = 1.5f;

  real h = GRAD_H;
  real tol = GRAD_TOL;

  u32 total_fail = 0;

  const char *names[] = {"W0", "b0", "W1", "b1", "W2", "b2"};
  Tensor *params[] = {W0, b0, W1, b1, W2, b2};

  for (u32 p = 0; p < 6; p++) {
    u32 n = params[p]->val->rows * params[p]->val->cols;
    printf("  %s (%ux%u, %u params)... ", names[p], params[p]->val->rows,
           params[p]->val->cols, n);
    u32 f = gradcheck_param(&graph, cost, params[p], h, tol);
    printf("%s (%u failures)\n", f == 0 ? "OK" : "FAIL", f);
    total_fail += f;
  }

  if (total_fail == 0)
    tests_passed++;
  else
    tests_failed++;

  arena_dispose(arena);
}

// ── Test: broadcast mat_add across batch ─────────────────────────────────

static void test_gradcheck_broadcast_add(void) {
  printf("\n[gradcheck: broadcast mat_add (batch B=2)]\n");

  Arena *arena = arena_new(MiB(32));
  Net *model = ARENA_OBJ(arena, Net);

  // X is (4x2), W is (3x4) -> z is (3x2), b is (3x1) broadcasted
  Tensor *X = tensor_create(arena, model, 4, 2, TENSOR_PLAIN);
  Tensor *W = tensor_create(arena, model, 3, 4,
                      TENSOR_PARAM | TENSOR_GRAD);
  Tensor *b = tensor_create(arena, model, 3, 1,
                      TENSOR_PARAM | TENSOR_GRAD);

  Tensor *z = tensor_mm(arena, model, W, X);
  Tensor *output = tensor_add(arena, model, z, b);

  Schedule graph = schedule_from(arena, model, output);
  CRL_ASSERT(graph.len > 0, "graph build failed");

  rng_seed_g(112233, 4);
  mat_rand(W->val, -0.5f, 0.5f);
  mat_rand(b->val, -0.5f, 0.5f);
  mat_rand(X->val, 0.5f, 1.5f);

  real h = GRAD_H;
  real tol = GRAD_TOL;

  printf("  W (3x4, 12 params)... ");
  u32 fw = gradcheck_param(&graph, output, W, h, tol);
  printf("%s (%u failures)\n", fw == 0 ? "OK" : "FAIL", fw);

  printf("  b (3x1 broadcast, 3 params)... ");
  u32 fb = gradcheck_param(&graph, output, b, h, tol);
  printf("%s (%u failures)\n", fb == 0 ? "OK" : "FAIL", fb);

  if (fw + fb == 0)
    tests_passed++;
  else
    tests_failed++;

  arena_dispose(arena);
}

// ── Test: elementwise mat_mul & tanh ─────────────────────────────────────

static void test_gradcheck_tanh_mul(void) {
  printf("\n[gradcheck: tanh & elementwise mat_mul]\n");

  Arena *arena = arena_new(MiB(32));
  Net *model = ARENA_OBJ(arena, Net);

  Tensor *input = tensor_create(arena, model, 3, 1, TENSOR_PLAIN);
  Tensor *W1 = tensor_create(arena, model, 3, 3,
                       TENSOR_PARAM | TENSOR_GRAD);
  Tensor *W2 = tensor_create(arena, model, 3, 3,
                       TENSOR_PARAM | TENSOR_GRAD);

  Tensor *z1 = tensor_mm(arena, model, W1, input);
  Tensor *z2 = tensor_mm(arena, model, W2, input);
  Tensor *t2 = tensor_tanh(arena, model, z2);
  Tensor *output = tensor_mul(arena, model, z1, t2);

  Schedule graph = schedule_from(arena, model, output);
  CRL_ASSERT(graph.len > 0, "graph build failed");

  rng_seed_g(445566, 5);
  mat_rand(W1->val, -0.5f, 0.5f);
  mat_rand(W2->val, -0.5f, 0.5f);
  mat_rand(input->val, 0.5f, 1.5f);

  real h = GRAD_H;
  real tol = GRAD_TOL;

  printf("  W1 (3x3, 9 params)... ");
  u32 f1 = gradcheck_param(&graph, output, W1, h, tol);
  printf("%s (%u failures)\n", f1 == 0 ? "OK" : "FAIL", f1);

  printf("  W2 (3x3, 9 params)... ");
  u32 f2 = gradcheck_param(&graph, output, W2, h, tol);
  printf("%s (%u failures)\n", f2 == 0 ? "OK" : "FAIL", f2);

  if (f1 + f2 == 0)
    tests_passed++;
  else
    tests_failed++;

  arena_dispose(arena);
}

// ── Test: MSE loss ───────────────────────────────────────────────────

static void test_gradcheck_mse(void) {
  printf("\n[gradcheck: MSE loss]\n");

  Arena *arena = arena_new(MiB(32));
  Net *model = ARENA_OBJ(arena, Net);

  Tensor *input = tensor_create(arena, model, 4, 2, TENSOR_PLAIN);
  Tensor *W = tensor_create(arena, model, 3, 4,
                      TENSOR_PARAM | TENSOR_GRAD);
  Tensor *target = tensor_create(arena, model, 3, 2, TENSOR_PLAIN);

  Tensor *pred = tensor_mm(arena, model, W, input);
  Tensor *loss = tensor_mse(arena, model, pred, target);

  Schedule graph = schedule_from(arena, model, loss);
  CRL_ASSERT(graph.len > 0, "graph build failed");

  rng_seed_g(778899, 6);
  mat_rand(W->val, -0.5f, 0.5f);
  mat_rand(input->val, 0.5f, 1.5f);
  mat_rand(target->val, 0.0f, 1.0f);

  real h = GRAD_H;
  real tol = GRAD_TOL;

  printf("  W (3x4, 12 params)... ");
  u32 fw = gradcheck_param(&graph, loss, W, h, tol);
  printf("%s (%u failures)\n", fw == 0 ? "OK" : "FAIL", fw);

  if (fw == 0)
    tests_passed++;
  else
    tests_failed++;

  arena_dispose(arena);
}

// ── Test: fused policy cross-entropy loss ────────────────────────────

static void test_gradcheck_fused_policy(void) {
  printf("\n[gradcheck: fused policy loss (batch B=3, act_dim=4)]\n");
  Arena *arena = arena_new(MiB(16));

  Mat *logits = mat_new(arena, 4, 3);
  Mat *logits_grad = mat_new(arena, 4, 3);
  Mat *loss_out = mat_new(arena, 1, 1);
  Mat *seed_grad = mat_new(arena, 1, 1);
  CRL_ASSERT(logits && logits_grad && loss_out && seed_grad,
             "arena out of memory in fused policy test");
  seed_grad->data[0] = 1.0f;

  u32 actions[3] = {1, 0, 3};
  real advantages[3] = {1.2f, -0.8f, 2.0f};

  rng_seed_g(9876, 5);
  mat_rand(logits, -1.0f, 1.0f);

  mat_zero(logits_grad);
  CRL_CHECK(mat_fused_policy_loss(loss_out, logits, actions, advantages, 3));
  CRL_CHECK(mat_fused_policy_add_grad(logits_grad, logits, actions, advantages,
                                      3, seed_grad));

  real h = GRAD_H;
  real tol = GRAD_TOL;
  u32 failures = 0;

  for (u32 i = 0; i < 12; i++) {
    real orig = logits->data[i];

    logits->data[i] = orig + h;
    Mat *up = mat_new(arena, 1, 1);
    CRL_CHECK(mat_fused_policy_loss(up, logits, actions, advantages, 3));

    logits->data[i] = orig - h;
    Mat *dn = mat_new(arena, 1, 1);
    CRL_CHECK(mat_fused_policy_loss(dn, logits, actions, advantages, 3));

    logits->data[i] = orig;

    real numeric = (up->data[0] - dn->data[0]) / (2.0f * h);
    real analytic = logits_grad->data[i];
    real denom = real_fmax((real)1.0, real_fmax(real_fabs(numeric),
                                                real_fabs(analytic)));
    real rel = real_fabs(numeric - analytic) / denom;

    total_params_checked++;
    if (rel < tol) {
      params_ok++;
    } else {
      failures++;
    }
  }

  printf("  logits (4x3, 12 elements)... %s (%u failures)\n",
         failures == 0 ? "OK" : "FAIL", failures);
  if (failures == 0)
    tests_passed++;
  else
    tests_failed++;

  arena_dispose(arena);
}

// ── Test: input gradient and mat_mul with both sides requiring grad ──────

static void test_gradcheck_input_and_mul(void) {
  printf("\n[gradcheck: input gradient & mat_mul both sides]\n");

  Arena *arena = arena_new(MiB(32));
  Net *model = ARENA_OBJ(arena, Net);

  // `input` and `gate` are activations (not parameters) that require grad, so
  // these checks cover the matmul B-side gradient and both branches of the
  // elementwise mul backward.
  Tensor *input = tensor_create(arena, model, 4, 2, TENSOR_GRAD);
  Tensor *W = tensor_create(arena, model, 3, 4,
                      TENSOR_PARAM | TENSOR_GRAD);
  Tensor *gate = tensor_create(arena, model, 3, 2, TENSOR_GRAD);

  Tensor *z = tensor_mm(arena, model, W, input);  // (3x2)
  Tensor *output = tensor_mul(arena, model, z, gate); // both sides require grad

  Schedule graph = schedule_from(arena, model, output);
  CRL_ASSERT(graph.len > 0, "graph build failed");

  rng_seed_g(246810, 8);
  mat_rand(W->val, -0.5f, 0.5f);
  mat_rand(gate->val, -1.0f, 1.0f);
  mat_rand(input->val, 0.5f, 1.5f);

  real h = GRAD_H;
  real tol = GRAD_TOL;

  printf("  W (3x4)... ");
  u32 fw = gradcheck_param(&graph, output, W, h, tol);
  printf("%s (%u failures)\n", fw == 0 ? "OK" : "FAIL", fw);

  printf("  gate (3x2, mul LHS)... ");
  u32 fg = gradcheck_param(&graph, output, gate, h, tol);
  printf("%s (%u failures)\n", fg == 0 ? "OK" : "FAIL", fg);

  printf("  input (4x2, mul RHS / matmul B-side)... ");
  u32 fi = gradcheck_param(&graph, output, input, h, tol);
  printf("%s (%u failures)\n", fi == 0 ? "OK" : "FAIL", fi);

  if (fw + fg + fi == 0)
    tests_passed++;
  else
    tests_failed++;

  arena_dispose(arena);
}

// ── Main ─────────────────────────────────────────────────────────────

int main(void) {
  printf("\n=== Autograd gradient checks (%s) ===\n\n", GRADCHECK_NAME);

  test_gradcheck_mlp();
  test_gradcheck_softmax();
  test_gradcheck_sub();
  test_gradcheck_broadcast_add();
  test_gradcheck_tanh_mul();
  test_gradcheck_mse();
  test_gradcheck_fused_policy();
  test_gradcheck_reinforce();
  test_gradcheck_input_and_mul();

  printf("\n---\nTests: %u passed, %u failed\n", tests_passed, tests_failed);
  printf("Parameters checked: %u/%u passed within tolerance\n\n", params_ok,
         total_params_checked);

  return tests_failed > 0 ? 1 : 0;
}
