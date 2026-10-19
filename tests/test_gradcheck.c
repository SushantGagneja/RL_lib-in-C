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

// Build the whole library with -DRL_LIB_USE_DOUBLE (see `make gradcheck-double`)
// to run these checks in double precision with a much tighter tolerance.
// The library is single precision by default, so the default build keeps a
// looser tolerance and is only a rough sanity check.
#ifdef RL_LIB_USE_DOUBLE
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
  RL_LIB_ASSERT(graph.len > 0, "graph build failed");

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
  RL_LIB_ASSERT(graph.len > 0, "graph build failed");

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
  RL_LIB_ASSERT(graph.len > 0, "graph build failed");

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
  RL_LIB_ASSERT(graph.len > 0, "graph build failed");

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
  RL_LIB_ASSERT(graph.len > 0, "graph build failed");

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
  RL_LIB_ASSERT(graph.len > 0, "graph build failed");

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
  RL_LIB_ASSERT(graph.len > 0, "graph build failed");

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
  RL_LIB_ASSERT(logits && logits_grad && loss_out && seed_grad,
             "arena out of memory in fused policy test");
  seed_grad->data[0] = 1.0f;

  u32 actions[3] = {1, 0, 3};
  real advantages[3] = {1.2f, -0.8f, 2.0f};

  rng_seed_g(9876, 5);
  mat_rand(logits, -1.0f, 1.0f);

  mat_zero(logits_grad);
  RL_LIB_CHECK(mat_fused_policy_loss(loss_out, logits, actions, advantages, 3));
  RL_LIB_CHECK(mat_fused_policy_add_grad(logits_grad, logits, actions, advantages,
                                      3, seed_grad));

  real h = GRAD_H;
  real tol = GRAD_TOL;
  u32 failures = 0;

  for (u32 i = 0; i < 12; i++) {
    real orig = logits->data[i];

    logits->data[i] = orig + h;
    Mat *up = mat_new(arena, 1, 1);
    RL_LIB_CHECK(mat_fused_policy_loss(up, logits, actions, advantages, 3));

    logits->data[i] = orig - h;
    Mat *dn = mat_new(arena, 1, 1);
    RL_LIB_CHECK(mat_fused_policy_loss(dn, logits, actions, advantages, 3));

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

// ── Test: PPO clipped surrogate policy loss ──────────────────────────

static void test_gradcheck_ppo_policy(void) {
  printf("\n[gradcheck: PPO clipped policy loss (B=4, act_dim=3)]\n");
  Arena *arena = arena_new(MiB(16));

  Mat *logits = mat_new(arena, 3, 4);
  Mat *logits_grad = mat_new(arena, 3, 4);
  Mat *loss_out = mat_new(arena, 1, 1);
  Mat *seed_grad = mat_new(arena, 1, 1);
  RL_LIB_ASSERT(logits && logits_grad && loss_out && seed_grad,
                "arena out of memory in ppo policy test");
  seed_grad->data[0] = 1.0f;

  u32 actions[4] = {2, 0, 1, 1};
  real advantages[4] = {1.1f, -0.6f, 0.9f, -1.4f};
  const real clip = 0.2f;

  rng_seed_g(13579, 11);
  mat_rand(logits, -1.5f, 1.5f);

  // Place each ratio clearly inside or outside the clip range by construction:
  // ratio = exp(new_logp - old_logp). Offsets of +/-0.5 give ratios of ~1.65
  // and ~0.61, far from the 0.8/1.2 boundaries, so finite differences never
  // straddle the kink. The advantage signs then make two columns clip-binding
  // (zero gradient) and two active (non-zero gradient).
  const real offsets[4] = {-0.5f, 0.5f, 0.5f, -0.5f};
  real old_logp[4];
  for (u32 b = 0; b < 4; b++) {
    real hi = logits->data[b];
    for (u32 a = 1; a < 3; a++) {
      const real z = logits->data[(u64)a * 4 + b];
      if (z > hi) hi = z;
    }
    real sum = 0.0f;
    for (u32 a = 0; a < 3; a++) {
      sum += real_exp(logits->data[(u64)a * 4 + b] - hi);
    }
    const real lse = hi + real_log(sum);
    const real new_logp = logits->data[(u64)actions[b] * 4 + b] - lse;
    old_logp[b] = new_logp - offsets[b];
  }

  mat_zero(logits_grad);
  RL_LIB_CHECK(mat_ppo_policy_loss(loss_out, logits, actions, advantages,
                                   old_logp, clip, 4));
  RL_LIB_CHECK(mat_ppo_policy_add_grad(logits_grad, logits, actions, advantages,
                                       old_logp, clip, 4, seed_grad));

  const real h = GRAD_H;
  const real tol = GRAD_TOL;
  u32 failures = 0;
  u32 nonzero = 0;

  for (u32 i = 0; i < 12; i++) {
    const real orig = logits->data[i];

    logits->data[i] = orig + h;
    Mat *up = mat_new(arena, 1, 1);
    RL_LIB_CHECK(mat_ppo_policy_loss(up, logits, actions, advantages, old_logp,
                                     clip, 4));

    logits->data[i] = orig - h;
    Mat *dn = mat_new(arena, 1, 1);
    RL_LIB_CHECK(mat_ppo_policy_loss(dn, logits, actions, advantages, old_logp,
                                     clip, 4));

    logits->data[i] = orig;

    const real numeric = (up->data[0] - dn->data[0]) / (2.0f * h);
    const real analytic = logits_grad->data[i];
    const real denom = real_fmax(
        (real)1.0, real_fmax(real_fabs(numeric), real_fabs(analytic)));
    const real rel = real_fabs(numeric - analytic) / denom;

    if (real_fabs(analytic) > 1e-4f) nonzero++;
    total_params_checked++;
    if (rel < tol) {
      params_ok++;
    } else {
      failures++;
    }
  }

  printf("  logits (3x4, 12 elements, %u non-zero)... %s (%u failures)\n",
         nonzero, failures == 0 ? "OK" : "FAIL", failures);
  if (failures == 0 && nonzero > 0)
    tests_passed++;
  else
    tests_failed++;

  arena_dispose(arena);
}

// ── Test: continuous (Gaussian) PPO surrogate ──────────────────────────
// Checks the clipped-surrogate gradient with respect to both the raw mean-head
// output and the learnable log standard deviation, on an asymmetric
// two-dimensional action box. `min_action` and `max_action` are deliberately
// different per dimension, so a per-dimension indexing mistake cannot cancel.
static void test_gradcheck_gaussian_ppo(void) {
  printf("\n[gradcheck: Gaussian PPO surrogate (B=4, act_dim=2)]\n");
  Arena *arena = arena_new(MiB(16));

  const u32 act_dim = 2;
  const u32 batch = 4;

  Mat *raw_mean = mat_new(arena, act_dim, batch);
  Mat *log_std = mat_new(arena, act_dim, 1);
  Mat *us = mat_new(arena, act_dim, batch);
  Mat *mean_grad = mat_new(arena, act_dim, batch);
  Mat *log_std_grad = mat_new(arena, act_dim, 1);
  Mat *loss_out = mat_new(arena, 1, 1);
  Mat *seed_grad = mat_new(arena, 1, 1);
  RL_LIB_ASSERT(raw_mean && log_std && us && mean_grad && log_std_grad &&
                    loss_out && seed_grad,
                "arena out of memory in Gaussian PPO test");
  seed_grad->data[0] = 1.0f;

  const real min_action[2] = {-2.0f, -0.5f};
  const real max_action[2] = {2.0f, 0.5f};
  const real clip = 0.2f;
  real advantages[4] = {1.1f, -0.6f, 0.9f, -1.4f};

  rng_seed_g(24680, 13);
  mat_rand(raw_mean, -1.5f, 1.5f);
  log_std->data[0] = -0.3f;
  log_std->data[1] = 0.4f;

  // Pre-squash samples (the surrogate's density is a function of u, not of the
  // action), well inside the mean clamp, so finite differences see a smooth
  // function.
  for (u32 b = 0; b < batch; b++) {
    us->data[b] = (real)(-1.4 + 0.7 * (double)b);
    us->data[(u64)batch + b] = (real)(-0.3 + 0.19 * (double)b);
  }

  Mat *logp = mat_new(arena, 1, batch);
  RL_LIB_CHECK(mat_gaussian_logp(logp, raw_mean, log_std, us, min_action,
                                 max_action, batch));

  // Place each ratio clearly inside or outside the clip range by construction:
  // ratio = exp(new_logp - old_logp), so an offset of +/-0.5 gives ~0.61 or
  // ~1.65, far from the 0.8/1.2 boundaries -- finite differences then never
  // straddle the kink. The advantage signs make two columns clip-binding (zero
  // gradient) and two active (non-zero gradient).
  const real offsets[4] = {-0.5f, 0.5f, 0.5f, -0.5f};
  real old_logp[4];
  for (u32 b = 0; b < batch; b++) {
    old_logp[b] = logp->data[b] - offsets[b];
  }

  // The log-density must agree with the textbook form, evaluated independently
  // in double precision: log N(u | mu, sigma) - log(half) - log(1 - tanh(u)^2).
  // That pins down both the log(sech^2(u)) identity and the -log(half) Jacobian
  // term of the squash.
  u32 lp_bad = 0;
  for (u32 b = 0; b < batch; b++) {
    double acc = 0.0;
    for (u32 d = 0; d < act_dim; d++) {
      const double uu = (double)us->data[(u64)d * batch + b];
      const double mu = (double)raw_mean->data[(u64)d * batch + b];
      const double sigma = exp((double)log_std->data[d]);
      const double half = 0.5 * ((double)max_action[d] - (double)min_action[d]);
      const double e = (uu - mu) / sigma;
      const double th = tanh(uu);
      acc += -0.5 * e * e - log(sigma) - 0.5 * log(2.0 * 3.141592653589793) -
             log(half) - log(1.0 - th * th);
    }
    if (fabs(acc - (double)logp->data[b]) > 1e-4 * (1.0 + fabs(acc))) {
      lp_bad++;
    }
  }
  printf("  log-density vs textbook form (4 columns)... %s (%u bad)\n",
         lp_bad == 0 ? "OK" : "FAIL", lp_bad);

  mat_zero(mean_grad);
  mat_zero(log_std_grad);
  RL_LIB_CHECK(mat_gaussian_ppo_loss(loss_out, raw_mean, log_std, us,
                                     advantages, old_logp, min_action,
                                     max_action, clip, batch));
  RL_LIB_CHECK(mat_gaussian_ppo_add_grad(
      mean_grad, log_std_grad, raw_mean, log_std, us, advantages, old_logp,
      min_action, max_action, clip, batch, seed_grad));

  const real h = GRAD_H;
  const real tol = GRAD_TOL;
  u32 failures = lp_bad;
  u32 nonzero = 0;

  for (u32 i = 0; i < act_dim * batch; i++) {
    const real orig = raw_mean->data[i];

    raw_mean->data[i] = orig + h;
    RL_LIB_CHECK(mat_gaussian_ppo_loss(loss_out, raw_mean, log_std, us,
                                       advantages, old_logp, min_action,
                                       max_action, clip, batch));
    const real up = loss_out->data[0];

    raw_mean->data[i] = orig - h;
    RL_LIB_CHECK(mat_gaussian_ppo_loss(loss_out, raw_mean, log_std, us,
                                       advantages, old_logp, min_action,
                                       max_action, clip, batch));
    const real dn = loss_out->data[0];

    raw_mean->data[i] = orig;

    const real numeric = (up - dn) / ((real)2 * h);
    const real analytic = mean_grad->data[i];
    const real denom = real_fmax(
        (real)1.0, real_fmax(real_fabs(numeric), real_fabs(analytic)));
    const real rel = real_fabs(numeric - analytic) / denom;

    if (real_fabs(analytic) > 1e-4f) nonzero++;
    total_params_checked++;
    if (rel < tol) {
      params_ok++;
    } else {
      failures++;
    }
  }
  printf("  raw mean (2x4, 8 elements, %u non-zero)... %s (%u failures)\n",
         nonzero, failures == 0 ? "OK" : "FAIL", failures);

  u32 lfailures = 0;
  u32 lnz = 0;
  for (u32 i = 0; i < act_dim; i++) {
    const real orig = log_std->data[i];

    log_std->data[i] = orig + h;
    RL_LIB_CHECK(mat_gaussian_ppo_loss(loss_out, raw_mean, log_std, us,
                                       advantages, old_logp, min_action,
                                       max_action, clip, batch));
    const real up = loss_out->data[0];

    log_std->data[i] = orig - h;
    RL_LIB_CHECK(mat_gaussian_ppo_loss(loss_out, raw_mean, log_std, us,
                                       advantages, old_logp, min_action,
                                       max_action, clip, batch));
    const real dn = loss_out->data[0];

    log_std->data[i] = orig;

    const real numeric = (up - dn) / ((real)2 * h);
    const real analytic = log_std_grad->data[i];
    const real denom = real_fmax(
        (real)1.0, real_fmax(real_fabs(numeric), real_fabs(analytic)));
    const real rel = real_fabs(numeric - analytic) / denom;

    if (real_fabs(analytic) > 1e-4f) lnz++;
    total_params_checked++;
    if (rel < tol) {
      params_ok++;
    } else {
      lfailures++;
    }
  }
  printf("  log_std (2x1, %u non-zero)... %s (%u failures)\n", lnz,
         lfailures == 0 ? "OK" : "FAIL", lfailures);

  // Gaussian entropy: d(mean H)/d(log_std_d) = 1.
  Mat *ent = mat_new(arena, 1, 1);
  mat_zero(log_std_grad);
  RL_LIB_CHECK(mat_gaussian_entropy(ent, log_std));
  RL_LIB_CHECK(mat_gaussian_entropy_add_grad(log_std_grad, log_std, seed_grad));
  u32 efailures = 0;
  for (u32 i = 0; i < act_dim; i++) {
    if (real_fabs(log_std_grad->data[i] - (real)1.0) > tol) efailures++;
    total_params_checked++;
    if (efailures == 0) params_ok++;
  }
  printf("  entropy d/d(log_std)... %s (%u failures)\n",
         efailures == 0 ? "OK" : "FAIL", efailures);

  if (failures + lfailures + efailures == 0 && nonzero > 0 && lnz > 0)
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
  RL_LIB_ASSERT(graph.len > 0, "graph build failed");

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

// ── Test: sigmoid / exp / log / div ──────────────────────────────────

static void test_gradcheck_elementwise_ops(void) {
  printf("\n[gradcheck: sigmoid, exp, log, div]\n");

  Arena *arena = arena_new(MiB(32));
  Net *model = ARENA_OBJ(arena, Net);

  Tensor *input = tensor_create(arena, model, 4, 1, TENSOR_PLAIN);
  Tensor *W = tensor_create(arena, model, 4, 4, TENSOR_PARAM | TENSOR_GRAD);
  Tensor *D = tensor_create(arena, model, 4, 1, TENSOR_PARAM | TENSOR_GRAD);

  // z = W x -> sigmoid / exp paths; log(exp(z)) exercises a composed chain;
  // the division puts both branches in the denominator.
  Tensor *z = tensor_mm(arena, model, W, input);
  Tensor *s = tensor_sigmoid(arena, model, z);
  Tensor *e = tensor_exp(arena, model, z);
  Tensor *l = tensor_log(arena, model, e);
  Tensor *denom = tensor_add(arena, model, l, D);
  Tensor *output = tensor_div(arena, model, s, denom);

  Schedule graph = schedule_from(arena, model, output);
  RL_LIB_ASSERT(graph.len > 0, "graph build failed");

  rng_seed_g(778899, 5);
  mat_rand(W->val, -0.3f, 0.3f);
  mat_rand(D->val, 2.0f, 4.0f); // keep the denominator away from zero
  mat_rand(input->val, 0.5f, 1.5f);

  real h = GRAD_H;
  real tol = GRAD_TOL;

  printf("  W (4x4, 16 params)... ");
  u32 f1 = gradcheck_param(&graph, output, W, h, tol);
  printf("%s (%u failures)\n", f1 == 0 ? "OK" : "FAIL", f1);

  printf("  D (4x1, 4 params)... ");
  u32 f2 = gradcheck_param(&graph, output, D, h, tol);
  printf("%s (%u failures)\n", f2 == 0 ? "OK" : "FAIL", f2);

  // Hand-computed forward values for the four new kernels.
  schedule_run(&graph);
  const real z0 = z->val->data[0];
  const real s_expected = (real)1.0 / ((real)1.0 + real_exp(-z0));
  const real e_expected = real_exp(z0);
  const real denom0 = l->val->data[0] + D->val->data[0];
  const real out_expected = s->val->data[0] / denom0;
  const b32 values_ok =
      real_fabs(s->val->data[0] - s_expected) < (real)1e-5 &&
      real_fabs(e->val->data[0] - e_expected) < (real)1e-5 &&
      real_fabs(l->val->data[0] - z0) < (real)1e-4 &&
      real_fabs(output->val->data[0] - out_expected) < (real)1e-5;
  printf("  sigmoid/exp/log/div vs hand-computed values... %s\n",
         values_ok ? "OK" : "FAIL");

  if (f1 + f2 == 0 && values_ok)
    tests_passed++;
  else
    tests_failed++;

  arena_dispose(arena);
}

// ── Test: reshape and broadcast ──────────────────────────────────────

static void test_gradcheck_shape_ops(void) {
  printf("\n[gradcheck: reshape and broadcast]\n");

  Arena *arena = arena_new(MiB(32));
  Net *model = ARENA_OBJ(arena, Net);

  Tensor *matrix = tensor_create(arena, model, 3, 4,
                                 TENSOR_PARAM | TENSOR_GRAD);
  Tensor *reshaped = tensor_reshape(arena, model, matrix, 4, 3);
  Tensor *reshape_out = tensor_sum(arena, model, reshaped, REDUCE_ALL);

  Tensor *row = tensor_create(arena, model, 1, 4,
                              TENSOR_PARAM | TENSOR_GRAD);
  Tensor *scalar = tensor_create(arena, model, 1, 1,
                                 TENSOR_PARAM | TENSOR_GRAD);
  Tensor *w = tensor_create(arena, model, 3, 4, TENSOR_PARAM | TENSOR_GRAD);
  Tensor *row_full = tensor_broadcast(arena, model, row, 3, 4);
  Tensor *scalar_full = tensor_broadcast(arena, model, scalar, 3, 4);
  Tensor *summed = tensor_add(arena, model, row_full, scalar_full);
  Tensor *broadcast_out = tensor_sum(
      arena, model, tensor_mul(arena, model, summed, w), REDUCE_ALL);

  Schedule reshape_graph = schedule_from(arena, model, reshape_out);
  Schedule broadcast_graph = schedule_from(arena, model, broadcast_out);

  rng_seed_g(112233, 4);
  mat_rand(matrix->val, -0.5f, 0.5f);
  mat_rand(row->val, -0.5f, 0.5f);
  mat_rand(scalar->val, -0.5f, 0.5f);
  mat_rand(w->val, -0.5f, 0.5f);

  printf("  matrix (3x4, reshaped to 4x3)... ");
  u32 f1 =
      gradcheck_param(&reshape_graph, reshape_out, matrix, GRAD_H, GRAD_TOL);
  printf("%s (%u failures)\n", f1 == 0 ? "OK" : "FAIL", f1);

  printf("  row (1x4, broadcast)... ");
  u32 f2 = gradcheck_param(&broadcast_graph, broadcast_out, row, GRAD_H,
                           GRAD_TOL);
  printf("%s (%u failures)\n", f2 == 0 ? "OK" : "FAIL", f2);

  printf("  scalar (1x1, broadcast)... ");
  u32 f3 = gradcheck_param(&broadcast_graph, broadcast_out, scalar, GRAD_H,
                           GRAD_TOL);
  printf("%s (%u failures)\n", f3 == 0 ? "OK" : "FAIL", f3);

  printf("  w (3x4, broadcast product)... ");
  u32 f4 =
      gradcheck_param(&broadcast_graph, broadcast_out, w, GRAD_H, GRAD_TOL);
  printf("%s (%u failures)\n", f4 == 0 ? "OK" : "FAIL", f4);

  if (f1 + f2 + f3 + f4 == 0)
    tests_passed++;
  else
    tests_failed++;

  arena_dispose(arena);
}

// ── Test: reductions ─────────────────────────────────────────────────

static void test_gradcheck_reduce(void) {
  printf("\n[gradcheck: sum and mean reductions]\n");

  Arena *arena = arena_new(MiB(32));
  Net *model = ARENA_OBJ(arena, Net);

  Tensor *rows_in = tensor_create(arena, model, 3, 4,
                                  TENSOR_PARAM | TENSOR_GRAD);
  Tensor *cols_in = tensor_create(arena, model, 3, 4,
                                  TENSOR_PARAM | TENSOR_GRAD);
  Tensor *all_in = tensor_create(arena, model, 3, 4,
                                 TENSOR_PARAM | TENSOR_GRAD);

  Tensor *rows_out = tensor_sum(arena, model, rows_in, REDUCE_ROWS);
  Tensor *cols_out = tensor_mean(arena, model, cols_in, REDUCE_COLS);
  Tensor *all_out = tensor_mean(arena, model, all_in, REDUCE_ALL);

  Schedule rows_graph = schedule_from(arena, model, rows_out);
  Schedule cols_graph = schedule_from(arena, model, cols_out);
  Schedule all_graph = schedule_from(arena, model, all_out);

  rng_seed_g(445566, 7);
  mat_rand(rows_in->val, -1.0f, 1.0f);
  mat_rand(cols_in->val, -1.0f, 1.0f);
  mat_rand(all_in->val, -1.0f, 1.0f);

  printf("  rows_in (3x4, sum over rows)... ");
  u32 f1 =
      gradcheck_param(&rows_graph, rows_out, rows_in, GRAD_H, GRAD_TOL);
  printf("%s (%u failures)\n", f1 == 0 ? "OK" : "FAIL", f1);

  printf("  cols_in (3x4, mean over cols)... ");
  u32 f2 =
      gradcheck_param(&cols_graph, cols_out, cols_in, GRAD_H, GRAD_TOL);
  printf("%s (%u failures)\n", f2 == 0 ? "OK" : "FAIL", f2);

  printf("  all_in (3x4, mean of all)... ");
  u32 f3 = gradcheck_param(&all_graph, all_out, all_in, GRAD_H, GRAD_TOL);
  printf("%s (%u failures)\n", f3 == 0 ? "OK" : "FAIL", f3);

  if (f1 + f2 + f3 == 0)
    tests_passed++;
  else
    tests_failed++;

  arena_dispose(arena);
}

// ── Test: gather ─────────────────────────────────────────────────────

static void test_gradcheck_gather(void) {
  printf("\n[gradcheck: gather]\n");

  Arena *arena = arena_new(MiB(32));
  Net *model = ARENA_OBJ(arena, Net);

  Tensor *logits = tensor_create(arena, model, 3, 4,
                                 TENSOR_PARAM | TENSOR_GRAD);
  Tensor *indices = tensor_create(arena, model, 1, 4, TENSOR_PLAIN);
  // Deliberately repeat index 2: the scatter in the backward pass must add.
  indices->val->data[0] = (real)0;
  indices->val->data[1] = (real)2;
  indices->val->data[2] = (real)1;
  indices->val->data[3] = (real)2;

  Tensor *picked = tensor_gather(arena, model, logits, indices);
  Schedule graph = schedule_from(arena, model, picked);

  rng_seed_g(667788, 9);
  mat_rand(logits->val, -1.0f, 1.0f);

  schedule_run(&graph);
  b32 values_ok = true;
  for (u32 c = 0; c < 4; c++) {
    const u32 idx = (u32)indices->val->data[c];
    const real expected = logits->val->data[(u64)idx * 4u + c];
    if (picked->val->data[c] != expected) {
      values_ok = false;
    }
  }
  printf("  forward selection (with a repeated index)... %s\n",
         values_ok ? "OK" : "FAIL");

  printf("  logits (3x4, repeated index)... ");
  u32 f = gradcheck_param(&graph, picked, logits, GRAD_H, GRAD_TOL);
  printf("%s (%u failures)\n", f == 0 ? "OK" : "FAIL", f);

  if (f == 0 && values_ok)
    tests_passed++;
  else
    tests_failed++;

  arena_dispose(arena);
}

// ── Test: min, max and clip ──────────────────────────────────────────

static void test_gradcheck_minmax_clip(void) {
  printf("\n[gradcheck: min, max, clip]\n");

  Arena *arena = arena_new(MiB(32));
  Net *model = ARENA_OBJ(arena, Net);

  Tensor *a = tensor_create(arena, model, 3, 2, TENSOR_PARAM | TENSOR_GRAD);
  Tensor *b = tensor_create(arena, model, 3, 2, TENSOR_PARAM | TENSOR_GRAD);
  Tensor *c = tensor_create(arena, model, 2, 2, TENSOR_PARAM | TENSOR_GRAD);

  Tensor *min_out = tensor_min(arena, model, a, b);
  Tensor *max_out = tensor_max(arena, model, a, b);
  Tensor *both = tensor_add(arena, model, min_out, max_out);
  Schedule mm_graph = schedule_from(arena, model, both);

  Tensor *clipped = tensor_clip(arena, model, c, REAL_C(-1.0), REAL_C(1.0));
  Schedule clip_graph = schedule_from(arena, model, clipped);

  rng_seed_g(889900, 11);
  mat_rand(a->val, -1.0f, 1.0f);
  mat_rand(b->val, -1.0f, 1.0f);
  // Keep every element well away from the clip bounds so the central
  // difference never straddles a kink.
  const real clip_vals[4] = {-2.0f, -0.4f, 0.4f, 2.0f};
  for (u32 i = 0; i < 4u; i++) {
    c->val->data[i] = clip_vals[i];
  }

  printf("  a (min and max LHS)... ");
  u32 f1 = gradcheck_param(&mm_graph, both, a, GRAD_H, GRAD_TOL);
  printf("%s (%u failures)\n", f1 == 0 ? "OK" : "FAIL", f1);

  printf("  b (min and max RHS)... ");
  u32 f2 = gradcheck_param(&mm_graph, both, b, GRAD_H, GRAD_TOL);
  printf("%s (%u failures)\n", f2 == 0 ? "OK" : "FAIL", f2);

  printf("  c (clip to [-1, 1])... ");
  u32 f3 = gradcheck_param(&clip_graph, clipped, c, GRAD_H, GRAD_TOL);
  printf("%s (%u failures)\n", f3 == 0 ? "OK" : "FAIL", f3);

  schedule_run(&clip_graph);
  const real expected[4] = {-1.0f, -0.4f, 0.4f, 1.0f};
  b32 values_ok = true;
  for (u32 i = 0; i < 4u; i++) {
    if (clipped->val->data[i] != expected[i]) {
      values_ok = false;
    }
  }
  printf("  clipped forward values... %s\n", values_ok ? "OK" : "FAIL");

  if (f1 + f2 + f3 == 0 && values_ok)
    tests_passed++;
  else
    tests_failed++;

  arena_dispose(arena);
}

int main(void) {
  printf("\n=== Autograd gradient checks (%s) ===\n\n", GRADCHECK_NAME);

  test_gradcheck_mlp();
  test_gradcheck_softmax();
  test_gradcheck_sub();
  test_gradcheck_broadcast_add();
  test_gradcheck_tanh_mul();
  test_gradcheck_elementwise_ops();
  test_gradcheck_shape_ops();
  test_gradcheck_reduce();
  test_gradcheck_gather();
  test_gradcheck_minmax_clip();
  test_gradcheck_mse();
  test_gradcheck_fused_policy();
  test_gradcheck_ppo_policy();
  test_gradcheck_gaussian_ppo();
  test_gradcheck_reinforce();
  test_gradcheck_input_and_mul();

  printf("\n---\nTests: %u passed, %u failed\n", tests_passed, tests_failed);
  printf("Parameters checked: %u/%u passed within tolerance\n\n", params_ok,
         total_params_checked);

  return tests_failed > 0 ? 1 : 0;
}
