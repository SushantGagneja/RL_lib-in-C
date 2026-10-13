// test_gradcheck.c — Numerical gradient check for the autograd engine
// Phase 0.1: The highest-value item. Compare analytic gradients from
// backward_pass against central finite differences.
#include "arena.h"
#include "autograd.h"
#include "base.h"
#include "mat.h"
#include "prng.h"

#include <math.h>

// ── Helpers ──────────────────────────────────────────────────────────

static u32 tests_passed = 0;
static u32 tests_failed = 0;
static u32 total_params_checked = 0;
static u32 params_ok = 0;

// Sum all elements of a matrix (the scalar "loss" when backward seeds with 1).
static f32 sum_of(const matrix *m) {
  f32 s = 0.0f;
  u64 n = (u64)m->rows * m->cols;
  for (u64 i = 0; i < n; i++)
    s += m->data[i];
  return s;
}

// Clear all parameter gradients in the graph to zero.
static void clear_param_grads(Graph *g) {
  for (u32 i = 0; i < g->size; i++) {
    Var *v = g->vars[i];
    if ((v->flags & VAR_FLAG_PARAMETER) && v->grad != NULL)
      clear(v->grad);
  }
}

// Clear all non-parameter activation gradients in the graph to zero.
static void clear_activation_grads(Graph *g) {
  for (u32 i = 0; i < g->size; i++) {
    Var *v = g->vars[i];
    if (!(v->flags & VAR_FLAG_PARAMETER) && v->grad != NULL)
      clear(v->grad);
  }
}

// ── Gradient check core ──────────────────────────────────────────────

// Check gradients for a single parameter Var against central differences.
// Returns the number of elements that failed the tolerance check.
static u32 gradcheck_param(Graph *graph, Var *output, Var *param, f32 h,
                           f32 tol) {
  u64 n = (u64)param->val->rows * param->val->cols;
  u32 failures = 0;

  // 1. Run forward + backward to get analytic gradients
  clear_param_grads(graph);
  clear_activation_grads(graph);
  forward_pass(graph);
  backward_pass(graph);

  // Save analytic gradients (backward accumulates into param->grad)
  f32 *analytic = malloc(sizeof(f32) * n);
  memcpy(analytic, param->grad->data, sizeof(f32) * n);

  // 2. Central differences for each element
  for (u64 i = 0; i < n; i++) {
    f32 orig = param->val->data[i];

    param->val->data[i] = orig + h;
    forward_pass(graph);
    f32 up = sum_of(output->val);

    param->val->data[i] = orig - h;
    forward_pass(graph);
    f32 dn = sum_of(output->val);

    param->val->data[i] = orig; // restore

    f32 numeric = (up - dn) / (2.0f * h);
    f32 denom = fmaxf(1e-6f, fabsf(numeric) + fabsf(analytic[i]));
    f32 rel = fabsf(numeric - analytic[i]) / denom;

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

// ── Test: matmul → relu → matmul → add → output ─────────────────────

static void test_gradcheck_mlp(void) {
  printf("[gradcheck: matmul → relu → matmul → add]\n");

  mem_arena *arena = arena_create(MiB(32));
  model_state *model = PUSH_STRUCT(arena, model_state);

  // Build a small 2-layer network: input(4x1) → W0(3x4) → relu → W1(2x3) +
  // b1(2x1)
  Var *input =  var_create(arena, model, 4, 1, VAR_FLAG_NONE);
  Var *W0 =     var_create(arena, model, 3, 4, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
  Var *W1 =     var_create(arena, model, 2, 3, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
  Var *b1 =     var_create(arena, model, 2, 1, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);

  Var *z0 = var_matmul(arena, model, W0, input);
  Var *a0 = var_relu(arena, model, z0);
  Var *z1 = var_matmul(arena, model, W1, a0);
  Var *output = var_add(arena, model, z1, b1);

  Graph graph = build_graph(arena, model, output);
  assert(graph.size > 0 && "Graph build failed");

  // Init parameters away from zero (ReLU kink)
  prng_seed(12345, 1);
  fill_rand(W0->val, 0.1f, 1.0f);
  fill_rand(W1->val, 0.1f, 1.0f);
  fill_rand(b1->val, 0.1f, 0.5f);

  // Input also away from zero
  fill_rand(input->val, 0.5f, 2.0f);

  f32 h = 1e-2f;
  f32 tol = 1e-2f;

  printf("  W0 (3x4, %d params)... ", 12);
  u32 f0 = gradcheck_param(&graph, output, W0, h, tol);
  printf("%s (%u failures)\n", f0 == 0 ? "OK" : "FAIL", f0);

  printf("  W1 (2x3, %d params)... ", 6);
  u32 f1 = gradcheck_param(&graph, output, W1, h, tol);
  printf("%s (%u failures)\n", f1 == 0 ? "OK" : "FAIL", f1);

  printf("  b1 (2x1, %d params)... ", 2);
  u32 fb = gradcheck_param(&graph, output, b1, h, tol);
  printf("%s (%u failures)\n", fb == 0 ? "OK" : "FAIL", fb);

  if (f0 + f1 + fb == 0) tests_passed++;
  else tests_failed++;

  arena_destroy(arena);
}

// ── Test: matmul → softmax ───────────────────────────────────────────

static void test_gradcheck_softmax(void) {
  printf("\n[gradcheck: matmul → softmax]\n");

  mem_arena *arena = arena_create(MiB(32));
  model_state *model = PUSH_STRUCT(arena, model_state);

  Var *input = var_create(arena, model, 3, 1, VAR_FLAG_NONE);
  Var *W =     var_create(arena, model, 5, 3, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
  Var *b =     var_create(arena, model, 5, 1, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);

  Var *z = var_matmul(arena, model, W, input);
  Var *zb = var_add(arena, model, z, b);
  Var *sm = var_softmax(arena, model, zb);

  // Project softmax with non-uniform weights to test softmax Jacobian non-trivially
  // (otherwise sum(softmax) is identically 1 with zero derivative everywhere).
  Var *target = var_create(arena, model, 1, 5, VAR_FLAG_NONE);
  target->val->data[0] = 1.0f;
  target->val->data[1] = 2.0f;
  target->val->data[2] = 0.5f;
  target->val->data[3] = -1.0f;
  target->val->data[4] = 3.0f;
  Var *output = var_matmul(arena, model, target, sm);

  Graph graph = build_graph(arena, model, output);
  assert(graph.size > 0);

  prng_seed(99999, 2);
  fill_rand(W->val, -0.5f, 0.5f);
  fill_rand(b->val, -0.2f, 0.2f);
  fill_rand(input->val, 0.5f, 1.5f);

  f32 h = 1e-2f;
  f32 tol = 1e-2f;

  printf("  W (5x3, %d params)... ", 15);
  u32 fw = gradcheck_param(&graph, output, W, h, tol);
  printf("%s (%u failures)\n", fw == 0 ? "OK" : "FAIL", fw);

  printf("  b (5x1, %d params)... ", 5);
  u32 fb = gradcheck_param(&graph, output, b, h, tol);
  printf("%s (%u failures)\n", fb == 0 ? "OK" : "FAIL", fb);

  if (fw + fb == 0) tests_passed++;
  else tests_failed++;

  arena_destroy(arena);
}

// ── Test: sub backward ───────────────────────────────────────────────

static void test_gradcheck_sub(void) {
  printf("\n[gradcheck: matmul → sub]\n");

  mem_arena *arena = arena_create(MiB(32));
  model_state *model = PUSH_STRUCT(arena, model_state);

  Var *input = var_create(arena, model, 3, 1, VAR_FLAG_NONE);
  Var *W =     var_create(arena, model, 4, 3, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
  Var *b =     var_create(arena, model, 4, 1, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);

  Var *z = var_matmul(arena, model, W, input);
  Var *output = var_sub(arena, model, z, b);

  Graph graph = build_graph(arena, model, output);
  assert(graph.size > 0);

  prng_seed(77777, 3);
  fill_rand(W->val, -1.0f, 1.0f);
  fill_rand(b->val, -0.5f, 0.5f);
  fill_rand(input->val, 0.5f, 1.5f);

  f32 h = 1e-2f;
  f32 tol = 1e-2f;

  printf("  W (4x3, %d params)... ", 12);
  u32 fw = gradcheck_param(&graph, output, W, h, tol);
  printf("%s (%u failures)\n", fw == 0 ? "OK" : "FAIL", fw);

  printf("  b (4x1, %d params)... ", 4);
  u32 fb = gradcheck_param(&graph, output, b, h, tol);
  printf("%s (%u failures)\n", fb == 0 ? "OK" : "FAIL", fb);

  if (fw + fb == 0) tests_passed++;
  else tests_failed++;

  arena_destroy(arena);
}

// ── Test: full actor network (matmul→add→relu→matmul→add→relu→matmul→add→softmax→reinforce) ──

static void test_gradcheck_reinforce(void) {
  printf("\n[gradcheck: full actor → reinforce loss]\n");

  mem_arena *arena = arena_create(MiB(64));
  model_state *model = PUSH_STRUCT(arena, model_state);

  // Smaller version of the actual model: input(8x1) → 6 → 4 → softmax(3)
  Var *input = var_create(arena, model, 8, 1, VAR_FLAG_NONE);

  Var *W0 = var_create(arena, model, 6, 8, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
  Var *b0 = var_create(arena, model, 6, 1, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
  Var *W1 = var_create(arena, model, 4, 6, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
  Var *b1 = var_create(arena, model, 4, 1, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
  Var *W2 = var_create(arena, model, 3, 4, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
  Var *b2 = var_create(arena, model, 3, 1, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);

  Var *z0 = var_matmul(arena, model, W0, input);
  Var *z0b = var_add(arena, model, z0, b0);
  Var *a0 = var_relu(arena, model, z0b);

  Var *z1 = var_matmul(arena, model, W1, a0);
  Var *z1b = var_add(arena, model, z1, b1);
  Var *a1 = var_relu(arena, model, z1b);

  Var *z2 = var_matmul(arena, model, W2, a1);
  Var *z2b = var_add(arena, model, z2, b2);
  Var *probs = var_softmax(arena, model, z2b);

  Var *advantage = var_create(arena, model, 3, 1, VAR_FLAG_NONE);
  Var *cost = var_reinforce_loss(arena, model, probs, advantage);

  Graph graph = build_graph(arena, model, cost);
  assert(graph.size > 0);

  // Init everything away from zero
  prng_seed(54321, 7);
  fill_rand(W0->val, 0.1f, 0.5f);
  fill_rand(b0->val, 0.01f, 0.1f);
  fill_rand(W1->val, 0.1f, 0.5f);
  fill_rand(b1->val, 0.01f, 0.1f);
  fill_rand(W2->val, 0.1f, 0.5f);
  fill_rand(b2->val, 0.01f, 0.1f);
  fill_rand(input->val, 0.5f, 2.0f);

  // Set advantage to a one-hot-like vector with a positive value at action 1
  clear(advantage->val);
  advantage->val->data[1] = 1.5f;

  f32 h = 1e-2f;
  f32 tol = 5e-2f; // Reinforce loss has the clamp, so be slightly more lenient

  u32 total_fail = 0;

  const char *names[] = {"W0", "b0", "W1", "b1", "W2", "b2"};
  Var *params[] = {W0, b0, W1, b1, W2, b2};

  for (u32 p = 0; p < 6; p++) {
    u32 n = params[p]->val->rows * params[p]->val->cols;
    printf("  %s (%ux%u, %u params)... ", names[p], params[p]->val->rows,
           params[p]->val->cols, n);
    u32 f = gradcheck_param(&graph, cost, params[p], h, tol);
    printf("%s (%u failures)\n", f == 0 ? "OK" : "FAIL", f);
    total_fail += f;
  }

  if (total_fail == 0) tests_passed++;
  else tests_failed++;

  arena_destroy(arena);
}

// ── Test: broadcast add across batch ─────────────────────────────────

static void test_gradcheck_broadcast_add(void) {
  printf("\n[gradcheck: broadcast add (batch B=2)]\n");

  mem_arena *arena = arena_create(MiB(32));
  model_state *model = PUSH_STRUCT(arena, model_state);

  // X is (4x2), W is (3x4) -> z is (3x2), b is (3x1) broadcasted
  Var *X = var_create(arena, model, 4, 2, VAR_FLAG_NONE);
  Var *W = var_create(arena, model, 3, 4, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
  Var *b = var_create(arena, model, 3, 1, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);

  Var *z = var_matmul(arena, model, W, X);
  Var *output = var_add(arena, model, z, b);

  Graph graph = build_graph(arena, model, output);
  assert(graph.size > 0);

  prng_seed(112233, 4);
  fill_rand(W->val, -0.5f, 0.5f);
  fill_rand(b->val, -0.5f, 0.5f);
  fill_rand(X->val, 0.5f, 1.5f);

  f32 h = 1e-2f;
  f32 tol = 1e-2f;

  printf("  W (3x4, 12 params)... ");
  u32 fw = gradcheck_param(&graph, output, W, h, tol);
  printf("%s (%u failures)\n", fw == 0 ? "OK" : "FAIL", fw);

  printf("  b (3x1 broadcast, 3 params)... ");
  u32 fb = gradcheck_param(&graph, output, b, h, tol);
  printf("%s (%u failures)\n", fb == 0 ? "OK" : "FAIL", fb);

  if (fw + fb == 0) tests_passed++;
  else tests_failed++;

  arena_destroy(arena);
}

// ── Test: elementwise mul & tanh ─────────────────────────────────────

static void test_gradcheck_tanh_mul(void) {
  printf("\n[gradcheck: tanh & elementwise mul]\n");

  mem_arena *arena = arena_create(MiB(32));
  model_state *model = PUSH_STRUCT(arena, model_state);

  Var *input = var_create(arena, model, 3, 1, VAR_FLAG_NONE);
  Var *W1 = var_create(arena, model, 3, 3, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
  Var *W2 = var_create(arena, model, 3, 3, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);

  Var *z1 = var_matmul(arena, model, W1, input);
  Var *z2 = var_matmul(arena, model, W2, input);
  Var *t2 = var_tanh(arena, model, z2);
  Var *output = var_mul(arena, model, z1, t2);

  Graph graph = build_graph(arena, model, output);
  assert(graph.size > 0);

  prng_seed(445566, 5);
  fill_rand(W1->val, -0.5f, 0.5f);
  fill_rand(W2->val, -0.5f, 0.5f);
  fill_rand(input->val, 0.5f, 1.5f);

  f32 h = 1e-2f;
  f32 tol = 1e-2f;

  printf("  W1 (3x3, 9 params)... ");
  u32 f1 = gradcheck_param(&graph, output, W1, h, tol);
  printf("%s (%u failures)\n", f1 == 0 ? "OK" : "FAIL", f1);

  printf("  W2 (3x3, 9 params)... ");
  u32 f2 = gradcheck_param(&graph, output, W2, h, tol);
  printf("%s (%u failures)\n", f2 == 0 ? "OK" : "FAIL", f2);

  if (f1 + f2 == 0) tests_passed++;
  else tests_failed++;

  arena_destroy(arena);
}

// ── Test: MSE loss ───────────────────────────────────────────────────

static void test_gradcheck_mse(void) {
  printf("\n[gradcheck: MSE loss]\n");

  mem_arena *arena = arena_create(MiB(32));
  model_state *model = PUSH_STRUCT(arena, model_state);

  Var *input = var_create(arena, model, 4, 2, VAR_FLAG_NONE);
  Var *W = var_create(arena, model, 3, 4, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
  Var *target = var_create(arena, model, 3, 2, VAR_FLAG_NONE);

  Var *pred = var_matmul(arena, model, W, input);
  Var *loss = var_mse(arena, model, pred, target);

  Graph graph = build_graph(arena, model, loss);
  assert(graph.size > 0);

  prng_seed(778899, 6);
  fill_rand(W->val, -0.5f, 0.5f);
  fill_rand(input->val, 0.5f, 1.5f);
  fill_rand(target->val, 0.0f, 1.0f);

  f32 h = 1e-2f;
  f32 tol = 1e-2f;

  printf("  W (3x4, 12 params)... ");
  u32 fw = gradcheck_param(&graph, loss, W, h, tol);
  printf("%s (%u failures)\n", fw == 0 ? "OK" : "FAIL", fw);

  if (fw == 0) tests_passed++;
  else tests_failed++;

  arena_destroy(arena);
}

// ── Main ─────────────────────────────────────────────────────────────

int main(void) {
  printf("\n=== Autograd gradient checks ===\n\n");

  test_gradcheck_mlp();
  test_gradcheck_softmax();
  test_gradcheck_sub();
  test_gradcheck_broadcast_add();
  test_gradcheck_tanh_mul();
  test_gradcheck_mse();
  test_gradcheck_reinforce();

  printf("\n---\nTests: %u passed, %u failed\n", tests_passed, tests_failed);
  printf("Parameters checked: %u/%u passed within tolerance\n\n", params_ok,
         total_params_checked);

  return tests_failed > 0 ? 1 : 0;
}
