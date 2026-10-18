// test_rl.c — Tests for the modules the original suite never touched:
// optimizers, network builders, GAE, the A2C update path, batched-vs-single
// equivalence, and end-to-end training smoke/determinism checks.
#include "memory.h"
#include "graph.h"
#include "base.h"
#include "envs/cartpole.h"
#include "linalg.h"
#include "layers.h"
#include "optim.h"
#include "random.h"
#include "rl.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

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
  ASSERT_TRUE(real_fabs((real)((a) - (b))) < (real)(tol), msg)

// ── GAE against a hand-computed 3-step trajectory ────────────────────

static void test_gae_handcomputed(void) {
  printf("  GAE hand-computed 3-step... ");

  Arena *arena = arena_new(MiB(1));
  Rollout *buf = rollout_new(arena, 1, 8, 1);
  Episode *t = &buf->trajectories[0];

  t->len = 3;
  t->rewards[0] = 1.0f; t->rewards[1] = 0.0f; t->rewards[2] = 2.0f;
  t->values[0] = 0.5f;  t->values[1] = 0.5f;  t->values[2] = 0.5f;
  t->dones[0] = 0;      t->dones[1] = 0;      t->dones[2] = 1;

  gae_compute(t, 0.9f, 0.8f, 0.0f);

  ASSERT_NEAR(t->advantages[2], 1.5f, 1e-4f, "adv[2] = 1.5");
  ASSERT_NEAR(t->advantages[1], 1.03f, 1e-4f, "adv[1] = 1.03");
  ASSERT_NEAR(t->advantages[0], 1.6916f, 1e-4f, "adv[0] = 1.6916");
  ASSERT_NEAR(t->returns[2], 2.0f, 1e-4f, "ret[2] = 2.0");
  ASSERT_NEAR(t->returns[0], 2.1916f, 1e-4f, "ret[0] = 2.1916");

  arena_dispose(arena);
  printf("OK\n");
}

// ── Adam single step against a hand-computed value ───────────────────

static void test_adam_step(void) {
  printf("  Adam single step... ");
  Arena *arena = arena_new(MiB(1));
  Net *model = net_new(arena);
  Tensor *p = tensor_create(arena, model, 1, 1,
                      TENSOR_PARAM | TENSOR_GRAD);
  p->val->data[0] = 1.0f;
  p->grad->data[0] = 1.0f;

  Adam *adam = adam_new(arena, model, 0.1f);
  adam_step(adam, model);
  // m=0.1, v=0.001, m_hat=1, v_hat=1 => update 0.1 => 0.9
  ASSERT_NEAR(p->val->data[0], 0.9f, 1e-5f, "Adam step 1.0 -> 0.9");

  arena_dispose(arena);
  printf("OK\n");
}

// ── Global-norm gradient clipping ────────────────────────────────────

static void test_grad_clip(void) {
  printf("  gradient norm clipping... ");
  Arena *arena = arena_new(MiB(1));
  Net *model = net_new(arena);
  Tensor *a = tensor_create(arena, model, 1, 1,
                      TENSOR_PARAM | TENSOR_GRAD);
  Tensor *b = tensor_create(arena, model, 1, 1,
                      TENSOR_PARAM | TENSOR_GRAD);
  a->grad->data[0] = 3.0f;
  b->grad->data[0] = 4.0f;

  real norm = clip_grad_norm(model, 1.0f);
  ASSERT_NEAR(norm, 5.0f, 1e-5f, "pre-clip norm = 5");
  real clipped = real_sqrt(a->grad->data[0] * a->grad->data[0] +
                      b->grad->data[0] * b->grad->data[0]);
  ASSERT_NEAR(clipped, 1.0f, 1e-5f, "post-clip norm = 1");
  ASSERT_NEAR(a->grad->data[0], 0.6f, 1e-5f, "a scaled to 0.6");
  ASSERT_NEAR(b->grad->data[0], 0.8f, 1e-5f, "b scaled to 0.8");

  arena_dispose(arena);
  printf("OK\n");
}

// ── Actor-critic builds and both heads run ───────────────────────────

static void test_actor_critic_build(void) {
  printf("  actor-critic builds and runs... ");
  Arena *arena = arena_new(MiB(16));
  Net *model = net_new(arena);
  Rng rng;
  rng_seed(&rng, 7, 1);

  const u32 hidden[] = {16, 16};
  actor_critic_build(arena, model, &rng, 4, hidden, 2, 3, 5);

  ASSERT_TRUE(model->fwd.len > 0, "forward graph built");
  ASSERT_TRUE(model->train.len > 0, "batch graph built");
  ASSERT_TRUE(model->output != NULL, "softmax output exists");
  ASSERT_TRUE(model->value != NULL, "value head exists");

  // Run the rollout graph: the softmax output must actually be a distribution.
  for (u32 i = 0; i < 4; i++) model->input->val->data[i] = 0.1f * (real)i;
  schedule_run(&model->fwd);

  real sum = 0.0f;
  for (u32 i = 0; i < 3; i++) sum += model->output->val->data[i];
  ASSERT_NEAR(sum, 1.0f, 1e-5f, "rollout softmax sums to 1");
  ASSERT_TRUE(isfinite(model->value->val->data[0]), "value head is finite");

  // Batch graph: every column must be a valid distribution.
  for (u32 c = 0; c < 5; c++) {
    for (u32 r = 0; r < 4; r++) {
      model->batch_input->val->data[r * 5 + c] = 0.05f * (real)(r + c);
    }
  }
  schedule_run(&model->train);
  u32 bad_cols = 0;
  for (u32 c = 0; c < 5; c++) {
    real cs = 0.0f;
    for (u32 a = 0; a < 3; a++) cs += model->batch_logits->val->data[a * 5 + c];
    // batch_logits are pre-softmax; check finite and column-shaped instead.
    if (!isfinite(cs)) bad_cols++;
  }
  ASSERT_TRUE(bad_cols == 0, "batched logits finite");

  arena_dispose(arena);
  printf("OK\n");
}

// ── Value head actually learns (the A2C P0 regression guard) ─────────

static void test_actor_critic_value_learns(void) {
  printf("  actor-critic value head learns... ");
  Arena *arena = arena_new(MiB(16));
  Net *model = net_new(arena);
  Rng rng;
  rng_seed(&rng, 11, 1);

  const u32 B = 8;
  actor_critic_build(arena, model, &rng, 4, (u32[]){16}, 1, 2, B);

  // Fixed target batch: value = sum of observation components.
  Mat *target = mat_new(arena, 1, B);
  for (u32 c = 0; c < B; c++) {
    real s = 0.0f;
    for (u32 r = 0; r < 4; r++) {
      real v = 0.1f * (real)(r + c) - 0.2f;
      model->batch_input->val->data[r * B + c] = v;
      s += v;
    }
    target->data[c] = s;
  }

  Adam *adam = adam_new(arena, model, 0.01f);
  Mat *seed = mat_new(arena, 1, 1);
  seed->data[0] = 1.0f;
  Mat *loss = mat_new(arena, 1, 1);

  RL_LIB_CHECK(mat_mse_loss(loss, model->batch_value->val, target));
  real first_loss = 0.0f;

  for (u32 step = 0; step < 400; step++) {
    schedule_run(&model->train);
    schedule_zero_act(&model->train);

    // Only the value head is trained here.
    RL_LIB_CHECK(mat_mse_add_grad(model->batch_value->grad, model->batch_value->val,
                               target, seed));
    schedule_grad_accum(&model->train);
    adam_step(adam, model);
    zero_grad(model);

    if (step == 0) {
      RL_LIB_CHECK(mat_mse_loss(loss, model->batch_value->val, target));
      first_loss = loss->data[0];
    }
  }

  schedule_run(&model->train);
  RL_LIB_CHECK(mat_mse_loss(loss, model->batch_value->val, target));
  real final_loss = loss->data[0];

  printf("(loss %.4f -> %.4f) ", first_loss, final_loss);
  ASSERT_TRUE(final_loss < first_loss * 0.1f, "value MSE dropped >90%");
  ASSERT_TRUE(isfinite(final_loss), "final value loss finite");

  arena_dispose(arena);
  printf("OK\n");
}

// ── Batched forward == per-sample forward; batched grad == sum of grads ──

static void test_batch_equivalence(void) {
  printf("  batched vs per-sample equivalence... ");
  Arena *arena = arena_new(MiB(32));
  Net *model = net_new(arena);
  Rng rng;
  rng_seed(&rng, 123, 1);

  const u32 obs_dim = 5, act_dim = 3, B = 6;
  actor_build(arena, model, &rng, obs_dim, (u32[]){8}, 1, act_dim, B);
  model->loss_plan = schedule_from(arena, model, model->cost);

  // Fixed observations.
  real obs[6][5];
  for (u32 c = 0; c < B; c++) {
    for (u32 r = 0; r < obs_dim; r++) {
      obs[c][r] = 0.3f * real_sin((real)(r + 1) * (real)(c + 1));
      model->batch_input->val->data[r * B + c] = obs[c][r];
    }
  }

  // (a) Forward equivalence: batched softmax columns match single-sample runs.
  schedule_run(&model->train);
  Mat *batch_probs = mat_new(arena, act_dim, B);
  RL_LIB_CHECK(mat_softmax(batch_probs, model->batch_logits->val));

  real max_diff = 0.0f;
  for (u32 c = 0; c < B; c++) {
    for (u32 r = 0; r < obs_dim; r++) model->input->val->data[r] = obs[c][r];
    schedule_run(&model->fwd);
    for (u32 a = 0; a < act_dim; a++) {
      real d = real_fabs(model->output->val->data[a] - batch_probs->data[a * B + c]);
      if (d > max_diff) max_diff = d;
    }
  }
  ASSERT_TRUE(max_diff < 1e-5f, "batched softmax == per-sample softmax");

  // (b) Gradient equivalence: fused batched policy gradient equals the sum of
  // per-sample reinforce gradients (scaled by 1/B).
  u32 actions[6] = {0, 2, 1, 1, 0, 2};
  real advs[6] = {0.5f, -1.0f, 0.7f, 0.2f, -0.4f, 1.3f};

  // Batched path. The fused loss is already a mean over the batch, so the seed
  // is 1.0; the per-sample path below divides each term by B instead.
  zero_grad(model);
  schedule_run(&model->train);
  schedule_zero_act(&model->train);
  Mat *seed = mat_new(arena, 1, 1);
  seed->data[0] = 1.0f;
  RL_LIB_CHECK(mat_fused_policy_add_grad(model->batch_logits->grad,
                                      model->batch_logits->val, actions, advs, B,
                                      seed));
  schedule_grad_accum(&model->train);

  // Snapshot batched parameter gradients.
  u32 nparams = model->n_params;
  real **batched = malloc(sizeof(real *) * nparams);
  real **per_pt = malloc(sizeof(real *) * nparams);
  for (u32 k = 0; k < nparams; k++) {
    u64 n = (u64)model->params[k]->val->rows * model->params[k]->val->cols;
    batched[k] = malloc(sizeof(real) * n);
    per_pt[k] = malloc(sizeof(real) * n);
    memcpy(batched[k], model->params[k]->grad->data, sizeof(real) * n);
  }

  // Per-sample path through the single-sample reinforce cost graph.
  zero_grad(model);
  for (u32 c = 0; c < B; c++) {
    for (u32 r = 0; r < obs_dim; r++) model->input->val->data[r] = obs[c][r];
    mat_zero(model->advantage->val);
    model->advantage->val->data[actions[c]] = advs[c] / (real)B;
    schedule_run(&model->loss_plan);
    schedule_grad(&model->loss_plan);
  }
  for (u32 k = 0; k < nparams; k++) {
    u64 n = (u64)model->params[k]->val->rows * model->params[k]->val->cols;
    memcpy(per_pt[k], model->params[k]->grad->data, sizeof(real) * n);
  }

  real worst = 0.0f;
  for (u32 k = 0; k < nparams; k++) {
    u64 n = (u64)model->params[k]->val->rows * model->params[k]->val->cols;
    for (u64 i = 0; i < n; i++) {
      real d = real_fabs(batched[k][i] - per_pt[k][i]);
      if (d > worst) worst = d;
    }
    free(batched[k]);
    free(per_pt[k]);
  }
  free(batched);
  free(per_pt);

  printf("(max grad diff %.2e) ", worst);
  ASSERT_TRUE(worst < 1e-4f, "batched gradient == sum of per-sample grads");

  arena_dispose(arena);
  printf("OK\n");
}

// ── Parameter checksum helper ────────────────────────────────────────

static real param_checksum(Net *model) {
  real s = 0.0f;
  for (u32 k = 0; k < model->n_params; k++) {
    Tensor *p = model->params[k];
    u64 n = (u64)p->val->rows * p->val->cols;
    for (u64 i = 0; i < n; i++) s += p->val->data[i] * (real)(i + 1);
  }
  return s;
}

// ── CartPole learning smoke test ─────────────────────────────────────

static void test_cartpole_learning_smoke(void) {
  printf("  CartPole REINFORCE learns (20 epochs)... ");
  Arena *arena = arena_new(GiB(1));
  Net *model = net_new(arena);
  Env *env = env_cartpole_create(arena);
  Rng rng;
  rng_seed(&rng, 42, 1);

  const u32 hidden[] = {64, 64};
  actor_build(arena, model, &rng, env->obs_dim, hidden, 2,
                          env->act_dim, 32);
  model->loss_plan = schedule_from(arena, model, model->cost);

  TrainCfg cfg = {
      .epochs = 20,
      .rollout_size = 32,
      .max_steps = env->max_steps,
      .learning_rate = 0.005f,
      .gamma = 0.99f,
      .seed = 42,
      .optim = OPT_ADAM,
      .max_grad_norm = 0.5f,
      .batch_size = 32,
  };
  train_reinforce(env, model, &cfg, arena);

  Rng eval_rng;
  rng_seed(&eval_rng, 999, 1);
  EvalResult res = evaluate(env, model, &eval_rng, 30, true);
  printf("(greedy return %.1f) ", res.avg_return);

  // Random scores ~22; require a clear margin above that.
  ASSERT_TRUE(res.avg_return > 100.0f, "learned policy beats random by a lot");

  arena_dispose(arena);
  printf("OK\n");
}

// ── Same-seed determinism ────────────────────────────────────────────

static void test_determinism(void) {
  printf("  same seed => identical weights... ");
  real checksum[2];

  for (u32 run = 0; run < 2; run++) {
    Arena *arena = arena_new(GiB(1));
    Net *model = net_new(arena);
    Env *env = env_cartpole_create(arena);
    Rng rng;
    rng_seed(&rng, 1234, 1);

    actor_build(arena, model, &rng, env->obs_dim, (u32[]){32, 32}, 2,
                            env->act_dim, 32);
    model->loss_plan = schedule_from(arena, model, model->cost);

    TrainCfg cfg = {
        .epochs = 5,
        .rollout_size = 16,
        .max_steps = env->max_steps,
        .learning_rate = 0.005f,
        .gamma = 0.99f,
        .seed = 1234,
        .optim = OPT_ADAM,
        .max_grad_norm = 0.5f,
        .batch_size = 32,
    };
    train_reinforce(env, model, &cfg, arena);

    checksum[run] = param_checksum(model);
    arena_dispose(arena);
  }

  printf("(%.6f vs %.6f) ", checksum[0], checksum[1]);
  ASSERT_TRUE(checksum[0] == checksum[1], "weights identical across runs");
  printf("OK\n");
}

int main(void) {
  printf("\n=== RL / optim / builder tests ===\n\n");

  test_gae_handcomputed();
  test_adam_step();
  test_grad_clip();
  test_actor_critic_build();
  test_actor_critic_value_learns();
  test_batch_equivalence();
  test_cartpole_learning_smoke();
  test_determinism();

  printf("\n---\nTests: %u passed, %u failed\n\n", tests_passed, tests_failed);
  return tests_failed > 0 ? 1 : 0;
}
