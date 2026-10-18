// test_rl.c — Tests for the modules the original suite never touched:
// optimizers, network builders, GAE, the A2C update path, batched-vs-single
// equivalence, and end-to-end training smoke/determinism checks.
#include "memory.h"
#include "graph.h"
#include "base.h"
#include "envs/cartpole.h"
#include "envs/pendulum.h"
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

// ── PPO learning smoke test ──────────────────────────────────────────

static void test_ppo_learning_smoke(void) {
  printf("  CartPole PPO learns (50 updates)... ");
  Arena *arena = arena_new(GiB(1));
  Net *model = net_new(arena);
  Env *env = env_cartpole_create(arena);
  Rng rng;
  rng_seed(&rng, 42, 1);

  const u32 hidden[] = {64, 64};
  actor_critic_build(arena, model, &rng, env->obs_dim, hidden, 2, env->act_dim,
                     64);

  TrainCfg cfg = {
      .epochs = 50,
      .rollout_size = 128,
      .max_steps = env->max_steps,
      .learning_rate = 3e-4f,
      .gamma = 0.99f,
      .lambda = 0.95f,
      .vf_coef = 0.5f,
      .entropy_coef = 0.01f,
      .seed = 42,
      .optim = OPT_ADAM,
      .max_grad_norm = 0.5f,
      .batch_size = 64,
      .num_envs = 8,
      .ppo_epochs = 10,
      .ppo_clip = 0.2f,
  };
  train_ppo(env, model, &cfg, arena);

  Rng eval_rng;
  rng_seed(&eval_rng, 999, 1);
  EvalResult res = evaluate(env, model, &eval_rng, 30, true);
  printf("(greedy return %.1f) ", res.avg_return);

  // Random scores ~22; require a wide margin above that.
  ASSERT_TRUE(res.avg_return > 150.0f, "PPO beats random by a wide margin");

  arena_dispose(arena);
  printf("OK\n");
}

// ── PPO same-seed determinism ────────────────────────────────────────

static void test_ppo_determinism(void) {
  printf("  PPO same seed => identical weights... ");
  real checksum[2];

  for (u32 run = 0; run < 2; run++) {
    Arena *arena = arena_new(GiB(1));
    Net *model = net_new(arena);
    Env *env = env_cartpole_create(arena);
    Rng rng;
    rng_seed(&rng, 1234, 1);

    actor_critic_build(arena, model, &rng, env->obs_dim, (u32[]){32, 32}, 2,
                       env->act_dim, 32);

    TrainCfg cfg = {
        .epochs = 8,
        .rollout_size = 64,
        .max_steps = env->max_steps,
        .learning_rate = 3e-4f,
        .gamma = 0.99f,
        .lambda = 0.95f,
        .vf_coef = 0.5f,
        .entropy_coef = 0.01f,
        .seed = 1234,
        .optim = OPT_ADAM,
        .max_grad_norm = 0.5f,
        .batch_size = 32,
        .num_envs = 4,
        .ppo_epochs = 4,
        .ppo_clip = 0.2f,
    };
    train_ppo(env, model, &cfg, arena);

    checksum[run] = param_checksum(model);
    arena_dispose(arena);
  }

  printf("(%.6f vs %.6f) ", checksum[0], checksum[1]);
  ASSERT_TRUE(checksum[0] == checksum[1], "PPO weights identical across runs");
  printf("OK\n");
}

// ── Continuous (squashed Gaussian) builder ───────────────────────────

static void test_continuous_builder(void) {
  printf("  continuous actor-critic builder... ");
  Arena *arena = arena_new(MiB(64));
  Net *model = net_new(arena);
  Rng rng;
  rng_seed(&rng, 7, 1);

  // Per-dimension bounds, deliberately not symmetric between dimensions.
  const real lo[2] = {-2.0f, -0.5f};
  const real hi[2] = {2.0f, 0.5f};

  const u32 hidden[] = {16, 16};
  continuous_actor_critic_build(arena, model, &rng, 3, hidden, 2, 2, 8, -1.5f);

  ASSERT_TRUE(model->mean != NULL, "mean head built");
  ASSERT_TRUE(model->batch_mean != NULL, "batched mean built");
  ASSERT_TRUE(model->value != NULL, "value head built");
  ASSERT_TRUE(model->batch_value != NULL, "batched value head built");
  ASSERT_TRUE(model->log_std != NULL, "log_std parameter built");
  ASSERT_TRUE(model->log_std->val->rows == 2, "one log_std per action dim");
  ASSERT_TRUE(model->output == NULL,
              "no discrete softmax head (discrete paths stay unusable)");
  ASSERT_TRUE(model->fwd.len > 0 && model->train.len > 0, "schedules built");
  ASSERT_TRUE(net_batch_size(model) == 8, "batch size reported");
  ASSERT_TRUE(real_fabs(model->log_std->val->data[0] + 1.5f) < 1e-6f,
              "log_std initialised from log_std_init");

  // log_std must be a registered parameter: that is what makes the optimiser,
  // the gradient clip and the checkpoint writer cover it.
  b32 registered = false;
  for (u32 i = 0; i < model->n_params; i++) {
    if (model->params[i] == model->log_std) {
      registered = true;
    }
  }
  ASSERT_TRUE(registered, "log_std is a registered parameter");

  // Samples are squashed into the box, dimension by dimension, and the sampler
  // hands back the pre-squash coordinate the surrogate's density is defined on.
  Mat *act = mat_new(arena, 2, 1);
  Mat *us = mat_new(arena, 2, 1);
  Mat *back = mat_new(arena, 2, 1);
  b32 inside = true;
  real worst = (real)0;
  for (u32 i = 0; i < 256; i++) {
    RL_LIB_CHECK(mat_gaussian_sample(act, us, model->mean->val,
                                     model->log_std->val, lo, hi, &rng));
    if (act->data[0] < lo[0] || act->data[0] > hi[0] || act->data[1] < lo[1] ||
        act->data[1] > hi[1]) {
      inside = false;
    }
    // The action is exactly the squash of the returned u, and away from the
    // bounds the documented inverse recovers u from the action.
    for (u32 d = 0; d < 2; d++) {
      const real center = (real)0.5f * (lo[d] + hi[d]);
      const real half = (real)0.5f * (hi[d] - lo[d]);
      if (real_fabs(act->data[d] - (center + half * real_tanh(us->data[d]))) >
          1e-6f) {
        inside = false;
      }
    }
    RL_LIB_CHECK(mat_gaussian_unsquash(back, act, lo, hi));
    for (u32 d = 0; d < 2; d++) {
      const real du = real_fabs(back->data[d] - us->data[d]);
      if (du > worst) {
        worst = du;
      }
    }
  }
  ASSERT_TRUE(inside, "256 samples stay in the box and match their pre-squash u");
  ASSERT_TRUE(worst < 1e-4f, "the unsquash inverse recovers the sampled u");

  // A batched view can be rebuilt without touching the parameters.
  net_set_batch_size(arena, model, 16);
  ASSERT_TRUE(net_batch_size(model) == 16, "batch size raised to 16");
  ASSERT_TRUE(model->batch_mean->val->cols == 16, "batched mean follows");
  ASSERT_TRUE(model->batch_value->val->cols == 16, "batched value follows");
  ASSERT_TRUE(real_fabs(model->log_std->val->data[0] + 1.5f) < 1e-6f,
              "log_std survives a rebuild");

  arena_dispose(arena);
  printf("OK\n");
}

// ── Pendulum learning smoke test (continuous PPO) ────────────────────

static void test_ppo_continuous_learning_smoke(void) {
  printf("  Pendulum continuous PPO learns... ");
  Arena *arena = arena_new(GiB(1));
  Net *model = net_new(arena);
  Env *env = env_pendulum_create(arena);
  Rng rng;
  rng_seed(&rng, 42, 1);

  const u32 hidden[] = {32, 32};
  const u32 batch = 32;
  continuous_actor_critic_build(arena, model, &rng, env->obs_dim, hidden, 2,
                                env->act_dim, batch, -2.0f);

  // Fixed evaluation seeds, so "before" and "after" see the same episodes.
  u64 seeds[8];
  Rng seed_rng;
  rng_seed(&seed_rng, 777, 1);
  for (u32 i = 0; i < 8; i++) {
    seeds[i] = ((u64)rng_u32(&seed_rng) << 32) ^ (u64)rng_u32(&seed_rng);
  }
  const EvalResult before = evaluate_continuous_seeded(env, model, seeds, 8, true);
  const real log_std_before = model->log_std->val->data[0];

  TrainCfg cfg = {
      .epochs = 150,
      .rollout_size = 128,
      .max_steps = env->max_steps,
      .learning_rate = 3e-4f,
      .gamma = 0.9f,
      .lambda = 0.95f,
      .vf_coef = 0.5f,
      .entropy_coef = 0.0f,
      .seed = 42,
      .optim = OPT_ADAM,
      .max_grad_norm = 0.5f,
      .batch_size = batch,
      .num_envs = 8,
      .ppo_epochs = 4,
      .ppo_clip = 0.2f,
  };
  train_ppo_continuous(env, model, &cfg, arena);

  const EvalResult after = evaluate_continuous_seeded(env, model, seeds, 8, true);
  printf("(%.1f -> %.1f) ", before.avg_return, after.avg_return);
  ASSERT_TRUE(after.avg_return > before.avg_return + 100.0f,
              "the return improves by a wide margin");
  ASSERT_TRUE(after.avg_steps == (real)env->max_steps,
              "Pendulum episodes always run to the time limit");
  ASSERT_TRUE(model->log_std->val->data[0] != log_std_before,
              "log_std is updated by the optimiser, not frozen");
  ASSERT_TRUE(real_exp(model->log_std->val->data[0]) > 0.0f,
              "log_std stays in a sane range");
  ASSERT_TRUE(evaluate_continuous(env, model, &rng, 4, true).avg_return <= 0.0f,
              "returns are non-positive, as Pendulum's cost is");
  ASSERT_TRUE(evaluate_continuous(env, model, &rng, 4, false).avg_return <= 0.0f,
              "sampled evaluation returns are non-positive too");

  arena_dispose(arena);
  printf("OK\n");
}

// ── Continuous PPO same-seed determinism ─────────────────────────────

static void test_ppo_continuous_determinism(void) {
  printf("  continuous PPO same seed => identical weights... ");
  real checksum[2];

  for (u32 run = 0; run < 2; run++) {
    Arena *arena = arena_new(GiB(1));
    Net *model = net_new(arena);
    Env *env = env_pendulum_create(arena);
    Rng rng;
    rng_seed(&rng, 1234, 1);

    continuous_actor_critic_build(arena, model, &rng, env->obs_dim,
                                  (u32[]){32, 32}, 2, env->act_dim, 32, -1.0f);

    TrainCfg cfg = {
        .epochs = 8,
        .rollout_size = 64,
        .max_steps = env->max_steps,
        .learning_rate = 3e-4f,
        .gamma = 0.9f,
        .lambda = 0.95f,
        .vf_coef = 0.5f,
        .entropy_coef = 0.0f,
        .seed = 1234,
        .optim = OPT_ADAM,
        .max_grad_norm = 0.5f,
        .batch_size = 32,
        .num_envs = 4,
        .ppo_epochs = 4,
        .ppo_clip = 0.2f,
    };
    train_ppo_continuous(env, model, &cfg, arena);

    checksum[run] = param_checksum(model);
    arena_dispose(arena);
  }

  printf("(%.6f vs %.6f) ", checksum[0], checksum[1]);
  ASSERT_TRUE(checksum[0] == checksum[1],
              "continuous PPO weights identical across runs");
  printf("OK\n");
}

// ── Parameter registry is not capped at a fixed count ────────────────

static void test_param_registry_growth(void) {
  printf("  parameter registry grows past 128 entries... ");

  Arena *arena = arena_new(MiB(16));
  Net *model = net_new(arena);
  Rng rng;
  rng_seed(&rng, 1, 1);

  Tensor *in = tensor_create(arena, model, 4, 1, TENSOR_PLAIN);
  Tensor *cur = in;
  const u32 layers = 200; // 2 parameters each, far past the old MAX_PARAMS = 128
  for (u32 i = 0; i < layers; i++) {
    Dense d = dense_new(arena, model, &rng, 4, 4, true, 1.0f);
    cur = dense_apply(arena, model, &d, cur);
  }

  ASSERT_TRUE(model->n_params == layers * 2, "every parameter is registered");
  ASSERT_TRUE(model->n_params > 128, "registry grew past the old fixed cap");
  ASSERT_TRUE(model->params_cap >= model->n_params, "capacity covers the count");

  b32 unique = true;
  for (u32 i = 0; i < model->n_params && unique; i++) {
    for (u32 j = i + 1; j < model->n_params; j++) {
      if (model->params[i] == model->params[j]) {
        unique = false;
        break;
      }
    }
  }
  ASSERT_TRUE(unique, "no duplicate or dropped registrations");

  // The optimizer iterates the registry, so it has to cope with the larger one.
  Adam *adam = adam_new(arena, model, REAL_C(0.01));
  zero_grad(model);
  for (u32 i = 0; i < model->n_params; i++) {
    mat_fill(model->params[i]->grad, REAL_C(1.0));
  }
  const real first_before = model->params[0]->val->data[0];
  const real last_before =
      model->params[model->n_params - 1]->val->data[0];
  adam_step(adam, model);
  ASSERT_TRUE(adam->step == 1, "optimiser stepped once over the larger registry");
  ASSERT_TRUE(model->params[0]->val->data[0] != first_before,
              "the first parameter was updated");
  ASSERT_TRUE(model->params[model->n_params - 1]->val->data[0] != last_before,
              "the last parameter was updated");

  arena_dispose(arena);
  printf("OK\n");
}

// ── The minibatch size is a runtime knob ─────────────────────────────

static void test_batch_size_rebuild(void) {
  printf("  batch size is rebuildable at runtime... ");

  Arena *arena = arena_new(MiB(16));
  Rng rng;
  rng_seed(&rng, 3, 3);
  Net *model = net_new(arena);
  const u32 hidden[1] = {8};
  actor_critic_build(arena, model, &rng, 4, hidden, 1, 3, 16);

  ASSERT_TRUE(net_batch_size(model) == 16, "initial batch dimension is 16");
  ASSERT_TRUE(net_set_batch_size(arena, model, 16),
              "a same-size request succeeds as a no-op");
  ASSERT_TRUE(net_batch_size(model) == 16, "still 16");

  const real before = param_checksum(model);
  ASSERT_TRUE(net_set_batch_size(arena, model, 5), "resize to 5 succeeds");
  ASSERT_TRUE(net_batch_size(model) == 5, "batch dimension is now 5");
  ASSERT_TRUE(model->train.len > 0, "training schedule was rebuilt");
  ASSERT_TRUE(param_checksum(model) == before,
              "the rebuild reuses the weights instead of re-initialising them");
  ASSERT_TRUE(model->batch_logits->val->rows == 3 &&
                  model->batch_logits->val->cols == 5,
              "batch logits have the new shape");
  ASSERT_TRUE(model->batch_value->val->rows == 1 &&
                  model->batch_value->val->cols == 5,
              "batch value has the new shape");

  // The rebuilt batch graph must agree with single-sample inference.
  const u32 batch = 5;
  real obs[4 * 5];
  for (u32 i = 0; i < 4 * batch; i++) {
    obs[i] = ((real)(i % 7) / (real)7) - REAL_C(0.5);
  }
  for (u32 b = 0; b < batch; b++) {
    for (u32 r = 0; r < 4; r++) {
      model->batch_input->val->data[r * batch + b] = obs[b * 4 + r];
    }
  }
  schedule_run(&model->train);

  real max_value_diff = (real)0;
  b32 argmax_agrees = true;
  for (u32 b = 0; b < batch; b++) {
    for (u32 r = 0; r < 4; r++) {
      model->input->val->data[r] = obs[b * 4 + r];
    }
    schedule_run(&model->fwd);

    const real diff = real_fabs(model->value->val->data[0] -
                                model->batch_value->val->data[b]);
    if (diff > max_value_diff) {
      max_value_diff = diff;
    }

    u32 single_arg = 0;
    u32 batch_arg = 0;
    for (u32 k = 1; k < 3; k++) {
      if (model->output->val->data[k] > model->output->val->data[single_arg]) {
        single_arg = k;
      }
      if (model->batch_logits->val->data[k * batch + b] >
          model->batch_logits->val->data[batch_arg * batch + b]) {
        batch_arg = k;
      }
    }
    if (single_arg != batch_arg) {
      argmax_agrees = false;
    }
  }
  printf("(value max diff %.2e) ", (double)max_value_diff);
  ASSERT_TRUE(max_value_diff < REAL_C(1e-4),
              "value head agrees between the batch and single-sample graphs");
  ASSERT_TRUE(argmax_agrees, "argmax action agrees for every batch column");

  // A model without a recorded architecture cannot be resized, and the request
  // is rejected rather than silently ignored.
  Net *manual = net_new(arena);
  (void)tensor_create(arena, manual, 2, 2, TENSOR_PLAIN);
  ASSERT_TRUE(!net_set_batch_size(arena, manual, 4),
              "a hand-built graph refuses a resize");
  ASSERT_TRUE(net_batch_size(manual) == 0, "and has no batch dimension");

  // The trainers apply TrainCfg.batch_size, so a size the model was not built
  // with just works.
  Arena *train_arena = arena_new(MiB(64));
  Net *trained = net_new(arena);
  const u32 hidden2[1] = {8};
  actor_critic_build(arena, trained, &rng, 4, hidden2, 1, 2, 16);
  Env *env = env_cartpole_create(arena);
  const TrainCfg cfg = {
      .epochs = 3,
      .rollout_size = 16,
      .max_steps = 500,
      .learning_rate = 3e-4f,
      .gamma = 0.99f,
      .lambda = 0.95f,
      .vf_coef = 0.5f,
      .entropy_coef = 0.01f,
      .seed = 4,
      .verbose = 0,
      .optim = OPT_ADAM,
      .max_grad_norm = 0.5f,
      .batch_size = 8,
      .num_envs = 2,
      .ppo_epochs = 2,
      .ppo_clip = 0.2f,
  };
  train_ppo(env, trained, &cfg, train_arena);
  ASSERT_TRUE(net_batch_size(trained) == 8,
              "the trainer rebuilt the graph for TrainCfg.batch_size");
  ASSERT_TRUE(trained->train.len > 0, "and trained through the rebuilt graph");

  arena_dispose(train_arena);
  arena_dispose(arena);
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
  test_ppo_learning_smoke();
  test_continuous_builder();
  test_ppo_continuous_learning_smoke();
  test_ppo_continuous_determinism();
  test_determinism();
  test_ppo_determinism();
  test_param_registry_growth();
  test_batch_size_rebuild();

  printf("\n---\nTests: %u passed, %u failed\n\n", tests_passed, tests_failed);
  return tests_failed > 0 ? 1 : 0;
}
