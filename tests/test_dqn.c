// test_dqn.c — Tests for the DQN module: the replay ring buffer, the
// Q-network builder and its batched masked-MSE graph, the epsilon schedule,
// target-network synchronization, and deterministic end-to-end CartPole
// training that must clearly beat the random-policy baseline.
#include "memory.h"
#include "base.h"
#include "dqn.h"
#include "envs/cartpole.h"
#include "graph.h"
#include "layers.h"
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

// The benchmark harness evaluates every policy on the same PRNG-derived seed
// list (base = training seed + 5000, stream 11); reuse it so the test's return
// is comparable with the published tables.
static void make_eval_seeds(u64 *seeds, u32 n, u64 base) {
  Rng rng;
  rng_seed(&rng, base, 11);
  for (u32 i = 0; i < n; i++) {
    const u64 hi = rng_u32(&rng);
    const u64 lo = rng_u32(&rng);
    seeds[i] = (hi << 32) ^ lo;
  }
}

static b32 params_equal(const Net *a, const Net *b) {
  if (a->n_params != b->n_params) {
    return false;
  }
  for (u32 i = 0; i < a->n_params; i++) {
    const Mat *x = a->params[i]->val;
    const Mat *y = b->params[i]->val;
    if (x->rows != y->rows || x->cols != y->cols) {
      return false;
    }
    if (memcmp(x->data, y->data,
               sizeof(real) * (u64)x->rows * (u64)x->cols) != 0) {
      return false;
    }
  }
  return true;
}

// ── Replay ring ──────────────────────────────────────────────────────

static void test_replay_ring(void) {
  printf("  Replay ring overwrite order... ");

  Arena *arena = arena_new(MiB(1));
  ReplayBuf *buf = replay_new(arena, 4, 2);
  ASSERT_TRUE(buf->size == 0u && buf->head == 0u, "buffer starts empty");

  for (u32 i = 0; i < 6; i++) {
    const real o[2] = {(real)i, (real)i * 10.0f};
    const real n[2] = {(real)i + 100.0f, (real)i * 10.0f + 1.0f};
    replay_add_ex(buf, o, i % 2u, (real)i, n, (i % 2u) == 1u, i == 4u);
  }

  ASSERT_TRUE(buf->size == 4u, "size caps at the capacity");
  ASSERT_TRUE(buf->head == 2u, "head wraps: six writes into four slots");
  // Write i lands in slot i % 4, so the oldest surviving write is i = 2 (slot
  // 2) and the newest is i = 5 (slot 1).
  ASSERT_NEAR(buf->rewards[2], 2.0f, 1e-6f, "slot 2 holds transition 2");
  ASSERT_NEAR(buf->obs[2 * 2 + 0], 2.0f, 1e-6f, "observation copied");
  ASSERT_NEAR(buf->next_obs[2 * 2 + 1], 2.0f * 10.0f + 1.0f, 1e-6f,
              "next observation copied");
  ASSERT_TRUE(buf->actions[1] == 1u, "action stored (i = 5 -> slot 1)");
  ASSERT_TRUE(buf->dones[1] == 1u && buf->dones[0] == 0u,
              "terminal flags stored (i = 5 -> slot 1, i = 4 -> slot 0)");
  ASSERT_TRUE(buf->truncated[1] == 0u && buf->truncated[0] == 1u,
              "truncated flags stored separately (i = 4 -> slot 0)");

  // The original entry point still marks a plain terminal transition.
  const real o[2] = {9.0f, 9.0f};
  const real n[2] = {10.0f, 10.0f};
  replay_add(buf, o, 0u, 1.0f, n, true);
  ASSERT_TRUE(buf->dones[2] == 1u && buf->truncated[2] == 0u,
              "replay_add marks terminal and leaves truncated clear");

  arena_dispose(arena);
  printf("OK\n");
}

// ── Q-network builder and batched graph ──────────────────────────────

static void test_q_net_build(void) {
  printf("  Q-network builder and rebuild... ");

  Arena *arena = arena_new(MiB(1));
  Net *q = net_new(arena);
  Rng rng;
  rng_seed(&rng, 7, 1);
  const u32 hidden[] = {16};
  q_net_build(arena, q, &rng, 4, hidden, 1, 3, 8);

  ASSERT_TRUE(q->input->val->rows == 4u && q->input->val->cols == 1u,
              "rollout input is 4 x 1");
  ASSERT_TRUE(q->output->val->rows == 3u && q->output->val->cols == 1u,
              "one Q value per action (3 x 1)");
  ASSERT_TRUE(q->batch_input->val->rows == 4u &&
                  q->batch_input->val->cols == 8u,
              "batch input is 4 x 8");
  ASSERT_TRUE(q->batch_logits->val->rows == 3u &&
                  q->batch_logits->val->cols == 8u,
              "batched Q values are 3 x 8");
  ASSERT_TRUE(q->target != NULL && q->target->val->rows == 3u &&
                  q->target->val->cols == 8u,
              "target tensor is 3 x 8");
  ASSERT_TRUE(q->cost != NULL && q->cost->val->rows == 1u &&
                  q->cost->val->cols == 1u,
              "cost is a scalar");
  ASSERT_TRUE(q->fwd.len > 0u && q->train.len > 0u && q->loss_plan.len > 0u,
              "rollout, training and loss schedules built");

  Tensor *w0 = q->params[0];
  ASSERT_TRUE(net_set_batch_size(arena, q, 16), "rebuild for batch 16");
  ASSERT_TRUE(q->batch_input->val->cols == 16u && q->target->val->cols == 16u,
              "batched views rebuilt at 16");
  ASSERT_TRUE(q->params[0] == w0, "parameters shared across a rebuild");

  mat_fill(q->batch_input->val, 1.0f);
  mat_fill(q->target->val, 0.0f);
  schedule_run(&q->loss_plan);
  ASSERT_TRUE(q->cost->val->data[0] > 0.0f,
              "MSE is positive when the network predicts non-zero values");

  arena_dispose(arena);
  printf("OK\n");
}

// ── Epsilon schedule ─────────────────────────────────────────────────

static void test_dqn_epsilon(void) {
  printf("  Epsilon schedule... ");

  TrainCfg cfg = {0};
  cfg.eps_start = 1.0f;
  cfg.eps_end = 0.1f;
  cfg.eps_decay_steps = 100u;

  ASSERT_NEAR(dqn_epsilon_at(&cfg, 0u), 1.0f, 1e-6f, "eps at step 0");
  ASSERT_NEAR(dqn_epsilon_at(&cfg, 50u), 0.55f, 1e-5f, "eps halfway");
  ASSERT_NEAR(dqn_epsilon_at(&cfg, 100u), 0.1f, 1e-6f, "eps at the end");
  ASSERT_NEAR(dqn_epsilon_at(&cfg, 250u), 0.1f, 1e-6f, "eps clamps");

  // Defaults: eps_start <= 0 => 1, eps_end <= 0 => 0.05, and the decay spans
  // half of epochs * rollout_size steps.
  TrainCfg dflt = {0};
  dflt.epochs = 100u;
  dflt.rollout_size = 32u;
  ASSERT_NEAR(dqn_epsilon_at(&dflt, 0u), 1.0f, 1e-6f, "default start");
  ASSERT_NEAR(dqn_epsilon_at(&dflt, 800u), 0.525f, 1e-5f, "default halfway");
  ASSERT_NEAR(dqn_epsilon_at(&dflt, 1600u), 0.05f, 1e-6f,
              "default end at half the run");

  printf("OK\n");
}

// ── Target sync and training effect ──────────────────────────────────

static void test_td_target(void) {
  printf("  One-step TD target and truncation... ");

  ASSERT_NEAR(dqn_td_target(1.0f, 2.0f, 0.5f, false), 2.0f, 1e-6f,
              "continuing transition bootstraps");
  ASSERT_NEAR(dqn_td_target(1.0f, 2.0f, 0.5f, true), 1.0f, 1e-6f,
              "terminal transition drops the bootstrap");
  ASSERT_NEAR(dqn_td_target(-1.0f, 100.0f, 0.99f, false), 98.0f, 1e-5f,
              "truncation keeps the same bootstrap as a continuing step");

  printf("OK\n");
}

static void test_dqn_sync_and_training(void) {
  printf("  Target sync and training effect... ");

  Arena *arena = arena_new(GiB(1));
  Env *env = env_cartpole_create(arena);
  const u32 hidden[] = {16};

  // `online` and `ref` share an initialization; `target` starts from a
  // different one and must be overwritten by the initial sync.
  Net *online = net_new(arena);
  Net *target = net_new(arena);
  Net *ref = net_new(arena);
  Rng r1, r2, r3;
  rng_seed(&r1, 11, 1);
  rng_seed(&r2, 99, 1);
  rng_seed(&r3, 11, 1);
  q_net_build(arena, online, &r1, env->obs_dim, hidden, 1, env->act_dim, 4);
  q_net_build(arena, target, &r2, env->obs_dim, hidden, 1, env->act_dim, 4);
  q_net_build(arena, ref, &r3, env->obs_dim, hidden, 1, env->act_dim, 4);
  ASSERT_TRUE(params_equal(online, ref), "same seed -> same initialization");

  dqn_sync_target(online, target);
  ASSERT_TRUE(params_equal(target, ref), "sync copies every parameter");

  TrainCfg cfg = {0};
  cfg.epochs = 5u;
  cfg.rollout_size = 8u;
  cfg.learning_rate = 1e-3f;
  cfg.gamma = 0.99f;
  cfg.seed = 3u;
  cfg.optim = OPT_ADAM;
  cfg.max_grad_norm = 10.0f;
  cfg.batch_size = 4u;
  cfg.replay_capacity = 128u;
  cfg.warmup_steps = 8u;
  cfg.train_every = 1u;
  cfg.target_sync = 1000000u; // never reached: the target must stay put
  cfg.eps_start = 1.0f;
  cfg.eps_end = 1.0f;
  cfg.eps_decay_steps = 1u;

  train_dqn(env, online, target, &cfg, arena);

  ASSERT_TRUE(!params_equal(online, ref),
              "gradient steps changed the online network");
  ASSERT_TRUE(params_equal(target, ref),
              "target network untouched before its sync interval");

  arena_dispose(arena);
  printf("OK\n");
}

// ── Determinism ──────────────────────────────────────────────────────

static void test_dqn_determinism(void) {
  printf("  Determinism across runs... ");

  Arena *arena = arena_new(GiB(1));
  const u32 hidden[] = {16};

  Net *q1 = net_new(arena);
  Net *t1 = net_new(arena);
  Net *q2 = net_new(arena);
  Net *t2 = net_new(arena);
  Env *env1 = env_cartpole_create(arena);
  Env *env2 = env_cartpole_create(arena);

  Rng r1, r2;
  rng_seed(&r1, 5, 1);
  rng_seed(&r2, 5, 1);
  q_net_build(arena, q1, &r1, env1->obs_dim, hidden, 1, env1->act_dim, 8);
  q_net_build(arena, t1, &r1, env1->obs_dim, hidden, 1, env1->act_dim, 8);
  q_net_build(arena, q2, &r2, env2->obs_dim, hidden, 1, env2->act_dim, 8);
  q_net_build(arena, t2, &r2, env2->obs_dim, hidden, 1, env2->act_dim, 8);

  TrainCfg cfg = {0};
  cfg.epochs = 30u;
  cfg.rollout_size = 16u;
  cfg.learning_rate = 1e-3f;
  cfg.gamma = 0.99f;
  cfg.seed = 21u;
  cfg.optim = OPT_ADAM;
  cfg.max_grad_norm = 10.0f;
  cfg.batch_size = 8u;
  cfg.replay_capacity = 256u;
  cfg.warmup_steps = 16u;
  cfg.train_every = 1u;
  cfg.target_sync = 50u;
  cfg.eps_start = 1.0f;
  cfg.eps_end = 0.05f;
  cfg.eps_decay_steps = 200u;

  train_dqn(env1, q1, t1, &cfg, arena);
  train_dqn(env2, q2, t2, &cfg, arena);

  ASSERT_TRUE(params_equal(q1, q2), "same seed -> identical online weights");
  ASSERT_TRUE(params_equal(t1, t2), "same seed -> identical target weights");

  arena_dispose(arena);
  printf("OK\n");
}

// ── End-to-end learning on CartPole ──────────────────────────────────

static void test_dqn_learning(void) {
  printf("  CartPole learning... ");

  Arena *arena = arena_new(GiB(1));
  Env *env = env_cartpole_create(arena);
  Net *online = net_new(arena);
  Net *target = net_new(arena);

  Rng rng;
  rng_seed(&rng, 7, 1);
  const u32 hidden[] = {64, 64};
  q_net_build(arena, online, &rng, env->obs_dim, hidden, 2, env->act_dim, 64);
  q_net_build(arena, target, &rng, env->obs_dim, hidden, 2, env->act_dim, 64);

  TrainCfg cfg = {0};
  cfg.epochs = 500u;
  cfg.rollout_size = 32u;
  cfg.max_steps = env->max_steps;
  cfg.learning_rate = 1e-3f;
  cfg.gamma = 0.99f;
  cfg.seed = 7u;
  cfg.optim = OPT_ADAM;
  cfg.max_grad_norm = 10.0f;
  cfg.batch_size = 64u;
  cfg.replay_capacity = 20000u;
  cfg.warmup_steps = 500u;
  cfg.train_every = 1u;
  cfg.target_sync = 250u;
  cfg.eps_start = 1.0f;
  cfg.eps_end = 0.05f;
  cfg.eps_decay_steps = 8000u;

  train_dqn(env, online, target, &cfg, arena);

  u64 eval_seeds[30];
  make_eval_seeds(eval_seeds, 30, 7 + 5000);
  const EvalResult res = evaluate_seeded(env, online, eval_seeds, 30, true);
  printf("(greedy %.1f) ", (double)res.avg_return);
  // The trained trajectory depends on the build's floating-point details (this
  // run reaches ~140 in the UBSan/-O0 test build and ~200 at -O2, against a
  // random baseline of ~22), so the bar is deliberately loose: the check is
  // that it learned, not how well.
  ASSERT_TRUE(res.avg_return > 100.0f,
              "greedy evaluation clears the random baseline by a wide margin");

  arena_dispose(arena);
  printf("OK\n");
}

int main(void) {
  printf("=== test_dqn ===\n");

  test_replay_ring();
  test_q_net_build();
  test_dqn_epsilon();
  test_td_target();
  test_dqn_sync_and_training();
  test_dqn_determinism();
  test_dqn_learning();

  printf("\nTests: %u passed, %u failed\n", tests_passed, tests_failed);
  return tests_failed == 0u ? 0 : 1;
}
