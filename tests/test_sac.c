// test_sac.c — Tests for the SAC module: the continuous replay ring, the
// gather layout, the soft target update and TD target, the actor/critic
// builders (including the batched rebuild and the split first layer), the actor
// cost composition against the fused kernels (value and gradient), determinism,
// and deterministic end-to-end Pendulum training that must clearly beat the
// random-policy baseline.
#include "base.h"
#include "checkpoint.h"
#include "envs/pendulum.h"
#include "graph.h"
#include "linalg.h"
#include "layers.h"
#include "memory.h"
#include "optim.h"
#include "random.h"
#include "rl.h"
#include "sac.h"

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
  printf("  Continuous replay ring... ");

  Arena *arena = arena_new(MiB(1));
  SacReplay *buf = sac_replay_new(arena, 3, 2, 2);
  ASSERT_TRUE(buf->size == 0u && buf->head == 0u, "buffer starts empty");

  for (u32 i = 0; i < 5; i++) {
    const real o[2] = {(real)i, (real)i * 10.0f};
    const real a[2] = {(real)i + 0.5f, -(real)i};
    const real n[2] = {(real)i + 100.0f, (real)i * 10.0f + 1.0f};
    sac_replay_add(buf, o, a, (real)i * 2.0f, n, (i % 2u) == 1u, i == 4u);
  }

  ASSERT_TRUE(buf->size == 3u, "size caps at the capacity");
  ASSERT_TRUE(buf->head == 2u, "head wraps: five writes into three slots");
  // Write i lands in slot i % 3, so the oldest surviving write is i = 2 (slot
  // 2), then 3 (slot 0) and 4 (slot 1).
  ASSERT_NEAR(buf->rewards[2], 4.0f, 1e-6f, "slot 2 holds transition 2");
  ASSERT_NEAR(buf->obs[2 * 2 + 1], 20.0f, 1e-6f, "observation copied");
  ASSERT_NEAR(buf->actions[2 * 2 + 1], -2.0f, 1e-6f, "action copied");
  ASSERT_NEAR(buf->next_obs[2 * 2 + 0], 102.0f, 1e-6f, "next observation");
  ASSERT_TRUE(buf->terminated[0] == 1u && buf->terminated[2] == 0u,
              "terminated flags stored (i = 3 -> slot 0)");
  ASSERT_TRUE(buf->truncated[1] == 1u && buf->truncated[0] == 0u,
              "truncated flags stored separately (i = 4 -> slot 1)");

  arena_dispose(arena);
  printf("OK\n");
}

static void test_replay_gather(void) {
  printf("  Replay gather layout... ");

  Arena *arena = arena_new(MiB(1));
  SacReplay *buf = sac_replay_new(arena, 4, 2, 2);
  for (u32 i = 0; i < 4; i++) {
    const real o[2] = {(real)i, (real)i + 10.0f};
    const real a[2] = {(real)i + 0.25f, (real)i + 0.75f};
    const real n[2] = {(real)i + 20.0f, (real)i + 30.0f};
    // Transition 1 ended for real, transition 2 hit a time limit.
    sac_replay_add(buf, o, a, (real)i, n, i == 1u, i == 2u);
  }

  const u32 idx[3] = {3u, 0u, 2u};
  Mat *obs = mat_new(arena, 2, 3);
  Mat *act = mat_new(arena, 2, 3);
  Mat *next = mat_new(arena, 2, 3);
  Mat *rew = mat_new(arena, 1, 3);
  Mat *term = mat_new(arena, 1, 3);
  sac_replay_gather(buf, idx, 3, obs, act, next, rew, term);

  // Column b is sampled transition idx[b]; row r is dimension r.
  ASSERT_NEAR(obs->data[0 * 3 + 0], 3.0f, 1e-6f, "obs column 0, row 0");
  ASSERT_NEAR(obs->data[1 * 3 + 1], 10.0f, 1e-6f, "obs column 1, row 1");
  ASSERT_NEAR(act->data[1 * 3 + 2], 2.75f, 1e-6f, "action copied");
  ASSERT_NEAR(next->data[0 * 3 + 2], 22.0f, 1e-6f, "next observation copied");
  ASSERT_NEAR(rew->data[0], 3.0f, 1e-6f, "reward column 0");
  ASSERT_NEAR(rew->data[1], 0.0f, 1e-6f, "reward column 1");
  ASSERT_NEAR(term->data[0], 0.0f, 1e-6f, "time limit does not set terminal");
  ASSERT_NEAR(term->data[1], 0.0f, 1e-6f, "continuing transition not terminal");
  ASSERT_TRUE(term->data[2] == 0.0f, "truncated transition not terminal");

  // A real terminal is reported as 1 for exactly that column.
  const u32 idx2[3] = {1u, 2u, 3u};
  sac_replay_gather(buf, idx2, 3, obs, act, next, rew, term);
  ASSERT_NEAR(term->data[0], 1.0f, 1e-6f, "terminal flag reported");
  ASSERT_NEAR(term->data[1], 0.0f, 1e-6f, "truncation stays non-terminal");
  ASSERT_NEAR(term->data[2], 0.0f, 1e-6f, "later transition not terminal");

  arena_dispose(arena);
  printf("OK\n");
}

// ── Targets and temperature plumbing ─────────────────────────────────

static void test_td_target(void) {
  printf("  TD target and truncation... ");

  ASSERT_NEAR(sac_td_target(1.0f, 2.0f, 0.5f, 1.0f, false), 2.5f, 1e-6f,
              "continuing transition bootstraps");
  ASSERT_NEAR(sac_td_target(1.0f, 2.0f, 0.5f, 0.5f, false), 1.75f, 1e-6f,
              "entropy bonus is subtracted inside the bootstrap");
  ASSERT_NEAR(sac_td_target(1.0f, 2.0f, 0.5f, 0.99f, true), 1.0f, 1e-6f,
              "terminal transition drops the bootstrap");
  ASSERT_NEAR(sac_td_target(-1.0f, 100.0f, -3.0f, 0.99f, false),
              -1.0f + 0.99f * 103.0f, 1e-4f,
              "truncation keeps the same bootstrap as a continuing step");

  printf("OK\n");
}

static void test_soft_update(void) {
  printf("  Soft target update... ");

  Arena *arena = arena_new(MiB(1));
  Net *src = net_new(arena);
  Net *dst = net_new(arena);
  Rng r1, r2;
  rng_seed(&r1, 3, 1);
  rng_seed(&r2, 4, 1);
  const u32 hidden[] = {8};
  sac_q_build(arena, src, &r1, 3, 2, hidden, 1, 0);
  sac_q_build(arena, dst, &r2, 3, 2, hidden, 1, 0);

  // Keep a copy of the pre-update destination and check the interpolation
  // parameter by parameter. The snapshot needs its own Rng at the same seed:
  // it was built from the same stream position as `dst`.
  Arena *snapshot = arena_new(MiB(1));
  Net *before = net_new(snapshot);
  Rng r3;
  rng_seed(&r3, 4, 1);
  sac_q_build(snapshot, before, &r3, 3, 2, hidden, 1, 0);
  ASSERT_TRUE(params_equal(dst, before), "snapshot matches the destination");

  sac_soft_update(src, dst, 0.25f);
  for (u32 i = 0; i < dst->n_params; i++) {
    const Mat *s = src->params[i]->val;
    const Mat *b = before->params[i]->val;
    const Mat *d = dst->params[i]->val;
    const u64 n = (u64)s->rows * (u64)s->cols;
    for (u64 k = 0; k < n; k++) {
      const real want = 0.75f * b->data[k] + 0.25f * s->data[k];
      if (real_fabs(d->data[k] - want) > 1e-6f) {
        ASSERT_TRUE(false, "parameter interpolation mismatch");
        k = n;
        i = dst->n_params;
      }
    }
  }
  ASSERT_TRUE(true, "tau = 0.25 interpolates every parameter");

  // tau = 1 copies, and out-of-range tau is clamped rather than extrapolated.
  sac_soft_update(src, dst, 1.0f);
  ASSERT_TRUE(params_equal(dst, src), "tau = 1 copies the source");
  sac_soft_update(src, dst, 5.0f);
  ASSERT_TRUE(params_equal(dst, src), "tau > 1 clamps to a copy");

  arena_dispose(arena);
  arena_dispose(snapshot);
  printf("OK\n");
}

// ── Builders ─────────────────────────────────────────────────────────

static void test_builders(void) {
  printf("  SAC builders and rebuild... ");

  Arena *arena = arena_new(MiB(1));
  Rng rng;
  rng_seed(&rng, 7, 1);
  const u32 hidden[] = {16, 8};
  Net *q = net_new(arena);
  sac_q_build(arena, q, &rng, 4, 3, hidden, 2, 6);

  ASSERT_TRUE(q->batch_input != NULL && q->batch_input->val->rows == 4u &&
                  q->batch_input->val->cols == 6u,
              "critic batch input is obs_dim x batch");
  ASSERT_TRUE(q->batch_action != NULL && q->batch_action->val->rows == 3u &&
                  q->batch_action->val->cols == 6u,
              "critic batch action is act_dim x batch");
  ASSERT_TRUE(q->batch_logits != NULL && q->batch_logits->val->rows == 1u &&
                  q->batch_logits->val->cols == 6u,
              "critic batch output is one value per column");
  ASSERT_TRUE(q->target != NULL && q->cost != NULL && q->loss_plan.len > 0u,
              "critic target and loss plan built");
  ASSERT_TRUE(q->fwd.len == 0u,
              "a critic has no single-sample rollout graph");

  // The recorded layer stack is reachable and precedes the head.
  SacQNet layers;
  ASSERT_TRUE(sac_q_layers(q, &layers), "critic layers readable");
  ASSERT_TRUE(layers.split_is_head == false && layers.num_hidden == 1u,
              "two hidden layers: split first, one plain hidden");
  ASSERT_TRUE(layers.split.out_features == 16u &&
                  layers.hidden[0].out_features == 8u,
              "layer widths recorded");
  ASSERT_TRUE(layers.split.bias != NULL && layers.head.bias != NULL,
              "split layer and head carry biases");

  // A non-SAC model has no recorded stack.
  Net *plain = net_new(arena);
  const u32 one[] = {8};
  q_net_build(arena, plain, &rng, 4, one, 1, 3, 4);
  SacQNet unused;
  ASSERT_TRUE(!sac_q_layers(plain, &unused), "non-SAC model has no layers");

  // Rebuilding keeps every parameter and resizes the data tensors.
  Tensor *w = q->params[0];
  ASSERT_TRUE(net_set_batch_size(arena, q, 10u), "critic rebuild");
  ASSERT_TRUE(q->batch_input->val->cols == 10u &&
                  q->batch_action->val->cols == 10u &&
                  q->target->val->cols == 10u,
              "critic batched views rebuilt at 10");
  ASSERT_TRUE(q->params[0] == w, "parameters shared across a rebuild");
  ASSERT_TRUE(net_arch_info(q, &(NetArchInfo){0}) == true,
              "critic records an architecture");

  // The actor keeps the mean/head and log_std but has no value head.
  Net *actor = net_new(arena);
  sac_actor_build(arena, actor, &rng, 4, hidden, 2, 3, 5, -0.5f);
  ASSERT_TRUE(actor->mean != NULL && actor->mean->val->rows == 3u,
              "actor mean head is act_dim x 1");
  ASSERT_TRUE(actor->log_std != NULL && actor->log_std->val->rows == 3u,
              "actor log_std parameter is act_dim x 1");
  ASSERT_TRUE(actor->value == NULL, "actor has no value head");
  ASSERT_TRUE(actor->batch_mean != NULL &&
                  actor->batch_mean->val->rows == 3u &&
                  actor->batch_mean->val->cols == 5u,
              "actor batched mean is act_dim x batch");
  ASSERT_TRUE(actor->train.len > 0u && actor->fwd.len > 0u,
              "actor rollout and training schedules built");
  ASSERT_TRUE(actor->log_std->val->data[0] == -0.5f,
              "log_std initialised from the argument");

  ASSERT_TRUE(net_set_batch_size(arena, actor, 9u), "actor rebuild");
  ASSERT_TRUE(actor->batch_mean->val->cols == 9u, "actor rebuilt at 9");
  ASSERT_TRUE(actor->mean != NULL && actor->log_std != NULL,
              "actor heads survive a rebuild");

  // A critic with no hidden layer is a linear Q(s, a): the split layer *is* the
  // output layer.
  Net *linear = net_new(arena);
  sac_q_build(arena, linear, &rng, 4, 3, NULL, 0, 7);
  SacQNet ll;
  ASSERT_TRUE(sac_q_layers(linear, &ll), "linear critic layers readable");
  ASSERT_TRUE(ll.split_is_head && ll.num_hidden == 0u,
              "no hidden layer: split layer is the head");
  ASSERT_TRUE(linear->batch_logits->val->rows == 1u &&
                  linear->batch_logits->val->cols == 7u,
              "linear critic still produces one value per column");

  // A model built at batch_size 0 gains its batched graph on the first call.
  Net *late = net_new(arena);
  sac_q_build(arena, late, &rng, 4, 3, hidden, 2, 0);
  ASSERT_TRUE(late->batch_input == NULL, "batch_size 0 builds no batched graph");
  ASSERT_TRUE(net_set_batch_size(arena, late, 5u),
              "batched graph created on demand");
  ASSERT_TRUE(late->batch_input->val->cols == 5u &&
                  late->batch_action->val->cols == 5u,
              "on-demand batched views have the requested size");

  arena_dispose(arena);
  printf("OK\n");
}

// ── The actor cost matches the fused kernels, in value and in gradient ──

static void test_actor_cost_and_gradient(void) {
  printf("  Actor cost vs fused kernels... ");

  Arena *arena = arena_new(GiB(1));
  Rng rng;
  rng_seed(&rng, 11, 1);
  const u32 hidden[] = {8};
  const u32 obs_dim = 3u;
  const u32 act_dim = 2u;
  const u32 B = 4u;
  const real min_a[2] = {-1.0f, -2.0f};
  const real max_a[2] = {1.0f, 3.0f};

  Net *actor = net_new(arena);
  Net *q1 = net_new(arena);
  Net *q2 = net_new(arena);
  sac_actor_build(arena, actor, &rng, obs_dim, hidden, 1, act_dim, B, 0.0f);
  sac_q_build(arena, q1, &rng, obs_dim, act_dim, hidden, 1, B);
  sac_q_build(arena, q2, &rng, obs_dim, act_dim, hidden, 1, B);

  SacQNet l1, l2;
  ASSERT_TRUE(sac_q_layers(q1, &l1) && sac_q_layers(q2, &l2),
              "critic layers readable");

  Tensor *d_obs = tensor_create(arena, actor, obs_dim, B, TENSOR_PLAIN);
  Tensor *d_eps = tensor_create(arena, actor, act_dim, B, TENSOR_PLAIN);
  Tensor *d_alpha = tensor_create(arena, actor, 1u, 1u, TENSOR_PLAIN);
  d_alpha->val->data[0] = 0.3f;

  Rng data_rng;
  rng_seed(&data_rng, 5, 1);
  for (u32 k = 0; k < (u64)obs_dim * B; k++) {
    d_obs->val->data[k] = rng_normal(&data_rng) * 0.5f;
  }
  for (u32 k = 0; k < (u64)act_dim * B; k++) {
    d_eps->val->data[k] = rng_normal(&data_rng);
  }

  Tensor *half = tensor_create(arena, actor, act_dim, 1u, TENSOR_PLAIN);
  Tensor *center = tensor_create(arena, actor, act_dim, 1u, TENSOR_PLAIN);
  for (u32 d = 0; d < act_dim; d++) {
    half->val->data[d] = 0.5f * (max_a[d] - min_a[d]);
    center->val->data[d] = 0.5f * (max_a[d] + min_a[d]);
  }

  Tensor *mu = tensor_clip_straight(arena, actor, actor->batch_mean,
                                    -RL_GAUSSIAN_MEAN_BOUND,
                                    RL_GAUSSIAN_MEAN_BOUND);
  Tensor *sigma = tensor_exp(
      arena, actor,
      tensor_clip(arena, actor, actor->log_std, RL_GAUSSIAN_LOG_STD_MIN,
                  RL_GAUSSIAN_LOG_STD_MAX));
  Tensor *u = tensor_add(
      arena, actor, mu,
      tensor_mul(arena, actor,
                 tensor_broadcast(arena, actor, sigma, act_dim, B), d_eps));
  Tensor *action = tensor_add(
      arena, actor,
      tensor_mul(arena, actor, tensor_tanh(arena, actor, u),
                 tensor_broadcast(arena, actor, half, act_dim, B)),
      tensor_broadcast(arena, actor, center, act_dim, B));
  Tensor *logp = tensor_gaussian_logp(arena, actor, actor->batch_mean,
                                      actor->log_std, u, min_a, max_a);
  Tensor *q_min = tensor_min(
      arena, actor, sac_q_apply(arena, actor, &l1, d_obs, action),
      sac_q_apply(arena, actor, &l2, d_obs, action));
  Tensor *cost = tensor_sub(
      arena, actor,
      tensor_mul(arena, actor, d_alpha,
                 tensor_mean(arena, actor, logp, REDUCE_ALL)),
      tensor_mean(arena, actor, q_min, REDUCE_ALL));
  Schedule plan = schedule_from(arena, actor, cost);

  // Run the composition. The graph's own batched mean and the sampled action
  // are needed by the reference, so read them back after the forward pass.
  memcpy(actor->batch_input->val->data, d_obs->val->data,
         sizeof(real) * (u64)obs_dim * B);
  schedule_run(&plan);
  const Mat *mu_val = actor->batch_mean->val;

  // Reference log-density: the fused kernel over the graph's own sample.
  Mat *u_ref = mat_new(arena, act_dim, B);
  Mat *a_ref = mat_new(arena, act_dim, B);
  for (u32 d = 0; d < act_dim; d++) {
    for (u32 b = 0; b < B; b++) {
      const u32 k = d * B + b;
      const real sigma_d = real_exp(actor->log_std->val->data[d]);
      const real uu = mu_val->data[k] + sigma_d * d_eps->val->data[k];
      u_ref->data[k] = uu;
      a_ref->data[k] = center->val->data[d] + half->val->data[d] *
                                                real_tanh(uu);
    }
  }
  Mat *logp_ref = mat_new(arena, 1, B);
  RL_LIB_CHECK(mat_gaussian_logp(logp_ref, mu_val, actor->log_std->val, u_ref,
                                 min_a, max_a, B));

  // Reference critic values: the same weights through their batched graphs.
  memcpy(q1->batch_input->val->data, d_obs->val->data,
         sizeof(real) * (u64)obs_dim * B);
  memcpy(q1->batch_action->val->data, a_ref->data,
         sizeof(real) * (u64)act_dim * B);
  memcpy(q2->batch_input->val->data, d_obs->val->data,
         sizeof(real) * (u64)obs_dim * B);
  memcpy(q2->batch_action->val->data, a_ref->data,
         sizeof(real) * (u64)act_dim * B);
  schedule_run(&q1->train);
  schedule_run(&q2->train);
  const Mat *qv1 = q1->batch_logits->val;
  const Mat *qv2 = q2->batch_logits->val;

  real logp_mean = 0.0f;
  real q_mean = 0.0f;
  for (u32 b = 0; b < B; b++) {
    logp_mean += logp_ref->data[b];
    q_mean += qv1->data[b] < qv2->data[b] ? qv1->data[b] : qv2->data[b];
  }
  logp_mean /= (real)B;
  q_mean /= (real)B;
  const real cost_ref = 0.3f * logp_mean - q_mean;

  ASSERT_NEAR(cost->val->data[0], cost_ref, 2e-5f,
              "actor cost matches the fused-kernel reference");
  ASSERT_TRUE(cost->val->data[0] != 0.0f, "cost is not degenerate");

  // Finite-difference check of the whole composition in one parameter: log_std
  // enters both the reparameterised sample and the log-density, so this covers
  // the policy-gradient path through the critics as well.
  zero_grad(actor);
  zero_grad(q1);
  zero_grad(q2);
  schedule_run(&plan);
  schedule_grad_from(&plan, cost, true);
  const real g = actor->log_std->grad->data[0];

  const real base = actor->log_std->val->data[0];
  const real h = 1e-2f;
  actor->log_std->val->data[0] = base + h;
  schedule_run(&plan);
  const real c_plus = cost->val->data[0];
  actor->log_std->val->data[0] = base - h;
  schedule_run(&plan);
  const real c_minus = cost->val->data[0];
  actor->log_std->val->data[0] = base;
  const real fd = (c_plus - c_minus) / (2.0f * h);
  printf("(g %.5f fd %.5f) ", (double)g, (double)fd);
  ASSERT_TRUE(real_fabs(g - fd) < 2e-2f * (real_fabs(fd) + 1.0f),
              "actor cost gradient matches central differences");

  // Restore the forward values for the perturbation-free weights, then check
  // that the squashed action tensor matches the reference element by element.
  schedule_run(&plan);
  b32 action_ok = true;
  for (u32 k = 0; k < (u64)act_dim * B; k++) {
    if (real_fabs(action->val->data[k] - a_ref->data[k]) > 1e-6f) {
      action_ok = false;
    }
  }
  ASSERT_TRUE(action_ok, "squashed action matches the reference");

  arena_dispose(arena);
  printf("OK\n");
}

// ── Checkpointing ────────────────────────────────────────────────────

static const char *SAC_CKPT = "sac_test_ckpt.bin.tmp";

static void test_checkpoint_roundtrip(void) {
  printf("  Checkpoint round-trip... ");

  Arena *arena = arena_new(GiB(1));
  const u32 obs_dim = 4u;
  const u32 act_dim = 2u;
  const u32 hidden[] = {12, 6};

  Net *actor = net_new(arena);
  Net *q = net_new(arena);
  Rng rng;
  rng_seed(&rng, 17, 1);
  sac_actor_build(arena, actor, &rng, obs_dim, hidden, 2, act_dim, 0, 0.0f);
  sac_q_build(arena, q, &rng, obs_dim, act_dim, hidden, 2, 0);

  ASSERT_TRUE(checkpoint_save(actor, NULL, SAC_CKPT), "save a SAC actor");
  Net *actor_copy = net_new(arena);
  Rng rng2;
  rng_seed(&rng2, 999, 1); // different init: the load must overwrite it
  sac_actor_build(arena, actor_copy, &rng2, obs_dim, hidden, 2, act_dim, 0,
                  0.0f);
  ASSERT_TRUE(checkpoint_load(actor_copy, NULL, SAC_CKPT),
              "load into the same architecture");
  ASSERT_TRUE(params_equal(actor_copy, actor),
              "actor weights and log_std restored");

  Net *wrong = net_new(arena);
  sac_q_build(arena, wrong, &rng2, obs_dim, act_dim, hidden, 2, 0);
  ASSERT_TRUE(!checkpoint_load(wrong, NULL, SAC_CKPT),
              "loading an actor file into a critic is rejected");

  ASSERT_TRUE(checkpoint_save(q, NULL, SAC_CKPT), "save a SAC critic");
  Net *q_copy = net_new(arena);
  sac_q_build(arena, q_copy, &rng2, obs_dim, act_dim, hidden, 2, 0);
  ASSERT_TRUE(checkpoint_load(q_copy, NULL, SAC_CKPT),
              "load a critic into the same architecture");
  ASSERT_TRUE(params_equal(q_copy, q), "critic weights restored");

  remove(SAC_CKPT);
  arena_dispose(arena);
  printf("OK\n");
}

// ── Determinism ──────────────────────────────────────────────────────

static void test_determinism(void) {
  printf("  Determinism across runs... ");

  Arena *arena = arena_new(GiB(1));
  const u32 hidden[] = {8};

  Net *actor1 = net_new(arena);
  Net *q1a = net_new(arena);
  Net *q2a = net_new(arena);
  Net *q1ta = net_new(arena);
  Net *q2ta = net_new(arena);
  Net *actor2 = net_new(arena);
  Net *q1b = net_new(arena);
  Net *q2b = net_new(arena);
  Net *q1tb = net_new(arena);
  Net *q2tb = net_new(arena);
  Env *env1 = env_pendulum_create(arena);
  Env *env2 = env_pendulum_create(arena);

  Rng r1, r2;
  rng_seed(&r1, 5, 1);
  rng_seed(&r2, 5, 1);
  sac_actor_build(arena, actor1, &r1, env1->obs_dim, hidden, 1, env1->act_dim,
                  8, 0.0f);
  sac_actor_build(arena, actor2, &r2, env2->obs_dim, hidden, 1, env2->act_dim,
                  8, 0.0f);
  sac_q_build(arena, q1a, &r1, env1->obs_dim, env1->act_dim, hidden, 1, 8);
  sac_q_build(arena, q2a, &r1, env1->obs_dim, env1->act_dim, hidden, 1, 8);
  sac_q_build(arena, q1ta, &r1, env1->obs_dim, env1->act_dim, hidden, 1, 8);
  sac_q_build(arena, q2ta, &r1, env1->obs_dim, env1->act_dim, hidden, 1, 8);
  sac_q_build(arena, q1b, &r2, env2->obs_dim, env2->act_dim, hidden, 1, 8);
  sac_q_build(arena, q2b, &r2, env2->obs_dim, env2->act_dim, hidden, 1, 8);
  sac_q_build(arena, q1tb, &r2, env2->obs_dim, env2->act_dim, hidden, 1, 8);
  sac_q_build(arena, q2tb, &r2, env2->obs_dim, env2->act_dim, hidden, 1, 8);
  ASSERT_TRUE(params_equal(actor1, actor2), "same seed -> same actor init");
  ASSERT_TRUE(params_equal(q1a, q1b), "same seed -> same critic init");

  // SGD exercises the non-Adam trainer branches; auto_alpha the temperature.
  TrainCfg cfg = {0};
  cfg.epochs = 4u;
  cfg.rollout_size = 16u;
  cfg.learning_rate = 1e-3f;
  cfg.gamma = 0.99f;
  cfg.seed = 21u;
  cfg.optim = OPT_SGD;
  cfg.max_grad_norm = 10.0f;
  cfg.batch_size = 8u;
  cfg.replay_capacity = 128u;
  cfg.warmup_steps = 16u;
  cfg.train_every = 1u;
  cfg.tau = 0.01f;
  cfg.alpha = 0.5f;
  cfg.auto_alpha = true;

  train_sac(env1, actor1, q1a, q2a, q1ta, q2ta, &cfg, arena);
  train_sac(env2, actor2, q1b, q2b, q1tb, q2tb, &cfg, arena);

  ASSERT_TRUE(params_equal(actor1, actor2),
              "same seed -> identical actor weights");
  ASSERT_TRUE(params_equal(q1a, q1b), "same seed -> identical critic weights");
  ASSERT_TRUE(params_equal(q1ta, q1tb), "same seed -> identical target weights");
  ASSERT_TRUE(params_equal(q2a, q2b), "same seed -> identical second critic");

  arena_dispose(arena);
  printf("OK\n");
}

// ── End-to-end learning on Pendulum ──────────────────────────────────

static void test_sac_learning(void) {
  printf("  Pendulum learning... ");

  Arena *arena = arena_new(GiB(1));
  Env *env = env_pendulum_create(arena);
  Net *actor = net_new(arena);
  Net *q1 = net_new(arena);
  Net *q2 = net_new(arena);
  Net *q1_target = net_new(arena);
  Net *q2_target = net_new(arena);

  Rng rng;
  rng_seed(&rng, 7, 1);
  const u32 hidden[] = {32, 32};
  const u32 batch_size = 32u;
  sac_actor_build(arena, actor, &rng, env->obs_dim, hidden, 2, env->act_dim,
                  batch_size, 0.0f);
  sac_q_build(arena, q1, &rng, env->obs_dim, env->act_dim, hidden, 2,
              batch_size);
  sac_q_build(arena, q2, &rng, env->obs_dim, env->act_dim, hidden, 2,
              batch_size);
  sac_q_build(arena, q1_target, &rng, env->obs_dim, env->act_dim, hidden, 2,
              batch_size);
  sac_q_build(arena, q2_target, &rng, env->obs_dim, env->act_dim, hidden, 2,
              batch_size);

  TrainCfg cfg = {0};
  cfg.epochs = 500u;
  cfg.rollout_size = 32u;
  cfg.max_steps = env->max_steps;
  cfg.learning_rate = 1e-3f;
  cfg.gamma = 0.99f;
  cfg.seed = 7u;
  cfg.optim = OPT_ADAM;
  cfg.max_grad_norm = 10.0f;
  cfg.batch_size = batch_size;
  cfg.replay_capacity = 10000u;
  cfg.warmup_steps = 500u;
  cfg.train_every = 1u;
  cfg.tau = 0.005f;
  cfg.alpha = 0.2f;
  cfg.auto_alpha = true;

  train_sac(env, actor, q1, q2, q1_target, q2_target, &cfg, arena);

  u64 eval_seeds[30];
  make_eval_seeds(eval_seeds, 30, 7 + 5000);
  const EvalResult res =
      evaluate_continuous_seeded(env, actor, eval_seeds, 30, true);
  printf("(mean %.1f) ", (double)res.avg_return);
  // The trained trajectory depends on the build's floating-point details (this
  // run reaches about -160 in the UBSan/-O0 test build), so the bar is
  // deliberately loose: a random policy scores about -1200, and the check is
  // that it learned, not how well.
  ASSERT_TRUE(res.avg_return > -700.0f,
              "deterministic evaluation clears the random baseline");

  arena_dispose(arena);
  printf("OK\n");
}

int main(void) {
  printf("=== test_sac ===\n");

  test_replay_ring();
  test_replay_gather();
  test_td_target();
  test_soft_update();
  test_builders();
  test_actor_cost_and_gradient();
  test_checkpoint_roundtrip();
  test_determinism();
  test_sac_learning();

  printf("\nTests: %u passed, %u failed\n", tests_passed, tests_failed);
  return tests_failed == 0u ? 0 : 1;
}
