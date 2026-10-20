#include "sac.h"

#include "linalg.h"
#include "optim.h"

#include <stdio.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Replay buffer
// ---------------------------------------------------------------------------
//
// A fixed-capacity ring over continuous transitions: `head` is the next write
// slot and `size` counts valid entries, so once full the buffer keeps the most
// recent `capacity` steps and overwrites the oldest. Every field lives in its
// own flat array, which makes the sampled-batch gather a per-field column copy.
// Termination and truncation are stored apart because the bootstrap differs:
// a time-limit cut still bootstraps from the next state, a real terminal does
// not.

SacReplay *sac_replay_new(Arena *arena, u32 capacity, u32 obs_dim,
                          u32 act_dim) {
  RL_LIB_ASSERT(arena != NULL, "sac_replay_new: NULL arena");
  RL_LIB_ASSERT(capacity > 0u && obs_dim > 0u && act_dim > 0u,
                "sac_replay_new: zero-sized ring buffer");

  SacReplay *buf = ARENA_OBJ(arena, SacReplay);
  buf->obs_dim = obs_dim;
  buf->act_dim = act_dim;
  buf->capacity = capacity;
  buf->size = 0u;
  buf->head = 0u;
  buf->obs = ARENA_ARR(arena, real, (u64)capacity * obs_dim);
  buf->next_obs = ARENA_ARR(arena, real, (u64)capacity * obs_dim);
  buf->actions = ARENA_ARR(arena, real, (u64)capacity * act_dim);
  buf->rewards = ARENA_ARR(arena, real, capacity);
  buf->terminated = ARENA_ARR(arena, u8, capacity);
  buf->truncated = ARENA_ARR(arena, u8, capacity);
  return buf;
}

void sac_replay_add(SacReplay *buf, const real *obs, const real *action,
                    real reward, const real *next_obs, b32 terminated,
                    b32 truncated) {
  RL_LIB_ASSERT(buf != NULL && obs != NULL && action != NULL &&
                    next_obs != NULL,
                "sac_replay_add: NULL argument");

  const u32 slot = buf->head;
  memcpy(buf->obs + (u64)slot * buf->obs_dim, obs,
         sizeof(real) * (u64)buf->obs_dim);
  memcpy(buf->actions + (u64)slot * buf->act_dim, action,
         sizeof(real) * (u64)buf->act_dim);
  memcpy(buf->next_obs + (u64)slot * buf->obs_dim, next_obs,
         sizeof(real) * (u64)buf->obs_dim);
  buf->rewards[slot] = reward;
  buf->terminated[slot] = terminated ? 1u : 0u;
  buf->truncated[slot] = truncated ? 1u : 0u;

  buf->head = (slot + 1u) % buf->capacity;
  if (buf->size < buf->capacity) {
    buf->size += 1u;
  }
}

void sac_replay_gather(const SacReplay *buf, const u32 *indices, u32 batch,
                       Mat *obs, Mat *actions, Mat *next_obs, Mat *rewards,
                       Mat *terminated) {
  RL_LIB_ASSERT(buf != NULL && indices != NULL && obs != NULL &&
                    actions != NULL && next_obs != NULL && rewards != NULL &&
                    terminated != NULL,
                "sac_replay_gather: NULL argument");
  RL_LIB_ASSERT(obs->rows == buf->obs_dim && obs->cols == batch &&
                    next_obs->rows == buf->obs_dim &&
                    next_obs->cols == batch && actions->rows == buf->act_dim &&
                    actions->cols == batch && rewards->rows == 1u &&
                    rewards->cols == batch && terminated->rows == 1u &&
                    terminated->cols == batch,
                "sac_replay_gather: destination shape mismatch");

  for (u32 b = 0; b < batch; b++) {
    const u32 slot = indices[b];
    RL_LIB_ASSERT(slot < buf->size, "sac_replay_gather: index out of range");

    const real *src_obs = buf->obs + (u64)slot * buf->obs_dim;
    const real *src_next = buf->next_obs + (u64)slot * buf->obs_dim;
    const real *src_act = buf->actions + (u64)slot * buf->act_dim;
    for (u32 r = 0; r < buf->obs_dim; r++) {
      obs->data[(u64)r * batch + b] = src_obs[r];
      next_obs->data[(u64)r * batch + b] = src_next[r];
    }
    for (u32 a = 0; a < buf->act_dim; a++) {
      actions->data[(u64)a * batch + b] = src_act[a];
    }
    rewards->data[b] = buf->rewards[slot];
    terminated->data[b] = buf->terminated[slot] != 0u ? (real)1 : (real)0;
  }
}

// ---------------------------------------------------------------------------
// Targets and the temperature
// ---------------------------------------------------------------------------

void sac_soft_update(const Net *src, Net *dst, real tau) {
  RL_LIB_ASSERT(src != NULL && dst != NULL,
                "sac_soft_update: NULL network");
  RL_LIB_ASSERT(src->n_params == dst->n_params,
                "sac_soft_update: parameter count mismatch");

  real t = tau;
  if (t < (real)0) {
    t = (real)0;
  } else if (t > (real)1) {
    t = (real)1;
  }

  for (u32 i = 0; i < src->n_params; i++) {
    const Mat *s = src->params[i]->val;
    Mat *d = dst->params[i]->val;
    RL_LIB_ASSERT(s->rows == d->rows && s->cols == d->cols,
                  "sac_soft_update: parameter shape mismatch");
    const u64 n = (u64)s->rows * (u64)s->cols;
    for (u64 k = 0; k < n; k++) {
      d->data[k] = ((real)1 - t) * d->data[k] + t * s->data[k];
    }
  }
}

real sac_td_target(real reward, real q_min, real alpha_logp, real gamma,
                   b32 terminal) {
  return terminal ? reward : reward + gamma * (q_min - alpha_logp);
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

// Everything one gradient step needs, so the step can live in its own function
// instead of a hundred locals.
typedef struct {
  const Env *env;
  Net *actor;
  Net *q1;
  Net *q2;
  Net *q1_target;
  Net *q2_target;
  Net *alpha_net;

  SacQNet critics[2];
  SacReplay *buf;
  Adam *adam_actor;
  Adam *adam_q1;
  Adam *adam_q2;
  Adam *adam_alpha;
  Tensor *log_alpha;
  Schedule actor_plan;
  Tensor *actor_cost;

  Tensor *d_obs;       // (obs_dim x B)
  Tensor *d_actions;   // (act_dim x B)
  Tensor *d_next_obs;  // (obs_dim x B)
  Tensor *d_eps;       // (act_dim x B) reparameterisation noise
  Tensor *d_alpha;     // (1 x 1) temperature used by the actor cost
  Mat *d_rewards;      // (1 x B)
  Mat *d_terminated;   // (1 x B) 1 for a real terminal
  Mat *d_next_actions; // (act_dim x B)
  Mat *d_next_u;       // (act_dim x B)
  Mat *d_logp;         // (1 x B) log pi(a'|s')
  u32 *indices;
  u32 batch_size;

  real gamma;
  real tau;
  real alpha0;
  real alpha_lr;
  real target_entropy;
  real max_grad_norm;
  b32 auto_alpha;
} SacTrainer;

// One gradient step over a minibatch sampled uniformly from the replay buffer:
// both critics regress the shared TD target, then the actor ascends the
// (entropy-adjusted) critic value through a reparameterised sample, then the
// temperature follows its scalar objective, and finally the target critics
// follow their online counterparts.
static void sac_grad_step(SacTrainer *t, Rng *rng, real lr_t) {
  const u32 obs_dim = t->env->obs_dim;
  const u32 act_dim = t->env->act_dim;
  const u32 B = t->batch_size;

  // 1. Draw a minibatch and gather it into (dim x batch) buffers.
  for (u32 b = 0; b < B; b++) {
    u32 pick = (u32)(rng_unit(rng) * (real)t->buf->size);
    if (pick >= t->buf->size) {
      pick = t->buf->size - 1u;
    }
    t->indices[b] = pick;
  }
  sac_replay_gather(t->buf, t->indices, B, t->d_obs->val, t->d_actions->val,
                    t->d_next_obs->val, t->d_rewards, t->d_terminated);

  // 2. Bootstrap. The next action comes from the current actor applied to the
  //    next states (its batched graph), then is scored by the target critics
  //    and the library's log-density kernel -- the same clamps and the same
  //    squash the policy's own sampler uses.
  memcpy(t->actor->batch_input->val->data, t->d_next_obs->val->data,
         sizeof(real) * (u64)obs_dim * B);
  schedule_run(&t->actor->train);

  const Mat *mu_next = t->actor->batch_mean->val; // act_dim x B
  Mat mu_col = {act_dim, 1u, NULL};
  Mat a_col = {act_dim, 1u, NULL};
  Mat u_col = {act_dim, 1u, NULL};
  for (u32 b = 0; b < B; b++) {
    mu_col.data = mu_next->data + (u64)b * act_dim;
    a_col.data = t->d_next_actions->data + (u64)b * act_dim;
    u_col.data = t->d_next_u->data + (u64)b * act_dim;
    RL_LIB_CHECK(mat_gaussian_sample(&a_col, &u_col, &mu_col,
                                     t->actor->log_std->val,
                                     t->env->action_min, t->env->action_max,
                                     rng));
  }
  RL_LIB_CHECK(mat_gaussian_logp(t->d_logp, mu_next, t->actor->log_std->val,
                                 t->d_next_u, t->env->action_min,
                                 t->env->action_max, B));

  Net *targets[2] = {t->q1_target, t->q2_target};
  for (u32 c = 0; c < 2u; c++) {
    Net *qt = targets[c];
    memcpy(qt->batch_input->val->data, t->d_next_obs->val->data,
           sizeof(real) * (u64)obs_dim * B);
    memcpy(qt->batch_action->val->data, t->d_next_actions->data,
           sizeof(real) * (u64)act_dim * B);
    schedule_run(&qt->train);
  }
  const Mat *q1_next = t->q1_target->batch_logits->val; // 1 x B
  const Mat *q2_next = t->q2_target->batch_logits->val;

  // 3. The shared TD target: a truncated transition still bootstraps, only a
  //    real terminal drops the continuation term.
  const real alpha_now =
      t->auto_alpha ? real_exp(t->log_alpha->val->data[0]) : t->alpha0;
  Mat *target = t->q1->target->val; // 1 x B
  for (u32 b = 0; b < B; b++) {
    const real q_min = (q1_next->data[b] < q2_next->data[b])
                           ? q1_next->data[b]
                           : q2_next->data[b];
    target->data[b] =
        sac_td_target(t->d_rewards->data[b], q_min,
                      alpha_now * t->d_logp->data[b], t->gamma,
                      t->d_terminated->data[b] > (real)0.5);
  }
  memcpy(t->q2->target->val->data, target->data, sizeof(real) * (u64)B);

  // 4. Both critics regress that target. Their gradients are independent, so a
  //    whole sweep of one is finished (including the parameter update) before
  //    the next starts.
  Net *critics[2] = {t->q1, t->q2};
  Adam *adams[2] = {t->adam_q1, t->adam_q2};
  for (u32 c = 0; c < 2u; c++) {
    Net *q = critics[c];
    memcpy(q->batch_input->val->data, t->d_obs->val->data,
           sizeof(real) * (u64)obs_dim * B);
    memcpy(q->batch_action->val->data, t->d_actions->val->data,
           sizeof(real) * (u64)act_dim * B);
    schedule_run(&q->loss_plan);
    schedule_grad_from(&q->loss_plan, q->cost, true);
    if (t->max_grad_norm > 0.0f) {
      (void)clip_grad_norm(q, t->max_grad_norm);
    }
    if (adams[c] != NULL) {
      adams[c]->lr = lr_t;
      adam_step(adams[c], q);
    } else {
      sgd_step(q, lr_t, 1.0f);
    }
    zero_grad(q);
  }

  // 5. The actor ascends the critic value of its own reparameterised action
  //    while pushing the entropy term up. The sweep also accumulates gradients
  //    in the critics' weights (the action path runs through them); those are
  //    not part of the actor objective, so they are dropped afterwards.
  memcpy(t->actor->batch_input->val->data, t->d_obs->val->data,
         sizeof(real) * (u64)obs_dim * B);
  const u64 eps_count = (u64)act_dim * B;
  for (u64 k = 0; k < eps_count; k++) {
    t->d_eps->val->data[k] = rng_normal(rng);
  }
  t->d_alpha->val->data[0] = alpha_now;

  schedule_run(&t->actor_plan);
  schedule_grad_from(&t->actor_plan, t->actor_cost, true);
  if (t->max_grad_norm > 0.0f) {
    (void)clip_grad_norm(t->actor, t->max_grad_norm);
  }
  if (t->adam_actor != NULL) {
    t->adam_actor->lr = lr_t;
    adam_step(t->adam_actor, t->actor);
  } else {
    sgd_step(t->actor, lr_t, 1.0f);
  }
  zero_grad(t->actor);
  zero_grad(t->q1);
  zero_grad(t->q2);

  // 6. Temperature: d/d(log alpha) of -log_alpha * (mean log pi + target).
  if (t->auto_alpha) {
    real mean_logp = 0.0f;
    for (u32 b = 0; b < B; b++) {
      mean_logp += t->d_logp->data[b];
    }
    mean_logp /= (real)B;
    t->log_alpha->grad->data[0] = -(mean_logp + t->target_entropy);
    if (t->adam_alpha != NULL) {
      adam_step(t->adam_alpha, t->alpha_net);
    } else {
      sgd_step(t->alpha_net, t->alpha_lr, 1.0f);
    }
    zero_grad(t->alpha_net);
  }

  // 7. Targets follow their online counterparts.
  sac_soft_update(t->q1, t->q1_target, t->tau);
  sac_soft_update(t->q2, t->q2_target, t->tau);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

void train_sac(Env *env, Net *actor, Net *q1, Net *q2, Net *q1_target,
               Net *q2_target, const TrainCfg *config, Arena *train_arena) {
  RL_LIB_ASSERT(env != NULL && actor != NULL && q1 != NULL && q2 != NULL &&
                    q1_target != NULL && q2_target != NULL && config != NULL &&
                    train_arena != NULL,
                "train_sac: NULL argument");
  RL_LIB_ASSERT(env->continuous && env->step_continuous != NULL &&
                    env->action_min != NULL && env->action_max != NULL,
                "train_sac: a continuous-action environment with bounds is "
                "required");
  RL_LIB_ASSERT(actor != q1 && actor != q2 && actor != q1_target &&
                    actor != q2_target && q1 != q2 && q1 != q1_target &&
                    q1 != q2_target && q2 != q1_target && q2 != q2_target &&
                    q1_target != q2_target,
                "train_sac: the actor and the critics must be distinct "
                "networks");
  RL_LIB_ASSERT(actor->mean != NULL && actor->log_std != NULL,
                "train_sac: actor was not built by sac_actor_build");
  RL_LIB_ASSERT(!config->norm_obs && !config->norm_reward,
                "train_sac: normalization is PPO-only (use train_ppo)");

  const u32 obs_dim = env->obs_dim;
  const u32 act_dim = env->act_dim;
  const u32 batch_size = (config->batch_size > 0u) ? config->batch_size : 64u;
  const u32 updates = (config->epochs > 0u) ? config->epochs : 1u;
  const u32 per_update =
      (config->rollout_size > 0u) ? config->rollout_size : 32u;
  const u32 warmup = (config->warmup_steps > 0u) ? config->warmup_steps : 1000u;
  const u32 train_every =
      (config->train_every > 0u) ? config->train_every : 1u;
  const u32 capacity =
      (config->replay_capacity > 0u) ? config->replay_capacity : 20000u;
  const real gamma = (config->gamma > 0.0f) ? config->gamma : 0.99f;
  const real tau = (config->tau > 0.0f) ? config->tau : (real)0.005;
  const real alpha0 = (config->alpha > 0.0f) ? config->alpha : (real)0.2;
  const real alpha_lr =
      (config->alpha_lr > 0.0f) ? config->alpha_lr : config->learning_rate;
  const real target_entropy =
      (config->target_entropy > 0.0f) ? config->target_entropy
                                      : -(real)act_dim;

  RL_LIB_ASSERT(net_set_batch_size(train_arena, actor, batch_size) &&
                    net_set_batch_size(train_arena, q1, batch_size) &&
                    net_set_batch_size(train_arena, q2, batch_size) &&
                    net_set_batch_size(train_arena, q1_target, batch_size) &&
                    net_set_batch_size(train_arena, q2_target, batch_size),
                "train_sac: could not set the training batch size");
  RL_LIB_ASSERT(actor->batch_mean != NULL &&
                    actor->batch_input != NULL &&
                    q1->batch_action != NULL && q2->batch_action != NULL &&
                    q1->target != NULL &&
                    q1_target->batch_action != NULL,
                "train_sac: missing SAC graphs (use the SAC builders)");

  SacTrainer t;
  memset(&t, 0, sizeof(t));
  t.env = env;
  t.actor = actor;
  t.q1 = q1;
  t.q2 = q2;
  t.q1_target = q1_target;
  t.q2_target = q2_target;
  t.batch_size = batch_size;
  t.gamma = gamma;
  t.tau = tau;
  t.alpha0 = alpha0;
  t.alpha_lr = alpha_lr;
  t.target_entropy = target_entropy;
  t.max_grad_norm = config->max_grad_norm;
  t.auto_alpha = config->auto_alpha;

  RL_LIB_ASSERT(sac_q_layers(q1, &t.critics[0]) &&
                    sac_q_layers(q2, &t.critics[1]),
                "train_sac: critics were not built by sac_q_build");
  RL_LIB_ASSERT(t.critics[0].split.obs_features == obs_dim &&
                    t.critics[0].split.act_features == act_dim,
                "train_sac: the critics do not match the environment's "
                "observation or action size");

  t.buf = sac_replay_new(train_arena, capacity, obs_dim, act_dim);
  t.adam_actor = (config->optim == OPT_ADAM)
                     ? adam_new(train_arena, actor, config->learning_rate)
                     : NULL;
  t.adam_q1 = (config->optim == OPT_ADAM)
                  ? adam_new(train_arena, q1, config->learning_rate)
                  : NULL;
  t.adam_q2 = (config->optim == OPT_ADAM)
                  ? adam_new(train_arena, q2, config->learning_rate)
                  : NULL;
  t.indices = ARENA_ARR(train_arena, u32, batch_size);

  t.d_obs = tensor_create(train_arena, actor, obs_dim, batch_size,
                          TENSOR_PLAIN);
  t.d_actions =
      tensor_create(train_arena, actor, act_dim, batch_size, TENSOR_PLAIN);
  t.d_next_obs = tensor_create(train_arena, actor, obs_dim, batch_size,
                               TENSOR_PLAIN);
  t.d_eps =
      tensor_create(train_arena, actor, act_dim, batch_size, TENSOR_PLAIN);
  t.d_rewards = mat_new(train_arena, 1u, batch_size);
  t.d_terminated = mat_new(train_arena, 1u, batch_size);
  t.d_next_actions = mat_new(train_arena, act_dim, batch_size);
  t.d_next_u = mat_new(train_arena, act_dim, batch_size);
  t.d_logp = mat_new(train_arena, 1u, batch_size);

  // Actor cost: alpha * mean_b log pi(a_b | s_b) - mean_b min_i Q_i(s_b, a_b),
  // with a_b = squash(mu_b + sigma_b * eps_b). Both critics are applied to the
  // graph-computed action through sac_q_apply, so the reverse sweep yields the
  // policy gradient directly -- no dQ/da has to be supplied by hand.
  Tensor *d_alpha = tensor_create(train_arena, actor, 1u, 1u, TENSOR_PLAIN);
  d_alpha->val->data[0] = alpha0;
  t.d_alpha = d_alpha;

  Tensor *half = tensor_create(train_arena, actor, act_dim, 1u, TENSOR_PLAIN);
  Tensor *center = tensor_create(train_arena, actor, act_dim, 1u, TENSOR_PLAIN);
  for (u32 d = 0; d < act_dim; d++) {
    const real lo = env->action_min[d];
    const real hi = env->action_max[d];
    RL_LIB_ASSERT(hi > lo, "train_sac: action bounds must be increasing");
    half->val->data[d] = ((real)0.5) * (hi - lo);
    center->val->data[d] = ((real)0.5) * (hi + lo);
  }

  Tensor *mu = tensor_clip_straight(train_arena, actor, actor->batch_mean,
                                    -RL_GAUSSIAN_MEAN_BOUND,
                                    RL_GAUSSIAN_MEAN_BOUND);
  Tensor *sigma = tensor_exp(
      train_arena, actor,
      tensor_clip(train_arena, actor, actor->log_std, RL_GAUSSIAN_LOG_STD_MIN,
                  RL_GAUSSIAN_LOG_STD_MAX));
  Tensor *u = tensor_add(
      train_arena, actor, mu,
      tensor_mul(train_arena, actor,
                 tensor_broadcast(train_arena, actor, sigma, act_dim,
                                  batch_size),
                 t.d_eps));
  Tensor *action = tensor_add(
      train_arena, actor,
      tensor_mul(train_arena, actor, tensor_tanh(train_arena, actor, u),
                 tensor_broadcast(train_arena, actor, half, act_dim,
                                  batch_size)),
      tensor_broadcast(train_arena, actor, center, act_dim, batch_size));

  Tensor *logp = tensor_gaussian_logp(train_arena, actor, actor->batch_mean,
                                      actor->log_std, u, env->action_min,
                                      env->action_max);
  Tensor *q_min = tensor_min(
      train_arena, actor,
      sac_q_apply(train_arena, actor, &t.critics[0], t.d_obs, action),
      sac_q_apply(train_arena, actor, &t.critics[1], t.d_obs, action));
  t.actor_cost = tensor_sub(
      train_arena, actor,
      tensor_mul(train_arena, actor, d_alpha,
                 tensor_mean(train_arena, actor, logp, REDUCE_ALL)),
      tensor_mean(train_arena, actor, q_min, REDUCE_ALL));
  t.actor_plan = schedule_from(train_arena, actor, t.actor_cost);

  // Temperature: a one-parameter network holding log alpha. Its objective is
  // the closed-form SAC one, J = -log_alpha * (mean_b log pi(a'|s') + entropy
  // target), so the gradient is a scalar and no graph is needed.
  t.alpha_net = net_new(train_arena);
  t.log_alpha = tensor_create(train_arena, t.alpha_net, 1u, 1u,
                              TENSOR_PARAM | TENSOR_GRAD);
  t.log_alpha->val->data[0] = real_log(alpha0);
  t.adam_alpha = (t.auto_alpha && config->optim == OPT_ADAM)
                     ? adam_new(train_arena, t.alpha_net, alpha_lr)
                     : NULL;

  // The targets start as copies of the critics.
  sac_soft_update(q1, q1_target, (real)1);
  sac_soft_update(q2, q2_target, (real)1);
  zero_grad(actor);
  zero_grad(q1);
  zero_grad(q2);

  Rng rng;
  const u64 seed = (config->seed != 0u) ? config->seed : 42u;
  rng_seed(&rng, seed, 4);

  real *obs = ARENA_ARR(train_arena, real, obs_dim);
  real *next_obs = ARENA_ARR(train_arena, real, obs_dim);
  real *action_buf = ARENA_ARR(train_arena, real, act_dim);
  real *u_buf = ARENA_ARR(train_arena, real, act_dim);
  Mat action_out = {act_dim, 1u, action_buf};
  Mat u_out = {act_dim, 1u, u_buf};

  // Sampling needs a full minibatch even if warmup_steps is smaller.
  const u32 min_fill = (warmup > batch_size) ? warmup : batch_size;

  env->reset(env, seed);

  u32 env_steps = 0u;
  real episode_return = 0.0f;
  real return_sum = 0.0f;
  u64 episode_count = 0u;
  u64 total_episodes = 0u;

  for (u32 it = 0; it < updates; it++) {
    const real lr_t = train_lr_at(config, config->learning_rate, it);

    for (u32 s = 0; s < per_update; s++) {
      // Act from the current policy: mean from the rollout graph, then the
      // library's Gaussian sampler, which applies the same clamps the training
      // graph does.
      env->observe(env, obs);
      memcpy(actor->input->val->data, obs, sizeof(real) * obs_dim);
      schedule_run(&actor->fwd);
      RL_LIB_CHECK(mat_gaussian_sample(&action_out, &u_out, actor->mean->val,
                                       actor->log_std->val, env->action_min,
                                       env->action_max, &rng));

      b32 done = false;
      b32 truncated = false;
      const real reward =
          env->step_continuous(env, action_buf, &done, &truncated);

      env->observe(env, next_obs);
      sac_replay_add(t.buf, obs, action_buf, reward, next_obs, done,
                     truncated);

      episode_return += reward;
      env_steps += 1u;

      if (done || truncated) {
        return_sum += episode_return;
        episode_count += 1u;
        total_episodes += 1u;
        episode_return = 0.0f;
        // Deterministic per-episode seed, kept separate from the sampling
        // stream that draws minibatch indices and exploration noise.
        env->reset(env, seed + total_episodes);
      }

      if (t.buf->size >= min_fill && (env_steps % train_every) == 0u) {
        sac_grad_step(&t, &rng, lr_t);
      }
    }

    if (config->verbose || (it + 1u) % 50u == 0u || it == 0u ||
        it + 1u == updates) {
      const real mean_return =
          (episode_count > 0u) ? return_sum / (real)episode_count : 0.0f;
      printf("Update %4u/%u | Mean Episode Return: %7.2f | Episodes: %3llu | "
             "Steps: %u | Alpha: %7.4f\n",
             it + 1u, updates, (double)mean_return,
             (unsigned long long)episode_count, env_steps,
             (double)(t.auto_alpha ? real_exp(t.log_alpha->val->data[0])
                                   : alpha0));
      return_sum = 0.0f;
      episode_count = 0u;
    }
  }
}
