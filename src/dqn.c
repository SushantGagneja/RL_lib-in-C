#include "dqn.h"

#include "layers.h"
#include "optim.h"
#include "random.h"

#include <stdio.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Replay buffer
// ---------------------------------------------------------------------------
//
// A fixed-capacity ring: `head` is the next write slot and `size` counts valid
// transitions, so once full the buffer keeps the most recent `capacity` steps
// and overwrites the oldest. Each field lives in its own flat array, which
// keeps the sampled-batch gather a column copy per field.

ReplayBuf *replay_new(Arena *arena, u32 capacity, u32 obs_dim) {
  RL_LIB_ASSERT(capacity > 0u && obs_dim > 0u, "replay_new: zero capacity");

  ReplayBuf *buf = ARENA_OBJ(arena, ReplayBuf);
  buf->obs_dim = obs_dim;
  buf->capacity = capacity;
  buf->size = 0u;
  buf->head = 0u;
  buf->obs = ARENA_ARR(arena, real, capacity * obs_dim);
  buf->next_obs = ARENA_ARR(arena, real, capacity * obs_dim);
  buf->rewards = ARENA_ARR(arena, real, capacity);
  buf->actions = ARENA_ARR(arena, u32, capacity);
  buf->dones = ARENA_ARR(arena, u8, capacity);
  buf->truncated = ARENA_ARR(arena, u8, capacity);
  return buf;
}

void replay_add(ReplayBuf *buf, const real *obs, u32 action, real reward,
                const real *next_obs, b32 done) {
  replay_add_ex(buf, obs, action, reward, next_obs, done, false);
}

void replay_add_ex(ReplayBuf *buf, const real *obs, u32 action, real reward,
                   const real *next_obs, b32 terminal, b32 truncated) {
  RL_LIB_ASSERT(buf != NULL && obs != NULL && next_obs != NULL,
                "replay_add: NULL argument");

  const u32 slot = buf->head;
  real *dst_obs = buf->obs + (u64)slot * buf->obs_dim;
  real *dst_next = buf->next_obs + (u64)slot * buf->obs_dim;
  for (u32 i = 0; i < buf->obs_dim; i++) {
    dst_obs[i] = obs[i];
    dst_next[i] = next_obs[i];
  }
  buf->rewards[slot] = reward;
  buf->actions[slot] = action;
  buf->dones[slot] = terminal ? 1u : 0u;
  buf->truncated[slot] = truncated ? 1u : 0u;

  buf->head = (slot + 1u == buf->capacity) ? 0u : slot + 1u;
  if (buf->size < buf->capacity) {
    buf->size += 1u;
  }
}

real dqn_td_target(real reward, real q_max, real gamma, b32 terminal) {
  return terminal ? reward : reward + gamma * q_max;
}

void dqn_sync_target(const Net *src, Net *dst) {
  RL_LIB_ASSERT(src != NULL && dst != NULL, "dqn_sync_target: NULL network");
  RL_LIB_ASSERT(src->n_params == dst->n_params,
                "dqn_sync_target: parameter count mismatch");

  for (u32 i = 0; i < src->n_params; i++) {
    const Mat *s = src->params[i]->val;
    Mat *d = dst->params[i]->val;
    RL_LIB_ASSERT(s->rows == d->rows && s->cols == d->cols,
                  "dqn_sync_target: parameter shape mismatch");
    memcpy(d->data, s->data, sizeof(real) * (u64)s->rows * (u64)s->cols);
  }
}

// ---------------------------------------------------------------------------
// Epsilon schedule
// ---------------------------------------------------------------------------

real dqn_epsilon_at(const TrainCfg *config, u32 env_steps) {
  RL_LIB_ASSERT(config != NULL, "dqn_epsilon_at: NULL config");

  const real start = (config->eps_start > 0.0f) ? config->eps_start : 1.0f;
  const real end = (config->eps_end > 0.0f) ? config->eps_end : 0.05f;

  u32 decay = config->eps_decay_steps;
  if (decay == 0u) {
    // Default: reach eps_end halfway through the run (the classic shape:
    // explore for the first half, then mostly exploit).
    const u32 updates = (config->epochs > 0u) ? config->epochs : 1u;
    const u32 per_update =
        (config->rollout_size > 0u) ? config->rollout_size : 32u;
    decay = (updates / 2u) * per_update;
    if (decay == 0u) {
      decay = 1u;
    }
  }

  if (env_steps >= decay) {
    return end;
  }
  const real t = (real)env_steps / (real)decay;
  return start + (end - start) * t;
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

// Gathers the observations of `idx` into `dst`'s columns (Mat is row-major, one
// transition per column).
static void gather_obs(const ReplayBuf *buf, const real *src, const u32 *idx,
                       u32 batch_size, Mat *dst) {
  for (u32 b = 0; b < batch_size; b++) {
    const real *row = src + (u64)idx[b] * buf->obs_dim;
    for (u32 r = 0; r < buf->obs_dim; r++) {
      dst->data[r * batch_size + b] = row[r];
    }
  }
}

// One gradient step over a minibatch of transitions sampled uniformly from the
// replay buffer. The prediction forward is run first (the trainer reads the
// online Q values to build the masked targets), then the target network scores
// the sampled next states, then the loss backpropagates and the optimizer
// steps. `lr_t` is the (possibly decayed) learning rate for this update.
static void dqn_grad_step(Net *q_model, Net *q_target, const ReplayBuf *buf,
                          Adam *adam, const TrainCfg *config, Rng *rng,
                          u32 *idx, u32 batch_size, real lr_t) {
  for (u32 b = 0; b < batch_size; b++) {
    u32 pick = (u32)(rng_unit(rng) * (real)buf->size);
    if (pick >= buf->size) {
      pick = buf->size - 1u;
    }
    idx[b] = pick;
  }

  gather_obs(buf, buf->obs, idx, batch_size, q_model->batch_input->val);
  // Forward fills batch_logits (the online Q values). The cost node also runs
  // forward here, but against the previous step's targets — the value is not
  // used; the backward below recomputes the error from the fresh target.
  schedule_run(&q_model->loss_plan);

  gather_obs(buf, buf->next_obs, idx, batch_size, q_target->batch_input->val);
  schedule_run(&q_target->train);

  const Mat *q = q_model->batch_logits->val;  // act_dim x batch_size
  const Mat *q_next = q_target->batch_logits->val;
  Mat *target = q_model->target->val;
  const u32 act_dim = q->rows;

  for (u32 b = 0; b < batch_size; b++) {
    const u32 slot = idx[b];

    real q_max = q_next->data[b]; // row 0 of column b
    for (u32 a = 1; a < act_dim; a++) {
      const real v = q_next->data[a * batch_size + b];
      if (v > q_max) {
        q_max = v;
      }
    }

    // Masked regression: untaken actions keep the network's own value (zero
    // error, zero gradient), the taken action gets the bootstrap target. A
    // truncated transition bootstraps normally; only a real terminal drops it.
    for (u32 a = 0; a < act_dim; a++) {
      target->data[a * batch_size + b] = q->data[a * batch_size + b];
    }
    target->data[buf->actions[slot] * batch_size + b] = dqn_td_target(
        buf->rewards[slot], q_max, config->gamma, buf->dones[slot] != 0u);
  }

  schedule_grad_from(&q_model->loss_plan, q_model->cost, true);

  if (config->max_grad_norm > 0.0f) {
    (void)clip_grad_norm(q_model, config->max_grad_norm);
  }
  if (adam != NULL) {
    adam->lr = lr_t;
    adam_step(adam, q_model);
  } else {
    sgd_step(q_model, lr_t, 1.0f);
  }
  zero_grad(q_model);
}

void train_dqn(Env *env, Net *q_model, Net *q_target, const TrainCfg *config,
               Arena *train_arena) {
  RL_LIB_ASSERT(env != NULL && q_model != NULL && q_target != NULL &&
                    config != NULL && train_arena != NULL,
                "train_dqn: NULL argument");
  RL_LIB_ASSERT(!env->continuous,
                "train_dqn: only discrete-action environments are supported");
  RL_LIB_ASSERT(q_model != q_target,
                "train_dqn: online and target network are the same Net");

  const u32 batch_size = (config->batch_size > 0u) ? config->batch_size : 64u;
  const u32 updates = (config->epochs > 0u) ? config->epochs : 1u;
  const u32 per_update =
      (config->rollout_size > 0u) ? config->rollout_size : 32u;
  const u32 warmup = (config->warmup_steps > 0u) ? config->warmup_steps : 1000u;
  const u32 train_every =
      (config->train_every > 0u) ? config->train_every : 1u;
  const u32 target_sync =
      (config->target_sync > 0u) ? config->target_sync : 500u;
  const u32 capacity =
      (config->replay_capacity > 0u) ? config->replay_capacity : 20000u;

  RL_LIB_ASSERT(net_set_batch_size(train_arena, q_model, batch_size),
                "train_dqn: could not set the online batch size");
  RL_LIB_ASSERT(net_set_batch_size(train_arena, q_target, batch_size),
                "train_dqn: could not set the target batch size");
  RL_LIB_ASSERT(q_model->target != NULL && q_model->cost != NULL,
                "train_dqn: q_model was not built by q_net_build");

  // Sampling needs a full minibatch even if warmup_steps is smaller.
  const u32 min_fill = (warmup > batch_size) ? warmup : batch_size;

  ReplayBuf *buf = replay_new(train_arena, capacity, env->obs_dim);
  Adam *adam = (config->optim == OPT_ADAM)
                   ? adam_new(train_arena, q_model, config->learning_rate)
                   : NULL;
  u32 *idx = ARENA_ARR(train_arena, u32, batch_size);
  real *obs = ARENA_ARR(train_arena, real, env->obs_dim);
  real *next_obs = ARENA_ARR(train_arena, real, env->obs_dim);

  Rng rng;
  const u64 seed = (config->seed != 0u) ? config->seed : 42u;
  rng_seed(&rng, seed, 3);

  dqn_sync_target(q_model, q_target);
  zero_grad(q_model);
  env->reset(env, seed);

  u32 env_steps = 0u;
  u32 grad_steps = 0u;
  real episode_return = 0.0f;
  real return_sum = 0.0f;
  u64 episode_count = 0u;
  u64 total_episodes = 0u;

  for (u32 u = 0; u < updates; u++) {
    const real lr_t = train_lr_at(config, config->learning_rate, u);

    for (u32 s = 0; s < per_update; s++) {
      // Epsilon-greedy action from the online network's rollout graph.
      env->observe(env, obs);
      for (u32 r = 0; r < env->obs_dim; r++) {
        q_model->input->val->data[r] = obs[r];
      }
      schedule_run(&q_model->fwd);

      const Mat *qv = q_model->output->val; // act_dim x 1
      u32 action = 0u;
      real best = qv->data[0];
      for (u32 a = 1; a < env->act_dim; a++) {
        if (qv->data[a] > best) {
          best = qv->data[a];
          action = a;
        }
      }
      if (rng_unit(&rng) < dqn_epsilon_at(config, env_steps)) {
        action = rng_u32(&rng) % env->act_dim;
      }

      b32 done = false;
      b32 truncated = false;
      const real reward = env->step(env, action, &done, &truncated);

      env->observe(env, next_obs);
      replay_add_ex(buf, obs, action, reward, next_obs, done, truncated);

      episode_return += reward;
      env_steps += 1u;

      if (done || truncated) {
        return_sum += episode_return;
        episode_count += 1u;
        total_episodes += 1u;
        episode_return = 0.0f;
        // Deterministic per-episode seed, kept separate from the sampling RNG
        // stream that draws epsilon and replay indices.
        env->reset(env, seed + total_episodes);
      }

      if (buf->size >= min_fill && (env_steps % train_every) == 0u) {
        dqn_grad_step(q_model, q_target, buf, adam, config, &rng, idx,
                      batch_size, lr_t);
        grad_steps += 1u;
        if ((grad_steps % target_sync) == 0u) {
          dqn_sync_target(q_model, q_target);
        }
      }
    }

    if (config->verbose || (u + 1u) % 50u == 0u || u == 0u ||
        u + 1u == updates) {
      const real mean_return =
          (episode_count > 0u) ? return_sum / (real)episode_count : 0.0f;
      printf("Update %4u/%u | Mean Episode Return: %7.2f | Episodes: %3llu | "
             "Steps: %u\n",
             u + 1u, updates, (double)mean_return,
             (unsigned long long)episode_count, env_steps);
      return_sum = 0.0f;
      episode_count = 0u;
    }
  }
}
