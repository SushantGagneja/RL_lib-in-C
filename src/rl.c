#include "rl.h"
#include "layers.h"
#include "vecenv.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Rollout *rollout_new(Arena *arena, u32 num_trajectories,
                                u32 max_steps, u32 obs_dim) {
  RL_LIB_ASSERT(arena != NULL, "rollout_new: NULL arena");
  RL_LIB_ASSERT(num_trajectories > 0 && max_steps > 0 && obs_dim > 0,
             "rollout_new: zero-sized buffer");

  Rollout *buf = ARENA_OBJ(arena, Rollout);
  buf->count = num_trajectories;
  buf->max_steps = max_steps;
  buf->obs_dim = obs_dim;
  buf->trajectories = ARENA_ARR(arena, Episode, num_trajectories);

  for (u32 i = 0; i < num_trajectories; i++) {
    buf->trajectories[i].obs = ARENA_ARR(arena, real, (u64)max_steps * obs_dim);
    buf->trajectories[i].actions = ARENA_ARR(arena, u32, max_steps);
    buf->trajectories[i].rewards = ARENA_ARR(arena, real, max_steps);
    buf->trajectories[i].returns = ARENA_ARR(arena, real, max_steps);
    buf->trajectories[i].values = ARENA_ARR(arena, real, max_steps);
    buf->trajectories[i].advantages = ARENA_ARR(arena, real, max_steps);
    buf->trajectories[i].dones = ARENA_ARR(arena, b8, max_steps);
    buf->trajectories[i].truncateds = ARENA_ARR(arena, b8, max_steps);
    buf->trajectories[i].len = 0;
  }

  return buf;
}

// Derives a fresh 64-bit seed from a PRNG stream so each episode's env.reset()
// gets an independent but reproducible seed.
static u64 next_seed(Rng *rng) {
  u64 hi = rng_u32(rng);
  u64 lo = rng_u32(rng);
  return (hi << 32) ^ lo;
}

static u32 sample_action(Rng *rng, const Mat *probs) {  real r = rng_unit(rng);
  real cdf = 0.0f;
  u32 size = probs->rows * probs->cols;
  for (u32 i = 0; i < size; i++) {
    cdf += probs->data[i];
    // Strict `<` so r == 0 can never select a zero-probability action.
    if (r < cdf) {
      return i;
    }
  }
  return size - 1;
}

static u32 argmax_action(const Mat *probs) {
  u32 size = probs->rows * probs->cols;
  u32 best = 0;
  real max_p = probs->data[0];
  for (u32 i = 1; i < size; i++) {
    if (probs->data[i] > max_p) {
      max_p = probs->data[i];
      best = i;
    }
  }
  return best;
}

void gae_compute(Episode *traj, real gamma, real lambda, real last_val) {
  if (traj->len == 0) return;
  real gae = 0.0f;
  for (i32 t = (i32)traj->len - 1; t >= 0; t--) {
    real next_val = (t == (i32)traj->len - 1) ? last_val : traj->values[t + 1];
    real non_terminal = traj->dones[t] ? 0.0f : 1.0f;
    real delta =
        traj->rewards[t] + gamma * next_val * non_terminal - traj->values[t];
    gae = delta + gamma * lambda * non_terminal * gae;
    traj->advantages[t] = gae;
    traj->returns[t] = gae + traj->values[t];
  }
}

// Fisher-Yates shuffle of the sample index arrays so the batched loop's
// remainder (sample_count % batch_size) drops a random subset instead of the
// systematically-biased tail (samples are stored in reverse time order).
static void shuffle_samples(real **obs, u32 *actions, real *values, real *extra_b,
                            u32 *order, u32 n, Rng *rng) {
  for (u32 i = 0; i < n; i++) order[i] = i;
  for (u32 i = n; i > 1; i--) {
    u32 j = rng_below(rng, i);
    if (j != i - 1) {
      u32 tmp = order[i - 1];
      order[i - 1] = order[j];
      order[j] = tmp;
    }
  }

  // Apply to the parallel arrays via a scratch copy of each.
  real **obs_copy = malloc(sizeof(real *) * n);
  u32 *act_copy = malloc(sizeof(u32) * n);
  real *val_copy = malloc(sizeof(real) * n);
  real *b_copy = malloc(sizeof(real) * n);
  RL_LIB_ASSERT(obs_copy && act_copy && val_copy && b_copy,
             "shuffle_samples: allocation failure");

  memcpy(obs_copy, obs, sizeof(real *) * n);
  memcpy(act_copy, actions, sizeof(u32) * n);
  memcpy(val_copy, values, sizeof(real) * n);
  memcpy(b_copy, extra_b, sizeof(real) * n);

  for (u32 i = 0; i < n; i++) {
    obs[i] = obs_copy[order[i]];
    actions[i] = act_copy[order[i]];
    values[i] = val_copy[order[i]];
    extra_b[i] = b_copy[order[i]];
  }

  free(obs_copy);
  free(act_copy);
  free(val_copy);
  free(b_copy);
}

// Applies TrainCfg.batch_size to the model before anything is sized from the
// graph. A model built by actor_build / actor_critic_build can rebuild its
// batched subgraph for the requested minibatch size, which makes the batch size
// a runtime knob; a hand-built graph has no recorded architecture, so the size
// stays fixed and the assertions in the trainers report the mismatch.
static void apply_config_batch_size(Arena *arena, Net *model,
                                    const TrainCfg *config) {
  if (config->batch_size > 0) {
    (void)net_set_batch_size(arena, model, config->batch_size);
  }
}

void train_reinforce(Env *env, Net *model, const TrainCfg *config,
                        Arena *train_arena) {
  RL_LIB_ASSERT(env != NULL && model != NULL && config != NULL,
             "train_reinforce: NULL argument");
  RL_LIB_ASSERT(model->fwd.len > 0,
             "train_reinforce: forward_graph is empty (forgot schedule_from?)");
  RL_LIB_ASSERT(train_arena != NULL, "train_reinforce: NULL arena");
  apply_config_batch_size(train_arena, model, config);

  u32 rollout_size = config->rollout_size > 0 ? config->rollout_size : 32;
  u32 max_steps = config->max_steps > 0 ? config->max_steps : env->max_steps;
  real gamma = config->gamma > 0.0f ? config->gamma : 0.99f;
  real lr = config->learning_rate > 0.0f ? config->learning_rate : 0.01f;

  Rollout *buffer =
      rollout_new(train_arena, rollout_size, max_steps, env->obs_dim);

  Rng rng;
  rng_seed(&rng, config->seed != 0 ? config->seed : 42, 1);
  Rng shuf_rng;
  rng_seed(&shuf_rng, (config->seed != 0 ? config->seed : 42) ^ 0x9e3779b9ULL,
              2);

  Adam *adam = (config->optim == OPT_ADAM)
                        ? adam_new(train_arena, model, lr)
                        : NULL;

  b32 use_batching = (model->train.len > 0 && model->batch_input != NULL &&
                      model->batch_logits != NULL);
  u32 batch_size = use_batching ? model->batch_input->val->cols : 1;

  // The config batch size must agree with the shape frozen into the graph.
  RL_LIB_ASSERT(config->batch_size == 0 || !use_batching ||
                 config->batch_size == batch_size,
             "TrainCfg.batch_size does not match the graph's batch dimension");

  if (!use_batching) {
    RL_LIB_ASSERT(model->loss_plan.len > 0,
               "train_reinforce: no batch graph and no cost graph");
  }

  printf("Starting REINFORCE Training:\n");
  printf("  Obs Dim: %u | Act Dim: %u | Epochs: %u | Rollout: %u | Batch: %u | "
         "Optim: %s | LR: %.4f | Gamma: %.2f\n\n",
         env->obs_dim, env->act_dim, config->epochs, rollout_size, batch_size,
         config->optim == OPT_ADAM ? "Adam" : "SGD", lr, gamma);

  // Persistent buffers for batching (allocated once, not per epoch).
  u64 max_total_samples = (u64)rollout_size * max_steps;
  real **sample_obs = ARENA_ARR(train_arena, real *, max_total_samples);
  u32 *sample_actions = ARENA_ARR(train_arena, u32, max_total_samples);
  real *sample_values = ARENA_ARR(train_arena, real, max_total_samples);
  real *sample_returns = ARENA_ARR(train_arena, real, max_total_samples);
  u32 *order = ARENA_ARR(train_arena, u32, max_total_samples);
  u32 *batch_acts = ARENA_ARR(train_arena, u32, batch_size);
  real *batch_advs = ARENA_ARR(train_arena, real, batch_size);

  Mat *seed_grad = mat_new(train_arena, 1, 1);
  Mat *loss_out = mat_new(train_arena, 1, 1);

  for (u32 epoch = 0; epoch < config->epochs; epoch++) {
    real total_reward = 0.0f;
    u32 sample_count = 0;

    // 1. Rollout phase
    for (u32 i = 0; i < rollout_size; i++) {
      env->reset(env, next_seed(&rng));
      Episode *traj = &buffer->trajectories[i];
      traj->len = 0;

      for (u32 t = 0; t < max_steps; t++) {
        real *curr_obs = traj->obs + (u64)t * env->obs_dim;
        env->observe(env, curr_obs);

        memcpy(model->input->val->data, curr_obs, sizeof(real) * env->obs_dim);
        schedule_run(&model->fwd);

        u32 action = sample_action(&rng, model->output->val);

        b32 done = false;
        b32 truncated = false;
        real reward = env->step(env, action, &done, &truncated);

        traj->actions[t] = action;
        traj->rewards[t] = reward;
        traj->dones[t] = (b8)done;
        traj->truncateds[t] = (b8)truncated;
        traj->len = t + 1;
        total_reward += reward;

        if (done || truncated) {
          break;
        }
      }
    }

    // 2. Discounted Monte-Carlo returns. NOTE: a time-limit truncation is
    // treated as terminal here (G = 0), which biases the return for truncated
    // episodes. REINFORCE has no value head to bootstrap from; A2C does.
    for (u32 i = 0; i < rollout_size; i++) {
      Episode *traj = &buffer->trajectories[i];
      real G = 0.0f;

      for (i32 t = (i32)traj->len - 1; t >= 0; t--) {
        G = traj->rewards[t] + gamma * G;
        traj->returns[t] = G;

        sample_obs[sample_count] = traj->obs + (u64)t * env->obs_dim;
        sample_actions[sample_count] = traj->actions[t];
        sample_values[sample_count] = 0.0f;
        sample_returns[sample_count] = G;
        sample_count++;
      }
    }

    // 3. Batched training or legacy fallback
    zero_grad(model);

    // Baseline statistics over the whole rollout, used by both paths.
    real ret_sum = 0.0f;
    for (u32 i = 0; i < sample_count; i++) ret_sum += sample_returns[i];
    real ret_mean = ret_sum / (real)sample_count;
    real ret_sq = 0.0f;
    for (u32 i = 0; i < sample_count; i++) {
      real d = sample_returns[i] - ret_mean;
      ret_sq += d * d;
    }
    real ret_std = real_sqrt(ret_sq / (real)sample_count);

    if (use_batching && sample_count >= batch_size) {
      shuffle_samples(sample_obs, sample_actions, sample_values, sample_returns,
                      order, sample_count, &shuf_rng);

      u32 num_batches = sample_count / batch_size;
      seed_grad->data[0] = 1.0f / (real)num_batches;

      for (u32 k = 0; k < num_batches; k++) {
        for (u32 b = 0; b < batch_size; b++) {
          u32 idx = k * batch_size + b;
          real *obs = sample_obs[idx];
          for (u32 r = 0; r < env->obs_dim; r++) {
            model->batch_input->val->data[r * batch_size + b] = obs[r];
          }
          batch_acts[b] = sample_actions[idx];
          batch_advs[b] =
              (sample_returns[idx] - ret_mean) / (ret_std + 1e-8f);
        }

        schedule_run(&model->train);
        schedule_zero_act(&model->train);

        RL_LIB_CHECK(mat_fused_policy_loss(loss_out, model->batch_logits->val,
                                        batch_acts, batch_advs, batch_size));
        RL_LIB_CHECK(mat_fused_policy_add_grad(
            model->batch_logits->grad, model->batch_logits->val, batch_acts,
            batch_advs, batch_size, seed_grad));

        schedule_grad_accum(&model->train);
      }

      if (config->max_grad_norm > 0.0f) {
        clip_grad_norm(model, config->max_grad_norm);
      }
      if (config->optim == OPT_ADAM) {
        adam_step(adam, model);
      } else {
        sgd_step(model, lr, 1.0f);
      }
      zero_grad(model);
    } else {
      // Unbatched 1-by-1 fallback through the single-sample cost graph.
      for (u32 i = 0; i < sample_count; i++) {
        memcpy(model->input->val->data, sample_obs[i],
               sizeof(real) * env->obs_dim);
        mat_zero(model->advantage->val);
        real adv = (sample_returns[i] - ret_mean) / (ret_std + 1e-8f);
        model->advantage->val->data[sample_actions[i]] = adv;

        schedule_run(&model->loss_plan);
        schedule_grad(&model->loss_plan);
      }

      if (sample_count > 0) {
        if (config->max_grad_norm > 0.0f) {
          clip_grad_norm(model, config->max_grad_norm);
        }
        if (config->optim == OPT_ADAM) {
          adam_step(adam, model);
        } else {
          sgd_step(model, lr, 1.0f / (real)sample_count);
        }
      }
    }

    // Logging
    if (config->verbose || (epoch + 1) % 50 == 0 || epoch == 0 ||
        epoch + 1 == config->epochs) {
      printf("Epoch %4u/%u | Mean Return: %7.2f | Samples: %5u\n", epoch + 1,
             config->epochs, total_reward / (real)rollout_size, sample_count);
    }
  }
}

void train_a2c(Env *env, Net *model, const TrainCfg *config,
                  Arena *train_arena) {
  RL_LIB_ASSERT(env != NULL && model != NULL && config != NULL,
             "train_a2c: NULL argument");
  RL_LIB_ASSERT(train_arena != NULL, "train_a2c: NULL arena");
  // Size the batched graph (if the model supports it) before checking that it
  // exists, so a model built with batch_size = 0 can still be trained in batches.
  apply_config_batch_size(train_arena, model, config);
  RL_LIB_ASSERT(model->fwd.len > 0 && model->train.len > 0,
             "train_a2c: missing graph (use actor_critic_build)");
  RL_LIB_ASSERT(model->value != NULL && model->batch_value != NULL,
             "train_a2c: model has no value head");

  u32 rollout_size = config->rollout_size > 0 ? config->rollout_size : 32;
  u32 max_steps = config->max_steps > 0 ? config->max_steps : env->max_steps;
  real gamma = config->gamma > 0.0f ? config->gamma : 0.99f;
  real lambda = config->lambda > 0.0f ? config->lambda : 0.95f;
  real vf_coef = config->vf_coef > 0.0f ? config->vf_coef : 0.5f;
  real entropy_coef = config->entropy_coef;
  real lr = config->learning_rate > 0.0f ? config->learning_rate : 0.001f;

  Rollout *buffer =
      rollout_new(train_arena, rollout_size, max_steps, env->obs_dim);

  Rng rng;
  rng_seed(&rng, config->seed != 0 ? config->seed : 42, 1);
  Rng shuf_rng;
  rng_seed(&shuf_rng, (config->seed != 0 ? config->seed : 42) ^ 0x9e3779b9ULL,
              3);

  Adam *adam = (config->optim == OPT_ADAM)
                        ? adam_new(train_arena, model, lr)
                        : NULL;

  u32 batch_size = model->batch_input->val->cols;
  RL_LIB_ASSERT(config->batch_size == 0 || config->batch_size == batch_size,
             "TrainCfg.batch_size does not match the graph's batch dimension");

  printf("Starting A2C Training:\n");
  printf("  Obs Dim: %u | Act Dim: %u | Epochs: %u | Rollout: %u | Batch: %u | "
         "LR: %.4f | Gamma: %.2f | Lambda: %.2f | Entropy: %.4f\n\n",
         env->obs_dim, env->act_dim, config->epochs, rollout_size, batch_size,
         lr, gamma, lambda, entropy_coef);

  u64 max_total_samples = (u64)rollout_size * max_steps;
  real **sample_obs = ARENA_ARR(train_arena, real *, max_total_samples);
  u32 *sample_actions = ARENA_ARR(train_arena, u32, max_total_samples);
  real *sample_values = ARENA_ARR(train_arena, real, max_total_samples);
  real *sample_advantages = ARENA_ARR(train_arena, real, max_total_samples);
  u32 *order = ARENA_ARR(train_arena, u32, max_total_samples);

  u32 *batch_acts = ARENA_ARR(train_arena, u32, batch_size);
  real *batch_advs = ARENA_ARR(train_arena, real, batch_size);
  Mat *target_val_mat = mat_new(train_arena, 1, batch_size);

  Mat *seed_pi = mat_new(train_arena, 1, 1);
  Mat *seed_vf = mat_new(train_arena, 1, 1);
  Mat *seed_ent = mat_new(train_arena, 1, 1);

  real *last_obs = ARENA_ARR(train_arena, real, env->obs_dim);

  for (u32 epoch = 0; epoch < config->epochs; epoch++) {
    real total_reward = 0.0f;
    u32 sample_count = 0;

    // 1. Rollout collection with state values and GAE
    for (u32 i = 0; i < rollout_size; i++) {
      env->reset(env, next_seed(&rng));
      Episode *traj = &buffer->trajectories[i];
      traj->len = 0;
      b32 gae_done = false;

      for (u32 t = 0; t < max_steps; t++) {
        real *curr_obs = traj->obs + (u64)t * env->obs_dim;
        env->observe(env, curr_obs);

        memcpy(model->input->val->data, curr_obs, sizeof(real) * env->obs_dim);
        schedule_run(&model->fwd);

        u32 action = sample_action(&rng, model->output->val);
        real val = model->value->val->data[0];

        b32 done = false;
        b32 truncated = false;
        real reward = env->step(env, action, &done, &truncated);

        traj->actions[t] = action;
        traj->rewards[t] = reward;
        traj->values[t] = val;
        traj->dones[t] = (b8)done;
        traj->truncateds[t] = (b8)truncated;
        traj->len = t + 1;
        total_reward += reward;

        if (done || truncated) {
          real last_val = 0.0f;
          if (truncated && !done) {
            env->observe(env, last_obs);
            memcpy(model->input->val->data, last_obs,
                   sizeof(real) * env->obs_dim);
            schedule_run(&model->fwd);
            last_val = model->value->val->data[0];
          }
          gae_compute(traj, gamma, lambda, last_val);
          gae_done = true;
          break;
        }
      }

      // If the inner loop hit config->max_steps before the env terminated or
      // truncated, GAE was never computed. Bootstrap from the current value.
      if (!gae_done) {
        real last_val = 0.0f;
        env->observe(env, last_obs);
        memcpy(model->input->val->data, last_obs, sizeof(real) * env->obs_dim);
        schedule_run(&model->fwd);
        last_val = model->value->val->data[0];
        gae_compute(traj, gamma, lambda, last_val);
      }

      for (u32 t = 0; t < traj->len; t++) {
        sample_obs[sample_count] = traj->obs + (u64)t * env->obs_dim;
        sample_actions[sample_count] = traj->actions[t];
        sample_values[sample_count] = traj->values[t];
        sample_advantages[sample_count] = traj->advantages[t];
        sample_count++;
      }
    }

    // 2. Normalize GAE advantages (over the whole rollout)
    real adv_sum = 0.0f;
    for (u32 i = 0; i < sample_count; i++) adv_sum += sample_advantages[i];
    real adv_mean = adv_sum / (real)sample_count;
    real adv_sq = 0.0f;
    for (u32 i = 0; i < sample_count; i++) {
      real d = sample_advantages[i] - adv_mean;
      adv_sq += d * d;
    }
    real adv_std = real_sqrt(adv_sq / (real)sample_count);
    for (u32 i = 0; i < sample_count; i++) {
      sample_advantages[i] = (sample_advantages[i] - adv_mean) / (adv_std + 1e-8f);
    }

    // 3. Collect minibatches, accumulate gradients, then one update per rollout
    // (keeps the data on-policy: a single step per batch of collected data).
    zero_grad(model);
    u32 num_batches = sample_count / batch_size;
    RL_LIB_ASSERT(num_batches > 0, "train_a2c: not enough samples for one batch");

    shuffle_samples(sample_obs, sample_actions, sample_values, sample_advantages,
                    order, sample_count, &shuf_rng);

    if (entropy_coef != 0.0f) {
      seed_ent->data[0] = -entropy_coef / (real)num_batches;
    } else {
      seed_ent->data[0] = 0.0f;
    }
    seed_pi->data[0] = 1.0f / (real)num_batches;
    seed_vf->data[0] = vf_coef / (real)num_batches;

    for (u32 k = 0; k < num_batches; k++) {
      for (u32 b = 0; b < batch_size; b++) {
        u32 idx = k * batch_size + b;
        real *obs = sample_obs[idx];
        for (u32 r = 0; r < env->obs_dim; r++) {
          model->batch_input->val->data[r * batch_size + b] = obs[r];
        }
        batch_acts[b] = sample_actions[idx];
        batch_advs[b] = sample_advantages[idx];
        target_val_mat->data[b] = sample_values[idx] + sample_advantages[idx];
      }

      schedule_run(&model->train);
      // Clear activation grads, THEN seed the policy and value nodes. The old
      // code seeded first and then cleared, wiping every seeded gradient.
      schedule_zero_act(&model->train);

      RL_LIB_CHECK(mat_fused_policy_add_grad(model->batch_logits->grad,
                                          model->batch_logits->val, batch_acts,
                                          batch_advs, batch_size, seed_pi));
      if (entropy_coef != 0.0f) {
        RL_LIB_CHECK(mat_entropy_add_grad(model->batch_logits->grad,
                                       model->batch_logits->val, batch_size,
                                       seed_ent));
      }
      RL_LIB_CHECK(mat_mse_add_grad(model->batch_value->grad,
                                 model->batch_value->val, target_val_mat,
                                 seed_vf));

      schedule_grad_accum(&model->train);
    }

    if (config->max_grad_norm > 0.0f) {
      clip_grad_norm(model, config->max_grad_norm);
    }
    if (config->optim == OPT_ADAM) {
      adam_step(adam, model);
    } else {
      sgd_step(model, lr, 1.0f);
    }
    zero_grad(model);

    if (config->verbose || (epoch + 1) % 50 == 0 || epoch == 0 ||
        epoch + 1 == config->epochs) {
      printf("Epoch %4u/%u | Mean Return: %7.2f | Samples: %5u\n", epoch + 1,
             config->epochs, total_reward / (real)rollout_size, sample_count);
    }
  }
}

// Runs one forward pass on a single observation and returns the critic value.
// Clobbers the forward graph's activations, so call it between rollouts and
// before the training loop starts.
static real critic_value(Net *model, const real *obs, u32 obs_dim) {
  memcpy(model->input->val->data, obs, sizeof(real) * obs_dim);
  schedule_run(&model->fwd);
  return model->value->val->data[0];
}

// Samples one action from already-computed policy probabilities and returns it.
// The chosen action's log-probability is written to *logp_out, floored at 1e-8
// so the PPO ratio stays finite.
static u32 sample_action_logp(Rng *rng, const Mat *probs, real *logp_out) {
  const u32 action = sample_action(rng, probs);
  real p = probs->data[action];
  if (p < 1e-8f) {
    p = 1e-8f;
  }
  *logp_out = real_log(p);
  return action;
}

// PPO with the clipped surrogate objective. The rollout is gathered from
// `config->num_envs` parallel environments, then replayed for
// `config->ppo_epochs` passes of shuffled minibatches. Unlike A2C (one update
// per rollout), PPO takes one optimiser step per minibatch, which is what lets
// it reuse the rollout data on-policy.
void train_ppo(Env *env, Net *model, const TrainCfg *config,
               Arena *train_arena) {
  RL_LIB_ASSERT(env != NULL && model != NULL && config != NULL,
                "train_ppo: NULL argument");
  RL_LIB_ASSERT(train_arena != NULL, "train_ppo: NULL arena");
  RL_LIB_ASSERT(env->clone != NULL, "train_ppo: env does not implement clone");
  // Size the batched graph (if the model supports it) before checking that it
  // exists, so a model built with batch_size = 0 can still be trained in batches.
  apply_config_batch_size(train_arena, model, config);
  RL_LIB_ASSERT(model->fwd.len > 0 && model->train.len > 0,
                "train_ppo: missing graph (use actor_critic_build)");
  RL_LIB_ASSERT(model->value != NULL && model->batch_value != NULL,
                "train_ppo: model has no value head");

  const u32 num_envs = config->num_envs > 0 ? config->num_envs : 1;
  const u32 rollout_len = config->rollout_size > 0 ? config->rollout_size : 32;
  const real gamma = config->gamma > 0.0f ? config->gamma : 0.99f;
  const real lambda = config->lambda > 0.0f ? config->lambda : 0.95f;
  const real vf_coef = config->vf_coef > 0.0f ? config->vf_coef : 0.5f;
  const real entropy_coef = config->entropy_coef;
  const real lr = config->learning_rate > 0.0f ? config->learning_rate : 3e-4f;
  const real clip_eps = config->ppo_clip > 0.0f ? config->ppo_clip : 0.2f;
  const u32 ppo_epochs = config->ppo_epochs > 0 ? config->ppo_epochs : 10;

  const u32 obs_dim = env->obs_dim;
  const u32 batch_size = model->batch_input->val->cols;
  RL_LIB_ASSERT(config->batch_size == 0 || config->batch_size == batch_size,
                "TrainCfg.batch_size does not match the graph's batch dimension");

  const u32 total = num_envs * rollout_len;
  RL_LIB_ASSERT(total >= batch_size,
                "train_ppo: rollout smaller than one minibatch");

  const u64 seed = config->seed != 0 ? config->seed : 42;
  VecEnv *vec = vecenv_new(train_arena, env, num_envs);
  vecenv_reset(vec, seed);
  vecenv_set_threads(train_arena, vec, config->num_threads);

  Rng rng;
  rng_seed(&rng, seed, 1);
  Rng shuf_rng;
  rng_seed(&shuf_rng, seed ^ 0x9e3779b9ULL, 3);

  Adam *adam =
      (config->optim == OPT_ADAM) ? adam_new(train_arena, model, lr) : NULL;

  printf("Starting PPO Training:\n");
  printf("  Obs Dim: %u | Act Dim: %u | Updates: %u | Envs: %u | Rollout: %u | "
         "Batch: %u | Clip: %.2f | PPO Epochs: %u | LR: %.4f\n\n",
         obs_dim, env->act_dim, config->epochs, num_envs, rollout_len,
         batch_size, clip_eps, ppo_epochs, lr);

  // Rollout buffers, laid out env-major: index i * rollout_len + t.
  real *obs_buf = ARENA_ARR(train_arena, real, (u64)total * obs_dim);
  u32 *acts = ARENA_ARR(train_arena, u32, total);
  real *logp_old = ARENA_ARR(train_arena, real, total);
  real *rew_buf = ARENA_ARR(train_arena, real, total);
  real *val_buf = ARENA_ARR(train_arena, real, total);
  real *adv_buf = ARENA_ARR(train_arena, real, total);
  real *ret_buf = ARENA_ARR(train_arena, real, total);
  b8 *end_flag = ARENA_ARR(train_arena, b8, total);
  b8 *term_flag = ARENA_ARR(train_arena, b8, total);
  real *boot_val = ARENA_ARR(train_arena, real, total);
  real *tail_val = ARENA_ARR(train_arena, real, num_envs);
  real *ep_ret = ARENA_ARR(train_arena, real, num_envs);

  u32 *step_actions = ARENA_ARR(train_arena, u32, num_envs);
  u32 *order = ARENA_ARR(train_arena, u32, total);
  u32 *batch_acts = ARENA_ARR(train_arena, u32, batch_size);
  real *batch_advs = ARENA_ARR(train_arena, real, batch_size);
  real *batch_oldlp = ARENA_ARR(train_arena, real, batch_size);
  Mat *target_val_mat = mat_new(train_arena, 1, batch_size);
  Mat *seed_pi = mat_new(train_arena, 1, 1);
  Mat *seed_vf = mat_new(train_arena, 1, 1);
  Mat *seed_ent = mat_new(train_arena, 1, 1);

  for (u32 i = 0; i < num_envs; i++) {
    ep_ret[i] = 0.0f;
  }

  for (u32 update = 0; update < config->epochs; update++) {
    real episode_return_sum = 0.0f;
    u64 episode_count = 0;

    // 1. Collect `rollout_len` steps from every sub-environment.
    for (u32 t = 0; t < rollout_len; t++) {
      for (u32 i = 0; i < num_envs; i++) {
        const real *obs = vec->obs + (u64)i * obs_dim;
        memcpy(model->input->val->data, obs, sizeof(real) * obs_dim);
        schedule_run(&model->fwd);

        real logp = 0.0f;
        const u32 action = sample_action_logp(&rng, model->output->val, &logp);

        const u64 idx = (u64)i * rollout_len + t;
        memcpy(obs_buf + idx * obs_dim, obs, sizeof(real) * obs_dim);
        acts[idx] = action;
        logp_old[idx] = logp;
        val_buf[idx] = model->value->val->data[0];
        step_actions[i] = action;
      }

      vecenv_step(vec, step_actions);

      for (u32 i = 0; i < num_envs; i++) {
        const u64 idx = (u64)i * rollout_len + t;
        rew_buf[idx] = vec->rewards[i];
        end_flag[idx] = vec->dones[i];
        term_flag[idx] = vec->terminals[i];
        boot_val[idx] = 0.0f;

        ep_ret[i] += vec->rewards[i];
        if (vec->dones[i]) {
          episode_return_sum += ep_ret[i];
          episode_count++;
          ep_ret[i] = 0.0f;

          // Time-limit truncation: bootstrap from the value of the final state.
          if (vec->truncateds[i]) {
            boot_val[idx] =
                critic_value(model, vec->final_obs + (u64)i * obs_dim, obs_dim);
          }
        }
      }
    }

    // Value of each env's current (post-rollout) state, for the last-step
    // bootstrap of episodes that did not end inside the rollout.
    for (u32 i = 0; i < num_envs; i++) {
      tail_val[i] = critic_value(model, vec->obs + (u64)i * obs_dim, obs_dim);
    }

    // 2. GAE-Lambda per sub-environment (buffers are env-major).
    for (u32 i = 0; i < num_envs; i++) {
      real adv_next = 0.0f;
      for (u32 step = 0; step < rollout_len; step++) {
        const u32 t = rollout_len - 1 - step;
        const u64 idx = (u64)i * rollout_len + t;

        real v_next;
        if (term_flag[idx]) {
          v_next = 0.0f;
        } else if (end_flag[idx]) {
          v_next = boot_val[idx]; // truncated: value of the final observation
        } else if (t + 1 < rollout_len) {
          v_next = val_buf[idx + 1];
        } else {
          v_next = tail_val[i];
        }

        const real non_terminal = term_flag[idx] ? 0.0f : 1.0f;
        const real delta =
            rew_buf[idx] + gamma * v_next * non_terminal - val_buf[idx];
        // The episode continues into step t+1 only if it did not end at t.
        const real cont = end_flag[idx] ? 0.0f : 1.0f;
        const real adv = delta + gamma * lambda * cont * adv_next;
        adv_buf[idx] = adv;
        ret_buf[idx] = adv + val_buf[idx];
        adv_next = adv;
      }
    }

    // 3. Normalize advantages over the whole rollout.
    real adv_sum = 0.0f;
    for (u32 i = 0; i < total; i++) {
      adv_sum += adv_buf[i];
    }
    const real adv_mean = adv_sum / (real)total;
    real adv_sq = 0.0f;
    for (u32 i = 0; i < total; i++) {
      const real d = adv_buf[i] - adv_mean;
      adv_sq += d * d;
    }
    const real adv_std = real_sqrt(adv_sq / (real)total);
    for (u32 i = 0; i < total; i++) {
      adv_buf[i] = (adv_buf[i] - adv_mean) / (adv_std + 1e-8f);
    }

    // 4. PPO update: `ppo_epochs` passes of shuffled minibatches, one optimiser
    // step per minibatch.
    const u32 num_batches = total / batch_size;
    zero_grad(model);

    for (u32 pe = 0; pe < ppo_epochs; pe++) {
      for (u32 i = 0; i < total; i++) {
        order[i] = i;
      }
      for (u32 i = total; i > 1; i--) {
        const u32 j = rng_below(&shuf_rng, i);
        const u32 tmp = order[i - 1];
        order[i - 1] = order[j];
        order[j] = tmp;
      }

      for (u32 k = 0; k < num_batches; k++) {
        for (u32 b = 0; b < batch_size; b++) {
          const u32 idx = order[k * batch_size + b];
          const real *obs = obs_buf + (u64)idx * obs_dim;
          for (u32 r = 0; r < obs_dim; r++) {
            model->batch_input->val->data[r * batch_size + b] = obs[r];
          }
          batch_acts[b] = acts[idx];
          batch_advs[b] = adv_buf[idx];
          batch_oldlp[b] = logp_old[idx];
          target_val_mat->data[b] = ret_buf[idx];
        }

        schedule_run(&model->train);
        schedule_zero_act(&model->train);

        // The clipped surrogate is already a mean over the minibatch, so its
        // gradient seed is 1; the value and entropy terms are weighted.
        seed_pi->data[0] = 1.0f;
        RL_LIB_CHECK(mat_ppo_policy_add_grad(
            model->batch_logits->grad, model->batch_logits->val, batch_acts,
            batch_advs, batch_oldlp, clip_eps, batch_size, seed_pi));

        if (entropy_coef != 0.0f) {
          seed_ent->data[0] = -entropy_coef;
          RL_LIB_CHECK(mat_entropy_add_grad(model->batch_logits->grad,
                                            model->batch_logits->val,
                                            batch_size, seed_ent));
        }

        seed_vf->data[0] = vf_coef;
        RL_LIB_CHECK(mat_mse_add_grad(model->batch_value->grad,
                                      model->batch_value->val, target_val_mat,
                                      seed_vf));

        schedule_grad_accum(&model->train);

        if (config->max_grad_norm > 0.0f) {
          clip_grad_norm(model, config->max_grad_norm);
        }
        if (config->optim == OPT_ADAM) {
          adam_step(adam, model);
        } else {
          sgd_step(model, lr, 1.0f);
        }
        zero_grad(model);
      }
    }

    if (config->verbose || (update + 1) % 50 == 0 || update == 0 ||
        update + 1 == config->epochs) {
      const real mean_return =
          episode_count > 0 ? episode_return_sum / (real)episode_count : 0.0f;
      printf("Update %4u/%u | Mean Episode Return: %7.2f | Episodes: %3llu | "
             "Steps: %u\n",
             update + 1, config->epochs, mean_return,
             (unsigned long long)episode_count, total);
    }
  }

  vecenv_shutdown(vec);
}

EvalResult evaluate_seeded(Env *env, Net *model, const u64 *seeds,
                                u32 num_episodes, b32 greedy) {
  RL_LIB_ASSERT(env != NULL && model != NULL && seeds != NULL && num_episodes > 0,
             "evaluate_seeded: bad argument");

  real total_reward = 0.0f;
  u64 total_steps = 0;
  real *obs_buf = malloc(sizeof(real) * env->obs_dim);
  RL_LIB_ASSERT(obs_buf != NULL, "evaluate_seeded: allocation failure");

  for (u32 ep = 0; ep < num_episodes; ep++) {
    env->reset(env, seeds[ep]);
    Rng ep_rng;
    rng_seed(&ep_rng, seeds[ep] ^ 0x5851f42d4c957f2dULL, ep + 1);
    real ep_reward = 0.0f;

    for (u32 t = 0; t < env->max_steps; t++) {
      env->observe(env, obs_buf);
      memcpy(model->input->val->data, obs_buf, sizeof(real) * env->obs_dim);
      schedule_run(&model->fwd);

      u32 action = greedy ? argmax_action(model->output->val)
                          : sample_action(&ep_rng, model->output->val);

      b32 done = false;
      b32 truncated = false;
      real r = env->step(env, action, &done, &truncated);
      ep_reward += r;
      total_steps++;

      if (done || truncated) {
        break;
      }
    }

    total_reward += ep_reward;
  }

  free(obs_buf);
  return (EvalResult){
      .avg_return = total_reward / (real)num_episodes,
      .avg_steps = (real)total_steps / (real)num_episodes,
  };
}

EvalResult evaluate(Env *env, Net *model, Rng *rng,
                         u32 episodes, b32 greedy) {
  RL_LIB_ASSERT(env != NULL && model != NULL && rng != NULL && episodes > 0,
             "evaluate: bad argument");

  u64 *seeds = malloc(sizeof(u64) * episodes);
  RL_LIB_ASSERT(seeds != NULL, "evaluate: allocation failure");
  for (u32 i = 0; i < episodes; i++) {
    seeds[i] = next_seed(rng);
  }

  EvalResult res = evaluate_seeded(env, model, seeds, episodes, greedy);
  free(seeds);
  return res;
}

// ---------------------------------------------------------------------------
// Continuous-action PPO
// ---------------------------------------------------------------------------

// Deterministic action of the continuous policy: the pre-squash mean pushed
// through the same tanh squash the sampler uses (tanh is monotone, so this is
// the mode of the squashed density).
static void continuous_greedy_action(const Env *env, const Mat *raw_mean,
                                     real *action_out) {
  for (u32 d = 0; d < env->act_dim; d++) {
    const real center =
        (real)0.5 * (env->action_min[d] + env->action_max[d]);
    const real half = (real)0.5 * (env->action_max[d] - env->action_min[d]);
    // Same mean clamp the sampler applies; without it a greedy action could
    // differ from the mode of the very density the policy is trained on.
    real m = raw_mean->data[d];
    if (m > RL_GAUSSIAN_MEAN_BOUND) {
      m = RL_GAUSSIAN_MEAN_BOUND;
    }
    if (m < -RL_GAUSSIAN_MEAN_BOUND) {
      m = -RL_GAUSSIAN_MEAN_BOUND;
    }
    action_out[d] = center + half * real_tanh(m);
  }
}

// Same shape as train_ppo, with a diagonal-Gaussian policy instead of a
// categorical one. The commented steps mirror that trainer exactly; only the
// action sampling, the stored actions and the policy-gradient term differ.
void train_ppo_continuous(Env *env, Net *model, const TrainCfg *config,
                          Arena *train_arena) {
  RL_LIB_ASSERT(env != NULL && model != NULL && config != NULL,
                "train_ppo_continuous: NULL argument");
  RL_LIB_ASSERT(train_arena != NULL, "train_ppo_continuous: NULL arena");
  RL_LIB_ASSERT(env->continuous && env->step_continuous != NULL,
                "train_ppo_continuous: env is not continuous");
  RL_LIB_ASSERT(env->clone != NULL,
                "train_ppo_continuous: env does not implement clone");
  RL_LIB_ASSERT(env->action_min != NULL && env->action_max != NULL,
                "train_ppo_continuous: env has no action bounds");
  apply_config_batch_size(train_arena, model, config);
  RL_LIB_ASSERT(model->fwd.len > 0 && model->train.len > 0,
                "train_ppo_continuous: missing graph (use "
                "continuous_actor_critic_build)");
  RL_LIB_ASSERT(model->mean != NULL && model->batch_mean != NULL,
                "train_ppo_continuous: model has no mean head");
  RL_LIB_ASSERT(model->value != NULL && model->batch_value != NULL,
                "train_ppo_continuous: model has no value head");
  RL_LIB_ASSERT(model->log_std != NULL,
                "train_ppo_continuous: model has no log_std parameter");

  const u32 num_envs = config->num_envs > 0 ? config->num_envs : 1;
  const u32 rollout_len = config->rollout_size > 0 ? config->rollout_size : 32;
  const real gamma = config->gamma > 0.0f ? config->gamma : 0.99f;
  const real lambda = config->lambda > 0.0f ? config->lambda : 0.95f;
  const real vf_coef = config->vf_coef > 0.0f ? config->vf_coef : 0.5f;
  const real entropy_coef = config->entropy_coef;
  const real lr = config->learning_rate > 0.0f ? config->learning_rate : 3e-4f;
  const real clip_eps = config->ppo_clip > 0.0f ? config->ppo_clip : 0.2f;
  const u32 ppo_epochs = config->ppo_epochs > 0 ? config->ppo_epochs : 10;

  const u32 obs_dim = env->obs_dim;
  const u32 act_dim = env->act_dim;
  const u32 batch_size = model->batch_input->val->cols;
  RL_LIB_ASSERT(config->batch_size == 0 || config->batch_size == batch_size,
                "TrainCfg.batch_size does not match the graph's batch dimension");

  const u32 total = num_envs * rollout_len;
  RL_LIB_ASSERT(total >= batch_size,
                "train_ppo_continuous: rollout smaller than one minibatch");

  const u64 seed = config->seed != 0 ? config->seed : 42;
  VecEnv *vec = vecenv_new(train_arena, env, num_envs);
  vecenv_reset(vec, seed);
  vecenv_set_threads(train_arena, vec, config->num_threads);

  Rng rng;
  rng_seed(&rng, seed, 1);
  Rng shuf_rng;
  rng_seed(&shuf_rng, seed ^ 0x9e3779b9ULL, 3);

  Adam *adam =
      (config->optim == OPT_ADAM) ? adam_new(train_arena, model, lr) : NULL;

  printf("Starting continuous PPO Training:\n");
  printf("  Obs Dim: %u | Act Dim: %u | Updates: %u | Envs: %u | Rollout: %u | "
         "Batch: %u | Clip: %.2f | PPO Epochs: %u | LR: %.4f\n\n",
         obs_dim, act_dim, config->epochs, num_envs, rollout_len, batch_size,
         clip_eps, ppo_epochs, lr);

  // Rollout buffers, laid out env-major: index i * rollout_len + t. Policy
  // samples are act_dim reals per step: the pre-squash coordinate u is what the
  // surrogate needs, and it cannot be recovered from a saturated action.
  real *obs_buf = ARENA_ARR(train_arena, real, (u64)total * obs_dim);
  real *logp_old = ARENA_ARR(train_arena, real, total);
  real *rew_buf = ARENA_ARR(train_arena, real, total);
  real *val_buf = ARENA_ARR(train_arena, real, total);
  real *adv_buf = ARENA_ARR(train_arena, real, total);
  real *ret_buf = ARENA_ARR(train_arena, real, total);
  b8 *end_flag = ARENA_ARR(train_arena, b8, total);
  b8 *term_flag = ARENA_ARR(train_arena, b8, total);
  real *boot_val = ARENA_ARR(train_arena, real, total);
  real *tail_val = ARENA_ARR(train_arena, real, num_envs);
  real *ep_ret = ARENA_ARR(train_arena, real, num_envs);

  real *step_actions = ARENA_ARR(train_arena, real, (u64)num_envs * act_dim);
  u32 *order = ARENA_ARR(train_arena, u32, total);
  Mat *batch_us = mat_new(train_arena, act_dim, batch_size);
  real *batch_advs = ARENA_ARR(train_arena, real, batch_size);
  real *batch_oldlp = ARENA_ARR(train_arena, real, batch_size);
  Mat *target_val_mat = mat_new(train_arena, 1, batch_size);
  Mat *sample_action_mat = mat_new(train_arena, act_dim, 1);
  Mat *sample_u_mat = mat_new(train_arena, act_dim, 1);
  real *us = ARENA_ARR(train_arena, real, (u64)total * act_dim);
  Mat *logp_mat = mat_new(train_arena, 1, 1);
  Mat *seed_pi = mat_new(train_arena, 1, 1);
  Mat *seed_vf = mat_new(train_arena, 1, 1);
  Mat *seed_ent = mat_new(train_arena, 1, 1);

  for (u32 i = 0; i < num_envs; i++) {
    ep_ret[i] = 0.0f;
  }

  for (u32 update = 0; update < config->epochs; update++) {
    real episode_return_sum = 0.0f;
    u64 episode_count = 0;

    // 1. Collect `rollout_len` steps from every sub-environment.
    for (u32 t = 0; t < rollout_len; t++) {
      for (u32 i = 0; i < num_envs; i++) {
        const real *obs = vec->obs + (u64)i * obs_dim;
        memcpy(model->input->val->data, obs, sizeof(real) * obs_dim);
        schedule_run(&model->fwd);

        RL_LIB_CHECK(mat_gaussian_sample(sample_action_mat, sample_u_mat,
                                         model->mean->val,
                                         model->log_std->val, env->action_min,
                                         env->action_max, &rng));
        RL_LIB_CHECK(mat_gaussian_logp(logp_mat, model->mean->val,
                                      model->log_std->val, sample_u_mat,
                                      env->action_min, env->action_max, 1u));

        const u64 idx = (u64)i * rollout_len + t;
        memcpy(obs_buf + idx * obs_dim, obs, sizeof(real) * obs_dim);
        for (u32 d = 0; d < act_dim; d++) {
          us[idx * act_dim + d] = sample_u_mat->data[d];
          step_actions[(u64)i * act_dim + d] = sample_action_mat->data[d];
        }
        logp_old[idx] = logp_mat->data[0];
        val_buf[idx] = model->value->val->data[0];
      }

      vecenv_step_continuous(vec, step_actions);

      for (u32 i = 0; i < num_envs; i++) {
        const u64 idx = (u64)i * rollout_len + t;
        rew_buf[idx] = vec->rewards[i];
        end_flag[idx] = vec->dones[i];
        term_flag[idx] = vec->terminals[i];
        boot_val[idx] = 0.0f;

        ep_ret[i] += vec->rewards[i];
        if (vec->dones[i]) {
          episode_return_sum += ep_ret[i];
          episode_count++;
          ep_ret[i] = 0.0f;

          // Time-limit truncation: bootstrap from the value of the final state.
          if (vec->truncateds[i]) {
            boot_val[idx] =
                critic_value(model, vec->final_obs + (u64)i * obs_dim, obs_dim);
          }
        }
      }
    }

    // Value of each env's current (post-rollout) state, for the last-step
    // bootstrap of episodes that did not end inside the rollout.
    for (u32 i = 0; i < num_envs; i++) {
      tail_val[i] = critic_value(model, vec->obs + (u64)i * obs_dim, obs_dim);
    }

    // 2. GAE-Lambda per sub-environment (buffers are env-major).
    for (u32 i = 0; i < num_envs; i++) {
      real adv_next = 0.0f;
      for (u32 step = 0; step < rollout_len; step++) {
        const u32 t = rollout_len - 1 - step;
        const u64 idx = (u64)i * rollout_len + t;

        real v_next;
        if (term_flag[idx]) {
          v_next = 0.0f;
        } else if (end_flag[idx]) {
          v_next = boot_val[idx]; // truncated: value of the final observation
        } else if (t + 1 < rollout_len) {
          v_next = val_buf[idx + 1];
        } else {
          v_next = tail_val[i];
        }

        const real non_terminal = term_flag[idx] ? 0.0f : 1.0f;
        const real delta =
            rew_buf[idx] + gamma * v_next * non_terminal - val_buf[idx];
        const real cont = end_flag[idx] ? 0.0f : 1.0f;
        const real adv = delta + gamma * lambda * cont * adv_next;
        adv_buf[idx] = adv;
        ret_buf[idx] = adv + val_buf[idx];
        adv_next = adv;
      }
    }

    // 3. Normalize advantages over the whole rollout.
    real adv_sum = 0.0f;
    for (u32 i = 0; i < total; i++) {
      adv_sum += adv_buf[i];
    }
    const real adv_mean = adv_sum / (real)total;
    real adv_sq = 0.0f;
    for (u32 i = 0; i < total; i++) {
      const real d = adv_buf[i] - adv_mean;
      adv_sq += d * d;
    }
    const real adv_std = real_sqrt(adv_sq / (real)total);
    for (u32 i = 0; i < total; i++) {
      adv_buf[i] = (adv_buf[i] - adv_mean) / (adv_std + 1e-8f);
    }

    // 4. PPO update: `ppo_epochs` passes of shuffled minibatches, one optimiser
    // step per minibatch.
    const u32 num_batches = total / batch_size;
    zero_grad(model);

    for (u32 pe = 0; pe < ppo_epochs; pe++) {
      for (u32 i = 0; i < total; i++) {
        order[i] = i;
      }
      for (u32 i = total; i > 1; i--) {
        const u32 j = rng_below(&shuf_rng, i);
        const u32 tmp = order[i - 1];
        order[i - 1] = order[j];
        order[j] = tmp;
      }

      for (u32 k = 0; k < num_batches; k++) {
        for (u32 b = 0; b < batch_size; b++) {
          const u32 idx = order[k * batch_size + b];
          const real *obs = obs_buf + (u64)idx * obs_dim;
          for (u32 r = 0; r < obs_dim; r++) {
            model->batch_input->val->data[r * batch_size + b] = obs[r];
          }
          for (u32 d = 0; d < act_dim; d++) {
            batch_us->data[(u64)d * batch_size + b] = us[idx * act_dim + d];
          }
          batch_advs[b] = adv_buf[idx];
          batch_oldlp[b] = logp_old[idx];
          target_val_mat->data[b] = ret_buf[idx];
        }

        schedule_run(&model->train);
        schedule_zero_act(&model->train);

        // The Gaussian surrogate is already a mean over the minibatch, so its
        // seed is 1; the value and entropy terms are weighted. The policy
        // gradient lands on the batch mean (an activation) and directly on the
        // log_std parameter, which no schedule edge reaches.
        seed_pi->data[0] = 1.0f;
        RL_LIB_CHECK(mat_gaussian_ppo_add_grad(
            model->batch_mean->grad, model->log_std->grad,
            model->batch_mean->val, model->log_std->val, batch_us, batch_advs,
            batch_oldlp, env->action_min, env->action_max, clip_eps, batch_size,
            seed_pi));

        if (entropy_coef != 0.0f) {
          seed_ent->data[0] = -entropy_coef;
          RL_LIB_CHECK(mat_gaussian_entropy_add_grad(model->log_std->grad,
                                                    model->log_std->val,
                                                    seed_ent));
        }

        seed_vf->data[0] = vf_coef;
        RL_LIB_CHECK(mat_mse_add_grad(model->batch_value->grad,
                                      model->batch_value->val, target_val_mat,
                                      seed_vf));

        schedule_grad_accum(&model->train);

        if (config->max_grad_norm > 0.0f) {
          clip_grad_norm(model, config->max_grad_norm);
        }
        if (config->optim == OPT_ADAM) {
          adam_step(adam, model);
        } else {
          sgd_step(model, lr, 1.0f);
        }
        zero_grad(model);
      }
    }

    if (config->verbose || (update + 1) % 50 == 0 || update == 0 ||
        update + 1 == config->epochs) {
      const real mean_return =
          episode_count > 0 ? episode_return_sum / (real)episode_count : 0.0f;
      printf("Update %4u/%u | Mean Episode Return: %7.2f | Episodes: %3llu | "
             "Steps: %u\n",
             update + 1, config->epochs, mean_return,
             (unsigned long long)episode_count, total);
    }
  }

  vecenv_shutdown(vec);
}

EvalResult evaluate_continuous_seeded(Env *env, Net *model, const u64 *seeds,
                                      u32 num_episodes, b32 deterministic) {
  RL_LIB_ASSERT(env != NULL && model != NULL && seeds != NULL &&
                    num_episodes > 0,
                "evaluate_continuous_seeded: bad argument");
  RL_LIB_ASSERT(env->continuous && env->step_continuous != NULL,
                "evaluate_continuous_seeded: env is not continuous");
  RL_LIB_ASSERT(model->mean != NULL && model->log_std != NULL,
                "evaluate_continuous_seeded: model has no continuous policy");
  RL_LIB_ASSERT(env->action_min != NULL && env->action_max != NULL,
                "evaluate_continuous_seeded: env has no action bounds");

  real total_reward = 0.0f;
  u64 total_steps = 0;
  real *obs_buf = malloc(sizeof(real) * env->obs_dim);
  real *act_buf = malloc(sizeof(real) * env->act_dim);
  real *u_buf = malloc(sizeof(real) * env->act_dim);
  RL_LIB_ASSERT(obs_buf != NULL && act_buf != NULL && u_buf != NULL,
                "evaluate_continuous_seeded: allocation failure");

  // Stack Mats wrapping the rollout buffers, so sampling needs no arena here.
  Mat action_mat = {
      .rows = env->act_dim,
      .cols = 1u,
      .data = act_buf,
  };
  Mat u_mat = {
      .rows = env->act_dim,
      .cols = 1u,
      .data = u_buf,
  };

  for (u32 ep = 0; ep < num_episodes; ep++) {
    env->reset(env, seeds[ep]);
    Rng ep_rng;
    rng_seed(&ep_rng, seeds[ep] ^ 0x5851f42d4c957f2dULL, ep + 1);
    real ep_reward = 0.0f;

    for (u32 t = 0; t < env->max_steps; t++) {
      env->observe(env, obs_buf);
      memcpy(model->input->val->data, obs_buf, sizeof(real) * env->obs_dim);
      schedule_run(&model->fwd);

      if (deterministic) {
        continuous_greedy_action(env, model->mean->val, act_buf);
      } else {
        RL_LIB_CHECK(mat_gaussian_sample(&action_mat, &u_mat, model->mean->val,
                                        model->log_std->val, env->action_min,
                                        env->action_max, &ep_rng));
      }

      b32 done = false;
      b32 truncated = false;
      const real r = env->step_continuous(env, act_buf, &done, &truncated);
      ep_reward += r;
      total_steps++;

      if (done || truncated) {
        break;
      }
    }

    total_reward += ep_reward;
  }

  free(u_buf);
  free(act_buf);
  free(obs_buf);
  return (EvalResult){
      .avg_return = total_reward / (real)num_episodes,
      .avg_steps = (real)total_steps / (real)num_episodes,
  };
}

EvalResult evaluate_continuous(Env *env, Net *model, Rng *rng, u32 episodes,
                               b32 deterministic) {
  RL_LIB_ASSERT(env != NULL && model != NULL && rng != NULL && episodes > 0,
                "evaluate_continuous: bad argument");

  u64 *seeds = malloc(sizeof(u64) * episodes);
  RL_LIB_ASSERT(seeds != NULL, "evaluate_continuous: allocation failure");
  for (u32 i = 0; i < episodes; i++) {
    seeds[i] = next_seed(rng);
  }

  EvalResult res =
      evaluate_continuous_seeded(env, model, seeds, episodes, deterministic);
  free(seeds);
  return res;
}
