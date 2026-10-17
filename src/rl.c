#include "rl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

RolloutBuffer *rl_buffer_create(mem_arena *arena, u32 num_trajectories,
                                u32 max_steps, u32 obs_dim) {
  CRL_ASSERT(arena != NULL, "rl_buffer_create: NULL arena");
  CRL_ASSERT(num_trajectories > 0 && max_steps > 0 && obs_dim > 0,
             "rl_buffer_create: zero-sized buffer");

  RolloutBuffer *buf = PUSH_STRUCT(arena, RolloutBuffer);
  buf->count = num_trajectories;
  buf->max_steps = max_steps;
  buf->obs_dim = obs_dim;
  buf->trajectories = PUSH_ARRAY(arena, RLTrajectory, num_trajectories);

  for (u32 i = 0; i < num_trajectories; i++) {
    buf->trajectories[i].obs = PUSH_ARRAY(arena, real, (u64)max_steps * obs_dim);
    buf->trajectories[i].actions = PUSH_ARRAY(arena, u32, max_steps);
    buf->trajectories[i].rewards = PUSH_ARRAY(arena, real, max_steps);
    buf->trajectories[i].returns = PUSH_ARRAY(arena, real, max_steps);
    buf->trajectories[i].values = PUSH_ARRAY(arena, real, max_steps);
    buf->trajectories[i].advantages = PUSH_ARRAY(arena, real, max_steps);
    buf->trajectories[i].dones = PUSH_ARRAY(arena, b8, max_steps);
    buf->trajectories[i].truncateds = PUSH_ARRAY(arena, b8, max_steps);
    buf->trajectories[i].len = 0;
  }

  return buf;
}

// Derives a fresh 64-bit seed from a PRNG stream so each episode's env.reset()
// gets an independent but reproducible seed.
static u64 next_seed(prng_state *rng) {
  u64 hi = prng_rand_r(rng);
  u64 lo = prng_rand_r(rng);
  return (hi << 32) ^ lo;
}

static u32 sample_action(prng_state *rng, const matrix *probs) {  real r = prng_randf_r(rng);
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

static u32 argmax_action(const matrix *probs) {
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

void rl_compute_gae(RLTrajectory *traj, real gamma, real lambda, real last_val) {
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
                            u32 *order, u32 n, prng_state *rng) {
  for (u32 i = 0; i < n; i++) order[i] = i;
  for (u32 i = n; i > 1; i--) {
    u32 j = prng_range_r(rng, i);
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
  CRL_ASSERT(obs_copy && act_copy && val_copy && b_copy,
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

void rl_train_reinforce(Env *env, model_state *model, const RLConfig *config,
                        mem_arena *train_arena) {
  CRL_ASSERT(env != NULL && model != NULL && config != NULL,
             "rl_train_reinforce: NULL argument");
  CRL_ASSERT(model->forward_graph.size > 0,
             "rl_train_reinforce: forward_graph is empty (forgot build_graph?)");
  CRL_ASSERT(train_arena != NULL, "rl_train_reinforce: NULL arena");

  u32 rollout_size = config->rollout_size > 0 ? config->rollout_size : 32;
  u32 max_steps = config->max_steps > 0 ? config->max_steps : env->max_steps;
  real gamma = config->gamma > 0.0f ? config->gamma : 0.99f;
  real lr = config->learning_rate > 0.0f ? config->learning_rate : 0.01f;

  RolloutBuffer *buffer =
      rl_buffer_create(train_arena, rollout_size, max_steps, env->obs_dim);

  prng_state rng;
  prng_seed_r(&rng, config->seed != 0 ? config->seed : 42, 1);
  prng_state shuf_rng;
  prng_seed_r(&shuf_rng, (config->seed != 0 ? config->seed : 42) ^ 0x9e3779b9ULL,
              2);

  AdamOptim *adam = (config->optim == OPTIM_ADAM)
                        ? optim_adam_create(train_arena, model, lr)
                        : NULL;

  b32 use_batching = (model->batch_graph.size > 0 && model->batch_input != NULL &&
                      model->batch_logits != NULL);
  u32 batch_size = use_batching ? model->batch_input->val->cols : 1;

  // The config batch size must agree with the shape frozen into the graph.
  CRL_ASSERT(config->batch_size == 0 || !use_batching ||
                 config->batch_size == batch_size,
             "RLConfig.batch_size does not match the graph's batch dimension");

  if (!use_batching) {
    CRL_ASSERT(model->cost_graph.size > 0,
               "rl_train_reinforce: no batch graph and no cost graph");
  }

  printf("Starting REINFORCE Training:\n");
  printf("  Obs Dim: %u | Act Dim: %u | Epochs: %u | Rollout: %u | Batch: %u | "
         "Optim: %s | LR: %.4f | Gamma: %.2f\n\n",
         env->obs_dim, env->act_dim, config->epochs, rollout_size, batch_size,
         config->optim == OPTIM_ADAM ? "Adam" : "SGD", lr, gamma);

  // Persistent buffers for batching (allocated once, not per epoch).
  u64 max_total_samples = (u64)rollout_size * max_steps;
  real **sample_obs = PUSH_ARRAY(train_arena, real *, max_total_samples);
  u32 *sample_actions = PUSH_ARRAY(train_arena, u32, max_total_samples);
  real *sample_values = PUSH_ARRAY(train_arena, real, max_total_samples);
  real *sample_returns = PUSH_ARRAY(train_arena, real, max_total_samples);
  u32 *order = PUSH_ARRAY(train_arena, u32, max_total_samples);
  u32 *batch_acts = PUSH_ARRAY(train_arena, u32, batch_size);
  real *batch_advs = PUSH_ARRAY(train_arena, real, batch_size);

  matrix *seed_grad = mat_create(train_arena, 1, 1);
  matrix *loss_out = mat_create(train_arena, 1, 1);

  for (u32 epoch = 0; epoch < config->epochs; epoch++) {
    real total_reward = 0.0f;
    u32 sample_count = 0;

    // 1. Rollout phase
    for (u32 i = 0; i < rollout_size; i++) {
      env->reset(env, next_seed(&rng));
      RLTrajectory *traj = &buffer->trajectories[i];
      traj->len = 0;

      for (u32 t = 0; t < max_steps; t++) {
        real *curr_obs = traj->obs + (u64)t * env->obs_dim;
        env->observe(env, curr_obs);

        memcpy(model->input->val->data, curr_obs, sizeof(real) * env->obs_dim);
        forward_pass(&model->forward_graph);

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
      RLTrajectory *traj = &buffer->trajectories[i];
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
    optim_zero_grad(model);

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

        forward_pass(&model->batch_graph);
        graph_zero_activation_grads(&model->batch_graph);

        CRL_CHECK(mat_fused_policy_loss(loss_out, model->batch_logits->val,
                                        batch_acts, batch_advs, batch_size));
        CRL_CHECK(mat_fused_policy_add_grad(
            model->batch_logits->grad, model->batch_logits->val, batch_acts,
            batch_advs, batch_size, seed_grad));

        backward_pass_accumulate(&model->batch_graph);
      }

      if (config->max_grad_norm > 0.0f) {
        optim_clip_grad_norm(model, config->max_grad_norm);
      }
      if (config->optim == OPTIM_ADAM) {
        optim_adam_step(adam, model);
      } else {
        optim_sgd_step(model, lr, 1.0f);
      }
      optim_zero_grad(model);
    } else {
      // Unbatched 1-by-1 fallback through the single-sample cost graph.
      for (u32 i = 0; i < sample_count; i++) {
        memcpy(model->input->val->data, sample_obs[i],
               sizeof(real) * env->obs_dim);
        mat_clear(model->advantage->val);
        real adv = (sample_returns[i] - ret_mean) / (ret_std + 1e-8f);
        model->advantage->val->data[sample_actions[i]] = adv;

        forward_pass(&model->cost_graph);
        backward_pass(&model->cost_graph);
      }

      if (sample_count > 0) {
        if (config->max_grad_norm > 0.0f) {
          optim_clip_grad_norm(model, config->max_grad_norm);
        }
        if (config->optim == OPTIM_ADAM) {
          optim_adam_step(adam, model);
        } else {
          optim_sgd_step(model, lr, 1.0f / (real)sample_count);
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

void rl_train_a2c(Env *env, model_state *model, const RLConfig *config,
                  mem_arena *train_arena) {
  CRL_ASSERT(env != NULL && model != NULL && config != NULL,
             "rl_train_a2c: NULL argument");
  CRL_ASSERT(model->forward_graph.size > 0 && model->batch_graph.size > 0,
             "rl_train_a2c: missing graph (use nn_build_actor_critic)");
  CRL_ASSERT(model->value != NULL && model->batch_value != NULL,
             "rl_train_a2c: model has no value head");
  CRL_ASSERT(train_arena != NULL, "rl_train_a2c: NULL arena");

  u32 rollout_size = config->rollout_size > 0 ? config->rollout_size : 32;
  u32 max_steps = config->max_steps > 0 ? config->max_steps : env->max_steps;
  real gamma = config->gamma > 0.0f ? config->gamma : 0.99f;
  real lambda = config->lambda > 0.0f ? config->lambda : 0.95f;
  real vf_coef = config->vf_coef > 0.0f ? config->vf_coef : 0.5f;
  real entropy_coef = config->entropy_coef;
  real lr = config->learning_rate > 0.0f ? config->learning_rate : 0.001f;

  RolloutBuffer *buffer =
      rl_buffer_create(train_arena, rollout_size, max_steps, env->obs_dim);

  prng_state rng;
  prng_seed_r(&rng, config->seed != 0 ? config->seed : 42, 1);
  prng_state shuf_rng;
  prng_seed_r(&shuf_rng, (config->seed != 0 ? config->seed : 42) ^ 0x9e3779b9ULL,
              3);

  AdamOptim *adam = (config->optim == OPTIM_ADAM)
                        ? optim_adam_create(train_arena, model, lr)
                        : NULL;

  u32 batch_size = model->batch_input->val->cols;
  CRL_ASSERT(config->batch_size == 0 || config->batch_size == batch_size,
             "RLConfig.batch_size does not match the graph's batch dimension");

  printf("Starting A2C Training:\n");
  printf("  Obs Dim: %u | Act Dim: %u | Epochs: %u | Rollout: %u | Batch: %u | "
         "LR: %.4f | Gamma: %.2f | Lambda: %.2f | Entropy: %.4f\n\n",
         env->obs_dim, env->act_dim, config->epochs, rollout_size, batch_size,
         lr, gamma, lambda, entropy_coef);

  u64 max_total_samples = (u64)rollout_size * max_steps;
  real **sample_obs = PUSH_ARRAY(train_arena, real *, max_total_samples);
  u32 *sample_actions = PUSH_ARRAY(train_arena, u32, max_total_samples);
  real *sample_values = PUSH_ARRAY(train_arena, real, max_total_samples);
  real *sample_advantages = PUSH_ARRAY(train_arena, real, max_total_samples);
  u32 *order = PUSH_ARRAY(train_arena, u32, max_total_samples);

  u32 *batch_acts = PUSH_ARRAY(train_arena, u32, batch_size);
  real *batch_advs = PUSH_ARRAY(train_arena, real, batch_size);
  matrix *target_val_mat = mat_create(train_arena, 1, batch_size);

  matrix *seed_pi = mat_create(train_arena, 1, 1);
  matrix *seed_vf = mat_create(train_arena, 1, 1);
  matrix *seed_ent = mat_create(train_arena, 1, 1);

  real *last_obs = PUSH_ARRAY(train_arena, real, env->obs_dim);

  for (u32 epoch = 0; epoch < config->epochs; epoch++) {
    real total_reward = 0.0f;
    u32 sample_count = 0;

    // 1. Rollout collection with state values and GAE
    for (u32 i = 0; i < rollout_size; i++) {
      env->reset(env, next_seed(&rng));
      RLTrajectory *traj = &buffer->trajectories[i];
      traj->len = 0;
      b32 gae_done = false;

      for (u32 t = 0; t < max_steps; t++) {
        real *curr_obs = traj->obs + (u64)t * env->obs_dim;
        env->observe(env, curr_obs);

        memcpy(model->input->val->data, curr_obs, sizeof(real) * env->obs_dim);
        forward_pass(&model->forward_graph);

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
            forward_pass(&model->forward_graph);
            last_val = model->value->val->data[0];
          }
          rl_compute_gae(traj, gamma, lambda, last_val);
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
        forward_pass(&model->forward_graph);
        last_val = model->value->val->data[0];
        rl_compute_gae(traj, gamma, lambda, last_val);
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
    optim_zero_grad(model);
    u32 num_batches = sample_count / batch_size;
    CRL_ASSERT(num_batches > 0, "rl_train_a2c: not enough samples for one batch");

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

      forward_pass(&model->batch_graph);
      // Clear activation grads, THEN seed the policy and value nodes. The old
      // code seeded first and then cleared, wiping every seeded gradient.
      graph_zero_activation_grads(&model->batch_graph);

      CRL_CHECK(mat_fused_policy_add_grad(model->batch_logits->grad,
                                          model->batch_logits->val, batch_acts,
                                          batch_advs, batch_size, seed_pi));
      if (entropy_coef != 0.0f) {
        CRL_CHECK(mat_entropy_add_grad(model->batch_logits->grad,
                                       model->batch_logits->val, batch_size,
                                       seed_ent));
      }
      CRL_CHECK(mat_mse_add_grad(model->batch_value->grad,
                                 model->batch_value->val, target_val_mat,
                                 seed_vf));

      backward_pass_accumulate(&model->batch_graph);
    }

    if (config->max_grad_norm > 0.0f) {
      optim_clip_grad_norm(model, config->max_grad_norm);
    }
    if (config->optim == OPTIM_ADAM) {
      optim_adam_step(adam, model);
    } else {
      optim_sgd_step(model, lr, 1.0f);
    }
    optim_zero_grad(model);

    if (config->verbose || (epoch + 1) % 50 == 0 || epoch == 0 ||
        epoch + 1 == config->epochs) {
      printf("Epoch %4u/%u | Mean Return: %7.2f | Samples: %5u\n", epoch + 1,
             config->epochs, total_reward / (real)rollout_size, sample_count);
    }
  }
}

RLEvalResult rl_evaluate_seeded(Env *env, model_state *model, const u64 *seeds,
                                u32 num_episodes, b32 greedy) {
  CRL_ASSERT(env != NULL && model != NULL && seeds != NULL && num_episodes > 0,
             "rl_evaluate_seeded: bad argument");

  real total_reward = 0.0f;
  u64 total_steps = 0;
  real *obs_buf = malloc(sizeof(real) * env->obs_dim);
  CRL_ASSERT(obs_buf != NULL, "rl_evaluate_seeded: allocation failure");

  for (u32 ep = 0; ep < num_episodes; ep++) {
    env->reset(env, seeds[ep]);
    prng_state ep_rng;
    prng_seed_r(&ep_rng, seeds[ep] ^ 0x5851f42d4c957f2dULL, ep + 1);
    real ep_reward = 0.0f;

    for (u32 t = 0; t < env->max_steps; t++) {
      env->observe(env, obs_buf);
      memcpy(model->input->val->data, obs_buf, sizeof(real) * env->obs_dim);
      forward_pass(&model->forward_graph);

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
  return (RLEvalResult){
      .mean_reward = total_reward / (real)num_episodes,
      .mean_steps = (real)total_steps / (real)num_episodes,
  };
}

RLEvalResult rl_evaluate(Env *env, model_state *model, prng_state *rng,
                         u32 episodes, b32 greedy) {
  CRL_ASSERT(env != NULL && model != NULL && rng != NULL && episodes > 0,
             "rl_evaluate: bad argument");

  u64 *seeds = malloc(sizeof(u64) * episodes);
  CRL_ASSERT(seeds != NULL, "rl_evaluate: allocation failure");
  for (u32 i = 0; i < episodes; i++) {
    seeds[i] = next_seed(rng);
  }

  RLEvalResult res = rl_evaluate_seeded(env, model, seeds, episodes, greedy);
  free(seeds);
  return res;
}
