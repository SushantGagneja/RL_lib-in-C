#include "rl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

RolloutBuffer *rl_buffer_create(mem_arena *arena, u32 num_trajectories,
                                u32 max_steps, u32 obs_dim) {
  RolloutBuffer *buf = PUSH_STRUCT(arena, RolloutBuffer);
  buf->count = num_trajectories;
  buf->max_steps = max_steps;
  buf->obs_dim = obs_dim;
  buf->trajectories = PUSH_ARRAY(arena, RLTrajectory, num_trajectories);

  for (u32 i = 0; i < num_trajectories; i++) {
    buf->trajectories[i].obs = PUSH_ARRAY(arena, f32, (u64)max_steps * obs_dim);
    buf->trajectories[i].actions = PUSH_ARRAY(arena, u32, max_steps);
    buf->trajectories[i].rewards = PUSH_ARRAY(arena, f32, max_steps);
    buf->trajectories[i].returns = PUSH_ARRAY(arena, f32, max_steps);
    buf->trajectories[i].values = PUSH_ARRAY(arena, f32, max_steps);
    buf->trajectories[i].advantages = PUSH_ARRAY(arena, f32, max_steps);
    buf->trajectories[i].dones = PUSH_ARRAY(arena, b8, max_steps);
    buf->trajectories[i].truncateds = PUSH_ARRAY(arena, b8, max_steps);
    buf->trajectories[i].len = 0;
  }

  return buf;
}

static u32 sample_action(prng_state *rng, const matrix *probs) {
  f32 r = prng_randf_r(rng);
  f32 cdf = 0.0f;
  u32 size = probs->rows * probs->cols;
  for (u32 i = 0; i < size; i++) {
    cdf += probs->data[i];
    if (r <= cdf) {
      return i;
    }
  }
  return size - 1;
}

static u32 argmax_action(const matrix *probs) {
  u32 size = probs->rows * probs->cols;
  u32 best = 0;
  f32 max_p = probs->data[0];
  for (u32 i = 1; i < size; i++) {
    if (probs->data[i] > max_p) {
      max_p = probs->data[i];
      best = i;
    }
  }
  return best;
}

void rl_compute_gae(RLTrajectory *traj, f32 gamma, f32 lambda, f32 last_val) {
  if (traj->len == 0) return;
  f32 gae = 0.0f;
  for (i32 t = (i32)traj->len - 1; t >= 0; t--) {
    f32 next_val = (t == (i32)traj->len - 1) ? last_val : traj->values[t + 1];
    f32 non_terminal = traj->dones[t] ? 0.0f : 1.0f;
    f32 delta = traj->rewards[t] + gamma * next_val * non_terminal - traj->values[t];
    gae = delta + gamma * lambda * non_terminal * gae;
    traj->advantages[t] = gae;
    traj->returns[t] = gae + traj->values[t];
  }
}

void rl_train_reinforce(Env *env, model_state *model, const RLConfig *config,
                        mem_arena *train_arena) {
  u32 rollout_size = config->rollout_size > 0 ? config->rollout_size : 32;
  u32 max_steps = config->max_steps > 0 ? config->max_steps : env->max_steps;
  f32 gamma = config->gamma > 0.0f ? config->gamma : 0.99f;
  f32 lr = config->learning_rate > 0.0f ? config->learning_rate : 0.01f;

  RolloutBuffer *buffer =
      rl_buffer_create(train_arena, rollout_size, max_steps, env->obs_dim);

  prng_state rng;
  prng_seed_r(&rng, config->seed != 0 ? config->seed : 42, 1);

  AdamOptim *adam = (config->optim == OPTIM_ADAM)
                        ? optim_adam_create(train_arena, model, lr)
                        : NULL;

  b32 use_batching = (model->batch_graph.size > 0 && model->batch_input != NULL &&
                      model->batch_logits != NULL);
  u32 batch_size = use_batching ? model->batch_input->val->cols : 1;

  printf("Starting REINFORCE Training:\n");
  printf("  Obs Dim: %u | Act Dim: %u | Epochs: %u | Rollout: %u | Batch: %u | "
         "Optim: %s | LR: %.4f | Gamma: %.2f\n\n",
         env->obs_dim, env->act_dim, config->epochs, rollout_size, batch_size,
         config->optim == OPTIM_ADAM ? "Adam" : "SGD", lr, gamma);

  // Buffer pointers for batching
  u64 max_total_samples = (u64)rollout_size * max_steps;
  f32 **sample_obs = PUSH_ARRAY(train_arena, f32 *, max_total_samples);
  u32 *sample_actions = PUSH_ARRAY(train_arena, u32, max_total_samples);
  f32 *sample_returns = PUSH_ARRAY(train_arena, f32, max_total_samples);

  matrix *seed_grad = mat_create(train_arena, 1, 1);
  matrix *loss_out = mat_create(train_arena, 1, 1);

  for (u32 epoch = 0; epoch < config->epochs; epoch++) {
    f32 total_reward = 0.0f;
    u32 sample_count = 0;

    // 1. Rollout phase
    for (u32 i = 0; i < rollout_size; i++) {
      env->reset(env, &rng);
      RLTrajectory *traj = &buffer->trajectories[i];
      traj->len = 0;

      for (u32 t = 0; t < max_steps; t++) {
        f32 *curr_obs = traj->obs + (u64)t * env->obs_dim;
        env->observe(env, curr_obs);

        memcpy(model->input->val->data, curr_obs, sizeof(f32) * env->obs_dim);
        forward_pass(&model->forward_graph);

        u32 action = sample_action(&rng, model->output->val);

        b32 done = false;
        b32 truncated = false;
        f32 reward = env->step(env, action, &done, &truncated);

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

    // 2. Compute discounted Monte-Carlo returns
    f32 return_sum = 0.0f;
    f32 return_sq_sum = 0.0f;

    for (u32 i = 0; i < rollout_size; i++) {
      RLTrajectory *traj = &buffer->trajectories[i];
      f32 G = 0.0f;

      for (i32 t = (i32)traj->len - 1; t >= 0; t--) {
        G = traj->rewards[t] + gamma * G;
        traj->returns[t] = G;
        return_sum += G;
        return_sq_sum += G * G;

        sample_obs[sample_count] = traj->obs + (u64)t * env->obs_dim;
        sample_actions[sample_count] = traj->actions[t];
        sample_returns[sample_count] = G;
        sample_count++;
      }
    }

    f32 return_mean = return_sum / (f32)sample_count;
    f32 return_var =
        return_sq_sum / (f32)sample_count - return_mean * return_mean;
    f32 return_std = sqrtf(MAX(return_var, 0.0f));

    // 3. Batched training or legacy fallback
    optim_zero_grad(model);

    if (use_batching && sample_count >= batch_size) {
      u32 num_batches = sample_count / batch_size;
      u32 *batch_acts = PUSH_ARRAY(train_arena, u32, batch_size);
      f32 *batch_advs = PUSH_ARRAY(train_arena, f32, batch_size);
      seed_grad->data[0] = 1.0f / (f32)num_batches;

      for (u32 k = 0; k < num_batches; k++) {
        for (u32 b = 0; b < batch_size; b++) {
          u32 idx = k * batch_size + b;
          f32 *obs = sample_obs[idx];
          for (u32 r = 0; r < env->obs_dim; r++) {
            model->batch_input->val->data[r * batch_size + b] = obs[r];
          }
          batch_acts[b] = sample_actions[idx];
          batch_advs[b] =
              (sample_returns[idx] - return_mean) / (return_std + 1e-8f);
        }

        forward_pass(&model->batch_graph);
        mat_clear(model->batch_logits->grad);

        mat_fused_policy_loss(loss_out, model->batch_logits->val, batch_acts,
                              batch_advs, batch_size);
        mat_fused_policy_add_grad(model->batch_logits->grad,
                                  model->batch_logits->val, batch_acts,
                                  batch_advs, batch_size, seed_grad);

        backward_pass_from(&model->batch_graph, model->batch_logits, false);
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
      // Unbatched 1-by-1 fallback
      for (u32 i = 0; i < sample_count; i++) {
        memcpy(model->input->val->data, sample_obs[i],
               sizeof(f32) * env->obs_dim);
        mat_clear(model->advantage->val);
        f32 adv = (sample_returns[i] - return_mean) / (return_std + 1e-8f);
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
          optim_sgd_step(model, lr, 1.0f / (f32)sample_count);
        }
      }
    }

    // Logging
    if ((epoch + 1) % 50 == 0 || epoch == 0 || epoch + 1 == config->epochs) {
      printf("Epoch %4u/%u | Mean Return: %7.2f | Samples: %5u | Return Std: "
             "%6.2f\n",
             epoch + 1, config->epochs, total_reward / (f32)rollout_size,
             sample_count, return_std);
    }
  }
}

void rl_train_a2c(Env *env, model_state *model, const RLConfig *config,
                  mem_arena *train_arena) {
  u32 rollout_size = config->rollout_size > 0 ? config->rollout_size : 32;
  u32 max_steps = config->max_steps > 0 ? config->max_steps : env->max_steps;
  f32 gamma = config->gamma > 0.0f ? config->gamma : 0.99f;
  f32 lambda = config->lambda > 0.0f ? config->lambda : 0.95f;
  f32 vf_coef = config->vf_coef > 0.0f ? config->vf_coef : 0.5f;
  f32 lr = config->learning_rate > 0.0f ? config->learning_rate : 0.001f;

  RolloutBuffer *buffer =
      rl_buffer_create(train_arena, rollout_size, max_steps, env->obs_dim);

  prng_state rng;
  prng_seed_r(&rng, config->seed != 0 ? config->seed : 42, 1);

  AdamOptim *adam = (config->optim == OPTIM_ADAM)
                        ? optim_adam_create(train_arena, model, lr)
                        : NULL;

  u32 batch_size =
      (model->batch_input != NULL) ? model->batch_input->val->cols : 32;

  printf("Starting A2C Training:\n");
  printf("  Obs Dim: %u | Act Dim: %u | Epochs: %u | Rollout: %u | Batch: %u | "
         "LR: %.4f | Gamma: %.2f | Lambda: %.2f\n\n",
         env->obs_dim, env->act_dim, config->epochs, rollout_size, batch_size,
         lr, gamma, lambda);

  u64 max_total_samples = (u64)rollout_size * max_steps;
  f32 **sample_obs = PUSH_ARRAY(train_arena, f32 *, max_total_samples);
  u32 *sample_actions = PUSH_ARRAY(train_arena, u32, max_total_samples);
  f32 *sample_advantages = PUSH_ARRAY(train_arena, f32, max_total_samples);
  f32 *sample_targets = PUSH_ARRAY(train_arena, f32, max_total_samples);

  matrix *target_val_mat = mat_create(train_arena, 1, batch_size);
  matrix *seed_pi = mat_create(train_arena, 1, 1);
  seed_pi->data[0] = 1.0f / (f32)batch_size;
  matrix *seed_vf = mat_create(train_arena, 1, 1);
  seed_vf->data[0] = vf_coef / (f32)batch_size;

  f32 *last_obs = PUSH_ARRAY(train_arena, f32, env->obs_dim);

  for (u32 epoch = 0; epoch < config->epochs; epoch++) {
    f32 total_reward = 0.0f;
    u32 sample_count = 0;

    // 1. Rollout collection with state values
    for (u32 i = 0; i < rollout_size; i++) {
      env->reset(env, &rng);
      RLTrajectory *traj = &buffer->trajectories[i];
      traj->len = 0;

      for (u32 t = 0; t < max_steps; t++) {
        f32 *curr_obs = traj->obs + (u64)t * env->obs_dim;
        env->observe(env, curr_obs);

        memcpy(model->input->val->data, curr_obs, sizeof(f32) * env->obs_dim);
        forward_pass(&model->forward_graph);

        u32 action = sample_action(&rng, model->output->val);
        f32 val = model->value != NULL ? model->value->val->data[0] : 0.0f;

        b32 done = false;
        b32 truncated = false;
        f32 reward = env->step(env, action, &done, &truncated);

        traj->actions[t] = action;
        traj->rewards[t] = reward;
        traj->values[t] = val;
        traj->dones[t] = (b8)done;
        traj->truncateds[t] = (b8)truncated;
        traj->len = t + 1;
        total_reward += reward;

        if (done || truncated) {
          f32 last_val = 0.0f;
          if (truncated && !done && model->value != NULL) {
            env->observe(env, last_obs);
            memcpy(model->input->val->data, last_obs, sizeof(f32) * env->obs_dim);
            forward_pass(&model->forward_graph);
            last_val = model->value->val->data[0];
          }
          rl_compute_gae(traj, gamma, lambda, last_val);
          break;
        }
      }

      for (u32 t = 0; t < traj->len; t++) {
        sample_obs[sample_count] = traj->obs + (u64)t * env->obs_dim;
        sample_actions[sample_count] = traj->actions[t];
        sample_advantages[sample_count] = traj->advantages[t];
        sample_targets[sample_count] = traj->returns[t];
        sample_count++;
      }
    }

    // 2. Normalize GAE advantages
    f32 adv_sum = 0.0f, adv_sq = 0.0f;
    for (u32 i = 0; i < sample_count; i++) {
      adv_sum += sample_advantages[i];
      adv_sq += sample_advantages[i] * sample_advantages[i];
    }
    f32 adv_mean = adv_sum / (f32)sample_count;
    f32 adv_std = sqrtf(MAX(adv_sq / (f32)sample_count - adv_mean * adv_mean, 0.0f));

    for (u32 i = 0; i < sample_count; i++) {
      sample_advantages[i] = (sample_advantages[i] - adv_mean) / (adv_std + 1e-8f);
    }

    // 3. Batched Actor-Critic training
    optim_zero_grad(model);
    u32 num_batches = sample_count / batch_size;
    u32 *batch_acts = PUSH_ARRAY(train_arena, u32, batch_size);
    f32 *batch_advs = PUSH_ARRAY(train_arena, f32, batch_size);

    for (u32 k = 0; k < num_batches; k++) {
      for (u32 b = 0; b < batch_size; b++) {
        u32 idx = k * batch_size + b;
        f32 *obs = sample_obs[idx];
        for (u32 r = 0; r < env->obs_dim; r++) {
          model->batch_input->val->data[r * batch_size + b] = obs[r];
        }
        batch_acts[b] = sample_actions[idx];
        batch_advs[b] = sample_advantages[idx];
        target_val_mat->data[b] = sample_targets[idx];
      }

      forward_pass(&model->batch_graph);
      mat_clear(model->batch_logits->grad);
      mat_clear(model->batch_value->grad);

      mat_fused_policy_add_grad(model->batch_logits->grad,
                                model->batch_logits->val, batch_acts,
                                batch_advs, batch_size, seed_pi);
      mat_mse_add_grad(model->batch_value->grad, model->batch_value->val,
                       target_val_mat, seed_vf);

      backward_pass_from(&model->batch_graph, NULL, false);

      if (config->max_grad_norm > 0.0f) {
        optim_clip_grad_norm(model, config->max_grad_norm);
      }

      if (config->optim == OPTIM_ADAM) {
        optim_adam_step(adam, model);
      } else {
        optim_sgd_step(model, lr, 1.0f);
      }
      optim_zero_grad(model);
    }

    if ((epoch + 1) % 50 == 0 || epoch == 0 || epoch + 1 == config->epochs) {
      printf("Epoch %4u/%u | Mean Return: %7.2f | Samples: %5u\n",
             epoch + 1, config->epochs, total_reward / (f32)rollout_size,
             sample_count);
    }
  }
}

RLEvalResult rl_evaluate(Env *env, model_state *model, prng_state *rng,
                         u32 episodes, b32 greedy) {
  f32 total_reward = 0.0f;
  u64 total_steps = 0;
  f32 *obs_buf = malloc(sizeof(f32) * env->obs_dim);

  for (u32 ep = 0; ep < episodes; ep++) {
    env->reset(env, rng);
    f32 ep_reward = 0.0f;

    for (u32 t = 0; t < env->max_steps; t++) {
      env->observe(env, obs_buf);
      memcpy(model->input->val->data, obs_buf, sizeof(f32) * env->obs_dim);
      forward_pass(&model->forward_graph);

      u32 action = greedy ? argmax_action(model->output->val)
                          : sample_action(rng, model->output->val);

      b32 done = false;
      b32 truncated = false;
      f32 r = env->step(env, action, &done, &truncated);
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
      .mean_reward = total_reward / (f32)episodes,
      .mean_steps = (f32)total_steps / (f32)episodes,
  };
}
