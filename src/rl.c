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
    buf->trajectories[i].dones = PUSH_ARRAY(arena, b8, max_steps);
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

  printf("Starting REINFORCE Training:\n");
  printf("  Obs Dim: %u | Act Dim: %u | Epochs: %u | Rollout: %u | Optim: %s | "
         "LR: %.4f | Gamma: %.2f\n\n",
         env->obs_dim, env->act_dim, config->epochs, rollout_size,
         config->optim == OPTIM_ADAM ? "Adam" : "SGD", lr, gamma);

  for (u32 epoch = 0; epoch < config->epochs; epoch++) {
    f32 total_reward = 0.0f;

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
        traj->len = t + 1;
        total_reward += reward;

        if (done || truncated) {
          break;
        }
      }
    }

    // 2. Returns and advantage statistics
    u32 sample_count = 0;
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
        sample_count++;
      }
    }

    f32 return_mean = return_sum / (f32)sample_count;
    f32 return_var =
        return_sq_sum / (f32)sample_count - return_mean * return_mean;
    f32 return_std = sqrtf(MAX(return_var, 0.0f));

    // 3. Backward pass across all rollout samples
    optim_zero_grad(model);

    for (u32 i = 0; i < rollout_size; i++) {
      RLTrajectory *traj = &buffer->trajectories[i];

      for (u32 t = 0; t < traj->len; t++) {
        f32 *curr_obs = traj->obs + (u64)t * env->obs_dim;
        memcpy(model->input->val->data, curr_obs, sizeof(f32) * env->obs_dim);

        clear(model->advantage->val);
        f32 advantage = (traj->returns[t] - return_mean) / (return_std + 1e-8f);
        model->advantage->val->data[traj->actions[t]] = advantage;

        forward_pass(&model->cost_graph);
        backward_pass(&model->cost_graph);
      }
    }

    // 4. Update model parameters
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

    // Logging
    if ((epoch + 1) % 50 == 0 || epoch == 0 || epoch + 1 == config->epochs) {
      printf("Epoch %4u/%u | Mean Return: %7.2f | Samples: %5u | Return Std: "
             "%6.2f\n",
             epoch + 1, config->epochs, total_reward / (f32)rollout_size,
             sample_count, return_std);
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
