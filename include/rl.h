#ifndef RL_H
#define RL_H

#include "arena.h"
#include "autograd.h"
#include "env.h"
#include "optim.h"

typedef enum {
  OPTIM_SGD = 0,
  OPTIM_ADAM = 1,
} OptimType;

// Configuration for training
typedef struct {
  u32 epochs;
  u32 rollout_size;
  u32 max_steps;
  f32 learning_rate;
  f32 gamma;
  f32 lambda;         // GAE lambda parameter (e.g. 0.95f)
  f32 vf_coef;        // Value loss coefficient (e.g. 0.5f)
  u64 seed;
  b32 verbose;
  OptimType optim;
  f32 max_grad_norm;
  u32 batch_size;
} RLConfig;

// Evaluation metrics
typedef struct {
  f32 mean_reward;
  f32 mean_steps;
} RLEvalResult;

// Rollout buffer holding trajectories
typedef struct {
  f32 *obs;         // [max_steps * obs_dim]
  u32 *actions;     // [max_steps]
  f32 *rewards;     // [max_steps]
  f32 *returns;     // [max_steps]
  f32 *values;      // [max_steps] (Critic estimated values)
  f32 *advantages;  // [max_steps] (GAE advantages)
  b8 *dones;        // [max_steps] (True environment termination)
  b8 *truncateds;   // [max_steps] (Time limit truncation)
  u32 len;
} RLTrajectory;

typedef struct {
  RLTrajectory *trajectories;
  u32 count;
  u32 max_steps;
  u32 obs_dim;
} RolloutBuffer;

// Allocates a rollout buffer from the arena
RolloutBuffer *rl_buffer_create(mem_arena *arena, u32 num_trajectories,
                                u32 max_steps, u32 obs_dim);

// Computes Generalized Advantage Estimation (GAE-Lambda) for a trajectory
void rl_compute_gae(RLTrajectory *traj, f32 gamma, f32 lambda, f32 last_val);

// Trains an actor model on any Env using batched REINFORCE
void rl_train_reinforce(Env *env, model_state *model, const RLConfig *config,
                        mem_arena *train_arena);

// Trains an Actor-Critic model on any Env using Advantage Actor-Critic (A2C) with GAE
void rl_train_a2c(Env *env, model_state *model, const RLConfig *config,
                  mem_arena *train_arena);

// Evaluates an actor model on an Env across a number of episodes
RLEvalResult rl_evaluate(Env *env, model_state *model, prng_state *rng,
                         u32 episodes, b32 greedy);

#endif // RL_H
