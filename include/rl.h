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
  real learning_rate;
  real gamma;
  real lambda;         // GAE lambda parameter (e.g. 0.95f)
  real vf_coef;        // Value loss coefficient (e.g. 0.5f)
  real entropy_coef;   // Entropy bonus coefficient (e.g. 0.01f)
  u64 seed;
  b32 verbose;
  OptimType optim;
  real max_grad_norm;
  u32 batch_size;
} RLConfig;

// Evaluation metrics
typedef struct {
  real mean_reward;
  real mean_steps;
} RLEvalResult;

// Rollout buffer holding trajectories
typedef struct {
  real *obs;         // [max_steps * obs_dim]
  u32 *actions;     // [max_steps]
  real *rewards;     // [max_steps]
  real *returns;     // [max_steps]
  real *values;      // [max_steps] (Critic estimated values)
  real *advantages;  // [max_steps] (GAE advantages)
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
void rl_compute_gae(RLTrajectory *traj, real gamma, real lambda, real last_val);

// Trains an actor model on any Env using batched REINFORCE
void rl_train_reinforce(Env *env, model_state *model, const RLConfig *config,
                        mem_arena *train_arena);

// Trains an Actor-Critic model on any Env using Advantage Actor-Critic (A2C) with GAE
void rl_train_a2c(Env *env, model_state *model, const RLConfig *config,
                  mem_arena *train_arena);

// Evaluates an actor model on an Env across a number of episodes. `greedy`
// selects argmax actions; otherwise actions are sampled.
// The seed-less variant derives per-episode seeds from `rng`; the seeded
// variant uses the caller's seeds so different policies can be compared on
// identical start states.
RLEvalResult rl_evaluate(Env *env, model_state *model, prng_state *rng,
                         u32 episodes, b32 greedy);
RLEvalResult rl_evaluate_seeded(Env *env, model_state *model, const u64 *seeds,
                                u32 num_episodes, b32 greedy);

#endif // RL_H
