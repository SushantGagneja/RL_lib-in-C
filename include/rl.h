#ifndef RL_H
#define RL_H

#include "memory.h"
#include "graph.h"
#include "env.h"
#include "optim.h"

typedef enum {
  OPT_SGD = 0,
  OPT_ADAM = 1,
} OptimKind;

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
  OptimKind optim;
  real max_grad_norm;
  u32 batch_size;

  // PPO-only knobs.
  u32 num_envs;     // parallel envs stepped per update (0 => 1)
  u32 ppo_epochs;   // inner optimisation passes over a rollout (0 => 10)
  real ppo_clip;    // PPO clip epsilon (<= 0 => 0.2)
} TrainCfg;

// Evaluation metrics
typedef struct {
  real avg_return;
  real avg_steps;
} EvalResult;

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
} Episode;

typedef struct {
  Episode *trajectories;
  u32 count;
  u32 max_steps;
  u32 obs_dim;
} Rollout;

// Allocates a rollout buffer from the arena
Rollout *rollout_new(Arena *arena, u32 num_trajectories,
                                u32 max_steps, u32 obs_dim);

// Computes Generalized Advantage Estimation (GAE-Lambda) for a trajectory
void gae_compute(Episode *traj, real gamma, real lambda, real last_val);

// Trains an actor model on any Env using batched REINFORCE
void train_reinforce(Env *env, Net *model, const TrainCfg *config,
                        Arena *train_arena);

// Trains an Actor-Critic model on any Env using Advantage Actor-Critic (A2C) with GAE
void train_a2c(Env *env, Net *model, const TrainCfg *config,
                  Arena *train_arena);

// Trains an Actor-Critic model using PPO with the clipped surrogate objective.
// The rollout is collected from `config->num_envs` parallel environments (a
// vectorized env), then replayed for `config->ppo_epochs` passes of shuffled
// minibatches, one optimiser step per minibatch. Requires a model built with
// actor_critic_build(). TrainCfg.ppo_clip selects the clip range.
void train_ppo(Env *env, Net *model, const TrainCfg *config,
               Arena *train_arena);

// Evaluates an actor model on an Env across a number of episodes. `greedy`
// selects argmax actions; otherwise actions are sampled.
// The seed-less variant derives per-episode seeds from `rng`; the seeded
// variant uses the caller's seeds so different policies can be compared on
// identical start states.
EvalResult evaluate(Env *env, Net *model, Rng *rng,
                         u32 episodes, b32 greedy);
EvalResult evaluate_seeded(Env *env, Net *model, const u64 *seeds,
                                u32 num_episodes, b32 greedy);

#endif // RL_H
