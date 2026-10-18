#ifndef VECENV_H
#define VECENV_H

#include "memory.h"
#include "env.h"

// A synchronised vector of independent environments — the standard shape for
// batched rollout collection. Sub-environments are stepped together, one action
// each. Finished sub-environments are auto-reset in place with a fresh seed
// (Gymnasium-style), and the observation *before* the reset is preserved in
// `final_obs` so a trainer can bootstrap the value of a truncated episode.
typedef struct {
  u32 num_envs;
  u32 obs_dim;
  u32 act_dim;

  Env **envs;

  real *obs;        // num_envs * obs_dim, current observation of each env
  real *final_obs;  // num_envs * obs_dim, terminal observation (valid if dones)
  real *rewards;    // num_envs
  b8 *dones;        // num_envs, terminal OR truncated on this step
  b8 *terminals;    // num_envs, true terminal (not a time-limit truncation)
  b8 *truncateds;   // num_envs, time-limit truncation

  Rng rng;  // seeds the automatic resets
  u64 step_count;
} VecEnv;

// Clones `prototype` num_envs times (each clone owns its own Env state) and
// resets every sub-environment. `prototype` is not consumed.
VecEnv *vecenv_new(Arena *arena, const Env *prototype, u32 num_envs);

// Resets every sub-environment: from one seed (derived per index) or from an
// explicit per-env seed array of length num_envs.
void vecenv_reset(VecEnv *v, u64 seed);
void vecenv_reset_seeds(VecEnv *v, const u64 *seeds);

// Steps every sub-environment with an explicit per-env action. On return,
// obs/rewards/dones/terminals/truncateds are filled; finished envs have been
// reset, so obs[] holds their post-reset observation and final_obs[] the
// pre-reset (terminal) one.
void vecenv_step(VecEnv *v, const u32 *actions);

// Samples a uniform-random action for every env into actions_out.
void vecenv_sample_uniform(VecEnv *v, Rng *rng, u32 *actions_out);

#endif // VECENV_H
