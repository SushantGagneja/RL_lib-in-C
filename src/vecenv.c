#include "vecenv.h"

// Derives a fresh 64-bit seed from the vector env's own PRNG for each reset so
// that auto-resets are reproducible for a given vecenv_new + reset sequence.
static u64 vecenv_next_seed(VecEnv *v) {
  const u64 hi = rng_u32(&v->rng);
  const u64 lo = rng_u32(&v->rng);
  return (hi << 32) ^ lo;
}

VecEnv *vecenv_new(Arena *arena, const Env *prototype, u32 num_envs) {
  RL_LIB_ASSERT(arena != NULL && prototype != NULL && num_envs > 0,
                "vecenv_new: bad argument");
  RL_LIB_ASSERT(prototype->clone != NULL,
                "vecenv_new: prototype env does not implement clone");

  VecEnv *v = ARENA_OBJ(arena, VecEnv);
  v->num_envs = num_envs;
  v->obs_dim = prototype->obs_dim;
  v->act_dim = prototype->act_dim;
  v->step_count = 0;
  rng_seed(&v->rng, 0x5ca1ab1e5eedULL, 7);

  const u64 obs_total = (u64)num_envs * v->obs_dim;
  v->envs = ARENA_ARR(arena, Env *, num_envs);
  v->obs = ARENA_ARR(arena, real, obs_total);
  v->final_obs = ARENA_ARR(arena, real, obs_total);
  v->rewards = ARENA_ARR(arena, real, num_envs);
  v->dones = ARENA_ARR(arena, b8, num_envs);
  v->terminals = ARENA_ARR(arena, b8, num_envs);
  v->truncateds = ARENA_ARR(arena, b8, num_envs);

  for (u32 i = 0; i < num_envs; i++) {
    v->envs[i] = prototype->clone(arena, prototype);
  }
  return v;
}

void vecenv_reset(VecEnv *v, u64 seed) {
  RL_LIB_ASSERT(v != NULL, "vecenv_reset: NULL vecenv");
  for (u32 i = 0; i < v->num_envs; i++) {
    v->envs[i]->reset(v->envs[i], seed + (u64)i);
    v->envs[i]->observe(v->envs[i], v->obs + (u64)i * v->obs_dim);
    v->rewards[i] = 0.0f;
    v->dones[i] = 0;
    v->terminals[i] = 0;
    v->truncateds[i] = 0;
  }
  v->step_count = 0;
}

void vecenv_reset_seeds(VecEnv *v, const u64 *seeds) {
  RL_LIB_ASSERT(v != NULL && seeds != NULL, "vecenv_reset_seeds: NULL argument");
  for (u32 i = 0; i < v->num_envs; i++) {
    v->envs[i]->reset(v->envs[i], seeds[i]);
    v->envs[i]->observe(v->envs[i], v->obs + (u64)i * v->obs_dim);
  }
  v->step_count = 0;
}

void vecenv_step(VecEnv *v, const u32 *actions) {
  RL_LIB_ASSERT(v != NULL && actions != NULL, "vecenv_step: NULL argument");
  for (u32 i = 0; i < v->num_envs; i++) {
    Env *e = v->envs[i];
    b32 done = false;
    b32 truncated = false;
    const real r = e->step(e, actions[i], &done, &truncated);
    v->rewards[i] = r;

    if (done || truncated) {
      // Capture the terminal observation, then auto-reset in place.
      e->observe(e, v->final_obs + (u64)i * v->obs_dim);
      e->reset(e, vecenv_next_seed(v));
    }
    e->observe(e, v->obs + (u64)i * v->obs_dim);

    v->dones[i] = (b8)(done || truncated);
    v->terminals[i] = (b8)done;
    v->truncateds[i] = (b8)truncated;
  }
  v->step_count++;
}

void vecenv_sample_uniform(VecEnv *v, Rng *rng, u32 *actions_out) {
  RL_LIB_ASSERT(v != NULL && rng != NULL && actions_out != NULL,
                "vecenv_sample_uniform: NULL argument");
  for (u32 i = 0; i < v->num_envs; i++) {
    actions_out[i] = rng_below(rng, v->act_dim);
  }
}
