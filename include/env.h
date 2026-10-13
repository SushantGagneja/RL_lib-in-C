#ifndef ENV_H
#define ENV_H

#include "base.h"
#include "prng.h"

// Unified RL Environment Interface
// Supports discrete action environments with separate done (terminal) and truncated (timeout) flags.
typedef struct Env Env;
struct Env {
  u32 obs_dim;
  u32 act_dim;
  u32 max_steps;
  void *state;
  void (*reset)(Env *e, prng_state *rng);
  f32 (*step)(Env *e, u32 action, b32 *done, b32 *truncated);
  void (*observe)(Env *e, f32 *obs_out);
  void (*destroy)(Env *e);
};

#endif // ENV_H
