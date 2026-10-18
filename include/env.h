#ifndef ENV_H
#define ENV_H

#include "base.h"
#include "random.h"

// Unified RL Environment Interface.
// Supports discrete action environments with separate done (terminal) and
// truncated (timeout) flags.
//
// Each environment owns its own PRNG state, seeded through reset(). This makes
// an Env self-contained: it can be copied, driven from any thread, and later
// vectorised without sharing a dangling pointer into a trainer's RNG.
typedef struct Env Env;
struct Env {
  u32 obs_dim;
  u32 act_dim;
  u32 max_steps;
  void *state;
  void (*reset)(Env *e, u64 seed);
  real (*step)(Env *e, u32 action, b32 *done, b32 *truncated);
  void (*observe)(Env *e, real *obs_out);
  // Allocates a fresh Env with an independent copy of this one's state. Used by
  // the vectorized environment wrapper to build N identical-but-separate envs.
  Env *(*clone)(Arena *arena, const Env *src);
  void (*destroy)(Env *e);
};

#endif // ENV_H
