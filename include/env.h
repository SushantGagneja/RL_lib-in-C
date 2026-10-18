#ifndef ENV_H
#define ENV_H

#include "base.h"
#include "random.h"

// Unified RL Environment Interface.
// Supports discrete and continuous action environments, with separate done
// (terminal) and truncated (timeout) flags.
//
// A discrete environment implements `step` with a u32 action. A continuous one
// sets `continuous`, implements `step_continuous` with an `act_dim`-long real
// action, and points `action_min`/`action_max` at `act_dim`-long bounds (its
// policy clamps to those). Environment code does not have to clamp defensively:
// the trainers clamp before calling.
//
// Each environment owns its own PRNG state, seeded through reset(). This makes
// an Env self-contained: it can be copied, driven from any thread, and
// vectorised without sharing a dangling pointer into a trainer's RNG.
typedef struct Env Env;
struct Env {
  u32 obs_dim;
  u32 act_dim;
  u32 max_steps;
  void *state;

  // Continuous-action marker plus the bounds a policy should clamp to. Both
  // pointers refer to `act_dim` entries and are NULL for discrete envs.
  b32 continuous;
  const real *action_min;
  const real *action_max;

  void (*reset)(Env *e, u64 seed);
  real (*step)(Env *e, u32 action, b32 *done, b32 *truncated);
  real (*step_continuous)(Env *e, const real *action, b32 *done,
                          b32 *truncated);
  void (*observe)(Env *e, real *obs_out);
  // Allocates a fresh Env with an independent copy of this one's state. Used by
  // the vectorized environment wrapper to build N identical-but-separate envs.
  Env *(*clone)(Arena *arena, const Env *src);
  void (*destroy)(Env *e);
};

#endif // ENV_H
