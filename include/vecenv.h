#ifndef VECENV_H
#define VECENV_H

#include "memory.h"
#include "env.h"

// Threaded stepping needs POSIX threads. Define RL_LIB_NO_THREADS to build the
// serial path only; a thread count above 1 is then refused loudly instead of
// silently running serially.
#if !defined(RL_LIB_NO_THREADS) && !defined(_WIN32)
#define RL_LIB_USE_PTHREADS 1
#endif

// A synchronised vector of independent environments — the standard shape for
// batched rollout collection. Sub-environments are stepped together, one action
// each. Finished sub-environments are auto-reset in place with a fresh seed
// (Gymnasium-style), and the observation *before* the reset is preserved in
// `final_obs` so a trainer can bootstrap the value of a truncated episode.
//
// Stepping is serial by default. `vecenv_set_threads` spreads the per-env work
// over worker threads; because each sub-environment owns its state and its PRNG
// and nothing else is shared, the trajectories are bit-identical to the serial
// ones (see tests/test_envs.c).
typedef struct VecEnvPool VecEnvPool;

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

  // Auto-reset streams, one per sub-environment: reset_base[i] is the seed that
  // vecenv_reset[_seeds] gave env i, reset_counts[i] counts its auto-resets.
  // Keeping the stream per env is what makes a sub-environment's trajectory
  // independent of the order in which its siblings are stepped — and therefore
  // identical whether they are stepped serially or in parallel.
  u64 *reset_base;
  u64 *reset_counts;

  u64 step_count;

  VecEnvPool *pool;  // NULL while stepping serially
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

// Continuous-action counterpart of vecenv_step: `actions` holds
// num_envs * act_dim values, one act_dim-long action per sub-environment
// (env-major, like `obs`). Auto-reset behaves exactly as in vecenv_step.
void vecenv_step_continuous(VecEnv *v, const real *actions);

// Copies the action bounds advertised by the sub-environments (act_dim entries
// each) into *min_out / *max_out. Either output may be NULL. Only valid for
// continuous-action vecenvs.
void vecenv_action_bounds(const VecEnv *v, const real **min_out,
                          const real **max_out);

// ── Threaded stepping (optional) ────────────────────────────────────────────
//
// Runs the per-env steps on `num_threads` threads (0 or 1 restores serial
// stepping, the default). Threads are spawned once and reused across steps, so
// this is cheap per call but still only pays off when a single env step costs
// more than the handful of microseconds a barrier round-trip takes — CartPole
// and Pendulum are far below that, where the serial loop is faster. Call it
// only while no step is in flight.
void vecenv_set_threads(Arena *arena, VecEnv *v, u32 num_threads);

// Number of threads currently used to step (1 when serial).
u32 vecenv_threads(const VecEnv *v);

// Joins the worker threads and returns the vecenv to serial stepping. Idempotent
// and safe to call at any point between steps; call it before the arena the
// vecenv lives in is disposed, or the workers outlive their own bookkeeping.
void vecenv_shutdown(VecEnv *v);

#endif // VECENV_H
