#ifndef SAC_H
#define SAC_H

// Soft actor-critic for continuous action spaces: an off-policy,
// maximum-entropy trainer over a replay buffer of (s, a, r, s', done) tuples.
//
// The actor is a sac_actor_build network (mean head plus a learnable log_std),
// the twin critics are sac_q_build networks, and the trainer keeps soft-updated
// copies of both critics as the bootstrap targets. Every cost is assembled as a
// graph: the actor's cost backpropagates through the graph-computed action into
// the critic weights, so no gradient is passed by hand. The only data-side
// quantity is the TD target, built from the target critics' minimum and the
// temperature times the next action's log-density.

#include "memory.h"
#include "random.h"
#include "graph.h"
#include "layers.h"
#include "rl.h"
#include "env.h"

// Replay buffer for continuous transitions: one flat arena-backed array per
// field, storing transitions in insertion order and overwriting the oldest
// entry once `capacity` is reached. The terminal and truncated flags are kept
// apart because they bootstrap differently: a time-limit cut still bootstraps,
// a real terminal state does not.
typedef struct {
  u32 obs_dim;
  u32 act_dim;
  u32 capacity;
  u32 size;    // valid transitions, <= capacity
  u32 head;    // next write index
  real *obs;      // capacity * obs_dim
  real *next_obs; // capacity * obs_dim
  real *actions;  // capacity * act_dim
  real *rewards;  // capacity
  u8 *terminated; // episode ended: cut the bootstrap
  u8 *truncated;  // time limit: keep bootstrapping
} SacReplay;

// Allocates an empty buffer holding `capacity` transitions.
SacReplay *sac_replay_new(Arena *arena, u32 capacity, u32 obs_dim,
                          u32 act_dim);

// Appends one transition, overwriting the oldest one once the buffer is full.
void sac_replay_add(SacReplay *buf, const real *obs, const real *action,
                    real reward, const real *next_obs, b32 terminated,
                    b32 truncated);

// Copies `batch` transitions (by index, as drawn by the trainer) into
// contiguous (rows x batch) matrices: observations into `obs` and `next_obs`
// (obs_dim x batch), actions into `actions` (act_dim x batch), and the reward
// and terminal flags into 1 x batch floats. `terminated` receives 1/0 for a
// real episode end, ignoring truncations.
void sac_replay_gather(const SacReplay *buf, const u32 *indices, u32 batch,
                       Mat *obs, Mat *actions, Mat *next_obs, Mat *rewards,
                       Mat *terminated);

// Soft target update over every parameter: dst <- (1 - tau) * dst + tau * src.
// `tau` is clamped to [0, 1]; 1 copies the source outright. Both networks must
// have the same parameter shapes.
void sac_soft_update(const Net *src, Net *dst, real tau);

// Bootstrap value of one transition: the reward alone when the episode
// terminated, otherwise reward + gamma * (q_min - alpha_logp), where
// `alpha_logp` is the temperature times the next action's log-density.
real sac_td_target(real reward, real q_min, real alpha_logp, real gamma,
                   b32 terminal);

// Trains `actor` against the twin critics `q1`/`q2`, using `q1_target` and
// `q2_target` as the soft-updated bootstrap targets. All five networks must come
// from the SAC builders (sac_actor_build / sac_q_build) and be pairwise
// distinct. `config` supplies the SAC knobs (tau, alpha, auto_alpha) and the
// shared off-policy settings (epochs, rollout_size, batch_size, warmup_steps,
// train_every, replay_capacity).
void train_sac(Env *env, Net *actor, Net *q1, Net *q2, Net *q1_target,
               Net *q2_target, const TrainCfg *config, Arena *train_arena);

#endif // SAC_H
