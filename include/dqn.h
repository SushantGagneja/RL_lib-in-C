#ifndef DQN_H
#define DQN_H

#include "memory.h"
#include "graph.h"
#include "env.h"
#include "rl.h"

// Deep Q-Network training: experience replay, a periodically synchronized
// target network, and an epsilon-greedy behaviour policy.
//
// The Q-network is built with q_net_build() (layers.h). Its rollout graph gives
// one Q value per action for inference, and its batched graph adds
//   batch_input (obs_dim x B) -> batch_logits (act_dim x B) -> cost (MSE)
// against a plain `target` tensor the trainer refills on every gradient step.
//
// The loss is the classic DQN regression in masked form: column b of the target
// holds the one-step bootstrap value for the action that was taken, and a copy
// of the network's own current Q value for every other action. The MSE only
// backpropagates into its prediction operand, so those copies have exactly zero
// error and contribute no gradient — the update is a one-hot squared error on
// the taken action, with no gather/scatter op needed. Parameter updates are
// accumulated by the graph's reverse sweep and applied by the configured
// optimizer (SGD or Adam), with the optional global gradient-norm clip.

// A fixed-capacity ring buffer of transitions, stored as flat arrays (one per
// field) so a sampled batch gathers by copying a column per field. Once full it
// overwrites the oldest transition, so training samples a window of the most
// recent `capacity` steps.
typedef struct {
  u32 obs_dim;
  u32 capacity;
  u32 size;       // valid transitions (<= capacity)
  u32 head;       // next write slot
  real *obs;      // capacity * obs_dim
  real *next_obs; // capacity * obs_dim
  real *rewards;  // capacity
  u32 *actions;   // capacity
  u8 *dones;      // capacity, 0/1: terminal *or* truncated
} ReplayBuf;

// Allocates an empty buffer. Both arguments must be non-zero.
ReplayBuf *replay_new(Arena *arena, u32 capacity, u32 obs_dim);

// Stores one transition, overwriting the oldest once the buffer is full.
// `done` must be true for terminal and truncated steps alike: neither
// bootstraps through the next state.
void replay_add(ReplayBuf *buf, const real *obs, u32 action, real reward,
                const real *next_obs, b32 done);

// Copies every parameter tensor of `src` into `dst`. Both must come from the
// same builder with the same architecture (creation order defines the pairing).
// train_dqn calls this once at the start and again every `target_sync`
// gradient steps.
void dqn_sync_target(const Net *src, Net *dst);

// Epsilon of the linear decay at `env_steps` collected steps: eps_start at 0,
// eps_end at and after eps_decay_steps. Public so logging and tests agree with
// what the trainer actually explores with.
real dqn_epsilon_at(const TrainCfg *config, u32 env_steps);

// Trains `q_model` on a discrete-action `env` for `config->epochs` updates of
// `config->rollout_size` environment steps each (0 => 32), running one gradient
// step per `config->train_every` collected steps (0 => 1) once the replay
// buffer holds `config->warmup_steps` transitions (0 => 1000; a batch is always
// required first). Every `config->target_sync` gradient steps (0 => 500) the
// target network is hard-copied from the online network. Episodes are seeded
// deterministically from `config->seed`, so a run is reproducible.
//
// Both networks must be built by q_net_build() with the same architecture;
// `q_target` is overwritten by the initial sync, so its own initialization does
// not matter. `train_arena` holds the replay buffer, the sampling scratch and
// the optimizer state. `config->batch_size` (0 => 64) is the minibatch size,
// and the optimizer is `config->optim` with `config->learning_rate`, decayed by
// `config->lr_schedule` exactly like the policy-gradient trainers.
void train_dqn(Env *env, Net *q_model, Net *q_target, const TrainCfg *config,
               Arena *train_arena);

#endif // DQN_H
