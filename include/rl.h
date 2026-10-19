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

// Learning-rate decay for a training run. LR_CONSTANT keeps `learning_rate`
// for every update; the other two interpolate from `learning_rate` at update 0
// to `lr_end` at the final update.
typedef enum {
  LR_CONSTANT = 0,
  LR_LINEAR = 1,
  LR_COSINE = 2,
} LrSchedule;

// Running statistics for observation normalization and reward scaling,
// Welford-updated while training (one instance serves every environment of a
// run) and read-only afterwards. Both transformations start from a
// standard-normal prior, so the first observations and returns pass through
// nearly unchanged instead of being divided by a near-zero variance.
typedef struct {
  u32 obs_dim;
  u64 count;      // observations folded in
  real *mean;     // [obs_dim]
  real *m2;       // [obs_dim] sum of squared deltas
  real eps;       // variance floor (1e-8)
  real clip_obs;  // observation clip
  u64 ret_count;  // discounted returns folded in
  real ret_mean;
  real ret_m2;
  real gamma;     // discount of the reward-scale return estimate (0.99)
} NormStats;

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
  // Worker threads used to step those envs (0 or 1 => serial, the default). It
  // never changes the trajectories, only how fast they are produced, and it only
  // pays off when one env step is more expensive than a barrier round-trip.
  u32 num_threads;

  // Learning-rate decay. See LrSchedule; ignored when lr_schedule is
  // LR_CONSTANT. Every trainer refreshes Adam's stored rate each update.
  LrSchedule lr_schedule;
  real lr_end;

  // PPO value-loss clipping, relative to the rollout's value estimate (the
  // worse of the clipped and unclipped branches, as in the clipped surrogate).
  // <= 0 disables it.
  real vf_clip;

  // PPO early stop: once the mean approximate KL of an update exceeds
  // 1.5 * target_kl, the remaining epochs of that update are skipped. <= 0
  // disables it.
  real target_kl;

  // Optional linear decay of the entropy bonus across the run, from
  // `entropy_coef` at update 0 to `entropy_coef_end` at the final update.
  b32 anneal_entropy;
  real entropy_coef_end;

  // Observation normalization and reward scaling over the trainer's rollout
  // stream. When either flag is set, `norm` must point at a norm_stats_new()
  // instance: it is folded during training and can then be handed to the
  // *_normed evaluation functions so evaluation sees the same transform.
  b32 norm_obs;
  b32 norm_reward;
  real norm_clip;         // observation clip (<= 0 => 10)
  real norm_reward_clip;  // scaled-reward clip (<= 0 => 10)
  NormStats *norm;

  // DQN-only knobs (train_dqn). `epochs` is the number of updates and
  // `rollout_size` the number of environment steps collected per update. The
  // replay buffer is arena-backed and lives for the training call, after which
  // the online network can be checkpointed and evaluated like any other.
  u32 replay_capacity; // transitions stored (0 => 20000)
  u32 warmup_steps;    // env steps collected before the first gradient step
                       // (0 => 1000; at least one full batch is always needed)
  u32 train_every;     // collected env steps per gradient step (0 => 1)
  u32 target_sync;     // gradient steps between target-network copies (0 => 500)
  real eps_start;      // initial epsilon (<= 0 => 1)
  real eps_end;        // final epsilon (<= 0 => 0.05)
  u32 eps_decay_steps; // env steps over which epsilon decays linearly
                       // (0 => half the run: epochs * rollout_size / 2)

  // SAC-only knobs (train_sac). The DQN block above supplies the shared
  // off-policy settings: `epochs` updates, `rollout_size` env steps per update,
  // `replay_capacity`, `warmup_steps` and `train_every`. `target_sync` is not
  // used: the twin critics are tracked with soft updates instead of copies.
  real tau;            // soft target rate (<= 0 => 0.005)
  real alpha;          // initial temperature (<= 0 => 0.2)
  b32 auto_alpha;      // learn log_alpha against `target_entropy` (default off)
  real alpha_lr;       // temperature learning rate (<= 0 => learning_rate)
  real target_entropy; // entropy target for auto_alpha (<= 0 => -act_dim)
} TrainCfg;

// Allocates arena-backed observation/reward statistics.
NormStats *norm_stats_new(Arena *arena, u32 obs_dim);

// Welford folds: the trainer calls these on every observation (and terminal
// observation) it feeds the model and on every discounted return it tracks.
void norm_stats_fold_obs(NormStats *stats, const real *obs);
void norm_stats_fold_return(NormStats *stats, real discounted_return);

// Maps one observation through the current statistics, clipped to
// stats->clip_obs. `out` must not alias `obs`.
void norm_stats_obs(const NormStats *stats, const real *obs, real *out);

// Current reward scale: 1 / sqrt(var + eps) of the folded discounted returns.
real norm_reward_factor(const NormStats *stats);

// The learning rate / entropy coefficient a trainer uses at `iteration` under
// the configured schedules. `lr_start` is the trainer's resolved starting rate
// (its default differs per trainer).
real train_lr_at(const TrainCfg *config, real lr_start, u32 iteration);
real train_entropy_coef_at(const TrainCfg *config, u32 iteration);

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

// Continuous-action PPO. Same rollout/GAE/minibatching structure as train_ppo,
// but the policy is the diagonal Gaussian produced by
// continuous_actor_critic_build(), so actions are real vectors of
// env->act_dim values clamped to env->action_min/action_max, and the learned
// log standard deviation is updated by the same optimiser. Requires a
// continuous-action environment (env->continuous).
void train_ppo_continuous(Env *env, Net *model, const TrainCfg *config,
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

// Continuous counterparts. `deterministic` uses the policy mean (clamped to the
// environment's bounds) instead of sampling from the Gaussian.
EvalResult evaluate_continuous(Env *env, Net *model, Rng *rng, u32 episodes,
                               b32 deterministic);
EvalResult evaluate_continuous_seeded(Env *env, Net *model, const u64 *seeds,
                                      u32 num_episodes, b32 deterministic);

// Normalized-observation counterparts, for models trained with
// TrainCfg.norm_obs. The observation is mapped through `stats` first; the
// reported return is the raw environment return (reward scaling is a training
// transform only).
EvalResult evaluate_normed(Env *env, Net *model, const NormStats *stats,
                           Rng *rng, u32 episodes, b32 greedy);
EvalResult evaluate_seeded_normed(Env *env, Net *model, const NormStats *stats,
                                  const u64 *seeds, u32 num_episodes,
                                  b32 greedy);
EvalResult evaluate_continuous_normed(Env *env, Net *model,
                                      const NormStats *stats, Rng *rng,
                                      u32 episodes, b32 deterministic);
EvalResult evaluate_continuous_seeded_normed(Env *env, Net *model,
                                             const NormStats *stats,
                                             const u64 *seeds,
                                             u32 num_episodes,
                                             b32 deterministic);

#endif // RL_H
