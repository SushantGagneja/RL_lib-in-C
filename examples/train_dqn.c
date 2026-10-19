#include "memory.h"
#include "envs/cartpole.h"
#include "layers.h"
#include "rl.h"
#include "dqn.h"
#include <stdio.h>
#include <stdlib.h>

// Trains CartPole-v1 with a DQN (experience replay, a target network that is
// hard-copied every 250 gradient steps, and epsilon-greedy exploration decaying
// from 1.0 to 0.05 over the first half of the run).
//
//   ./train_dqn [seed] [updates]
//
// One update collects 32 environment steps and takes 32 gradient steps (one per
// collected step, after a 500-step warmup), so the default 500 updates see
// 16000 environment steps. Evaluation is greedy on the benchmark harness's
// seed list (training seed + 5000).

// The benchmark harness evaluates every policy on the same PRNG-derived seed
// list (base = training seed + 5000, stream 11). Reuse that protocol here so
// these returns can be dropped into the harness table unchanged.
static void make_eval_seeds(u64 *seeds, u32 n, u64 base) {
  Rng rng;
  rng_seed(&rng, base, 11);
  for (u32 i = 0; i < n; i++) {
    const u64 hi = rng_u32(&rng);
    const u64 lo = rng_u32(&rng);
    seeds[i] = (hi << 32) ^ lo;
  }
}

int main(int argc, char **argv) {
  const u64 seed = (argc > 1) ? (u64)strtoull(argv[1], NULL, 10) : 42;
  const u32 updates = (argc > 2) ? (u32)atoi(argv[2]) : 500;

  printf("=== DQN on CartPole-v1 (seed: %llu, updates: %u) ===\n",
         (unsigned long long)seed, updates);

  Arena *arena = arena_new(GiB(1));
  Net *q_model = net_new(arena);
  Net *q_target = net_new(arena);
  Env *env = env_cartpole_create(arena);

  Rng model_rng;
  rng_seed(&model_rng, seed, 1);

  const u32 hidden_dims[] = {64, 64};
  q_net_build(arena, q_model, &model_rng, env->obs_dim, hidden_dims, 2,
              env->act_dim, 64);
  // The earliest sync in train_dqn overwrites these weights; a separate Net is
  // all that matters.
  q_net_build(arena, q_target, &model_rng, env->obs_dim, hidden_dims, 2,
              env->act_dim, 64);

  TrainCfg cfg = {
      .epochs = updates,          // updates, each collecting 32 env steps
      .rollout_size = 32,
      .max_steps = env->max_steps,
      .learning_rate = 1e-3f,
      .gamma = 0.99f,
      .seed = seed,
      .optim = OPT_ADAM,
      .max_grad_norm = 10.0f,
      .batch_size = 64,
      .replay_capacity = 20000,
      .warmup_steps = 500,
      .train_every = 1,
      .target_sync = 250,
      .eps_start = 1.0f,
      .eps_end = 0.05f,
      .eps_decay_steps = 0, // 0 => decay over half the run (8000 steps here)
  };
  train_dqn(env, q_model, q_target, &cfg, arena);

  enum { EVAL_EPISODES = 100 };
  u64 eval_seeds[EVAL_EPISODES];
  make_eval_seeds(eval_seeds, EVAL_EPISODES, seed + 5000);
  const EvalResult greedy =
      evaluate_seeded(env, q_model, eval_seeds, EVAL_EPISODES, true);
  printf("\nFinal Evaluation (raw env, 100 episodes, harness seeds):\n");
  printf("  Greedy: mean return %7.2f | mean steps %5.1f\n", greedy.avg_return,
         greedy.avg_steps);
  printf("  (For comparison: a random policy scores ~22.)\n\n");

  arena_dispose(arena);
  return 0;
}
