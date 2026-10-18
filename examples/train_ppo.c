#include "memory.h"
#include "envs/cartpole.h"
#include "layers.h"
#include "rl.h"
#include <stdio.h>
#include <stdlib.h>

// Trains CartPole-v1 with PPO (clipped surrogate objective). The rollout is
// collected from a vectorized environment (num_envs parallel CartPoles), then
// replayed for several epochs of shuffled minibatches.
//
//   ./train_ppo [seed] [updates] [threads]
//
// `threads` spreads the per-env stepping of the rollout over worker threads
// (0 = serial, the default). It cannot change the result, only the wall time —
// and CartPole is far too cheap to profit from it (see the benchmark section of
// the README).
int main(int argc, char **argv) {
  u64 seed = (argc > 1) ? (u64)strtoull(argv[1], NULL, 10) : 42;
  u32 updates = (argc > 2) ? (u32)atoi(argv[2]) : 300;
  u32 threads = (argc > 3) ? (u32)atoi(argv[3]) : 0;

  printf("=== PPO on CartPole-v1 (seed: %llu, updates: %u, threads: %u) ===\n",
         (unsigned long long)seed, updates, threads);

  Arena *arena = arena_new(GiB(1));
  Net *model = net_new(arena);
  Env *env = env_cartpole_create(arena);

  Rng model_rng;
  rng_seed(&model_rng, seed, 1);

  const u32 hidden_dims[] = {64, 64};
  actor_critic_build(arena, model, &model_rng, env->obs_dim, hidden_dims, 2,
                     env->act_dim, 64);

  TrainCfg cfg = {
      .epochs = updates,
      .rollout_size = 128,
      .max_steps = env->max_steps,
      .learning_rate = 3e-4f,
      .gamma = 0.99f,
      .lambda = 0.95f,
      .vf_coef = 0.5f,
      .entropy_coef = 0.01f,
      .seed = seed,
      .optim = OPT_ADAM,
      .max_grad_norm = 0.5f,
      .batch_size = 64,
      // PPO-only: 8 parallel envs, 10 passes per rollout, clip range 0.2.
      .num_envs = 8,
      .num_threads = threads,
      .ppo_epochs = 10,
      .ppo_clip = 0.2f,
  };

  train_ppo(env, model, &cfg, arena);

  Rng eval_rng;
  rng_seed(&eval_rng, seed + 1000, 1);
  EvalResult res = evaluate(env, model, &eval_rng, 100, true);
  printf("\nFinal Evaluation (100 episodes, greedy):\n");
  printf("  Mean Return: %.2f | Mean Steps: %.1f\n\n", res.avg_return,
         res.avg_steps);

  arena_dispose(arena);
  return 0;
}
