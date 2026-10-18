#include "memory.h"
#include "envs/pendulum.h"
#include "layers.h"
#include "rl.h"
#include <stdio.h>
#include <stdlib.h>

// Trains Pendulum-v1 with continuous-action PPO: the policy is a diagonal
// Gaussian (mean head + learned log standard deviation) and every action is a
// real torque clamped to [-2, 2]. The same rollout/GAE/minibatch machinery as
// the discrete trainer is reused; only the policy and the action storage differ.
//
// A well-trained Pendulum returns roughly -200 or better; a random policy sits
// near -1200 (and a policy that never moves costs about -1000), so the number to
// watch is the mean episode return, which should climb from around -1000 to a
// few hundred below zero and stay there.
//
//   ./train_ppo_continuous [seed] [updates] [threads]
//
// The defaults (seed 42, 1200 updates) reproduce the numbers in the README:
// -182.64 greedy after training, from -1200 or so at update 0. `threads` spreads
// the per-env stepping of the rollout over worker threads (0 = serial, the
// default); it cannot change the result, only the wall time — and on a cheap env
// like this one it is slower, see the README's limitations.
int main(int argc, char **argv) {
  const u64 seed = (argc > 1) ? (u64)strtoull(argv[1], NULL, 10) : 42;
  const u32 updates = (argc > 2) ? (u32)atoi(argv[2]) : 1200;
  const u32 threads = (argc > 3) ? (u32)atoi(argv[3]) : 0;

  printf("=== Continuous PPO on Pendulum-v1 (seed: %llu, updates: %u, "
         "threads: %u) ===\n",
         (unsigned long long)seed, updates, threads);

  Arena *arena = arena_new(GiB(1));
  Net *model = net_new(arena);
  Env *env = env_pendulum_create(arena);

  Rng model_rng;
  rng_seed(&model_rng, seed, 1);

  const u32 hidden_dims[] = {64, 64};
  const u32 batch_size = 64;
  // log_std_init = -2 starts the exploration noise at sigma = exp(-2) ~ 0.135
  // radians, a small but non-trivial fraction of the [-2, 2] torque range.
  continuous_actor_critic_build(arena, model, &model_rng, env->obs_dim,
                                hidden_dims, 2, env->act_dim, batch_size, -2.0f);

  TrainCfg cfg = {
      .epochs = updates,
      .rollout_size = 128,
      .max_steps = env->max_steps,
      // lr = 1e-4 is deliberately conservative: Adam's steps are ~lr per
      // parameter per minibatch, and Pendulum's good policies live in a narrow
      // basin. At 3e-4 seeds 42/7/123 peaked around -150 and then fell into the
      // "hang still, apply no torque" basin (-1200); at 2e-4 seed 42 still
      // collapsed. 1e-4 with 5 PPO epochs sits at -130..-240 for every seed
      // tried and does not degrade over 1200 updates.
      .learning_rate = 1e-4f,
      .gamma = 0.9f, // 200-step episodes, so a 10-step horizon is enough
      .lambda = 0.95f,
      .vf_coef = 0.5f,
      // A small exploration bonus keeps the learned sigma from random-walking
      // (the log_std update is mean-zero noise as long as the sampled actions
      // sit inside the box).
      .entropy_coef = 0.01f,
      .seed = seed,
      .optim = OPT_ADAM,
      .max_grad_norm = 0.5f,
      .batch_size = batch_size,
      // PPO-only: 8 parallel envs, 5 passes per rollout, clip range 0.2.
      .num_envs = 8,
      .num_threads = threads,
      .ppo_epochs = 5,
      .ppo_clip = 0.2f,
  };

  train_ppo_continuous(env, model, &cfg, arena);

  Rng eval_rng;
  rng_seed(&eval_rng, seed + 1000, 1);
  EvalResult res = evaluate_continuous(env, model, &eval_rng, 100, true);
  printf("\nFinal Evaluation (100 episodes, deterministic):\n");
  printf("  Mean Return: %.2f | Mean Steps: %.1f\n\n", (double)res.avg_return,
         (double)res.avg_steps);

  arena_dispose(arena);
  return 0;
}
