#include "memory.h"
#include "envs/pendulum.h"
#include "layers.h"
#include "rl.h"
#include "sac.h"
#include <stdio.h>
#include <stdlib.h>

// Trains Pendulum-v1 with soft actor-critic: twin critics with a split first
// layer, a reparameterised tanh-Gaussian actor, soft target updates, and an
// automatically tuned temperature.
//
//   ./train_sac [seed] [updates]
//
// One update collects `rollout_size` (32 below) environment steps and takes one
// gradient step per collected step, after a 1000-step warmup, so the default
// 400 updates see 12800 environment steps. Evaluation is deterministic (the
// policy mean) on the benchmark harness's seed list (training seed + 5000).

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
  const u32 updates = (argc > 2) ? (u32)atoi(argv[2]) : 400;
  const u32 batch_size = 64;

  printf("=== SAC on Pendulum-v1 (seed: %llu, updates: %u) ===\n",
         (unsigned long long)seed, updates);

  Arena *arena = arena_new(GiB(1));
  Env *env = env_pendulum_create(arena);

  Net *actor = net_new(arena);
  Net *q1 = net_new(arena);
  Net *q2 = net_new(arena);
  Net *q1_target = net_new(arena);
  Net *q2_target = net_new(arena);

  // Separate streams so a change to one network's initialisation cannot shift
  // the others' weights.
  Rng actor_rng;
  Rng q1_rng;
  Rng q2_rng;
  rng_seed(&actor_rng, seed, 1);
  rng_seed(&q1_rng, seed, 2);
  rng_seed(&q2_rng, seed, 3);

  const u32 hidden_dims[] = {64, 64};
  sac_actor_build(arena, actor, &actor_rng, env->obs_dim, hidden_dims, 2,
                  env->act_dim, batch_size, 0.0f);
  sac_q_build(arena, q1, &q1_rng, env->obs_dim, env->act_dim, hidden_dims, 2,
              batch_size);
  sac_q_build(arena, q2, &q2_rng, env->obs_dim, env->act_dim, hidden_dims, 2,
              batch_size);
  // The first soft update overwrites these; separate Nets are all that matters.
  sac_q_build(arena, q1_target, &q1_rng, env->obs_dim, env->act_dim,
              hidden_dims, 2, batch_size);
  sac_q_build(arena, q2_target, &q2_rng, env->obs_dim, env->act_dim,
              hidden_dims, 2, batch_size);

  TrainCfg cfg = {
      .epochs = updates,      // updates, each collecting 32 env steps
      .rollout_size = 32,
      .max_steps = env->max_steps,
      .learning_rate = 3e-4f,
      .gamma = 0.99f,
      .seed = seed,
      .optim = OPT_ADAM,
      .max_grad_norm = 10.0f,
      .batch_size = batch_size,
      .replay_capacity = 100000,
      .warmup_steps = 1000,
      .train_every = 1,
      .tau = 0.005f,
      .alpha = 0.2f,
      .auto_alpha = true,
      .target_entropy = -1.0f, // the -act_dim heuristic, written out
  };
  train_sac(env, actor, q1, q2, q1_target, q2_target, &cfg, arena);

  enum { EVAL_EPISODES = 100 };
  u64 eval_seeds[EVAL_EPISODES];
  make_eval_seeds(eval_seeds, EVAL_EPISODES, seed + 5000);
  const EvalResult det = evaluate_continuous_seeded(
      env, actor, eval_seeds, EVAL_EPISODES, true);
  printf("\nFinal Evaluation (raw env, 100 episodes, harness seeds):\n");
  printf("  Mean:   mean return %7.2f | mean steps %5.1f\n", det.avg_return,
         det.avg_steps);
  printf("  (For comparison: a random policy scores about -1200.)\n\n");

  arena_dispose(arena);
  return 0;
}
