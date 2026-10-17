#include "arena.h"
#include "envs/cartpole.h"
#include "nn.h"
#include "rl.h"
#include <stdio.h>
#include <stdlib.h>

// Trains CartPole-v1 with A2C (advantage actor-critic) + GAE.
// Previously this path could not run at all; see the P0 fixes in rl.c/nn.c.
int main(int argc, char **argv) {
  u64 seed = (argc > 1) ? (u64)strtoull(argv[1], NULL, 10) : 42;
  u32 epochs = (argc > 2) ? (u32)atoi(argv[2]) : 200;

  printf("=== A2C on CartPole-v1 (seed: %llu, epochs: %u) ===\n",
         (unsigned long long)seed, epochs);

  mem_arena *arena = arena_create(GiB(1));
  model_state *model = model_create(arena);
  Env *env = env_cartpole_create(arena);

  prng_state model_rng;
  prng_seed_r(&model_rng, seed, 1);

  const u32 hidden_dims[] = {64, 64};
  nn_build_actor_critic(arena, model, &model_rng, env->obs_dim, hidden_dims, 2,
                        env->act_dim, 32);

  RLConfig cfg = {
      .epochs = epochs,
      .rollout_size = 32,
      .max_steps = env->max_steps,
      .learning_rate = 0.001f,
      .gamma = 0.99f,
      .lambda = 0.95f,
      .vf_coef = 0.5f,
      .entropy_coef = 0.01f,
      .seed = seed,
      .optim = OPTIM_ADAM,
      .max_grad_norm = 0.5f,
      .batch_size = 32,
  };

  rl_train_a2c(env, model, &cfg, arena);

  prng_state eval_rng;
  prng_seed_r(&eval_rng, seed + 1000, 1);
  RLEvalResult res = rl_evaluate(env, model, &eval_rng, 100, true);
  printf("\nFinal Evaluation (100 episodes, greedy):\n");
  printf("  Mean Return: %.2f | Mean Steps: %.1f\n\n", res.mean_reward,
         res.mean_steps);

  arena_destroy(arena);
  return 0;
}
