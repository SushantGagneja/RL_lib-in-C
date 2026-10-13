#include "arena.h"
#include "envs/snake.h"
#include "nn.h"
#include "rl.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  u64 seed = (argc > 1) ? (u64)strtoull(argv[1], NULL, 10) : 42;
  u32 epochs = (argc > 2) ? (u32)atoi(argv[2]) : 200;

  printf("=== Training Snake (seed: %llu, epochs: %u) ===\n",
         (unsigned long long)seed, epochs);

  mem_arena *arena = arena_create(GiB(1));
  model_state *model = model_create(arena);
  Env *env = env_snake_create(arena, 36);

  const u32 hidden_dims[] = {128, 128};
  nn_build_discrete_actor(arena, model, env->obs_dim, hidden_dims, 2,
                          env->act_dim);
  model->forward_graph = build_graph(arena, model, model->output);
  model->cost_graph = build_graph(arena, model, model->cost);

  RLConfig cfg = {
      .epochs = epochs,
      .rollout_size = 64,
      .max_steps = env->max_steps,
      .learning_rate = 0.05f,
      .gamma = 0.99f,
      .seed = seed,
  };

  rl_train_reinforce(env, model, &cfg, arena);

  prng_state eval_rng;
  prng_seed_r(&eval_rng, seed + 1000, 1);
  RLEvalResult res = rl_evaluate(env, model, &eval_rng, 100, true);
  printf("\nFinal Evaluation (100 episodes):\n");
  printf("  Mean Return: %.2f | Mean Steps: %.1f\n\n", res.mean_reward,
         res.mean_steps);

  arena_destroy(arena);
  return 0;
}
