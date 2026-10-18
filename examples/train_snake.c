#include "memory.h"
#include "envs/snake.h"
#include "layers.h"
#include "rl.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  u64 seed = (argc > 1) ? (u64)strtoull(argv[1], NULL, 10) : 42;
  u32 epochs = (argc > 2) ? (u32)atoi(argv[2]) : 200;

  printf("=== Training Snake (seed: %llu, epochs: %u) ===\n",
         (unsigned long long)seed, epochs);

  Arena *arena = arena_new(GiB(1));
  Net *model = net_new(arena);
  Env *env = env_snake_create(arena, 36);

  Rng model_rng;
  rng_seed(&model_rng, seed, 1);

  const u32 hidden_dims[] = {128, 128};
  actor_build(arena, model, &model_rng, env->obs_dim, hidden_dims, 2,
                          env->act_dim, 32);
  model->loss_plan = schedule_from(arena, model, model->cost);

  TrainCfg cfg = {
      .epochs = epochs,
      .rollout_size = 64,
      .max_steps = env->max_steps,
      .learning_rate = 0.001f,
      .gamma = 0.99f,
      .seed = seed,
      .optim = OPT_ADAM,
      .max_grad_norm = 0.5f,
      .batch_size = 32,
  };

  train_reinforce(env, model, &cfg, arena);

  Rng eval_rng;
  rng_seed(&eval_rng, seed + 1000, 1);
  EvalResult res = evaluate(env, model, &eval_rng, 100, true);
  printf("\nFinal Evaluation (100 episodes):\n");
  printf("  Mean Return: %.2f | Mean Steps: %.1f\n\n", res.avg_return,
         res.avg_steps);

  arena_dispose(arena);
  return 0;
}
