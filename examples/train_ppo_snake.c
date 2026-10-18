#include "memory.h"
#include "envs/snake.h"
#include "layers.h"
#include "rl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Trains 6x6 Snake with PPO over a vectorized environment.
//
//   ./train_ppo_snake [seed] [updates] [shaping_coef] [threads]
//
// The environment's reward is sparse (once per food, +20). The default is to
// train on it as-is: in the reported runs the sparse reward was learnable at
// this budget, and a control trained with no shaping matched the shaped runs,
// so shaping is offered rather than required. Pass a non-zero coefficient to
// enable it; the term is the standard telescoping potential
//
//   r' = r + coef * (gamma * Phi(s') - Phi(s)),
//   Phi(s) = -manhattan(head, food)
//
// (Ng, Harada & Russell 1999): it leaves the set of optimal policies unchanged
// and only tells the learner which way the food is. It is applied by a wrapper
// environment used *for training only*; the final evaluation always runs on the
// raw environment, so the numbers are comparable with the other Snake baselines
// in the README. The per-update returns the trainer prints are shaped, the
// evaluation at the end is not. The final evaluation runs on the benchmark
// harness's seed list (training seed + 5000) so the numbers slot straight into
// its tables.
//
// The wrapper is also a worked example of composing environments: it clones its
// inner environment, forwards observe(), and owns nothing but two observation
// buffers, so the vectorized wrapper can replicate it like any other env.

typedef struct {
  Env base;
  Env *inner;
  real *prev_obs; // Phi(s) for the shaping term
  real *cur_obs;  // scratch for Phi(s')
  u32 grid;
  real coef;
  real gamma;
} ShapedSnake;

// Phi(s) from the observation alone (channel 0 is the head, channel 1 the food,
// both grid-major as x = i % grid, y = i / grid).
static real snake_potential(const ShapedSnake *s, const real *obs) {
  const u32 g = s->grid;
  i32 head = -1;
  i32 food = -1;
  for (u32 i = 0; i < g; i++) {
    if (obs[i] > 0.5f) {
      head = (i32)i;
    }
    if (obs[g + i] > 0.5f) {
      food = (i32)i;
    }
  }
  if (head < 0 || food < 0) {
    return 0.0f;
  }
  const i32 hx = head % (i32)g;
  const i32 hy = head / (i32)g;
  const i32 fx = food % (i32)g;
  const i32 fy = food / (i32)g;
  const i32 dx = (hx > fx) ? (hx - fx) : (fx - hx);
  const i32 dy = (hy > fy) ? (hy - fy) : (fy - hy);
  return -(real)(dx + dy);
}

static void shaped_reset(Env *env, u64 seed) {
  ShapedSnake *s = (ShapedSnake *)env;
  s->inner->reset(s->inner, seed);
  s->inner->observe(s->inner, s->prev_obs);
}

static real shaped_step(Env *env, u32 action, b32 *done, b32 *truncated) {
  ShapedSnake *s = (ShapedSnake *)env;
  const real reward = s->inner->step(s->inner, action, done, truncated);
  s->inner->observe(s->inner, s->cur_obs);

  // A terminal state absorbs: Phi(s') = 0 there, per the shaping theorem. A
  // *truncated* episode is not terminal, so it keeps its potential.
  const real phi_next = *done ? 0.0f : snake_potential(s, s->cur_obs);
  const real phi = snake_potential(s, s->prev_obs);
  real *swap = s->prev_obs;
  s->prev_obs = s->cur_obs;
  s->cur_obs = swap;
  return reward + s->coef * (s->gamma * phi_next - phi);
}

static void shaped_observe(Env *env, real *obs_out) {
  ShapedSnake *s = (ShapedSnake *)env;
  s->inner->observe(s->inner, obs_out);
}

static Env *shaped_clone(Arena *arena, const Env *src) {
  const ShapedSnake *s = (const ShapedSnake *)src;
  ShapedSnake *dst = ARENA_OBJ(arena, ShapedSnake);
  dst->base = s->base;
  dst->inner = s->inner->clone(arena, s->inner);
  dst->prev_obs = ARENA_ARR(arena, real, s->base.obs_dim);
  dst->cur_obs = ARENA_ARR(arena, real, s->base.obs_dim);
  dst->grid = s->grid;
  dst->coef = s->coef;
  dst->gamma = s->gamma;
  memcpy(dst->prev_obs, s->prev_obs, sizeof(real) * s->base.obs_dim);
  return (Env *)dst;
}

static Env *shaped_snake_create(Arena *arena, Env *inner, real coef,
                                real gamma) {
  ShapedSnake *s = ARENA_OBJ(arena, ShapedSnake);
  s->base = *inner;
  s->base.reset = shaped_reset;
  s->base.step = shaped_step;
  s->base.observe = shaped_observe;
  s->base.clone = shaped_clone;
  s->inner = inner;
  s->prev_obs = ARENA_ARR(arena, real, inner->obs_dim);
  s->cur_obs = ARENA_ARR(arena, real, inner->obs_dim);
  s->grid = inner->obs_dim / 3u;
  s->coef = coef;
  s->gamma = gamma;
  shaped_reset((Env *)s, 1);
  return (Env *)s;
}

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
  const u32 updates = (argc > 2) ? (u32)atoi(argv[2]) : 1500;
  const real coef = (argc > 3) ? (real)atof(argv[3]) : 0.0f;
  const u32 threads = (argc > 4) ? (u32)atoi(argv[4]) : 0;

  printf("=== PPO on Snake 6x6 (seed: %llu, updates: %u, shaping: %.2f, "
         "threads: %u) ===\n",
         (unsigned long long)seed, updates, (double)coef, threads);

  Arena *arena = arena_new(GiB(1));
  Net *model = net_new(arena);
  Env *env = env_snake_create(arena, 36);

  Rng model_rng;
  rng_seed(&model_rng, seed, 1);

  const u32 hidden_dims[] = {128, 128};
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
      .num_envs = 8,
      .num_threads = threads,
      .ppo_epochs = 10,
      .ppo_clip = 0.2f,
  };

  Env *train_env =
      (coef > 0.0f) ? shaped_snake_create(arena, env, coef, cfg.gamma) : env;
  train_ppo(train_env, model, &cfg, arena);

  // The raw environment, so these returns are on the same scale as the other
  // Snake baselines (a food is +20, a win +100 on top), on the benchmark
  // harness's evaluation seeds (training seed + 5000).
  enum { EVAL_EPISODES = 100 };
  u64 eval_seeds[EVAL_EPISODES];
  make_eval_seeds(eval_seeds, EVAL_EPISODES, seed + 5000);
  const EvalResult greedy =
      evaluate_seeded(env, model, eval_seeds, EVAL_EPISODES, true);
  const EvalResult stochastic =
      evaluate_seeded(env, model, eval_seeds, EVAL_EPISODES, false);
  printf("\nFinal Evaluation (raw env, %d episodes, harness seeds):\n",
         (int)EVAL_EPISODES);
  printf("  Greedy:     mean return %7.2f | mean steps %5.1f\n",
         (double)greedy.avg_return, (double)greedy.avg_steps);
  printf("  Stochastic: mean return %7.2f | mean steps %5.1f\n",
         (double)stochastic.avg_return, (double)stochastic.avg_steps);
  printf("  (For comparison: random ~-5, BFS planner ~299, "
         "REINFORCE ~40)\n\n");

  arena_dispose(arena);
  return 0;
}
