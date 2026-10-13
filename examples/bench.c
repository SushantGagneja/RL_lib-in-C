/**
 * bench.c — Multi-seed benchmark harness for cRL.
 *
 * Evaluates Random, Greedy-Heuristic, and Trained policies on CartPole-v1
 * and Snake. Every number in the README benchmark table must come from
 * running this binary.
 *
 * Usage:
 *   ./bench                  # Run all benchmarks
 *   ./bench cartpole         # CartPole only
 *   ./bench snake            # Snake only
 */

#include "arena.h"
#include "envs/cartpole.h"
#include "envs/snake.h"
#include "nn.h"
#include "rl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_SEEDS 5
#define EVAL_EPISODES 100

static const u64 SEEDS[NUM_SEEDS] = {42, 100, 2024, 777, 9999};

typedef struct {
  f32 mean;
  f32 std;
} Stat;

static Stat compute_stat(const f32 *vals, u32 n) {
  f32 sum = 0.0f, sq = 0.0f;
  for (u32 i = 0; i < n; i++) {
    sum += vals[i];
    sq += vals[i] * vals[i];
  }
  f32 mean = sum / (f32)n;
  f32 var = sq / (f32)n - mean * mean;
  return (Stat){.mean = mean, .std = sqrtf(var > 0.0f ? var : 0.0f)};
}

/* ── Random policy baseline ───────────────────────────────────────── */

static f32 eval_random(Env *env, prng_state *rng, u32 episodes) {
  f32 total = 0.0f;
  for (u32 ep = 0; ep < episodes; ep++) {
    env->reset(env, rng);
    for (u32 t = 0; t < env->max_steps; t++) {
      u32 act = prng_randn_r(rng, env->act_dim);
      b32 done = false, trunc = false;
      total += env->step(env, act, &done, &trunc);
      if (done || trunc) break;
    }
  }
  return total / (f32)episodes;
}

/* ── Greedy heuristic baseline for CartPole ────────────────────────  */

static f32 eval_cartpole_greedy(Env *env, prng_state *rng, u32 episodes) {
  f32 total = 0.0f;
  f32 obs[4];
  for (u32 ep = 0; ep < episodes; ep++) {
    env->reset(env, rng);
    for (u32 t = 0; t < env->max_steps; t++) {
      env->observe(env, obs);
      // PD controller: push right if angle + angular_vel > 0
      u32 act = (obs[2] + obs[3] > 0.0f) ? 1 : 0;
      b32 done = false, trunc = false;
      total += env->step(env, act, &done, &trunc);
      if (done || trunc) break;
    }
  }
  return total / (f32)episodes;
}

/* ── Greedy heuristic baseline for Snake ──────────────────────────  */

static f32 eval_snake_greedy(Env *env, prng_state *rng, u32 episodes) {
  f32 total = 0.0f;
  u32 grid_size = env->obs_dim / 3; // obs_dim = 3 * grid_size
  u32 side = (u32)sqrtf((f32)grid_size);
  f32 *obs = malloc(sizeof(f32) * env->obs_dim);

  for (u32 ep = 0; ep < episodes; ep++) {
    env->reset(env, rng);
    for (u32 t = 0; t < env->max_steps; t++) {
      env->observe(env, obs);

      // Find head and food positions from one-hot channels
      i32 hx = -1, hy = -1, fx = -1, fy = -1;
      for (u32 i = 0; i < grid_size; i++) {
        if (obs[i] == 1.0f) {
          hx = (i32)(i % side);
          hy = (i32)(i / side);
        }
        if (obs[grid_size + i] == 1.0f) {
          fx = (i32)(i % side);
          fy = (i32)(i / side);
        }
      }

      // Step along axis of max food displacement
      u32 act = 0;
      if (hx >= 0 && fx >= 0) {
        i32 dx = fx - hx;
        i32 dy = fy - hy;
        if (abs(dx) >= abs(dy)) {
          act = (dx > 0) ? 1 : 0; // RIGHT : LEFT
        } else {
          act = (dy > 0) ? 2 : 3; // UP : DOWN
        }
      }

      b32 done = false, trunc = false;
      total += env->step(env, act, &done, &trunc);
      if (done || trunc) break;
    }
  }
  free(obs);
  return total / (f32)episodes;
}

/* ── Trained policy evaluation ────────────────────────────────────  */

static f32 train_and_eval_cartpole(mem_arena *arena, u64 seed, u32 epochs) {
  model_state *model = model_create(arena);
  Env *env = env_cartpole_create(arena);

  prng_state model_rng;
  prng_seed_r(&model_rng, seed, 1);

  const u32 hidden[] = {64, 64};
  nn_build_discrete_actor(arena, model, &model_rng, env->obs_dim, hidden, 2,
                          env->act_dim, 32);
  model->forward_graph = build_graph(arena, model, model->output);
  model->cost_graph = build_graph(arena, model, model->cost);

  RLConfig cfg = {
      .epochs = epochs,
      .rollout_size = 32,
      .max_steps = env->max_steps,
      .learning_rate = 0.005f,
      .gamma = 0.99f,
      .seed = seed,
      .optim = OPTIM_ADAM,
      .max_grad_norm = 0.5f,
      .batch_size = 32,
  };

  rl_train_reinforce(env, model, &cfg, arena);

  prng_state eval_rng;
  prng_seed_r(&eval_rng, seed + 5000, 1);
  RLEvalResult res = rl_evaluate(env, model, &eval_rng, EVAL_EPISODES, true);
  return res.mean_reward;
}

static f32 train_and_eval_snake(mem_arena *arena, u64 seed, u32 epochs) {
  model_state *model = model_create(arena);
  Env *env = env_snake_create(arena, 36);

  prng_state model_rng;
  prng_seed_r(&model_rng, seed, 1);

  const u32 hidden[] = {128, 128};
  nn_build_discrete_actor(arena, model, &model_rng, env->obs_dim, hidden, 2,
                          env->act_dim, 32);
  model->forward_graph = build_graph(arena, model, model->output);
  model->cost_graph = build_graph(arena, model, model->cost);

  RLConfig cfg = {
      .epochs = epochs,
      .rollout_size = 64,
      .max_steps = env->max_steps,
      .learning_rate = 0.001f,
      .gamma = 0.99f,
      .seed = seed,
      .optim = OPTIM_ADAM,
      .max_grad_norm = 0.5f,
      .batch_size = 32,
  };

  rl_train_reinforce(env, model, &cfg, arena);

  prng_state eval_rng;
  prng_seed_r(&eval_rng, seed + 5000, 1);
  RLEvalResult res = rl_evaluate(env, model, &eval_rng, EVAL_EPISODES, true);
  return res.mean_reward;
}

/* ── Print helpers ────────────────────────────────────────────────  */

static void print_row(const char *policy, Stat s) {
  printf("| %-30s | %8.2f ± %5.2f |\n", policy, s.mean, s.std);
}

static void print_table_header(const char *env_name) {
  printf("\n╔══════════════════════════════════════════════════╗\n");
  printf("║  %s Benchmark (%d seeds × %d episodes)  \n",
         env_name, NUM_SEEDS, EVAL_EPISODES);
  printf("╚══════════════════════════════════════════════════╝\n\n");
  printf("| %-30s | %16s |\n", "Policy", "Return (mean±std)");
  printf("|%s|%s|\n",
         "--------------------------------", "------------------");
}

/* ── Main ─────────────────────────────────────────────────────────  */

int main(int argc, char **argv) {
  b32 do_cartpole = true;
  b32 do_snake = true;

  if (argc > 1) {
    if (strcmp(argv[1], "cartpole") == 0) {
      do_snake = false;
    } else if (strcmp(argv[1], "snake") == 0) {
      do_cartpole = false;
    }
  }

  if (do_cartpole) {
    print_table_header("CartPole-v1");

    f32 random_vals[NUM_SEEDS], greedy_vals[NUM_SEEDS],
        trained_vals[NUM_SEEDS];

    for (u32 s = 0; s < NUM_SEEDS; s++) {
      mem_arena *arena = arena_create(GiB(1));
      Env *env = env_cartpole_create(arena);
      prng_state rng;
      prng_seed_r(&rng, SEEDS[s], 1);

      random_vals[s] = eval_random(env, &rng, EVAL_EPISODES);
      greedy_vals[s] = eval_cartpole_greedy(env, &rng, EVAL_EPISODES);
      trained_vals[s] = train_and_eval_cartpole(arena, SEEDS[s], 50);

      printf("  [seed %llu] Random: %.1f | Greedy: %.1f | Trained: %.1f\n",
             (unsigned long long)SEEDS[s], random_vals[s], greedy_vals[s],
             trained_vals[s]);

      arena_destroy(arena);
    }

    printf("\n");
    print_row("Random", compute_stat(random_vals, NUM_SEEDS));
    print_row("Greedy (PD controller)", compute_stat(greedy_vals, NUM_SEEDS));
    print_row("Trained (REINFORCE, 50ep)", compute_stat(trained_vals, NUM_SEEDS));
    printf("\n");
  }

  if (do_snake) {
    print_table_header("Snake 6×6");

    f32 random_vals[NUM_SEEDS], greedy_vals[NUM_SEEDS],
        trained_vals[NUM_SEEDS];

    for (u32 s = 0; s < NUM_SEEDS; s++) {
      mem_arena *arena = arena_create(GiB(1));
      Env *env = env_snake_create(arena, 36);
      prng_state rng;
      prng_seed_r(&rng, SEEDS[s], 1);

      random_vals[s] = eval_random(env, &rng, EVAL_EPISODES);
      greedy_vals[s] = eval_snake_greedy(env, &rng, EVAL_EPISODES);
      trained_vals[s] = train_and_eval_snake(arena, SEEDS[s], 200);

      printf("  [seed %llu] Random: %.1f | Greedy: %.1f | Trained: %.1f\n",
             (unsigned long long)SEEDS[s], random_vals[s], greedy_vals[s],
             trained_vals[s]);

      arena_destroy(arena);
    }

    printf("\n");
    print_row("Random", compute_stat(random_vals, NUM_SEEDS));
    print_row("Greedy (food heuristic)", compute_stat(greedy_vals, NUM_SEEDS));
    print_row("Trained (REINFORCE, 200ep)", compute_stat(trained_vals, NUM_SEEDS));
    printf("\n");
  }

  return 0;
}
