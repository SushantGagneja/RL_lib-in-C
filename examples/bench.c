/**
 * bench.c — Multi-seed benchmark harness for RL_lib-in-C.
 *
 * Evaluates Random, a stronger Greedy baseline, and a Trained policy on
 * CartPole-v1 and Snake, using identical evaluation seeds for every policy so
 * the comparison is apples-to-apples. Statistics use Welford's method in f64
 * and report the sample (n-1) standard deviation.
 *
 * Output is written to bench_results.md and echoed to stdout.
 *
 * Usage:
 *   ./rl_lib_bench                  # both environments
 *   ./rl_lib_bench cartpole         # CartPole only
 *   ./rl_lib_bench snake            # Snake only
 */

#include "memory.h"
#include "envs/cartpole.h"
#include "envs/snake.h"
#include "layers.h"
#include "rl.h"
#include "simd.h"
#include "vecenv.h"
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h> // QueryPerformanceCounter for the wall-clock sections
#endif

#define NUM_SEEDS 5
#define EVAL_EPISODES 100

static const u64 SEEDS[NUM_SEEDS] = {42, 100, 2024, 777, 9999};

static FILE *g_out = NULL;

// Prints to stdout and, if open, to the markdown results file.
static void emit(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);
  if (g_out != NULL) {
    va_start(args, fmt);
    vfprintf(g_out, fmt, args);
    va_end(args);
  }
}

typedef struct {
  double mean;
  double std; // sample (n-1) std
  double worst;
  double best;
} Stat;

// Welford's online algorithm in double: numerically stable and returns the
// sample standard deviation (n-1), unlike the naive E[x^2]-mean^2 in real.
static Stat compute_stat(const double *v, u32 n) {
  double mean = 0.0, m2 = 0.0;
  double worst = v[0], best = v[0];
  u32 count = 0;
  for (u32 i = 0; i < n; i++) {
    count++;
    double d = v[i] - mean;
    mean += d / (double)count;
    m2 += d * (v[i] - mean);
    if (v[i] < worst) worst = v[i];
    if (v[i] > best) best = v[i];
  }
  double std = (count > 1) ? sqrt(m2 / (double)(count - 1)) : 0.0;
  return (Stat){.mean = mean, .std = std, .worst = worst, .best = best};
}

// Identical per-episode start seeds for every policy.
static void make_eval_seeds(u64 *seeds, u32 n, u64 base) {
  Rng r;
  rng_seed(&r, base, 11);
  for (u32 i = 0; i < n; i++) {
    u64 hi = rng_u32(&r);
    u64 lo = rng_u32(&r);
    seeds[i] = (hi << 32) ^ lo;
  }
}

/* ── Random baseline ──────────────────────────────────────────────── */

static double eval_random(Env *env, const u64 *seeds, u32 n) {
  double total = 0.0;
  for (u32 ep = 0; ep < n; ep++) {
    env->reset(env, seeds[ep]);
    Rng r;
    rng_seed(&r, seeds[ep] ^ 0xdeadbeefULL, 1);
    for (u32 t = 0; t < env->max_steps; t++) {
      u32 act = rng_below(&r, env->act_dim);
      b32 done = false, trunc = false;
      total += env->step(env, act, &done, &trunc);
      if (done || trunc) break;
    }
  }
  return total / (double)n;
}

/* ── Greedy CartPole baseline (PD controller) ─────────────────────── */

static double eval_cartpole_greedy(Env *env, const u64 *seeds, u32 n) {
  double total = 0.0;
  real obs[4];
  for (u32 ep = 0; ep < n; ep++) {
    env->reset(env, seeds[ep]);
    for (u32 t = 0; t < env->max_steps; t++) {
      env->observe(env, obs);
      u32 act = (obs[2] + obs[3] > 0.0f) ? 1 : 0;
      b32 done = false, trunc = false;
      total += env->step(env, act, &done, &trunc);
      if (done || trunc) break;
    }
  }
  return total / (double)n;
}

/* ── Greedy Snake baseline (BFS to food, avoiding body/walls) ─────── */

// Action encoding matches envs/snake.c: LEFT=0, RIGHT=1, UP=2, DOWN=3
// (UP increases y, DOWN decreases y).
static u32 snake_greedy_action(const real *obs, u32 grid, u32 side) {
  if (grid > 256) return 0;

  i32 hx = -1, hy = -1, fx = -1, fy = -1;
  for (u32 i = 0; i < grid; i++) {
    if (obs[i] == 1.0f) {
      hx = (i32)(i % side);
      hy = (i32)(i / side);
    }
    if (obs[grid + i] == 1.0f) {
      fx = (i32)(i % side);
      fy = (i32)(i / side);
    }
  }
  if (hx < 0 || fx < 0) return 0;

  bool blocked[256];
  for (u32 i = 0; i < grid; i++) blocked[i] = (obs[2 * grid + i] == 1.0f);
  i32 h_idx = hy * (i32)side + hx;
  blocked[h_idx] = false; // the head is the start, not an obstacle

  static const i32 DX[4] = {-1, 1, 0, 0};
  static const i32 DY[4] = {0, 0, 1, -1};

  i32 dist[256], par[256], q[256];
  for (u32 i = 0; i < grid; i++) {
    dist[i] = -1;
    par[i] = -1;
  }
  u32 head_q = 0, tail_q = 0;
  q[tail_q++] = h_idx;
  dist[h_idx] = 0;
  i32 food_idx = fy * (i32)side + fx;

  while (head_q < tail_q) {
    i32 cur = q[head_q++];
    if (cur == food_idx) break;
    i32 cx = cur % (i32)side, cy = cur / (i32)side;
    for (u32 d = 0; d < 4; d++) {
      i32 nx = cx + DX[d], ny = cy + DY[d];
      if (nx < 0 || nx >= (i32)side || ny < 0 || ny >= (i32)side) continue;
      i32 nidx = ny * (i32)side + nx;
      if (dist[nidx] != -1) continue;
      if (blocked[nidx] && nidx != food_idx) continue;
      dist[nidx] = dist[cur] + 1;
      par[nidx] = cur;
      q[tail_q++] = nidx;
    }
  }

  if (dist[food_idx] >= 0 && par[food_idx] >= 0) {
    // Walk back to the first step away from the head.
    i32 step = food_idx;
    while (par[step] != h_idx && par[step] != -1) {
      step = par[step];
    }
    i32 sx = step % (i32)side, sy = step / (i32)side;
    if (sx > hx) return 1; // RIGHT
    if (sx < hx) return 0; // LEFT
    if (sy > hy) return 2; // UP
    if (sy < hy) return 3; // DOWN
    return 0;
  }

  // No path to food: take any safe move (avoid walls and body), else turn.
  for (u32 d = 0; d < 4; d++) {
    i32 nx = hx + DX[d], ny = hy + DY[d];
    if (nx < 0 || nx >= (i32)side || ny < 0 || ny >= (i32)side) continue;
    i32 nidx = ny * (i32)side + nx;
    if (blocked[nidx]) continue;
    if (d == 0) return 0;
    if (d == 1) return 1;
    if (d == 2) return 2;
    return 3;
  }
  return 0;
}

static double eval_snake_greedy(Env *env, const u64 *seeds, u32 n) {
  double total = 0.0;
  u32 grid = env->obs_dim / 3;
  u32 side = (u32)sqrt((double)grid);
  real *obs = malloc(sizeof(real) * env->obs_dim);

  for (u32 ep = 0; ep < n; ep++) {
    env->reset(env, seeds[ep]);
    for (u32 t = 0; t < env->max_steps; t++) {
      env->observe(env, obs);
      u32 act = snake_greedy_action(obs, grid, side);
      b32 done = false, trunc = false;
      total += env->step(env, act, &done, &trunc);
      if (done || trunc) break;
    }
  }
  free(obs);
  return total / (double)n;
}

/* ── Training ─────────────────────────────────────────────────────── */

static Net *train_cartpole(Arena *arena, u64 seed, u32 epochs) {
  Net *model = net_new(arena);
  Env *env = env_cartpole_create(arena);
  Rng model_rng;
  rng_seed(&model_rng, seed, 1);

  const u32 hidden[] = {64, 64};
  actor_build(arena, model, &model_rng, env->obs_dim, hidden, 2,
                          env->act_dim, 32);
  model->loss_plan = schedule_from(arena, model, model->cost);

  TrainCfg cfg = {
      .epochs = epochs,
      .rollout_size = 32,
      .max_steps = env->max_steps,
      .learning_rate = 0.005f,
      .gamma = 0.99f,
      .seed = seed,
      .optim = OPT_ADAM,
      .max_grad_norm = 0.5f,
      .batch_size = 32,
  };
  train_reinforce(env, model, &cfg, arena);
  return model;
}

static Net *train_snake(Arena *arena, u64 seed, u32 epochs) {
  Net *model = net_new(arena);
  Env *env = env_snake_create(arena, 36);
  Rng model_rng;
  rng_seed(&model_rng, seed, 1);

  const u32 hidden[] = {128, 128};
  actor_build(arena, model, &model_rng, env->obs_dim, hidden, 2,
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
  return model;
}

/* ── Throughput micro-benchmarks ─────────────────────────────────── */

static double now_seconds(void) {
  return (double)clock() / (double)CLOCKS_PER_SEC;
}

// Environment stepping throughput with random actions (no rendering).
static void bench_env_throughput(void) {
  emit("\n=== Throughput ===\n");
  emit("\n`Env.step()` throughput (random policy, single core, excludes "
       "`observe()`):\n\n");
  emit("| Environment | Steps/sec |\n|---|---|\n");

  const char *names[2] = {"CartPole-v1", "Snake 6x6"};
  for (u32 e = 0; e < 2; e++) {
    Arena *arena = arena_new(GiB(1));
    Env *env = (e == 0) ? env_cartpole_create(arena) : env_snake_create(arena, 36);
    Rng r;
    rng_seed(&r, 12345, 7);
    env->reset(env, 1);

    const u64 target = 200000;
    u64 steps = 0;
    double t0 = now_seconds();
    while (steps < target) {
      u32 act = rng_below(&r, env->act_dim);
      b32 done = false, trunc = false;
      env->step(env, act, &done, &trunc);
      if (done || trunc) env->reset(env, rng_u32(&r));
      steps++;
    }
    double dt = now_seconds() - t0;
    double rate = (double)steps / dt;
    printf("  [throughput] %-12s %8.0f steps/sec\n", names[e], rate);
    emit("| %s | %.0f |\n", names[e], rate);
    arena_dispose(arena);
  }
}

// Naive triple-loop GEMM used as the comparison point for the NN kernel.
static void matmul_naive(Mat *out, const Mat *a, const Mat *b) {
  for (u32 i = 0; i < out->rows; i++) {
    for (u32 j = 0; j < out->cols; j++) {
      real sum = 0.0f;
      for (u32 k = 0; k < a->cols; k++) {
        sum += a->data[i * a->cols + k] * b->data[k * b->cols + j];
      }
      out->data[i * out->cols + j] = sum;
    }
  }
}

// Time `reps` multiplies of an M x K by K x N and return GFLOPS.
static double time_matmul(u32 M, u32 K, u32 N, u32 reps, b32 want_naive) {
  Arena *arena = arena_new(MiB(64));
  Mat *a = mat_new(arena, M, K);
  Mat *b = mat_new(arena, K, N);
  Mat *c = mat_new(arena, M, N);
  rng_seed_g(2024, 1);
  mat_rand(a, -1.0f, 1.0f);
  mat_rand(b, -1.0f, 1.0f);

  volatile real sink = 0.0f;
  const double t0 = now_seconds();
  for (u32 i = 0; i < reps; i++) {
    if (want_naive) {
      matmul_naive(c, a, b);
    } else {
      mat_mm(c, a, b, 1, 0, 0);
    }
    sink += c->data[0];
  }
  const double dt = now_seconds() - t0;
  (void)sink;

  const double flops = 2.0 * (double)M * (double)K * (double)N * (double)reps;
  arena_dispose(arena);
  return flops / dt / 1e9;
}

static const char *precision_name(void) {
  return sizeof(real) == sizeof(double) ? "f64" : "f32";
}

static void bench_matmul_throughput(void) {
  // A large square case (compute/L2 bound) and a representative RL layer
  // (64 features x 64-sample batch), where the register blocking matters most.
  const u32 shapes[2][3] = {{256, 256, 256}, {64, 64, 64}};
  const u32 reps_lib[2] = {200, 20000};
  const u32 reps_naive[2] = {2, 200};

  emit("\nMatMul (single core, -O3, `%s` backend, %s):\n\n", simd_backend_name(),
       precision_name());
  emit("| Shape (M x K x N) | `mat_mm` GFLOPS | Naive GFLOPS |\n|---|---|---|\n");

  for (u32 s = 0; s < 2; s++) {
    const u32 M = shapes[s][0];
    const u32 K = shapes[s][1];
    const u32 N = shapes[s][2];
    const double g_lib = time_matmul(M, K, N, reps_lib[s], false);
    const double g_naive = time_matmul(M, K, N, reps_naive[s], true);
    emit("| %ux%ux%u | %.3f | %.3f |\n", M, K, N, g_lib, g_naive);
    printf("  [throughput] matmul %ux%ux%u %.3f GFLOPS | naive %.3f GFLOPS\n", M,
           K, N, g_lib, g_naive);
  }
}

// Throughput of stepping several environments through the vectorized wrapper.
static void bench_vecenv_throughput(void) {
  const u32 num_envs = 8;
  Arena *arena = arena_new(GiB(1));
  Env *proto = env_cartpole_create(arena);
  VecEnv *vec = vecenv_new(arena, proto, num_envs);
  vecenv_reset(vec, 1);

  Rng r;
  rng_seed(&r, 54321, 7);
  u32 *acts = malloc(sizeof(u32) * num_envs);

  const u64 target = 200000; // env-steps
  u64 steps = 0;
  double t0 = now_seconds();
  while (steps < target) {
    vecenv_sample_uniform(vec, &r, acts);
    vecenv_step(vec, acts);
    steps += num_envs;
  }
  double dt = now_seconds() - t0;
  double rate = (double)steps / dt;

  emit("\n`VecEnv` throughput (%u synchronized CartPole envs):\n\n", num_envs);
  emit("| Implementation | Env steps/sec |\n|---|---|\n");
  emit("| `vecenv_step` (x%u) | %.0f |\n", num_envs, rate);
  printf("  [throughput] vecenv x%u %8.0f env-steps/sec\n", num_envs, rate);

  free(acts);
  arena_dispose(arena);
}

// Wall-clock seconds. The scalar benchmarks use CPU time (`clock()`), which is
// the steadier metric for single-threaded work, but a threaded speedup is
// invisible in CPU time by construction — the same work just moves to another
// core — so the worker pool is measured on the wall clock.
static double now_wall_seconds(void) {
#if defined(_WIN32)
  // The Windows monotonic clock is the performance counter; clock_gettime is not
  // part of the C library there.
  LARGE_INTEGER freq;
  LARGE_INTEGER now;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&now);
  return (double)now.QuadPart / (double)freq.QuadPart;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

// ── Threaded VecEnv scaling (synthetic) ──────────────────────────────
//
// The worker pool only pays off when a single env step costs more than the
// barrier round-trip (tens of microseconds). CartPole and Pendulum are far
// below that, so this measures a deliberately artificial env whose step burns a
// fixed amount of floating-point work — a known cost per step and nothing more.
// What is being measured is the pool, not the environment.

#define BURN_SINF 4000

typedef struct {
  Env base;
  real x;
  u32 count;
} BurnEnv;

// Keeps the optimiser from deleting the loop.
static volatile real g_burn_sink;

static real burn_work(real x) {
  for (u32 i = 0; i < BURN_SINF; i++) {
    x = real_sin(x) * 1.0000001f + 0.0000001f;
  }
  return x;
}

static void burn_reset(Env *env, u64 seed) {
  BurnEnv *s = (BurnEnv *)env;
  s->x = 1.0f + (real)(seed % 1000) * 1e-3f;
  s->count = 0;
}

static void burn_observe(Env *env, real *out) {
  BurnEnv *s = (BurnEnv *)env;
  out[0] = s->x;
  out[1] = s->x * s->x;
  out[2] = real_sin(s->x);
  out[3] = 1.0f;
}

static real burn_step(Env *env, u32 action, b32 *done, b32 *truncated) {
  BurnEnv *s = (BurnEnv *)env;
  s->x = burn_work(s->x + (real)action * 1e-3f);
  s->count++;
  *done = false;
  *truncated = (s->count % 200) == 0;
  g_burn_sink = s->x;
  return -s->x;
}

static Env *burn_clone(Arena *arena, const Env *src) {
  BurnEnv *dst = ARENA_OBJ(arena, BurnEnv);
  *dst = *(const BurnEnv *)src;
  return (Env *)dst;
}

static Env *burn_create(Arena *arena) {
  BurnEnv *s = ARENA_OBJ(arena, BurnEnv);
  s->base.obs_dim = 4;
  s->base.act_dim = 1;
  s->base.max_steps = 200;
  s->base.reset = burn_reset;
  s->base.step = burn_step;
  s->base.observe = burn_observe;
  s->base.clone = burn_clone;
  burn_reset((Env *)s, 1);
  return (Env *)s;
}

static void bench_vecenv_thread_scaling(void) {
  const u32 num_envs = 8;
  const u32 steps = 200;
  u32 acts[8] = {0};

  emit("\n`VecEnv` threaded stepping, synthetic compute-bound env (%u envs, "
       "%u `sinf` per step, %u lockstep steps, wall clock):\n\n",
       num_envs, BURN_SINF, steps);
  emit("| Threads | Env steps/sec | Speedup |\n|---|---|---|\n");

  double base = 0.0;
  for (u32 threads = 1; threads <= 8; threads *= 2) {
    Arena *arena = arena_new(GiB(1));
    Env *proto = burn_create(arena);
    VecEnv *vec = vecenv_new(arena, proto, num_envs);
    vecenv_reset(vec, 1);
    vecenv_set_threads(arena, vec, threads);

    double t0 = now_wall_seconds();
    for (u32 i = 0; i < steps; i++) {
      vecenv_step(vec, acts);
    }
    double dt = now_wall_seconds() - t0;
    double rate = (double)num_envs * (double)steps / dt;
    if (threads == 1) {
      base = rate;
    }

    emit("| %u | %.0f | %.2fx |\n", threads, rate, rate / base);
    printf("  [throughput] vecenv threaded x%u (actual %u) %8.0f env-steps/sec "
           "(%.2fx)\n",
           threads, vecenv_threads(vec), rate, rate / base);

    vecenv_shutdown(vec);
    arena_dispose(arena);
  }
}

/* ── Main ─────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
  b32 do_cartpole = true;
  b32 do_snake = true;
  b32 do_throughput = true;

  if (argc > 1) {
    if (strcmp(argv[1], "cartpole") == 0) {
      do_snake = false;
      do_throughput = false;
    } else if (strcmp(argv[1], "snake") == 0) {
      do_cartpole = false;
      do_throughput = false;
    } else if (strcmp(argv[1], "throughput") == 0) {
      do_cartpole = false;
      do_snake = false;
    }
  }

  g_out = fopen("bench_results.md", "w");
  if (g_out != NULL) {
    emit("# RL_lib-in-C benchmark results\n\n");
    emit("Generated by `make bench`. %d seeds x %d eval episodes per policy.\n",
         NUM_SEEDS, EVAL_EPISODES);
    emit("Evaluation seeds are identical across policies; std is the sample "
         "(n-1) standard deviation computed in f64.\n");
  }

  if (do_cartpole) {
    double rnd[NUM_SEEDS], grd[NUM_SEEDS], tr_grd[NUM_SEEDS], tr_sto[NUM_SEEDS];
    emit("\n=== CartPole-v1 ===\n");
    for (u32 s = 0; s < NUM_SEEDS; s++) {
      Arena *arena = arena_new(GiB(1));
      Env *env = env_cartpole_create(arena);
      u64 eval_seeds[EVAL_EPISODES];
      make_eval_seeds(eval_seeds, EVAL_EPISODES, SEEDS[s] + 5000);

      rnd[s] = eval_random(env, eval_seeds, EVAL_EPISODES);
      grd[s] = eval_cartpole_greedy(env, eval_seeds, EVAL_EPISODES);

      Net *model = train_cartpole(arena, SEEDS[s], 50);
      tr_grd[s] = evaluate_seeded(env, model, eval_seeds, EVAL_EPISODES, true)
                      .avg_return;
      tr_sto[s] =
          evaluate_seeded(env, model, eval_seeds, EVAL_EPISODES, false)
              .avg_return;

      printf("  [seed %-4llu] Random: %6.1f | Greedy: %6.1f | Trained(greedy): "
             "%6.1f | Trained(stoch): %6.1f\n",
             (unsigned long long)SEEDS[s], rnd[s], grd[s], tr_grd[s], tr_sto[s]);
      arena_dispose(arena);
    }

    emit("\n| Policy | Return (mean ± sample std) | Best seed | Worst seed |\n");
    emit("|---|---|---|---|\n");
    Stat a = compute_stat(rnd, NUM_SEEDS);
    Stat b = compute_stat(grd, NUM_SEEDS);
    Stat c = compute_stat(tr_grd, NUM_SEEDS);
    Stat d = compute_stat(tr_sto, NUM_SEEDS);
    emit("| Random | %.2f ± %.2f | %.2f | %.2f |\n", a.mean, a.std, a.best,
         a.worst);
    emit("| Greedy (PD controller) | %.2f ± %.2f | %.2f | %.2f |\n", b.mean,
         b.std, b.best, b.worst);
    emit("| **Trained (REINFORCE, greedy eval)** | %.2f ± %.2f | %.2f | %.2f |\n",
         c.mean, c.std, c.best, c.worst);
    emit("| Trained (REINFORCE, stochastic eval) | %.2f ± %.2f | %.2f | %.2f |\n",
         d.mean, d.std, d.best, d.worst);
  }

  if (do_snake) {
    double rnd[NUM_SEEDS], grd[NUM_SEEDS], tr_grd[NUM_SEEDS], tr_sto[NUM_SEEDS];
    emit("\n=== Snake 6x6 ===\n");
    for (u32 s = 0; s < NUM_SEEDS; s++) {
      Arena *arena = arena_new(GiB(1));
      Env *env = env_snake_create(arena, 36);
      u64 eval_seeds[EVAL_EPISODES];
      make_eval_seeds(eval_seeds, EVAL_EPISODES, SEEDS[s] + 5000);

      rnd[s] = eval_random(env, eval_seeds, EVAL_EPISODES);
      grd[s] = eval_snake_greedy(env, eval_seeds, EVAL_EPISODES);

      Net *model = train_snake(arena, SEEDS[s], 200);
      tr_grd[s] = evaluate_seeded(env, model, eval_seeds, EVAL_EPISODES, true)
                      .avg_return;
      tr_sto[s] =
          evaluate_seeded(env, model, eval_seeds, EVAL_EPISODES, false)
              .avg_return;

      printf("  [seed %-4llu] Random: %6.1f | Greedy(BFS): %6.1f | "
             "Trained(greedy): %6.1f | Trained(stoch): %6.1f\n",
             (unsigned long long)SEEDS[s], rnd[s], grd[s], tr_grd[s], tr_sto[s]);
      arena_dispose(arena);
    }

    emit("\n| Policy | Return (mean ± sample std) | Best seed | Worst seed |\n");
    emit("|---|---|---|---|\n");
    Stat a = compute_stat(rnd, NUM_SEEDS);
    Stat b = compute_stat(grd, NUM_SEEDS);
    Stat c = compute_stat(tr_grd, NUM_SEEDS);
    Stat d = compute_stat(tr_sto, NUM_SEEDS);
    emit("| Random | %.2f ± %.2f | %.2f | %.2f |\n", a.mean, a.std, a.best,
         a.worst);
    emit("| Greedy (BFS to food) | %.2f ± %.2f | %.2f | %.2f |\n", b.mean, b.std,
         b.best, b.worst);
    emit("| **Trained (REINFORCE, greedy eval)** | %.2f ± %.2f | %.2f | %.2f |\n",
         c.mean, c.std, c.best, c.worst);
    emit("| Trained (REINFORCE, stochastic eval) | %.2f ± %.2f | %.2f | %.2f |\n",
         d.mean, d.std, d.best, d.worst);
  }

  if (do_throughput) {
    bench_env_throughput();
    bench_vecenv_throughput();
    bench_vecenv_thread_scaling();
    bench_matmul_throughput();
  }

  if (g_out != NULL) {
    fclose(g_out);
    printf("\nWrote bench_results.md\n");
  }
  return 0;
}
