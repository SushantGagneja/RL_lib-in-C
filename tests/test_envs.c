// test_envs.c — Unit tests for CartPole and Snake environments
#include "arena.h"
#include "base.h"
#include "envs/cartpole.h"
#include "envs/snake.h"
#include "prng.h"
#include <math.h>
#include <stdio.h>

static u32 tests_passed = 0;
static u32 tests_failed = 0;

#define ASSERT_TRUE(cond, msg)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: %s (line %d)\n", msg, __LINE__);                         \
      tests_failed++;                                                          \
    } else {                                                                   \
      tests_passed++;                                                          \
    }                                                                          \
  } while (0)

static void test_cartpole_reset_and_bounds(mem_arena *arena) {
  printf("  cartpole reset & bounds... ");
  Env *env = env_cartpole_create(arena);
  ASSERT_TRUE(env != NULL, "env create");
  ASSERT_TRUE(env->obs_dim == 4, "obs dim 4");
  ASSERT_TRUE(env->act_dim == 2, "act dim 2");
  ASSERT_TRUE(env->max_steps == 500, "max steps 500");

  prng_state rng;
  prng_seed_r(&rng, 12345, 1);
  env->reset(env, &rng);

  f32 obs[4] = {0};
  env->observe(env, obs);
  ASSERT_TRUE(fabsf(obs[0]) <= 0.051f, "x init within bounds");
  ASSERT_TRUE(fabsf(obs[1]) <= 0.051f, "x_dot init within bounds");
  ASSERT_TRUE(fabsf(obs[2]) <= 0.051f, "theta init within bounds");
  ASSERT_TRUE(fabsf(obs[3]) <= 0.051f, "theta_dot init within bounds");

  b32 done = false;
  b32 truncated = false;
  f32 r = env->step(env, 1, &done, &truncated);
  ASSERT_TRUE(r == 1.0f, "step reward 1.0");
  ASSERT_TRUE(!done, "not done after 1 step");
  ASSERT_TRUE(!truncated, "not truncated after 1 step");

  printf("OK\n");
}

static void test_cartpole_termination(mem_arena *arena) {
  printf("  cartpole termination on fall... ");
  Env *env = env_cartpole_create(arena);
  prng_state rng;
  prng_seed_r(&rng, 42, 1);
  env->reset(env, &rng);

  b32 done = false;
  b32 truncated = false;
  // Push right repeatedly until pole falls or boundary crossed
  for (u32 t = 0; t < 200; t++) {
    env->step(env, 1, &done, &truncated);
    if (done) break;
  }
  ASSERT_TRUE(done, "cartpole terminates when falling");
  printf("OK\n");
}

static void test_snake_reset_and_step(mem_arena *arena) {
  printf("  snake reset & observation... ");
  Env *env = env_snake_create(arena, 36);
  ASSERT_TRUE(env != NULL, "snake create");
  ASSERT_TRUE(env->obs_dim == 76, "snake obs dim 76");
  ASSERT_TRUE(env->act_dim == 5, "snake act dim 5");

  prng_state rng;
  prng_seed_r(&rng, 999, 1);
  env->reset(env, &rng);

  f32 obs[76] = {0};
  env->observe(env, obs);

  // Check one-hot activations
  u32 head_count = 0, food_count = 0, pov_count = 0;
  for (u32 i = 0; i < 36; i++) {
    if (obs[i] == 1.0f) head_count++;
    if (obs[36 + i] == 1.0f) food_count++;
  }
  for (u32 i = 72; i < 76; i++) {
    if (obs[i] == 1.0f) pov_count++;
  }
  ASSERT_TRUE(head_count == 1, "exactly 1 head one-hot");
  ASSERT_TRUE(food_count == 1, "exactly 1 food one-hot");
  ASSERT_TRUE(pov_count == 1, "exactly 1 pov one-hot");

  b32 done = false;
  b32 truncated = false;
  env->step(env, 1, &done, &truncated); // Step right
  ASSERT_TRUE(!done, "not done on first step");

  printf("OK\n");
}

static void test_snake_wall_death(mem_arena *arena) {
  printf("  snake wall collision... ");
  Env *env = env_snake_create(arena, 36);
  prng_state rng;
  prng_seed_r(&rng, 555, 1);
  env->reset(env, &rng);

  b32 done = false;
  b32 truncated = false;
  f32 r = 0.0f;
  // Step left 5 times from middle (3), hits wall at x < 0
  for (u32 i = 0; i < 5; i++) {
    r = env->step(env, 0, &done, &truncated);
    if (done) break;
  }
  ASSERT_TRUE(done, "snake dies on wall");
  ASSERT_TRUE(r == -10.0f, "wall death penalty -10.0");
  printf("OK\n");
}

int main(void) {
  mem_arena *arena = arena_create(MiB(16));
  printf("\n=== Environment Interface Unit Tests ===\n\n");

  test_cartpole_reset_and_bounds(arena);
  test_cartpole_termination(arena);
  test_snake_reset_and_step(arena);
  test_snake_wall_death(arena);

  printf("\n---\nPassed: %u  Failed: %u\n\n", tests_passed, tests_failed);
  arena_destroy(arena);
  return tests_failed > 0 ? 1 : 0;
}
