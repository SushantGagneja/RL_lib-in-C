// test_envs.c — Unit tests for CartPole and Snake environments
#include "memory.h"
#include "base.h"
#include "envs/cartpole.h"
#include "envs/snake.h"
#include "envs/pendulum.h"
#include "random.h"
#include "vecenv.h"
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

static void test_cartpole_reset_and_bounds(Arena *arena) {
  printf("  cartpole reset & bounds... ");
  Env *env = env_cartpole_create(arena);
  ASSERT_TRUE(env != NULL, "env create");
  ASSERT_TRUE(env->obs_dim == 4, "obs dim 4");
  ASSERT_TRUE(env->act_dim == 2, "act dim 2");
  ASSERT_TRUE(env->max_steps == 500, "max steps 500");

  env->reset(env, 12345);

  real obs[4] = {0};
  env->observe(env, obs);
  ASSERT_TRUE(real_fabs(obs[0]) <= 0.051f, "x init within bounds");
  ASSERT_TRUE(real_fabs(obs[1]) <= 0.051f, "x_dot init within bounds");
  ASSERT_TRUE(real_fabs(obs[2]) <= 0.051f, "theta init within bounds");
  ASSERT_TRUE(real_fabs(obs[3]) <= 0.051f, "theta_dot init within bounds");

  b32 done = false;
  b32 truncated = false;
  real r = env->step(env, 1, &done, &truncated);
  ASSERT_TRUE(r == 1.0f, "step reward 1.0");
  ASSERT_TRUE(!done, "not done after 1 step");
  ASSERT_TRUE(!truncated, "not truncated after 1 step");

  printf("OK\n");
}

static void test_cartpole_determinism(Arena *arena) {
  printf("  cartpole same seed gives same trajectory... ");
  Env *a = env_cartpole_create(arena);
  Env *b = env_cartpole_create(arena);
  a->reset(a, 777);
  b->reset(b, 777);

  b32 same = true;
  for (u32 t = 0; t < 50; t++) {
    real oa[4], ob[4];
    a->observe(a, oa);
    b->observe(b, ob);
    for (u32 i = 0; i < 4; i++) {
      if (oa[i] != ob[i]) same = false;
    }
    b32 da = false, ta = false, db = false, tb = false;
    real ra = a->step(a, t % 2, &da, &ta);
    real rb = b->step(b, t % 2, &db, &tb);
    if (ra != rb || da != db || ta != tb) same = false;
    if (da || ta) break;
  }
  ASSERT_TRUE(same, "identical seed => identical CartPole trajectories");
  printf("OK\n");
}

static void test_cartpole_termination(Arena *arena) {
  printf("  cartpole termination on fall... ");
  Env *env = env_cartpole_create(arena);
  env->reset(env, 42);

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

static void test_snake_reset_and_step(Arena *arena) {
  printf("  snake reset & observation... ");
  Env *env = env_snake_create(arena, 36);
  ASSERT_TRUE(env != NULL, "snake create");
  ASSERT_TRUE(env->obs_dim == 108, "snake obs dim 108 (3x36 channels)");
  ASSERT_TRUE(env->act_dim == 4, "snake act dim 4 (left, right, up, down)");

  env->reset(env, 999);

  real obs[108] = {0};
  env->observe(env, obs);

  // Check channel activations: channel 0 (head), channel 1 (food), channel 2
  // (body segments)
  u32 head_count = 0, food_count = 0, body_count = 0;
  for (u32 i = 0; i < 36; i++) {
    if (obs[i] == 1.0f) head_count++;
    if (obs[36 + i] == 1.0f) food_count++;
    if (obs[72 + i] == 1.0f) body_count++;
  }
  ASSERT_TRUE(head_count == 1, "exactly 1 head one-hot");
  ASSERT_TRUE(food_count == 1, "exactly 1 food one-hot");
  ASSERT_TRUE(body_count == 0, "0 body segments at initial length 1");

  b32 done = false;
  b32 truncated = false;
  env->step(env, 1, &done, &truncated); // Step right
  ASSERT_TRUE(!done, "not done on first step");

  printf("OK\n");
}

static void test_snake_grid_validation(Arena *arena) {
  printf("  snake grid validation... ");
  // Not a perfect square.
  ASSERT_TRUE(env_snake_create(arena, 35) == NULL, "non-square grid rejected");
  // Larger than MAX_SNAKE_LEN (289 > 256).
  ASSERT_TRUE(env_snake_create(arena, 289) == NULL, "oversized grid rejected");
  ASSERT_TRUE(env_snake_create(arena, 0) == NULL, "zero grid rejected");
  ASSERT_TRUE(env_snake_create(arena, 36) != NULL, "valid 6x6 grid accepted");
  printf("OK\n");
}

static void test_snake_wall_death(Arena *arena) {
  printf("  snake wall collision... ");
  Env *env = env_snake_create(arena, 36);
  env->reset(env, 555);

  b32 done = false;
  b32 truncated = false;
  real r = 0.0f;
  // Step left 5 times from middle (3), hits wall at x < 0
  for (u32 i = 0; i < 5; i++) {
    r = env->step(env, 0, &done, &truncated);
    if (done) break;
  }
  ASSERT_TRUE(done, "snake dies on wall");
  ASSERT_TRUE(!truncated, "wall death is not timeout truncation");
  ASSERT_TRUE(r == -10.0f, "wall death penalty -10.0");
  printf("OK\n");
}

static void test_snake_self_collision(Arena *arena) {
  printf("  snake self-collision... ");
  Env *env = env_snake_create(arena, 36);
  env->reset(env, 101);

  // A 2x2 loop; moving DOWN from the head (2,2) enters the neck (2,1), which is
  // NOT the vacating tail here (the tail is (3,3) after the extra segment), so
  // this is a genuine self-collision.
  const i32 xs[5] = {2, 2, 3, 3, 3};
  const i32 ys[5] = {2, 1, 1, 2, 3};
  snake_test_set_body(env, xs, ys, 5);
  snake_test_set_food(env, 0, 0);

  b32 done = false;
  b32 truncated = false;
  real r = env->step(env, 3, &done, &truncated); // DOWN -> (2,1) = neck
  ASSERT_TRUE(done, "snake detects true self-collision");
  ASSERT_TRUE(!truncated, "self-collision is not timeout");
  ASSERT_TRUE(r == -10.0f, "self-collision penalty -10.0");
  printf("OK\n");
}

static void test_snake_tail_vacate(Arena *arena) {
  printf("  snake can follow its own tail... ");
  Env *env = env_snake_create(arena, 36);
  env->reset(env, 202);

  // 2x2 loop: head (2,2), tail (2,1). Moving DOWN into the tail cell is legal
  // because the tail vacates this step (no food eaten).
  const i32 xs[4] = {2, 3, 3, 2};
  const i32 ys[4] = {2, 2, 1, 1};
  snake_test_set_body(env, xs, ys, 4);
  snake_test_set_food(env, 0, 0); // ensure no growth

  b32 done = false;
  b32 truncated = false;
  env->step(env, 3, &done, &truncated); // DOWN -> (2,1) = vacating tail
  ASSERT_TRUE(!done, "moving into the vacating tail is legal");
  printf("OK\n");
}

static void test_snake_eat_grows(Arena *arena) {
  printf("  snake eating grows and respawns food... ");
  Env *env = env_snake_create(arena, 36);
  env->reset(env, 303);

  const i32 xs[1] = {2};
  const i32 ys[1] = {2};
  snake_test_set_body(env, xs, ys, 1);
  snake_test_set_food(env, 3, 2); // immediately to the right

  real obs[108];
  env->observe(env, obs);
  real food_before = 0.0f;
  for (u32 i = 0; i < 36; i++) food_before += obs[36 + i];

  b32 done = false, truncated = false;
  real r = env->step(env, 1, &done, &truncated); // RIGHT into food

  ASSERT_TRUE(r == 20.0f, "food reward +20");
  ASSERT_TRUE(!done, "not done after eating");

  env->observe(env, obs);
  u32 head_count = 0, body_count = 0;
  for (u32 i = 0; i < 36; i++) {
    if (obs[i] == 1.0f) head_count++;
    if (obs[72 + i] == 1.0f) body_count++;
  }
  ASSERT_TRUE(head_count == 1, "head still present");
  ASSERT_TRUE(body_count == 1, "body grew to 1 segment after eating");
  ASSERT_TRUE(food_before == 1.0f, "food present before eating");
  printf("OK\n");
}

// ── Vectorized environment ───────────────────────────────────────────

static void test_vecenv_basics(Arena *arena) {
  printf("  vecenv shape, determinism & independent clones... ");
  Env *proto = env_cartpole_create(arena);
  VecEnv *a = vecenv_new(arena, proto, 4);
  VecEnv *b = vecenv_new(arena, proto, 4);

  ASSERT_TRUE(a->num_envs == 4 && a->obs_dim == 4 && a->act_dim == 2,
              "vecenv shape matches the prototype");
  ASSERT_TRUE(a->envs[0] != a->envs[1] &&
                  a->envs[0]->state != a->envs[1]->state,
              "each clone allocates its own Env and state");

  vecenv_reset(a, 1234);
  vecenv_reset(b, 1234);

  b32 same = true;
  for (u64 i = 0; i < (u64)4 * 4; i++) {
    if (a->obs[i] != b->obs[i]) same = false;
  }
  ASSERT_TRUE(same, "same reset seed => identical initial observations");

  // Identical seed + identical per-step actions => identical trajectories.
  u32 seq[50][4];
  Rng r;
  rng_seed(&r, 55, 1);
  for (u32 t = 0; t < 50; t++) {
    for (u32 i = 0; i < 4; i++) seq[t][i] = rng_below(&r, 2);
  }
  b32 traj_same = true;
  for (u32 t = 0; t < 50; t++) {
    vecenv_step(a, seq[t]);
    vecenv_step(b, seq[t]);
    for (u64 i = 0; i < (u64)4 * 4; i++) {
      if (a->obs[i] != b->obs[i]) traj_same = false;
    }
    for (u32 i = 0; i < 4; i++) {
      if (a->rewards[i] != b->rewards[i]) traj_same = false;
    }
  }
  ASSERT_TRUE(traj_same, "same seed + actions => identical trajectories");

  // Stepping one sub-environment by hand must not perturb a sibling.
  real e1_before[4];
  real e1_after[4];
  for (u32 k = 0; k < 4; k++) e1_before[k] = a->obs[4u + k];
  b32 d = false;
  b32 tr = false;
  a->envs[0]->step(a->envs[0], 1, &d, &tr);
  a->envs[1]->observe(a->envs[1], e1_after);
  b32 indep = true;
  for (u32 k = 0; k < 4; k++) {
    if (e1_before[k] != e1_after[k]) indep = false;
  }
  ASSERT_TRUE(indep, "cloned sub-environments have independent state");
  printf("OK\n");
}

static void test_vecenv_autoreset(Arena *arena) {
  printf("  vecenv auto-reset preserves the terminal observation... ");
  Env *proto = env_cartpole_create(arena);
  VecEnv *v = vecenv_new(arena, proto, 1);
  // A second, stand-alone env replays the same actions to know what the
  // terminal observation must have been.
  Env *mirror = env_cartpole_create(arena);

  vecenv_reset(v, 7);
  mirror->reset(mirror, 7);

  const u32 act = 1; // always push right: the pole always falls
  b32 completed = false;
  for (u32 t = 0; t < 600; t++) {
    b32 md = false;
    b32 mt = false;
    mirror->step(mirror, act, &md, &mt);
    real mirror_obs[4];
    mirror->observe(mirror, mirror_obs);

    u32 actions[1] = {act};
    vecenv_step(v, actions);

    if (v->dones[0]) {
      b32 final_ok = true;
      for (u32 k = 0; k < 4; k++) {
        if (v->final_obs[k] != mirror_obs[k]) final_ok = false;
      }
      ASSERT_TRUE(final_ok, "final_obs == the terminal observation");
      ASSERT_TRUE(v->terminals[0] != 0, "a fall is a terminal, not a truncation");
      ASSERT_TRUE(v->truncateds[0] == 0, "not truncated");
      completed = true;
      break;
    }
  }
  ASSERT_TRUE(completed, "an episode completed within the horizon");
  printf("OK\n");
}

// ── Pendulum (continuous action) ────────────────────────────────────────

static void test_pendulum_reset_and_bounds(Arena *arena) {
  printf("  pendulum reset & bounds... ");
  Env *env = env_pendulum_create(arena);
  ASSERT_TRUE(env != NULL, "env create");
  ASSERT_TRUE(env->obs_dim == 3, "obs dim 3");
  ASSERT_TRUE(env->act_dim == 1, "act dim 1");
  ASSERT_TRUE(env->max_steps == 200, "max steps 200 (Gymnasium v1)");
  ASSERT_TRUE(env->continuous != 0, "flags itself continuous");
  ASSERT_TRUE(env->step == NULL, "has no discrete step");
  ASSERT_TRUE(env->step_continuous != NULL, "has a continuous step");
  ASSERT_TRUE(env->action_min != NULL && env->action_max != NULL,
              "advertises action bounds");
  ASSERT_TRUE(env->action_min[0] == -2.0f && env->action_max[0] == 2.0f,
              "torque bounds are [-2, 2]");

  for (u64 seed = 1; seed <= 16; seed++) {
    env->reset(env, seed);
    real obs[3];
    env->observe(env, obs);
    ASSERT_TRUE(real_fabs(obs[0] * obs[0] + obs[1] * obs[1] - 1.0f) < 1e-5f,
                "cos^2 + sin^2 == 1");
    ASSERT_TRUE(obs[2] >= -1.0f && obs[2] <= 1.0f,
                "initial angular velocity in [-1, 1]");
  }
  printf("OK\n");
}

static void test_pendulum_dynamics(Arena *arena) {
  printf("  pendulum dynamics match the reference Euler step... ");
  Env *env = env_pendulum_create(arena);
  real obs[3];
  b32 done = false;
  b32 trunc = false;

  // Upright at rest with no torque: zero cost, and gravity exerts no torque.
  pendulum_test_set_state(env, 0.0f, 0.0f);
  const real zero_torque[1] = {0.0f};
  real r = env->step_continuous(env, zero_torque, &done, &trunc);
  ASSERT_TRUE(real_fabs(r) < 1e-6f, "upright at rest pays no cost");
  ASSERT_TRUE(done == 0 && trunc == 0, "not finished after one step");
  env->observe(env, obs);
  ASSERT_TRUE(real_fabs(obs[0] - 1.0f) < 1e-6f, "cos(theta) stays 1");
  ASSERT_TRUE(real_fabs(obs[2]) < 1e-6f, "angular velocity stays 0");

  // Horizontal at rest with the maximum torque:
  //   cost  = 1 * (pi/2)^2 + 0.1 * 0 + 0.001 * 2^2 = 2.4714
  //   wdot  = (3 * 10 / (2 * 1) * sin(pi/2) + 3 / (1 * 1^2) * 2) * 0.05 = 1.05
  //   theta = pi/2 + 1.05 * 0.05 = 1.6233
  pendulum_test_set_state(env, 1.5707963f, 0.0f);
  const real full_torque[1] = {2.0f};
  r = env->step_continuous(env, full_torque, &done, &trunc);
  ASSERT_TRUE(real_fabs(r + 2.47141f) < 1e-4f, "cost matches the formula");
  env->observe(env, obs);
  ASSERT_TRUE(real_fabs(obs[2] - 1.05f) < 1e-5f, "new angular velocity is 1.05");
  ASSERT_TRUE(real_fabs(obs[0] - real_cos(1.6233002f)) < 1e-5f,
              "cos(theta) advanced by wdot * dt");
  ASSERT_TRUE(real_fabs(obs[1] - real_sin(1.6233002f)) < 1e-5f,
              "sin(theta) advanced by wdot * dt");

  // Torque outside the bounds is clamped: 100 behaves exactly like 2.
  Env *loud = env_pendulum_create(arena);
  pendulum_test_set_state(loud, 1.5707963f, 0.0f);
  const real huge_torque[1] = {100.0f};
  const real r_loud = loud->step_continuous(loud, huge_torque, &done, &trunc);
  ASSERT_TRUE(real_fabs(r_loud - r) < 1e-6f, "clamped torque gives the same cost");
  real loud_obs[3];
  loud->observe(loud, loud_obs);
  ASSERT_TRUE(real_fabs(loud_obs[2] - obs[2]) < 1e-6f,
              "clamped torque gives the same dynamics");

  // The angular velocity is capped at +-8 rad/s.
  pendulum_test_set_state(env, 0.0f, 7.9f);
  r = env->step_continuous(env, full_torque, &done, &trunc);
  env->observe(env, obs);
  ASSERT_TRUE(obs[2] <= 8.0f + 1e-6f, "angular velocity is capped at max_speed");
  printf("OK\n");
}

static void test_pendulum_determinism_and_clone(Arena *arena) {
  printf("  pendulum determinism & clone... ");
  Env *a = env_pendulum_create(arena);
  Env *b = env_pendulum_create(arena);
  a->reset(a, 999);
  b->reset(b, 999);

  const real torque[1] = {0.75f};
  b32 da = false;
  b32 ta = false;
  b32 db = false;
  b32 tb = false;
  for (u32 t = 0; t < 50; t++) {
    const real ra = a->step_continuous(a, torque, &da, &ta);
    const real rb = b->step_continuous(b, torque, &db, &tb);
    if (ra != rb) {
      ASSERT_TRUE(false, "same seed, same reward");
      break;
    }
  }
  real oa[3];
  real ob[3];
  a->observe(a, oa);
  b->observe(b, ob);
  ASSERT_TRUE(oa[0] == ob[0] && oa[1] == ob[1] && oa[2] == ob[2],
              "same seed, same trajectory");

  // A clone continues from exactly where the original is, without sharing state.
  Env *c = a->clone(arena, a);
  ASSERT_TRUE(c != NULL, "clone returns an env");
  ASSERT_TRUE(c->continuous != 0, "clone stays continuous");
  const real rc = c->step_continuous(c, torque, &da, &ta);
  const real ra2 = a->step_continuous(a, torque, &db, &tb);
  ASSERT_TRUE(rc == ra2, "clone and original take the same step");
  c->observe(c, ob);
  a->observe(a, oa);
  ASSERT_TRUE(oa[0] == ob[0] && oa[1] == ob[1] && oa[2] == ob[2],
              "clone and original agree after stepping");
  a->reset(a, 4321); // moving on must not disturb the clone
  c->observe(c, oa);
  ASSERT_TRUE(oa[0] == ob[0] && oa[1] == ob[1] && oa[2] == ob[2],
              "clone state is independent");
  printf("OK\n");
}

static void test_pendulum_truncation(Arena *arena) {
  printf("  pendulum truncates at max_steps... ");
  Env *env = env_pendulum_create(arena);
  env->reset(env, 5);

  const real torque[1] = {0.5f};
  b32 done = false;
  b32 trunc = false;
  u32 steps = 0;
  for (u32 t = 0; t < 400; t++) {
    env->step_continuous(env, torque, &done, &trunc);
    steps++;
    if (done || trunc) {
      break;
    }
  }
  ASSERT_TRUE(steps == env->max_steps, "episode length is max_steps");
  ASSERT_TRUE(trunc != 0, "the limit is a truncation");
  ASSERT_TRUE(done == 0, "Pendulum has no terminal state");
  printf("OK\n");
}

static void test_vecenv_continuous(Arena *arena) {
  printf("  vecenv continuous stepping... ");
  Env *proto = env_pendulum_create(arena);
  const u32 n = 8;
  VecEnv *v = vecenv_new(arena, proto, n);
  ASSERT_TRUE(v != NULL, "vecenv create");
  ASSERT_TRUE(v->obs_dim == 3 && v->act_dim == 1, "dims carried over");

  const real *lo = NULL;
  const real *hi = NULL;
  vecenv_action_bounds(v, &lo, &hi);
  ASSERT_TRUE(lo != NULL && hi != NULL, "bounds are exposed");
  ASSERT_TRUE(lo[0] == -2.0f && hi[0] == 2.0f, "bounds are [-2, 2]");

  vecenv_reset(v, 11);
  real actions[8];
  for (u32 i = 0; i < n; i++) {
    actions[i] = -2.0f + (real)i * 0.5f;
  }
  vecenv_step_continuous(v, actions);

  real reward_span = 0.0f;
  for (u32 i = 0; i < n; i++) {
    ASSERT_TRUE(v->rewards[i] <= 0.0f, "costs are non-positive");
    ASSERT_TRUE(v->dones[i] == 0, "nothing finished on step 1");
    const real *o = v->obs + (u64)i * 3;
    ASSERT_TRUE(real_fabs(o[0] * o[0] + o[1] * o[1] - 1.0f) < 1e-5f,
                "observation stays normalised");
    for (u32 j = 0; j < n; j++) {
      if (i != j && v->rewards[i] != v->rewards[j]) {
        reward_span = 1.0f;
      }
    }
  }
  ASSERT_TRUE(reward_span > 0.0f, "per-env actions are used");

  // Run to the time limit: every env truncates together, and their
  // observations are reset in place to a valid fresh start state.
  for (u32 t = 1; t < 200; t++) {
    vecenv_step_continuous(v, actions);
  }
  for (u32 i = 0; i < n; i++) {
    ASSERT_TRUE(v->truncateds[i] != 0, "time limit reached");
    ASSERT_TRUE(v->terminals[i] == 0, "a time limit is not a terminal state");
    const real *o = v->obs + (u64)i * 3;
    ASSERT_TRUE(real_fabs(o[0] * o[0] + o[1] * o[1] - 1.0f) < 1e-5f,
                "auto-reset gives a normalised observation");
    ASSERT_TRUE(real_fabs(o[2]) <= 1.0f,
                "auto-reset gives a fresh angular velocity");
    const real *fo = v->final_obs + (u64)i * 3;
    ASSERT_TRUE(real_fabs(fo[0] * fo[0] + fo[1] * fo[1] - 1.0f) < 1e-5f,
                "final_obs holds the pre-reset observation");
  }
  printf("OK\n");
}

int main(void) {
  Arena *arena = arena_new(MiB(16));
  printf("\n=== Environment Interface Unit Tests ===\n\n");

  test_cartpole_reset_and_bounds(arena);
  test_cartpole_determinism(arena);
  test_cartpole_termination(arena);
  test_snake_reset_and_step(arena);
  test_snake_grid_validation(arena);
  test_snake_wall_death(arena);
  test_snake_self_collision(arena);
  test_snake_tail_vacate(arena);
  test_snake_eat_grows(arena);

  test_vecenv_basics(arena);
  test_vecenv_autoreset(arena);

  test_pendulum_reset_and_bounds(arena);
  test_pendulum_dynamics(arena);
  test_pendulum_determinism_and_clone(arena);
  test_pendulum_truncation(arena);
  test_vecenv_continuous(arena);

  printf("\n---\nPassed: %u  Failed: %u\n\n", tests_passed, tests_failed);
  arena_dispose(arena);
  return tests_failed > 0 ? 1 : 0;
}
