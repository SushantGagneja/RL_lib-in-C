#include "cartpole.h"
#include <math.h>
#include <stdlib.h>

typedef struct {
  real x;
  real x_dot;
  real theta;
  real theta_dot;
  u32 steps;
  prng_state rng; // owned by the environment, seeded in reset()
} CartPoleState;

static const real GRAVITY = 9.8f;
static const real MASS_POLE = 0.1f;
static const real TOTAL_MASS = 1.1f;
static const real LENGTH_HALF = 0.5f;
static const real POLEMASS_LENGTH = 0.05f; // MASS_POLE * LENGTH_HALF
static const real TAU = 0.02f;
static const real X_THRESHOLD = 2.4f;
static const real THETA_THRESHOLD = 0.2095f; // ~12 degrees
static const u32 MAX_STEPS = 500;

static void cartpole_reset(Env *e, u64 seed) {
  CartPoleState *s = (CartPoleState *)e->state;
  prng_seed_r(&s->rng, seed, 1);
  s->x = (prng_randf_r(&s->rng) * 0.1f) - 0.05f;
  s->x_dot = (prng_randf_r(&s->rng) * 0.1f) - 0.05f;
  s->theta = (prng_randf_r(&s->rng) * 0.1f) - 0.05f;
  s->theta_dot = (prng_randf_r(&s->rng) * 0.1f) - 0.05f;
  s->steps = 0;
}

static real cartpole_step(Env *e, u32 action, b32 *done, b32 *truncated) {
  CartPoleState *s = (CartPoleState *)e->state;

  real force = (action == 1) ? 10.0f : -10.0f;
  real costh = real_cos(s->theta);
  real sinth = real_sin(s->theta);

  real temp = (force + POLEMASS_LENGTH * s->theta_dot * s->theta_dot * sinth) /
             TOTAL_MASS;
  real theta_acc =
      (GRAVITY * sinth - costh * temp) /
      (LENGTH_HALF *
       (4.0f / 3.0f - MASS_POLE * costh * costh / TOTAL_MASS));
  real x_acc = temp - POLEMASS_LENGTH * theta_acc * costh / TOTAL_MASS;

  s->x += TAU * s->x_dot;
  s->x_dot += TAU * x_acc;
  s->theta += TAU * s->theta_dot;
  s->theta_dot += TAU * theta_acc;
  s->steps++;

  *done = (real_fabs(s->x) > X_THRESHOLD || real_fabs(s->theta) > THETA_THRESHOLD);
  *truncated = (s->steps >= MAX_STEPS);

  return 1.0f; // +1 reward for every step balanced
}

static void cartpole_observe(Env *e, real *obs_out) {
  CartPoleState *s = (CartPoleState *)e->state;
  obs_out[0] = s->x;
  obs_out[1] = s->x_dot;
  obs_out[2] = s->theta;
  obs_out[3] = s->theta_dot;
}

static void cartpole_destroy(Env *e) {
  (void)e;
  // Allocated on arena; no-op
}

Env *env_cartpole_create(mem_arena *arena) {
  Env *env = PUSH_STRUCT(arena, Env);
  CartPoleState *s = PUSH_STRUCT(arena, CartPoleState);

  env->obs_dim = 4;
  env->act_dim = 2;
  env->max_steps = MAX_STEPS;
  env->state = s;
  env->reset = cartpole_reset;
  env->step = cartpole_step;
  env->observe = cartpole_observe;
  env->destroy = cartpole_destroy;

  return env;
}
