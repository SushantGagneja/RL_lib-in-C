#include "cartpole.h"
#include <math.h>
#include <stdlib.h>

typedef struct {
  f32 x;
  f32 x_dot;
  f32 theta;
  f32 theta_dot;
  u32 steps;
} CartPoleState;

static const f32 GRAVITY = 9.8f;
static const f32 MASS_POLE = 0.1f;
static const f32 TOTAL_MASS = 1.1f;
static const f32 LENGTH_HALF = 0.5f;
static const f32 POLEMASS_LENGTH = 0.05f; // MASS_POLE * LENGTH_HALF
static const f32 TAU = 0.02f;
static const f32 X_THRESHOLD = 2.4f;
static const f32 THETA_THRESHOLD = 0.2095f; // ~12 degrees
static const u32 MAX_STEPS = 500;

static void cartpole_reset(Env *e, prng_state *rng) {
  CartPoleState *s = (CartPoleState *)e->state;
  s->x = (prng_randf_r(rng) * 0.1f) - 0.05f;
  s->x_dot = (prng_randf_r(rng) * 0.1f) - 0.05f;
  s->theta = (prng_randf_r(rng) * 0.1f) - 0.05f;
  s->theta_dot = (prng_randf_r(rng) * 0.1f) - 0.05f;
  s->steps = 0;
}

static f32 cartpole_step(Env *e, u32 action, b32 *done, b32 *truncated) {
  CartPoleState *s = (CartPoleState *)e->state;

  f32 force = (action == 1) ? 10.0f : -10.0f;
  f32 costh = cosf(s->theta);
  f32 sinth = sinf(s->theta);

  f32 temp = (force + POLEMASS_LENGTH * s->theta_dot * s->theta_dot * sinth) /
             TOTAL_MASS;
  f32 theta_acc =
      (GRAVITY * sinth - costh * temp) /
      (LENGTH_HALF *
       (4.0f / 3.0f - MASS_POLE * costh * costh / TOTAL_MASS));
  f32 x_acc = temp - POLEMASS_LENGTH * theta_acc * costh / TOTAL_MASS;

  s->x += TAU * s->x_dot;
  s->x_dot += TAU * x_acc;
  s->theta += TAU * s->theta_dot;
  s->theta_dot += TAU * theta_acc;
  s->steps++;

  *done = (fabsf(s->x) > X_THRESHOLD || fabsf(s->theta) > THETA_THRESHOLD);
  *truncated = (s->steps >= MAX_STEPS);

  return 1.0f; // +1 reward for every step balanced
}

static void cartpole_observe(Env *e, f32 *obs_out) {
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
