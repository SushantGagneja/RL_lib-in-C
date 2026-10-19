#include "cartpole.h"
#include <math.h>
#include <stdlib.h>

typedef struct {
  real x;
  real x_dot;
  real theta;
  real theta_dot;
  u32 steps;
  Rng rng; // owned by the environment, seeded in reset()
} CartPoleState;

static const real GRAVITY = REAL_C(9.8);
static const real MASS_POLE = REAL_C(0.1);
static const real TOTAL_MASS = REAL_C(1.1);
static const real LENGTH_HALF = REAL_C(0.5);
static const real POLEMASS_LENGTH = REAL_C(0.05); // MASS_POLE * LENGTH_HALF
static const real TAU = REAL_C(0.02);
static const real X_THRESHOLD = REAL_C(2.4);
static const real THETA_THRESHOLD = REAL_C(0.2095); // ~12 degrees
static const u32 MAX_STEPS = 500;

static void cartpole_reset(Env *e, u64 seed) {
  CartPoleState *s = (CartPoleState *)e->state;
  rng_seed(&s->rng, seed, 1);
  s->x = (rng_unit(&s->rng) * REAL_C(0.1)) - REAL_C(0.05);
  s->x_dot = (rng_unit(&s->rng) * REAL_C(0.1)) - REAL_C(0.05);
  s->theta = (rng_unit(&s->rng) * REAL_C(0.1)) - REAL_C(0.05);
  s->theta_dot = (rng_unit(&s->rng) * REAL_C(0.1)) - REAL_C(0.05);
  s->steps = 0;
}

static real cartpole_step(Env *e, u32 action, b32 *done, b32 *truncated) {
  CartPoleState *s = (CartPoleState *)e->state;

  real force = (action == 1) ? REAL_C(10.0) : -REAL_C(10.0);
  real costh = real_cos(s->theta);
  real sinth = real_sin(s->theta);

  real temp = (force + POLEMASS_LENGTH * s->theta_dot * s->theta_dot * sinth) /
             TOTAL_MASS;
  real theta_acc =
      (GRAVITY * sinth - costh * temp) /
      (LENGTH_HALF *
       (REAL_C(4.0) / REAL_C(3.0) - MASS_POLE * costh * costh / TOTAL_MASS));
  real x_acc = temp - POLEMASS_LENGTH * theta_acc * costh / TOTAL_MASS;

  s->x += TAU * s->x_dot;
  s->x_dot += TAU * x_acc;
  s->theta += TAU * s->theta_dot;
  s->theta_dot += TAU * theta_acc;
  s->steps++;

  *done = (real_fabs(s->x) > X_THRESHOLD || real_fabs(s->theta) > THETA_THRESHOLD);
  *truncated = (s->steps >= MAX_STEPS);

  return REAL_C(1.0); // +1 reward for every step balanced
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

static Env *cartpole_clone(Arena *arena, const Env *src) {
  RL_LIB_ASSERT(src != NULL && src->state != NULL, "cartpole_clone: NULL source");
  Env *env = env_cartpole_create(arena);
  *(CartPoleState *)env->state = *(const CartPoleState *)src->state;
  return env;
}

Env *env_cartpole_create(Arena *arena) {
  Env *env = ARENA_OBJ(arena, Env);
  CartPoleState *s = ARENA_OBJ(arena, CartPoleState);

  env->obs_dim = 4;
  env->act_dim = 2;
  env->max_steps = MAX_STEPS;
  env->state = s;
  env->reset = cartpole_reset;
  env->step = cartpole_step;
  env->observe = cartpole_observe;
  env->clone = cartpole_clone;
  env->destroy = cartpole_destroy;

  return env;
}
