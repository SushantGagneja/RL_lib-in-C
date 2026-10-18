#include "pendulum.h"
#include <math.h>

// Gymnasium's Pendulum-v1: a torque-driven pendulum with a continuous action
// space. The dynamics are a point mass on a massless rod, integrated with one
// Euler step of dt; the agent pays a quadratic penalty for being away from the
// upright position, for spinning, and for using torque.
typedef struct {
  real theta;      // angle from upright, radians
  real theta_dot;  // angular velocity, radians/second
  u32 steps;
  Rng rng; // owned by the environment, seeded in reset()
} PendulumState;

static const real MAX_TORQUE = 2.0f;
static const real MAX_SPEED = 8.0f;
static const real DT = 0.05f;
static const real GRAVITY = 10.0f;
static const real MASS = 1.0f;
static const real LENGTH = 1.0f;
static const real PI = 3.14159265f;
static const real TWO_PI = 6.28318531f;
static const u32 MAX_STEPS = 200;

// Action bounds a policy clamps to. One entry per action dimension.
static const real ACTION_MIN[1] = {-2.0f};
static const real ACTION_MAX[1] = {2.0f};

// Wraps an angle into [-pi, pi).
static real angle_normalize(real x) {
  real r = real_fmod(x + PI, TWO_PI);
  if (r < 0.0f) {
    r += TWO_PI;
  }
  return r - PI;
}

static void pendulum_reset(Env *e, u64 seed) {
  PendulumState *s = (PendulumState *)e->state;
  rng_seed(&s->rng, seed, 1);
  s->theta = rng_unit(&s->rng) * TWO_PI - PI;
  s->theta_dot = rng_unit(&s->rng) * 2.0f - 1.0f; // [-1, 1]
  s->steps = 0;
}

static real pendulum_step_continuous(Env *e, const real *action, b32 *done,
                                    b32 *truncated) {
  PendulumState *s = (PendulumState *)e->state;

  real torque = action[0];
  if (torque > MAX_TORQUE) {
    torque = MAX_TORQUE;
  }
  if (torque < -MAX_TORQUE) {
    torque = -MAX_TORQUE;
  }

  const real theta = angle_normalize(s->theta);
  const real theta_dot = s->theta_dot;
  const real reward =
      -(theta * theta + 0.1f * theta_dot * theta_dot +
        0.001f * torque * torque);

  real new_theta_dot =
      theta_dot +
      (3.0f * GRAVITY / (2.0f * LENGTH) * real_sin(s->theta) +
       3.0f / (MASS * LENGTH * LENGTH) * torque) *
          DT;
  if (new_theta_dot > MAX_SPEED) {
    new_theta_dot = MAX_SPEED;
  }
  if (new_theta_dot < -MAX_SPEED) {
    new_theta_dot = -MAX_SPEED;
  }

  s->theta = angle_normalize(s->theta + new_theta_dot * DT);
  s->theta_dot = new_theta_dot;
  s->steps++;

  *done = false; // Pendulum has no terminal state, only the time limit
  *truncated = (s->steps >= MAX_STEPS);
  return reward;
}

static void pendulum_observe(Env *e, real *obs_out) {
  PendulumState *s = (PendulumState *)e->state;
  obs_out[0] = real_cos(s->theta);
  obs_out[1] = real_sin(s->theta);
  obs_out[2] = s->theta_dot;
}

static void pendulum_destroy(Env *e) {
  (void)e;
  // Allocated on arena; no-op
}

static Env *pendulum_clone(Arena *arena, const Env *src) {
  RL_LIB_ASSERT(src != NULL && src->state != NULL,
                "pendulum_clone: NULL source");
  Env *env = env_pendulum_create(arena);
  *(PendulumState *)env->state = *(const PendulumState *)src->state;
  return env;
}

void pendulum_test_set_state(Env *e, real theta, real theta_dot) {
  PendulumState *s = (PendulumState *)e->state;
  s->theta = theta;
  s->theta_dot = theta_dot;
  s->steps = 0;
}

Env *env_pendulum_create(Arena *arena) {
  Env *env = ARENA_OBJ(arena, Env);
  PendulumState *s = ARENA_OBJ(arena, PendulumState);

  env->obs_dim = 3;
  env->act_dim = 1;
  env->max_steps = MAX_STEPS;
  env->state = s;
  env->continuous = true;
  env->action_min = ACTION_MIN;
  env->action_max = ACTION_MAX;
  env->reset = pendulum_reset;
  env->step = NULL; // continuous: use step_continuous
  env->step_continuous = pendulum_step_continuous;
  env->observe = pendulum_observe;
  env->clone = pendulum_clone;
  env->destroy = pendulum_destroy;

  return env;
}
