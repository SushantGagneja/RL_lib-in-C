#ifndef ENVS_PENDULUM_H
#define ENVS_PENDULUM_H

#include "memory.h"
#include "env.h"

// Creates an instance of Pendulum-v1 (Gymnasium dynamics).
// obs_dim: 3  (cos theta, sin theta, theta_dot)
// act_dim: 1  (torque in [-2, 2] N*m)
// max_steps: 200
//
// The single action dimension makes this the smallest possible continuous
// control task: the policy has to learn to swing the pendulum up and hold it
// there, which needs the continuous action path end to end.
Env *env_pendulum_create(Arena *arena);

// Test/debug helper: places the pendulum at (theta, theta_dot) in radians and
// radians/second.
void pendulum_test_set_state(Env *e, real theta, real theta_dot);

#endif // ENVS_PENDULUM_H
