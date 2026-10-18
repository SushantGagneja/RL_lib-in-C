#ifndef ENVS_CARTPOLE_H
#define ENVS_CARTPOLE_H

#include "memory.h"
#include "env.h"

// Creates an instance of the standard CartPole-v1 environment.
// obs_dim: 4 (x, x_dot, theta, theta_dot)
// act_dim: 2 (0: left, 1: right)
// max_steps: 500
Env *env_cartpole_create(Arena *arena);

#endif // ENVS_CARTPOLE_H
