#ifndef ENVS_SNAKE_H
#define ENVS_SNAKE_H

#include "arena.h"
#include "env.h"

// Creates an instance of the Grid Snake environment conforming to Env.
// grid_size: e.g. 36 (6x6 grid)
// obs_dim: 2 * grid_size + 4 (76 for 6x6)
// act_dim: 5 (LEFT=0, RIGHT=1, UP=2, DOWN=3, NONE=4)
// max_steps: 100
Env *env_snake_create(mem_arena *arena, u32 grid_size);

#endif // ENVS_SNAKE_H
