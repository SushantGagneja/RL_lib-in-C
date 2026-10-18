#ifndef ENVS_SNAKE_H
#define ENVS_SNAKE_H

#include "memory.h"
#include "env.h"

// Creates an instance of the Grid Snake environment conforming to Env.
// grid_size: e.g. 36 (6x6 grid)
// obs_dim: 3 * grid_size (108 for 6x6: head grid, food grid, body segments grid)
// act_dim: 4 (LEFT=0, RIGHT=1, UP=2, DOWN=3)
// max_steps: 100
Env *env_snake_create(Arena *arena, u32 grid_size);

// Test/debug helper. Overwrites the snake body with `len` segments, given head
// first as parallel (x, y) arrays. Lets tests construct positions without
// duplicating the private SnakeState layout.
void snake_test_set_body(Env *e, const i32 *xs, const i32 *ys, u32 len);

// Test/debug helper. Places the food at (x, y).
void snake_test_set_food(Env *e, i32 x, i32 y);

#endif // ENVS_SNAKE_H
