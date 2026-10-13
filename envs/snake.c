#include "snake.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef enum { SNAKE_LEFT = 0, SNAKE_RIGHT = 1, SNAKE_UP = 2, SNAKE_DOWN = 3, SNAKE_NONE = 4 } SnakeAction;

typedef struct {
  i32 x;
  i32 y;
} SnakePos;

typedef struct {
  SnakePos snake;
  SnakePos food;
  u32 rows;
  u32 cols;
  u32 grid_size;
  u32 steps;
  u32 max_steps;
  u32 foods_eaten;
  SnakeAction pov;
} SnakeState;

static SnakePos random_food_pos(prng_state *rng, u32 cols, u32 rows) {
  return (SnakePos){
      .x = (i32)prng_randn_r(rng, cols),
      .y = (i32)prng_randn_r(rng, rows),
  };
}

static void snake_reset(Env *e, prng_state *rng) {
  SnakeState *s = (SnakeState *)e->state;
  s->snake = (SnakePos){.x = (i32)(s->cols / 2), .y = (i32)(s->rows / 2)};

  do {
    s->food = random_food_pos(rng, s->cols, s->rows);
  } while (s->food.x == s->snake.x && s->food.y == s->snake.y);

  s->steps = 0;
  s->foods_eaten = 0;
  s->pov = SNAKE_RIGHT;
}

static f32 snake_step(Env *e, u32 action, b32 *done, b32 *truncated) {
  SnakeState *s = (SnakeState *)e->state;
  SnakeAction act = (SnakeAction)action;

  if (act == SNAKE_NONE) {
    act = s->pov;
  }

  if (act == SNAKE_LEFT) {
    s->snake.x -= 1;
  } else if (act == SNAKE_RIGHT) {
    s->snake.x += 1;
  } else if (act == SNAKE_UP) {
    s->snake.y += 1;
  } else if (act == SNAKE_DOWN) {
    s->snake.y -= 1;
  }
  s->pov = act;
  s->steps++;

  f32 reward = 0.0f;

  // Food check
  if (s->snake.x == s->food.x && s->snake.y == s->food.y) {
    reward += 20.0f;
    s->foods_eaten++;
    // Place new food
    // (Note: in step, generate next food using a deterministic hash of position & step)
    s->food.x = (s->food.x + 3) % (i32)s->cols;
    s->food.y = (s->food.y + 2) % (i32)s->rows;
    if (s->food.x == s->snake.x && s->food.y == s->snake.y) {
      s->food.x = (s->food.x + 1) % (i32)s->cols;
    }
  }

  // Wall collision check
  b32 hit_wall = (s->snake.x < 0 || s->snake.x >= (i32)s->cols ||
                  s->snake.y < 0 || s->snake.y >= (i32)s->rows);

  if (hit_wall) {
    reward -= 10.0f;
  }

  *done = hit_wall;
  *truncated = (s->steps >= s->max_steps);

  return reward;
}

static void snake_observe(Env *e, f32 *obs_out) {
  SnakeState *s = (SnakeState *)e->state;
  memset(obs_out, 0, sizeof(f32) * e->obs_dim);

  if (s->snake.x >= 0 && s->snake.x < (i32)s->cols &&
      s->snake.y >= 0 && s->snake.y < (i32)s->rows) {
    u32 snake_i = (u32)s->snake.y * s->cols + (u32)s->snake.x;
    obs_out[snake_i] = 1.0f;
  }

  if (s->food.x >= 0 && s->food.x < (i32)s->cols &&
      s->food.y >= 0 && s->food.y < (i32)s->rows) {
    u32 food_i = (u32)s->food.y * s->cols + (u32)s->food.x;
    obs_out[s->grid_size + food_i] = 1.0f;
  }

  obs_out[2 * s->grid_size + (u32)s->pov] = 1.0f;
}

static void snake_destroy(Env *e) {
  (void)e;
}

Env *env_snake_create(mem_arena *arena, u32 grid_size) {
  u32 side = (u32)sqrtf((f32)grid_size);
  if (side * side != grid_size) {
    return NULL;
  }

  Env *env = PUSH_STRUCT(arena, Env);
  SnakeState *s = PUSH_STRUCT(arena, SnakeState);

  s->rows = side;
  s->cols = side;
  s->grid_size = grid_size;
  s->max_steps = 100;

  env->obs_dim = 2 * grid_size + 4;
  env->act_dim = 5;
  env->max_steps = 100;
  env->state = s;
  env->reset = snake_reset;
  env->step = snake_step;
  env->observe = snake_observe;
  env->destroy = snake_destroy;

  return env;
}
