#include "snake.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
  SNAKE_LEFT = 0,
  SNAKE_RIGHT = 1,
  SNAKE_UP = 2,
  SNAKE_DOWN = 3,
} SnakeAction;

typedef struct {
  i32 x;
  i32 y;
} SnakePos;

#define MAX_SNAKE_LEN 256

typedef struct {
  SnakePos body[MAX_SNAKE_LEN];
  u32 body_len;

  SnakePos food;
  u32 rows;
  u32 cols;
  u32 grid_size;
  u32 steps;
  u32 max_steps;
  u32 foods_eaten;
  SnakeAction last_action;

  prng_state *rng;
} SnakeState;

static SnakePos random_food_pos(prng_state *rng, u32 cols, u32 rows) {
  return (SnakePos){
      .x = (i32)prng_randn_r(rng, cols),
      .y = (i32)prng_randn_r(rng, rows),
  };
}

static b32 is_body_at(const SnakeState *s, i32 x, i32 y) {
  for (u32 i = 0; i < s->body_len; i++) {
    if (s->body[i].x == x && s->body[i].y == y) {
      return true;
    }
  }
  return false;
}

static void respawn_food(SnakeState *s) {
  if (s->rng == NULL) return;
  u32 attempts = 0;
  do {
    s->food = random_food_pos(s->rng, s->cols, s->rows);
    attempts++;
  } while (is_body_at(s, s->food.x, s->food.y) && attempts < 100);
}

static void snake_reset(Env *e, prng_state *rng) {
  SnakeState *s = (SnakeState *)e->state;
  s->rng = rng;

  s->body_len = 1;
  s->body[0] = (SnakePos){.x = (i32)(s->cols / 2), .y = (i32)(s->rows / 2)};

  respawn_food(s);

  s->steps = 0;
  s->foods_eaten = 0;
  s->last_action = SNAKE_RIGHT;
}

static f32 snake_step(Env *e, u32 action, b32 *done, b32 *truncated) {
  SnakeState *s = (SnakeState *)e->state;
  SnakeAction act = (SnakeAction)(action % 4);

  // Prevent instant 180-degree reversal into own neck if length > 1
  if (s->body_len > 1) {
    if ((act == SNAKE_LEFT && s->last_action == SNAKE_RIGHT) ||
        (act == SNAKE_RIGHT && s->last_action == SNAKE_LEFT) ||
        (act == SNAKE_UP && s->last_action == SNAKE_DOWN) ||
        (act == SNAKE_DOWN && s->last_action == SNAKE_UP)) {
      act = s->last_action;
    }
  }

  SnakePos new_head = s->body[0];
  if (act == SNAKE_LEFT) {
    new_head.x -= 1;
  } else if (act == SNAKE_RIGHT) {
    new_head.x += 1;
  } else if (act == SNAKE_UP) {
    new_head.y += 1;
  } else if (act == SNAKE_DOWN) {
    new_head.y -= 1;
  }
  s->last_action = act;
  s->steps++;

  // 1. Wall collision check
  b32 hit_wall = (new_head.x < 0 || new_head.x >= (i32)s->cols ||
                  new_head.y < 0 || new_head.y >= (i32)s->rows);
  if (hit_wall) {
    *done = true;
    *truncated = false;
    return -10.0f;
  }

  // 2. Self collision check (head collides with any existing body segment)
  b32 hit_self = false;
  for (u32 i = 0; i < s->body_len; i++) {
    if (s->body[i].x == new_head.x && s->body[i].y == new_head.y) {
      hit_self = true;
      break;
    }
  }
  if (hit_self) {
    *done = true;
    *truncated = false;
    return -10.0f;
  }

  // 3. Move body
  b32 ate_food = (new_head.x == s->food.x && new_head.y == s->food.y);
  f32 reward = 0.0f;

  if (ate_food) {
    reward += 20.0f;
    s->foods_eaten++;
    if (s->body_len < MAX_SNAKE_LEN) {
      s->body_len++;
    }
  }

  // Shift body segments backwards
  for (u32 i = s->body_len - 1; i > 0; i--) {
    s->body[i] = s->body[i - 1];
  }
  s->body[0] = new_head;

  if (ate_food) {
    respawn_food(s);
  }

  *done = false;
  *truncated = (s->steps >= s->max_steps);
  return reward;
}

static void snake_observe(Env *e, f32 *obs_out) {
  SnakeState *s = (SnakeState *)e->state;
  memset(obs_out, 0, sizeof(f32) * e->obs_dim);

  // Channel 0: Head position
  if (s->body[0].x >= 0 && s->body[0].x < (i32)s->cols &&
      s->body[0].y >= 0 && s->body[0].y < (i32)s->rows) {
    u32 head_i = (u32)s->body[0].y * s->cols + (u32)s->body[0].x;
    obs_out[head_i] = 1.0f;
  }

  // Channel 1: Food position
  if (s->food.x >= 0 && s->food.x < (i32)s->cols &&
      s->food.y >= 0 && s->food.y < (i32)s->rows) {
    u32 food_i = (u32)s->food.y * s->cols + (u32)s->food.x;
    obs_out[s->grid_size + food_i] = 1.0f;
  }

  // Channel 2: Body segments (tail)
  for (u32 i = 1; i < s->body_len; i++) {
    if (s->body[i].x >= 0 && s->body[i].x < (i32)s->cols &&
        s->body[i].y >= 0 && s->body[i].y < (i32)s->rows) {
      u32 body_i = (u32)s->body[i].y * s->cols + (u32)s->body[i].x;
      obs_out[2 * s->grid_size + body_i] = 1.0f;
    }
  }
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
  s->body_len = 1;
  s->rng = NULL;

  env->obs_dim = 3 * grid_size;
  env->act_dim = 4;
  env->max_steps = 100;
  env->state = s;
  env->reset = snake_reset;
  env->step = snake_step;
  env->observe = snake_observe;
  env->destroy = snake_destroy;

  return env;
}
