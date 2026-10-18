#include "snake.h"
#include <math.h>
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

#define SNAKE_FOOD_REWARD 20.0f
#define SNAKE_DEATH_REWARD -10.0f
#define SNAKE_WIN_REWARD 100.0f

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
  b32 won;

  Rng rng; // owned by the environment, seeded in reset()
} SnakeState;

static SnakePos random_food_pos(Rng *rng, u32 cols, u32 rows) {
  return (SnakePos){
      .x = (i32)rng_below(rng, cols),
      .y = (i32)rng_below(rng, rows),
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
  u32 attempts = 0;
  do {
    s->food = random_food_pos(&s->rng, s->cols, s->rows);
    attempts++;
  } while (is_body_at(s, s->food.x, s->food.y) && attempts < 100);
}

static void snake_reset(Env *e, u64 seed) {
  SnakeState *s = (SnakeState *)e->state;
  rng_seed(&s->rng, seed, 1);

  s->body_len = 1;
  s->body[0] = (SnakePos){.x = (i32)(s->cols / 2), .y = (i32)(s->rows / 2)};

  s->steps = 0;
  s->foods_eaten = 0;
  s->won = false;

  respawn_food(s);
}

static real snake_step(Env *e, u32 action, b32 *done, b32 *truncated) {
  SnakeState *s = (SnakeState *)e->state;
  SnakeAction act = (SnakeAction)(action % 4);

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
  s->steps++;

  *done = false;
  *truncated = false;

  // 1. Wall collision.
  b32 hit_wall = (new_head.x < 0 || new_head.x >= (i32)s->cols ||
                  new_head.y < 0 || new_head.y >= (i32)s->rows);
  if (hit_wall) {
    *done = true;
    return SNAKE_DEATH_REWARD;
  }

  // 2. Self collision. The tail cell (body[body_len-1]) moves away this step
  // unless the snake is about to eat, so moving into it is legal.
  b32 ate_food = (new_head.x == s->food.x && new_head.y == s->food.y);
  u32 occupied = s->body_len;
  if (!ate_food && occupied > 0) {
    occupied -= 1; // exclude the vacating tail
  }
  for (u32 i = 0; i < occupied; i++) {
    if (s->body[i].x == new_head.x && s->body[i].y == new_head.y) {
      *done = true;
      return SNAKE_DEATH_REWARD;
    }
  }

  // 3. Move the body forward.
  real reward = 0.0f;
  if (ate_food) {
    reward += SNAKE_FOOD_REWARD;
    s->foods_eaten++;
    if (s->body_len < MAX_SNAKE_LEN) {
      s->body_len++;
    }
  }

  for (u32 i = s->body_len - 1; i > 0; i--) {
    s->body[i] = s->body[i - 1];
  }
  s->body[0] = new_head;

  // 4. Win when the snake fills the whole board.
  if (ate_food && s->body_len >= s->grid_size) {
    s->won = true;
    *done = true;
    return reward + SNAKE_WIN_REWARD;
  }

  if (ate_food) {
    respawn_food(s);
  }

  *truncated = (s->steps >= s->max_steps);
  return reward;
}

static void snake_observe(Env *e, real *obs_out) {
  SnakeState *s = (SnakeState *)e->state;
  memset(obs_out, 0, sizeof(real) * e->obs_dim);

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

static Env *snake_clone(Arena *arena, const Env *src) {
  RL_LIB_ASSERT(src != NULL && src->state != NULL, "snake_clone: NULL source");
  const SnakeState *s = (const SnakeState *)src->state;
  Env *env = env_snake_create(arena, s->grid_size);
  RL_LIB_ASSERT(env != NULL, "snake_clone: invalid grid size in source");
  *(SnakeState *)env->state = *s;
  return env;
}

void snake_test_set_body(Env *e, const i32 *xs, const i32 *ys, u32 len) {
  SnakeState *s = (SnakeState *)e->state;
  if (len > MAX_SNAKE_LEN) len = MAX_SNAKE_LEN;
  s->body_len = len;
  for (u32 i = 0; i < len; i++) {
    s->body[i] = (SnakePos){.x = xs[i], .y = ys[i]};
  }
}

void snake_test_set_food(Env *e, i32 x, i32 y) {
  SnakeState *s = (SnakeState *)e->state;
  s->food = (SnakePos){.x = x, .y = y};
}

Env *env_snake_create(Arena *arena, u32 grid_size) {
  if (grid_size == 0 || grid_size > MAX_SNAKE_LEN) {
    return NULL;
  }

  u32 side = (u32)real_sqrt((real)grid_size);
  if (side * side != grid_size) {
    return NULL;
  }

  Env *env = ARENA_OBJ(arena, Env);
  SnakeState *s = ARENA_OBJ(arena, SnakeState);

  s->rows = side;
  s->cols = side;
  s->grid_size = grid_size;
  s->max_steps = 100;
  s->body_len = 1;
  rng_seed(&s->rng, 0, 1);

  env->obs_dim = 3 * grid_size;
  env->act_dim = 4;
  env->max_steps = 100;
  env->state = s;
  env->reset = snake_reset;
  env->step = snake_step;
  env->observe = snake_observe;
  env->clone = snake_clone;
  env->destroy = snake_destroy;

  return env;
}
