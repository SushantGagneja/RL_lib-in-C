// env.c — Snake environment, evaluation harness, baselines, and REINFORCE training
#include "base.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "arena.h"
#include "autograd.h"
#include "mat.h"
#include "model.h"
#include "prng.h"

// ── Environment Types & Constants ────────────────────────────────────

typedef enum { LEFT = 0, RIGHT = 1, UP = 2, DOWN = 3, NONE = 4 } ACTION;

typedef struct {
  i32 x;
  i32 y;
} State;

typedef struct {
  State snake;
  State food;
  f32 score;
  u32 foods_eaten;

  u32 rows;
  u32 cols;
  u32 grid_size;
  u64 steps;
  ACTION pov;

  prng_state rng;
} SnakeENV;

#define EPISODE_LEN 100

typedef struct {
  State states[EPISODE_LEN];
  State food_states[EPISODE_LEN];
  ACTION povs[EPISODE_LEN];
  ACTION actions[EPISODE_LEN];
  f32 rewards[EPISODE_LEN];
  f32 returns[EPISODE_LEN];
  u32 len;
} Trajectory;

typedef struct {
  Trajectory *trajectories;
  u32 capacity;
  u32 count;
} RolloutBuffer;

// ── Environment Dynamics ─────────────────────────────────────────────

SnakeENV *create_env(u32 grid_size, u64 seed) {
  SnakeENV *env = malloc(sizeof(*env));
  if (env == NULL) {
    return NULL;
  }

  u32 side = (u32)sqrtf((f32)grid_size);
  if (side * side != grid_size) {
    free(env);
    return NULL;
  }

  prng_seed_r(&env->rng, seed, 1);

  env->snake = (State){.x = (i32)(side / 2), .y = (i32)(side / 2)};
  env->food = (State){.x = (i32)(side - 1), .y = (i32)(side - 1)};
  env->rows = side;
  env->cols = side;
  env->grid_size = grid_size;
  env->score = 0.0f;
  env->foods_eaten = 0;
  env->pov = RIGHT;
  env->steps = 0;
  return env;
}

void destroy_env(SnakeENV *env) {
  if (env != NULL) {
    free(env);
  }
}

State get_random_food_loc(prng_state *rng, u32 cols, u32 rows) {
  return (State){
      .x = (i32)prng_randn_r(rng, cols),
      .y = (i32)prng_randn_r(rng, rows),
  };
}

b32 game_over(const SnakeENV *env) {
  return env->snake.x < 0 || env->snake.x >= (i32)env->cols ||
         env->snake.y < 0 || env->snake.y >= (i32)env->rows;
}

f32 get_reward(SnakeENV *env) {
  f32 reward = 0.0f;

  if (env->snake.x == env->food.x && env->snake.y == env->food.y) {
    reward += 20.0f;
    env->foods_eaten++;

    State new_food;
    do {
      new_food = get_random_food_loc(&env->rng, env->cols, env->rows);
    } while (new_food.x == env->snake.x && new_food.y == env->snake.y);

    env->food = new_food;
  }

  if (game_over(env)) {
    reward -= 10.0f;
  }

  return reward;
}

void reset_state(SnakeENV *env) {
  env->snake = (State){
      .x = (i32)(env->cols / 2),
      .y = (i32)(env->rows / 2),
  };

  do {
    env->food = get_random_food_loc(&env->rng, env->cols, env->rows);
  } while (env->food.x == env->snake.x && env->food.y == env->snake.y);

  env->score = 0.0f;
  env->foods_eaten = 0;
  env->pov = RIGHT;
  env->steps = 0;
}

void take_action(SnakeENV *env, ACTION action) {
  if (action == NONE) {
    action = env->pov;
  }

  if (action == LEFT) {
    env->snake.x -= 1;
  } else if (action == RIGHT) {
    env->snake.x += 1;
  } else if (action == UP) {
    env->snake.y += 1;
  } else if (action == DOWN) {
    env->snake.y -= 1;
  } else {
    return;
  }

  env->pov = action;
}

ACTION sample_action(prng_state *rng, const matrix *probs) {
  f32 r = prng_randf_r(rng);
  f32 cdf = 0.0f;

  u32 size = probs->rows * probs->cols;
  for (u32 i = 0; i < size; i++) {
    cdf += probs->data[i];
    if (r <= cdf) {
      return (ACTION)i;
    }
  }
  return (ACTION)(size - 1);
}

ACTION argmax_action(const matrix *probs) {
  u32 size = probs->rows * probs->cols;
  u32 best = 0;
  f32 max_p = probs->data[0];
  for (u32 i = 1; i < size; i++) {
    if (probs->data[i] > max_p) {
      max_p = probs->data[i];
      best = i;
    }
  }
  return (ACTION)best;
}

void build_state_vector(matrix *in, State state, State food_state, ACTION pov,
                        u32 cols, u32 grid_size) {
  clear(in);

  u32 snake_i = (u32)state.y * cols + (u32)state.x;
  u32 food_i = (u32)food_state.y * cols + (u32)food_state.x;

  in->data[snake_i] = 1.0f;
  in->data[grid_size + food_i] = 1.0f;
  in->data[2 * grid_size + (u32)pov] = 1.0f;
}

// ── Evaluation & Baselines ───────────────────────────────────────────

typedef enum {
  POLICY_MODEL_GREEDY,
  POLICY_MODEL_STOCHASTIC,
  POLICY_RANDOM,
  POLICY_GREEDY
} PolicyType;

typedef struct {
  f32 mean_reward;
  f32 mean_foods;
  f32 mean_steps;
} EvalResult;

// Hand-coded greedy heuristic: step along the axis with the larger distance to food
static ACTION greedy_baseline_action(const SnakeENV *env) {
  i32 dx = env->food.x - env->snake.x;
  i32 dy = env->food.y - env->snake.y;
  if (abs(dx) >= abs(dy)) {
    return dx > 0 ? RIGHT : LEFT;
  } else {
    return dy > 0 ? UP : DOWN;
  }
}

EvalResult evaluate_policy(model_state *model, SnakeENV *env,
                           PolicyType policy_type, u32 episodes) {
  f32 total_reward = 0.0f;
  u32 total_foods = 0;
  u64 total_steps = 0;

  for (u32 ep = 0; ep < episodes; ep++) {
    reset_state(env);
    f32 ep_reward = 0.0f;

    for (u32 t = 0; t < EPISODE_LEN; t++) {
      ACTION action = NONE;

      switch (policy_type) {
      case POLICY_MODEL_GREEDY: {
        build_state_vector(model->input->val, env->snake, env->food, env->pov,
                           env->cols, env->grid_size);
        forward_pass(&model->forward_graph);
        action = argmax_action(model->output->val);
      } break;
      case POLICY_MODEL_STOCHASTIC: {
        build_state_vector(model->input->val, env->snake, env->food, env->pov,
                           env->cols, env->grid_size);
        forward_pass(&model->forward_graph);
        action = sample_action(&env->rng, model->output->val);
      } break;
      case POLICY_RANDOM: {
        action = (ACTION)prng_randn_r(&env->rng, 4);
      } break;
      case POLICY_GREEDY: {
        action = greedy_baseline_action(env);
      } break;
      }

      take_action(env, action);
      f32 r = get_reward(env);
      ep_reward += r;
      total_steps++;

      if (game_over(env)) {
        break;
      }
    }
    total_reward += ep_reward;
    total_foods += env->foods_eaten;
  }

  return (EvalResult){
      .mean_reward = total_reward / (f32)episodes,
      .mean_foods = (f32)total_foods / (f32)episodes,
      .mean_steps = (f32)total_steps / (f32)episodes,
  };
}

// ── Training ─────────────────────────────────────────────────────────

void train(model_state *model, SnakeENV *env, mem_arena *arena, u32 epochs,
           u32 rollout_size, f32 learning_rate, f32 gamma) {
  RolloutBuffer *buffer = PUSH_STRUCT(arena, RolloutBuffer);
  buffer->capacity = rollout_size;
  buffer->trajectories = PUSH_ARRAY(arena, Trajectory, rollout_size);

  printf("\nStarting REINFORCE Training:\n");
  printf("  Epochs: %u | Rollout size: %u | LR: %.3f | Gamma: %.2f\n\n",
         epochs, rollout_size, learning_rate, gamma);

  for (u32 epoch = 0; epoch < epochs; epoch++) {
    u32 foods_eaten = 0;

    for (u32 i = 0; i < rollout_size; i++) {
      reset_state(env);

      Trajectory *traj = &buffer->trajectories[i];
      traj->len = 0;

      for (u32 t = 0; t < EPISODE_LEN; t++) {
        State state = env->snake;
        State food_state = env->food;
        ACTION pov = env->pov;

        build_state_vector(model->input->val, state, food_state, pov, env->cols,
                           env->grid_size);

        forward_pass(&model->forward_graph);
        ACTION action = sample_action(&env->rng, model->output->val);

        take_action(env, action);

        f32 reward = get_reward(env);
        env->score += reward;
        b32 done = game_over(env);

        traj->states[t] = state;
        traj->food_states[t] = food_state;
        traj->povs[t] = pov;
        traj->actions[t] = action;
        traj->rewards[t] = reward;
        traj->len = t + 1;

        if (done) {
          break;
        }
      }

      foods_eaten += env->foods_eaten;
    }

    buffer->count = rollout_size;

    u32 sample_count = 0;
    f32 average_return = 0.0f;
    f32 return_sum = 0.0f;
    f32 return_sq_sum = 0.0f;

    for (u32 b = 0; b < buffer->count; b++) {
      Trajectory *traj = &buffer->trajectories[b];
      f32 G = 0.0f;

      for (i32 t = (i32)traj->len - 1; t >= 0; t--) {
        G = traj->rewards[t] + gamma * G;
        traj->returns[t] = G;
        return_sum += G;
        return_sq_sum += G * G;
        sample_count++;
      }

      average_return += traj->returns[0];
    }

    f32 return_mean = return_sum / (f32)sample_count;
    f32 return_variance =
        return_sq_sum / (f32)sample_count - return_mean * return_mean;
    f32 return_std = sqrtf(MAX(return_variance, 0.0f));

    for (u32 i = 0; i < model->cost_graph.size; i++) {
      Var *cur = model->cost_graph.vars[i];
      if (cur->flags & VAR_FLAG_PARAMETER) {
        clear(cur->grad);
      }
    }

    for (u32 b = 0; b < buffer->count; b++) {
      Trajectory *traj = &buffer->trajectories[b];

      for (u32 t = 0; t < traj->len; t++) {
        build_state_vector(model->input->val, traj->states[t],
                           traj->food_states[t], traj->povs[t], env->cols,
                           env->grid_size);

        clear(model->advantage->val);
        f32 advantage = (traj->returns[t] - return_mean) / (return_std + 1e-8f);
        model->advantage->val->data[traj->actions[t]] = advantage;

        forward_pass(&model->cost_graph);
        backward_pass(&model->cost_graph);
      }
    }

    if (sample_count > 0) {
      f32 gradient_scale = learning_rate / (f32)sample_count;

      for (u32 i = 0; i < model->cost_graph.size; i++) {
        Var *cur = model->cost_graph.vars[i];
        if ((cur->flags & VAR_FLAG_PARAMETER) == 0) {
          continue;
        }

        scale(cur->grad, gradient_scale);
        sub(cur->val, cur->val, cur->grad);
      }
    }

    if ((epoch + 1) % 50 == 0 || epoch == 0 || epoch + 1 == epochs) {
      printf("Epoch %4u/%u | Avg return: %7.2f | Foods: %4u | Samples: %5u | "
             "Std: %6.2f\n",
             epoch + 1, epochs, average_return / (f32)buffer->count,
             foods_eaten, sample_count, return_std);
    }
  }
}

// ── Multi-seed Benchmark Runner ───────────────────────────────────────

static void run_multiseed_benchmark(u32 num_seeds, u32 train_epochs) {
  printf("=================================================================\n");
  printf("  cRL Multi-Seed Evaluation Benchmark (%u seeds, %u epochs/seed)\n",
         num_seeds, train_epochs);
  printf("=================================================================\n\n");

  f32 trained_foods[16] = {0};
  f32 trained_returns[16] = {0};
  f32 trained_steps[16] = {0};

  f32 random_foods[16] = {0};
  f32 random_returns[16] = {0};
  f32 random_steps[16] = {0};

  f32 greedy_foods[16] = {0};
  f32 greedy_returns[16] = {0};
  f32 greedy_steps[16] = {0};

  u64 base_seeds[] = {42, 100, 2024, 777, 9999};

  for (u32 s = 0; s < num_seeds; s++) {
    u64 seed = s < 5 ? base_seeds[s] : (u64)(s * 1337 + 7);
    printf("--- Running Seed %u (seed=%llu) ---\n", s + 1, (unsigned long long)seed);

    prng_seed(seed, 1);
    mem_arena *arena = arena_create(GiB(1));
    model_state *model = PUSH_STRUCT(arena, model_state);
    create_actor_model(arena, model);
    model->forward_graph = build_graph(arena, model, model->output);
    model->cost_graph = build_graph(arena, model, model->cost);

    SnakeENV *env = create_env(36, seed);

    // Evaluate baselines on this seed
    EvalResult rand_res = evaluate_policy(model, env, POLICY_RANDOM, 100);
    EvalResult grdy_res = evaluate_policy(model, env, POLICY_GREEDY, 100);

    random_foods[s] = rand_res.mean_foods;
    random_returns[s] = rand_res.mean_reward;
    random_steps[s] = rand_res.mean_steps;

    greedy_foods[s] = grdy_res.mean_foods;
    greedy_returns[s] = grdy_res.mean_reward;
    greedy_steps[s] = grdy_res.mean_steps;

    // Train
    train(model, env, arena, train_epochs, 64, 0.05f, 0.99f);

    // Evaluate trained policy (greedy argmax)
    EvalResult train_res = evaluate_policy(model, env, POLICY_MODEL_GREEDY, 100);
    trained_foods[s] = train_res.mean_foods;
    trained_returns[s] = train_res.mean_reward;
    trained_steps[s] = train_res.mean_steps;

    printf("Seed %u Results (100 test episodes):\n", s + 1);
    printf("  Random:  Foods = %.2f | Return = %6.2f | Steps = %.1f\n",
           rand_res.mean_foods, rand_res.mean_reward, rand_res.mean_steps);
    printf("  Greedy:  Foods = %.2f | Return = %6.2f | Steps = %.1f\n",
           grdy_res.mean_foods, grdy_res.mean_reward, grdy_res.mean_steps);
    printf("  Trained: Foods = %.2f | Return = %6.2f | Steps = %.1f\n\n",
           train_res.mean_foods, train_res.mean_reward, train_res.mean_steps);

    destroy_env(env);
    arena_destroy(arena);
  }

  // Compute mean and std
  f32 tf_mean = 0, tr_mean = 0, ts_mean = 0;
  f32 rf_mean = 0, rr_mean = 0, rs_mean = 0;
  f32 gf_mean = 0, gr_mean = 0, gs_mean = 0;

  for (u32 s = 0; s < num_seeds; s++) {
    tf_mean += trained_foods[s]; tr_mean += trained_returns[s]; ts_mean += trained_steps[s];
    rf_mean += random_foods[s];  rr_mean += random_returns[s];  rs_mean += random_steps[s];
    gf_mean += greedy_foods[s];  gr_mean += greedy_returns[s];  gs_mean += greedy_steps[s];
  }
  tf_mean /= num_seeds; tr_mean /= num_seeds; ts_mean /= num_seeds;
  rf_mean /= num_seeds; rr_mean /= num_seeds; rs_mean /= num_seeds;
  gf_mean /= num_seeds; gr_mean /= num_seeds; gs_mean /= num_seeds;

  f32 tf_var = 0, tr_var = 0;
  f32 rf_var = 0, rr_var = 0;
  f32 gf_var = 0, gr_var = 0;
  for (u32 s = 0; s < num_seeds; s++) {
    tf_var += (trained_foods[s] - tf_mean) * (trained_foods[s] - tf_mean);
    tr_var += (trained_returns[s] - tr_mean) * (trained_returns[s] - tr_mean);
    rf_var += (random_foods[s] - rf_mean) * (random_foods[s] - rf_mean);
    rr_var += (random_returns[s] - rr_mean) * (random_returns[s] - rr_mean);
    gf_var += (greedy_foods[s] - gf_mean) * (greedy_foods[s] - gf_mean);
    gr_var += (greedy_returns[s] - gr_mean) * (greedy_returns[s] - gr_mean);
  }
  f32 tf_std = sqrtf(tf_var / num_seeds);
  f32 tr_std = sqrtf(tr_var / num_seeds);
  f32 rf_std = sqrtf(rf_var / num_seeds);
  f32 rr_std = sqrtf(rr_var / num_seeds);
  f32 gf_std = sqrtf(gf_var / num_seeds);
  f32 gr_std = sqrtf(gr_var / num_seeds);

  printf("=================================================================\n");
  printf("  Benchmark Summary over %u Seeds (100 eval episodes / seed)\n", num_seeds);
  printf("=================================================================\n");
  printf("| Policy          | Foods Eaten (mean ± std) | Return (mean ± std)   | Mean Steps |\n");
  printf("|-----------------|--------------------------|-----------------------|------------|\n");
  printf("| Random Baseline | %5.2f ± %4.2f             | %6.2f ± %5.2f         | %10.1f |\n",
         rf_mean, rf_std, rr_mean, rr_std, rs_mean);
  printf("| Greedy Baseline | %5.2f ± %4.2f             | %6.2f ± %5.2f         | %10.1f |\n",
         gf_mean, gf_std, gr_mean, gr_std, gs_mean);
  printf("| Trained (REINFORCE) | %5.2f ± %4.2f         | %6.2f ± %5.2f         | %10.1f |\n",
         tf_mean, tf_std, tr_mean, tr_std, ts_mean);
  printf("=================================================================\n\n");
}

// ── Main ─────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
  u64 seed = 42;
  u32 epochs = 300;

  if (argc > 1) {
    if (strcmp(argv[1], "--benchmark") == 0) {
      u32 num_seeds = 5;
      if (argc > 2) {
        num_seeds = (u32)atoi(argv[2]);
      }
      u32 b_epochs = (argc > 3) ? (u32)atoi(argv[3]) : 300;
      run_multiseed_benchmark(num_seeds, b_epochs);
      return 0;
    }
    seed = (u64)strtoull(argv[1], NULL, 10);
  }

  if (argc > 2) {
    epochs = (u32)atoi(argv[2]);
  }

  printf("=== cRL (seed: %llu, epochs: %u) ===\n", (unsigned long long)seed, epochs);

  prng_seed(seed, 1);
  mem_arena *arena = arena_create(GiB(1));
  assert(arena != NULL && "Arena creation failed");

  model_state *model = PUSH_STRUCT(arena, model_state);
  create_actor_model(arena, model);
  model->forward_graph = build_graph(arena, model, model->output);
  model->cost_graph = build_graph(arena, model, model->cost);

  SnakeENV *env = create_env(36, seed);
  assert(env != NULL && "Env creation failed");

  // Initial baseline evaluation
  printf("\n[Pre-training Evaluation — 100 episodes]\n");
  EvalResult rand_res = evaluate_policy(model, env, POLICY_RANDOM, 100);
  printf("  Random Baseline:  Foods: %.2f | Mean Return: %6.2f | Mean Steps: %.1f\n",
         rand_res.mean_foods, rand_res.mean_reward, rand_res.mean_steps);

  EvalResult greedy_res = evaluate_policy(model, env, POLICY_GREEDY, 100);
  printf("  Greedy Baseline:  Foods: %.2f | Mean Return: %6.2f | Mean Steps: %.1f\n",
         greedy_res.mean_foods, greedy_res.mean_reward, greedy_res.mean_steps);

  EvalResult untr_res = evaluate_policy(model, env, POLICY_MODEL_GREEDY, 100);
  printf("  Untrained Model:  Foods: %.2f | Mean Return: %6.2f | Mean Steps: %.1f\n",
         untr_res.mean_foods, untr_res.mean_reward, untr_res.mean_steps);

  // Train
  train(model, env, arena, epochs, 64, 0.05f, 0.99f);

  // Post-training evaluation
  printf("\n[Post-training Evaluation — 100 episodes]\n");
  EvalResult trained_greedy = evaluate_policy(model, env, POLICY_MODEL_GREEDY, 100);
  printf("  Trained (Greedy): Foods: %.2f | Mean Return: %6.2f | Mean Steps: %.1f\n",
         trained_greedy.mean_foods, trained_greedy.mean_reward, trained_greedy.mean_steps);

  EvalResult trained_stoch = evaluate_policy(model, env, POLICY_MODEL_STOCHASTIC, 100);
  printf("  Trained (Stoch):  Foods: %.2f | Mean Return: %6.2f | Mean Steps: %.1f\n",
         trained_stoch.mean_foods, trained_stoch.mean_reward, trained_stoch.mean_steps);

  printf("\n[Summary Table]\n");
  printf("| Policy          | Foods (mean) | Return (mean) | Steps (mean) |\n");
  printf("|-----------------|--------------|---------------|--------------|\n");
  printf("| Random          | %12.2f | %13.2f | %12.1f |\n", rand_res.mean_foods, rand_res.mean_reward, rand_res.mean_steps);
  printf("| Greedy Baseline | %12.2f | %13.2f | %12.1f |\n", greedy_res.mean_foods, greedy_res.mean_reward, greedy_res.mean_steps);
  printf("| Untrained Model | %12.2f | %13.2f | %12.1f |\n", untr_res.mean_foods, untr_res.mean_reward, untr_res.mean_steps);
  printf("| Trained Model   | %12.2f | %13.2f | %12.1f |\n\n", trained_greedy.mean_foods, trained_greedy.mean_reward, trained_greedy.mean_steps);

  destroy_env(env);
  arena_destroy(arena);

  return 0;
}
