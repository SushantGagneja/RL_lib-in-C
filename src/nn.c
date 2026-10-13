#include "nn.h"
#include <math.h>

Var *nn_linear(mem_arena *arena, model_state *model, prng_state *rng,
               Var *in, u32 out_features, b32 bias) {
  u32 in_features = in->val->rows;

  Var *W = var_create(arena, model, out_features, in_features,
                      VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);

  f32 limit = sqrtf(6.0f / (f32)(in_features + out_features));
  if (rng != NULL) {
    fill_rand_r(W->val, rng, -limit, limit);
  } else {
    fill_rand(W->val, -limit, limit);
  }

  Var *z = var_matmul(arena, model, W, in);

  if (bias) {
    Var *b = var_create(arena, model, out_features, 1,
                        VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
    clear(b->val);
    return var_add(arena, model, z, b);
  }

  return z;
}

Var *nn_mlp(mem_arena *arena, model_state *model, prng_state *rng, Var *in,
            const u32 *layer_dims, u32 num_layers) {
  Var *cur = in;
  for (u32 i = 0; i < num_layers; i++) {
    cur = nn_linear(arena, model, rng, cur, layer_dims[i], true);
    if (i + 1 < num_layers) {
      cur = var_relu(arena, model, cur);
    }
  }
  return cur;
}

Var *nn_linear_with_params(mem_arena *arena, model_state *model, Var *in,
                           Var *W, Var *b) {
  Var *z = var_matmul(arena, model, W, in);
  if (b != NULL) {
    z = var_add(arena, model, z, b);
  }
  return z;
}

void nn_build_discrete_actor(mem_arena *arena, model_state *model,
                             prng_state *rng, u32 obs_dim,
                             const u32 *hidden_dims, u32 num_hidden,
                             u32 act_dim, u32 batch_size) {
  model->input = var_create(arena, model, obs_dim, 1, VAR_FLAG_NONE);

  Var **W_layers = PUSH_ARRAY(arena, Var *, num_hidden + 1);
  Var **b_layers = PUSH_ARRAY(arena, Var *, num_hidden + 1);

  Var *cur = model->input;
  u32 in_dim = obs_dim;

  for (u32 i = 0; i < num_hidden; i++) {
    u32 out_dim = hidden_dims[i];
    Var *W = var_create(arena, model, out_dim, in_dim,
                        VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
    f32 limit = sqrtf(6.0f / (f32)(in_dim + out_dim));
    if (rng != NULL) {
      fill_rand_r(W->val, rng, -limit, limit);
    } else {
      fill_rand(W->val, -limit, limit);
    }

    Var *b = var_create(arena, model, out_dim, 1,
                        VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
    mat_clear(b->val);

    W_layers[i] = W;
    b_layers[i] = b;

    cur = nn_linear_with_params(arena, model, cur, W, b);
    cur = var_relu(arena, model, cur);
    in_dim = out_dim;
  }

  Var *W_last = var_create(arena, model, act_dim, in_dim,
                           VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
  f32 limit = sqrtf(6.0f / (f32)(in_dim + act_dim));
  if (rng != NULL) {
    fill_rand_r(W_last->val, rng, -limit, limit);
  } else {
    fill_rand(W_last->val, -limit, limit);
  }

  Var *b_last = var_create(arena, model, act_dim, 1,
                           VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
  mat_clear(b_last->val);

  W_layers[num_hidden] = W_last;
  b_layers[num_hidden] = b_last;

  Var *logits = nn_linear_with_params(arena, model, cur, W_last, b_last);
  model->output = var_softmax(arena, model, logits);

  model->advantage = var_create(arena, model, act_dim, 1, VAR_FLAG_NONE);
  model->cost = var_reinforce_loss(arena, model, model->output, model->advantage);

  if (batch_size > 0) {
    model->batch_input = var_create(arena, model, obs_dim, batch_size, VAR_FLAG_NONE);
    Var *b_cur = model->batch_input;

    for (u32 i = 0; i < num_hidden; i++) {
      b_cur = nn_linear_with_params(arena, model, b_cur, W_layers[i], b_layers[i]);
      b_cur = var_relu(arena, model, b_cur);
    }

    model->batch_logits = nn_linear_with_params(arena, model, b_cur,
                                                W_layers[num_hidden],
                                                b_layers[num_hidden]);
    model->batch_graph = build_graph(arena, model, model->batch_logits);
  }
}

void nn_build_actor_critic(mem_arena *arena, model_state *model,
                           prng_state *rng, u32 obs_dim,
                           const u32 *hidden_dims, u32 num_hidden,
                           u32 act_dim, u32 batch_size) {
  model->input = var_create(arena, model, obs_dim, 1, VAR_FLAG_NONE);

  Var **W_trunk = PUSH_ARRAY(arena, Var *, num_hidden);
  Var **b_trunk = PUSH_ARRAY(arena, Var *, num_hidden);

  Var *cur = model->input;
  u32 in_dim = obs_dim;

  for (u32 i = 0; i < num_hidden; i++) {
    u32 out_dim = hidden_dims[i];
    Var *W = var_create(arena, model, out_dim, in_dim,
                        VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
    f32 limit = sqrtf(6.0f / (f32)(in_dim + out_dim));
    if (rng != NULL) fill_rand_r(W->val, rng, -limit, limit);
    else fill_rand(W->val, -limit, limit);

    Var *b = var_create(arena, model, out_dim, 1,
                        VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
    mat_clear(b->val);

    W_trunk[i] = W;
    b_trunk[i] = b;

    cur = nn_linear_with_params(arena, model, cur, W, b);
    cur = var_relu(arena, model, cur);
    in_dim = out_dim;
  }

  // Actor head
  Var *W_act = var_create(arena, model, act_dim, in_dim,
                          VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
  f32 limit_act = sqrtf(6.0f / (f32)(in_dim + act_dim));
  if (rng != NULL) fill_rand_r(W_act->val, rng, -limit_act, limit_act);
  else fill_rand(W_act->val, -limit_act, limit_act);

  Var *b_act = var_create(arena, model, act_dim, 1,
                          VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
  mat_clear(b_act->val);

  Var *logits = nn_linear_with_params(arena, model, cur, W_act, b_act);
  model->output = var_softmax(arena, model, logits);

  // Critic head
  Var *W_val = var_create(arena, model, 1, in_dim,
                          VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
  f32 limit_val = sqrtf(6.0f / (f32)(in_dim + 1));
  if (rng != NULL) fill_rand_r(W_val->val, rng, -limit_val, limit_val);
  else fill_rand(W_val->val, -limit_val, limit_val);

  Var *b_val = var_create(arena, model, 1, 1,
                          VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
  mat_clear(b_val->val);

  model->value = nn_linear_with_params(arena, model, cur, W_val, b_val);

  // Build forward graph covering both actor output and value
  Var *combined_step = var_add(arena, model, logits, model->value);
  model->forward_graph = build_graph(arena, model, combined_step);

  if (batch_size > 0) {
    model->batch_input = var_create(arena, model, obs_dim, batch_size, VAR_FLAG_NONE);
    Var *b_cur = model->batch_input;

    for (u32 i = 0; i < num_hidden; i++) {
      b_cur = nn_linear_with_params(arena, model, b_cur, W_trunk[i], b_trunk[i]);
      b_cur = var_relu(arena, model, b_cur);
    }

    model->batch_logits = nn_linear_with_params(arena, model, b_cur, W_act, b_act);
    model->batch_value = nn_linear_with_params(arena, model, b_cur, W_val, b_val);

    Var *combined_batch = var_add(arena, model, model->batch_logits, model->batch_value);
    model->batch_graph = build_graph(arena, model, combined_batch);
  }
}
