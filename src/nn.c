#include "nn.h"
#include <math.h>

// ── Linear layer ─────────────────────────────────────────────────────

static Var *linear_param(mem_arena *arena, model_state *model, u32 rows,
                         u32 cols) {
  return var_create(arena, model, rows, cols,
                    VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
}

Linear nn_linear_create(mem_arena *arena, model_state *model, prng_state *rng,
                        u32 in_features, u32 out_features, b32 bias, real gain) {
  CRL_ASSERT(in_features > 0 && out_features > 0,
             "nn_linear_create: zero-sized layer");

  Linear layer = {
      .in_features = in_features,
      .out_features = out_features,
  };

  layer.W = linear_param(arena, model, out_features, in_features);
  real limit = gain * real_sqrt(6.0f / (real)in_features);
  if (rng != NULL) {
    mat_fill_rand_r(layer.W->val, rng, -limit, limit);
  } else {
    mat_fill_rand(layer.W->val, -limit, limit);
  }

  if (bias) {
    layer.b = linear_param(arena, model, out_features, 1);
    mat_clear(layer.b->val);
  } else {
    layer.b = NULL;
  }

  return layer;
}

Var *nn_linear_apply(mem_arena *arena, model_state *model, const Linear *layer,
                     Var *in) {
  Var *z = var_matmul(arena, model, layer->W, in);
  if (layer->b != NULL) {
    z = var_add(arena, model, z, layer->b);
  }
  return z;
}

// ── MLP ──────────────────────────────────────────────────────────────

Var *nn_mlp(mem_arena *arena, model_state *model, prng_state *rng, Var *in,
            const u32 *layer_dims, u32 num_layers) {
  Var *cur = in;
  u32 in_dim = in->val->rows;

  for (u32 i = 0; i < num_layers; i++) {
    Linear layer =
        nn_linear_create(arena, model, rng, in_dim, layer_dims[i], true, 1.0f);
    cur = nn_linear_apply(arena, model, &layer, cur);
    if (i + 1 < num_layers) {
      cur = var_relu(arena, model, cur);
    }
    in_dim = layer_dims[i];
  }

  return cur;
}

// ── Discrete actor ───────────────────────────────────────────────────

// Builds the shared trunk (hidden ReLU layers) returning the last activation
// and its width. `layers` receives the created Linear structs.
static Var *build_trunk(mem_arena *arena, model_state *model, prng_state *rng,
                        Var *in, u32 in_dim, const u32 *hidden_dims,
                        u32 num_hidden, Linear *layers) {
  Var *cur = in;
  for (u32 i = 0; i < num_hidden; i++) {
    layers[i] = nn_linear_create(arena, model, rng, in_dim, hidden_dims[i],
                                 true, 1.0f);
    cur = nn_linear_apply(arena, model, &layers[i], cur);
    cur = var_relu(arena, model, cur);
    in_dim = hidden_dims[i];
  }
  return cur;
}

void nn_build_discrete_actor(mem_arena *arena, model_state *model,
                             prng_state *rng, u32 obs_dim,
                             const u32 *hidden_dims, u32 num_hidden,
                             u32 act_dim, u32 batch_size) {
  model->input = var_create(arena, model, obs_dim, 1, VAR_FLAG_NONE);

  Linear *trunk = PUSH_ARRAY(arena, Linear, num_hidden);
  Var *cur = build_trunk(arena, model, rng, model->input, obs_dim, hidden_dims,
                         num_hidden, trunk);
  u32 trunk_dim = (num_hidden > 0) ? hidden_dims[num_hidden - 1] : obs_dim;

  // Policy head with a small gain so the initial policy is near-uniform.
  Linear actor =
      nn_linear_create(arena, model, rng, trunk_dim, act_dim, true, 0.01f);
  Var *logits = nn_linear_apply(arena, model, &actor, cur);
  model->output = var_softmax(arena, model, logits);

  model->advantage = var_create(arena, model, act_dim, 1, VAR_FLAG_NONE);
  model->cost =
      var_reinforce_loss(arena, model, model->output, model->advantage);
  model->forward_graph = build_graph(arena, model, model->output);

  if (batch_size > 0) {
    model->batch_input =
        var_create(arena, model, obs_dim, batch_size, VAR_FLAG_NONE);
    Var *b_cur = model->batch_input;
    for (u32 i = 0; i < num_hidden; i++) {
      b_cur = nn_linear_apply(arena, model, &trunk[i], b_cur);
      b_cur = var_relu(arena, model, b_cur);
    }
    model->batch_logits = nn_linear_apply(arena, model, &actor, b_cur);
    model->batch_graph = build_graph(arena, model, model->batch_logits);
  }
}

// ── Actor-critic ─────────────────────────────────────────────────────

void nn_build_actor_critic(mem_arena *arena, model_state *model,
                           prng_state *rng, u32 obs_dim,
                           const u32 *hidden_dims, u32 num_hidden,
                           u32 act_dim, u32 batch_size) {
  model->input = var_create(arena, model, obs_dim, 1, VAR_FLAG_NONE);

  Linear *trunk = PUSH_ARRAY(arena, Linear, num_hidden);
  Var *cur = build_trunk(arena, model, rng, model->input, obs_dim, hidden_dims,
                         num_hidden, trunk);
  u32 trunk_dim = (num_hidden > 0) ? hidden_dims[num_hidden - 1] : obs_dim;

  // Actor head (small gain) and value head (unit gain).
  Linear actor =
      nn_linear_create(arena, model, rng, trunk_dim, act_dim, true, 0.01f);
  Linear critic =
      nn_linear_create(arena, model, rng, trunk_dim, 1, true, 1.0f);

  Var *logits = nn_linear_apply(arena, model, &actor, cur);
  model->output = var_softmax(arena, model, logits);
  model->value = nn_linear_apply(arena, model, &critic, cur);

  // Forward graph must reach BOTH the softmax output and the value head. Rooting
  // it at a synthetic `logits + value` node (as the old code did) both failed
  // to build (shape mismatch) and never ran the softmax.
  Var *forward_roots[2] = {model->output, model->value};
  model->forward_graph =
      build_graph_multi(arena, model, forward_roots, 2);

  if (batch_size > 0) {
    model->batch_input =
        var_create(arena, model, obs_dim, batch_size, VAR_FLAG_NONE);
    Var *b_cur = model->batch_input;
    for (u32 i = 0; i < num_hidden; i++) {
      b_cur = nn_linear_apply(arena, model, &trunk[i], b_cur);
      b_cur = var_relu(arena, model, b_cur);
    }

    model->batch_logits = nn_linear_apply(arena, model, &actor, b_cur);
    model->batch_value = nn_linear_apply(arena, model, &critic, b_cur);

    Var *batch_roots[2] = {model->batch_logits, model->batch_value};
    model->batch_graph = build_graph_multi(arena, model, batch_roots, 2);
  }
}
