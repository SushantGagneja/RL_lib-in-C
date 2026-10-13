#include "nn.h"
#include <math.h>

Var *nn_linear(mem_arena *arena, model_state *model, Var *in, u32 out_features,
               b32 bias) {
  u32 in_features = in->val->rows;

  Var *W = var_create(arena, model, out_features, in_features,
                      VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);

  f32 limit = sqrtf(6.0f / (f32)(in_features + out_features));
  fill_rand(W->val, -limit, limit);

  Var *z = var_matmul(arena, model, W, in);

  if (bias) {
    Var *b = var_create(arena, model, out_features, 1,
                        VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
    clear(b->val);
    return var_add(arena, model, z, b);
  }

  return z;
}

Var *nn_mlp(mem_arena *arena, model_state *model, Var *in,
            const u32 *layer_dims, u32 num_layers) {
  Var *cur = in;
  for (u32 i = 0; i < num_layers; i++) {
    cur = nn_linear(arena, model, cur, layer_dims[i], true);
    if (i + 1 < num_layers) {
      cur = var_relu(arena, model, cur);
    }
  }
  return cur;
}

void nn_build_discrete_actor(mem_arena *arena, model_state *model, u32 obs_dim,
                             const u32 *hidden_dims, u32 num_hidden,
                             u32 act_dim) {
  model->input = var_create(arena, model, obs_dim, 1, VAR_FLAG_NONE);

  Var *cur = model->input;
  for (u32 i = 0; i < num_hidden; i++) {
    cur = nn_linear(arena, model, cur, hidden_dims[i], true);
    cur = var_relu(arena, model, cur);
  }

  Var *logits = nn_linear(arena, model, cur, act_dim, true);
  model->output = var_softmax(arena, model, logits);

  model->advantage = var_create(arena, model, act_dim, 1, VAR_FLAG_NONE);
  model->cost = var_reinforce_loss(arena, model, model->output, model->advantage);
}
