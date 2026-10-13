#ifndef NN_H
#define NN_H

#include "arena.h"
#include "autograd.h"

// Creates a linear layer: W (out_features x in_features) * in + b (out_features x 1)
// Uses Xavier/Glorot uniform initialization for W, zeros for b.
Var *nn_linear(mem_arena *arena, model_state *model, Var *in, u32 out_features, b32 bias);

// Creates an MLP with specified layer dimensions:
// in -> layer_dims[0] (ReLU) -> ... -> layer_dims[num_layers - 1] (linear output)
Var *nn_mlp(mem_arena *arena, model_state *model, Var *in, const u32 *layer_dims, u32 num_layers);

// Helper to build a discrete actor model:
// input (obs_dim x 1) -> hidden layers (ReLU) -> logits (act_dim x 1) -> softmax (act_dim x 1)
// Also attaches model->advantage and model->cost (reinforce loss).
void nn_build_discrete_actor(mem_arena *arena, model_state *model, u32 obs_dim,
                             const u32 *hidden_dims, u32 num_hidden, u32 act_dim);

#endif // NN_H
