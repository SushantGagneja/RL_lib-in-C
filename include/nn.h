#ifndef NN_H
#define NN_H

#include "arena.h"
#include "autograd.h"
#include "prng.h"

// Creates a linear layer: W (out_features x in_features) * in + b (out_features x 1)
// Uses Xavier/Glorot uniform initialization for W, zeros for b, seeded with rng.
Var *nn_linear(mem_arena *arena, model_state *model, prng_state *rng, Var *in,
               u32 out_features, b32 bias);

// Creates an MLP with specified layer dimensions:
// in -> layer_dims[0] (ReLU) -> ... -> layer_dims[num_layers - 1] (linear output)
Var *nn_mlp(mem_arena *arena, model_state *model, prng_state *rng, Var *in,
            const u32 *layer_dims, u32 num_layers);

// Creates a linear layer reusing existing parameter variables W and b:
Var *nn_linear_with_params(mem_arena *arena, model_state *model, Var *in,
                           Var *W, Var *b);

// Helper to build a discrete actor model:
// input (obs_dim x 1) -> hidden layers (ReLU) -> logits -> softmax (act_dim x 1)
// When batch_size > 0, also creates batch_input (obs_dim x B) and batch_logits (act_dim x B)
// sharing the exact same parameter weights.
void nn_build_discrete_actor(mem_arena *arena, model_state *model,
                             prng_state *rng, u32 obs_dim,
                             const u32 *hidden_dims, u32 num_hidden,
                             u32 act_dim, u32 batch_size);

// Builds an Actor-Critic network with shared trunk and separate actor & critic heads:
// - Rollout inference: model->input -> model->output (action probs) & model->value (1 x 1)
// - Batched training: model->batch_input -> model->batch_logits & model->batch_value (1 x B)
void nn_build_actor_critic(mem_arena *arena, model_state *model,
                           prng_state *rng, u32 obs_dim,
                           const u32 *hidden_dims, u32 num_hidden,
                           u32 act_dim, u32 batch_size);

#endif // NN_H
