#ifndef NN_H
#define NN_H

#include "arena.h"
#include "autograd.h"
#include "prng.h"

// A fully connected layer: y = W x + b, with W of shape (out x in) and b of
// shape (out x 1). The parameters are real Vars registered with the model, so
// the same layer can be applied to several inputs (rollout and batched graphs)
// without duplicating weights.
typedef struct {
  Var *W; // (out_features x in_features), always present
  Var *b; // (out_features x 1), or NULL when bias == false
  u32 in_features;
  u32 out_features;
} Linear;

// Allocates the parameters for one layer. Weights use He-uniform
// initialization (limit = gain * sqrt(6 / in_features)); bias starts at zero.
// Pass gain == 1 for layers followed by ReLU and a small gain (e.g. 0.01) for
// policy/value heads so the initial policy is close to uniform.
Linear nn_linear_create(mem_arena *arena, model_state *model, prng_state *rng,
                        u32 in_features, u32 out_features, b32 bias, real gain);

// Applies a layer to `in`. Creates a matmul (and optional broadcast add) node
// that shares the layer's parameter Vars. May be called repeatedly with
// different inputs.
Var *nn_linear_apply(mem_arena *arena, model_state *model, const Linear *layer,
                     Var *in);

// MLP: in -> layer_dims[0] (ReLU) -> ... -> layer_dims[n-1] (linear output).
Var *nn_mlp(mem_arena *arena, model_state *model, prng_state *rng, Var *in,
            const u32 *layer_dims, u32 num_layers);

// Builds a discrete actor:
//   input (obs_dim x 1) -> hidden (ReLU) -> logits -> softmax (act_dim x 1)
// When batch_size > 0 it also builds batch_input (obs_dim x B) and
// batch_logits (act_dim x B) sharing the exact same parameter weights.
void nn_build_discrete_actor(mem_arena *arena, model_state *model,
                             prng_state *rng, u32 obs_dim,
                             const u32 *hidden_dims, u32 num_hidden,
                             u32 act_dim, u32 batch_size);

// Builds an actor-critic with a shared trunk and separate actor/value heads:
//   - Rollout inference: input -> output (action probs) and value (1 x 1)
//   - Batched training: batch_input -> batch_logits and batch_value (1 x B)
void nn_build_actor_critic(mem_arena *arena, model_state *model,
                           prng_state *rng, u32 obs_dim,
                           const u32 *hidden_dims, u32 num_hidden,
                           u32 act_dim, u32 batch_size);

#endif // NN_H
