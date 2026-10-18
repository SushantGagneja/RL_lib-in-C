#ifndef LAYERS_H
#define LAYERS_H

#include "memory.h"
#include "graph.h"
#include "random.h"

// A fully connected layer: y = weight @ x + bias, with weight stored as
// (out_features x in_features) and bias as (out_features x 1). The parameters
// are Tensors registered with the model, so one layer can be applied to several
// inputs (rollout and batched graphs) without duplicating weights.
typedef struct {
  Tensor *weight; // (out_features x in_features)
  Tensor *bias;   // (out_features x 1), or NULL when bias == false
  u32 in_features;
  u32 out_features;
} Dense;

// Allocates one layer's parameters. Weights use He-uniform init
// (limit = gain * sqrt(6 / in_features)); bias starts at zero. Use gain == 1
// for layers followed by ReLU and a small gain (e.g. 0.01) for policy/value
// heads so the initial policy is close to uniform.
Dense dense_new(Arena *arena, Net *model, Rng *rng, u32 in_features,
                u32 out_features, b32 bias, real gain);

// Applies `layer` to `in`, creating a matrix-multiply (and optional broadcast
// add) node that shares the layer's parameters. May be called repeatedly with
// different inputs.
Tensor *dense_apply(Arena *arena, Net *model, const Dense *layer, Tensor *in);

// MLP: in -> layer_dims[0] (ReLU) -> ... -> layer_dims[n-1] (linear output).
Tensor *mlp_new(Arena *arena, Net *model, Rng *rng, Tensor *in,
                const u32 *layer_dims, u32 num_layers);

// Builds a discrete actor:
//   input (obs_dim x 1) -> hidden (ReLU) -> logits -> softmax (act_dim x 1)
// When batch_size > 0 it also builds batch_input (obs_dim x B) and
// batch_logits (act_dim x B) sharing the exact same parameter weights.
void actor_build(Arena *arena, Net *model, Rng *rng, u32 obs_dim,
                 const u32 *hidden_dims, u32 num_hidden, u32 act_dim,
                 u32 batch_size);

// Builds an actor-critic with a shared trunk and separate actor/value heads:
//   - Rollout inference: input -> output (action probs) and value (1 x 1)
//   - Batched training: batch_input -> batch_logits and batch_value (1 x B)
void actor_critic_build(Arena *arena, Net *model, Rng *rng, u32 obs_dim,
                        const u32 *hidden_dims, u32 num_hidden, u32 act_dim,
                        u32 batch_size);

#endif // LAYERS_H
