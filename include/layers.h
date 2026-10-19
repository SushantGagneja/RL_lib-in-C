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

// A dense layer whose input is split into an observation and an action block:
//   y = w_obs @ obs + w_act @ act + bias
// Used by the SAC critics, where Q(s, a) must be differentiable in a
// graph-computed action: `act` can be any graph tensor and the reverse sweep
// yields dQ/da through w_act, without a concatenation op. Both matrices use the
// He-uniform limit of the joint fan-in (obs_dim + act_dim).
typedef struct {
  Tensor *w_obs; // (out_features x obs_features)
  Tensor *w_act; // (out_features x act_features)
  Tensor *bias;  // (out_features x 1), or NULL
  u32 obs_features;
  u32 act_features;
  u32 out_features;
} SplitDense;

// The layer stack of a SAC critic, kept accessible so the trainer can apply the
// same weights to a graph-computed action for the actor loss. Every tensor
// pointer is a model parameter; copying the struct copies only the descriptors.
typedef struct {
  SplitDense split; // (obs, act) -> hidden_dims[0], or the Q value itself when
                    // the critic is linear (num_hidden == 0)
  Dense *hidden;    // (num_hidden - 1) ReLU layers, arena-owned
  u32 num_hidden;   // entries in `hidden`
  b32 split_is_head; // true when `split` already produces the scalar Q
  Dense head;       // trunk -> 1 (unused when split_is_head)
} SacQNet;

// Applies the critic's layer stack to an (observation, action) pair inside
// `model`'s graph. Both operands are (features x batch) matrices.
Tensor *sac_q_apply(Arena *arena, Net *model, const SacQNet *q, Tensor *obs,
                    Tensor *act);

// Builds a SAC critic: the parameter stack of Q(s, a), plus the batched graph
// (batch_input, batch_action) -> batch_logits (1 x B), a gradient-free `target`
// (1 x B) and the MSE `cost` with its `loss_plan`, mirroring q_net_build. The
// record lets net_set_batch_size rebuild it and a checkpoint prove it was
// written for the same architecture.
void sac_q_build(Arena *arena, Net *model, Rng *rng, u32 obs_dim, u32 act_dim,
                 const u32 *hidden_dims, u32 num_hidden, u32 batch_size);

// Reads the layer descriptors a sac_q_build model recorded. Returns false for
// any other kind of model.
b32 sac_q_layers(const Net *model, SacQNet *out);

// SAC actor: obs -> mean head (act_dim x 1) plus the same learnable log_std
// parameter as continuous_actor_critic_build, but no value head (SAC's critics
// are separate networks). `log_std_init` seeds log(sigma); pass 0 for sigma = 1.
void sac_actor_build(Arena *arena, Net *model, Rng *rng, u32 obs_dim,
                     const u32 *hidden_dims, u32 num_hidden, u32 act_dim,
                     u32 batch_size, real log_std_init);

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

// Continuous-action actor-critic: a mean head (`model->mean`, act_dim x 1) plus
// a value head, and a learnable log standard deviation (`model->log_std`,
// act_dim x 1) shared by every batch column. `log_std_init` seeds log(sigma);
// pass 0 for sigma = 1, or something like -0.5 to start with a smaller
// exploration noise. `model->output` stays NULL: this model has no action
// probabilities, so the discrete-only paths cannot be used on it by mistake.
void continuous_actor_critic_build(Arena *arena, Net *model, Rng *rng,
                                   u32 obs_dim, const u32 *hidden_dims,
                                   u32 num_hidden, u32 act_dim, u32 batch_size,
                                   real log_std_init);

// Builds a discrete-action Q-network (DQN):
//   - Rollout inference: input -> output, one Q value per action (act_dim x 1)
//   - Batched training: batch_input -> batch_logits (act_dim x B), plus a
//     plain `target` tensor (act_dim x B) and `cost` = MSE(batch_logits,
//     target). Regression targets are written by the trainer, so they must not
//     receive gradients — `target` is a plain leaf, and the MSE propagates into
//     its prediction operand only.
// The output head is linear (no softmax: these are values, not probabilities),
// so `model->output` cannot be used with the stochastic sampling paths.
void q_net_build(Arena *arena, Net *model, Rng *rng, u32 obs_dim,
                 const u32 *hidden_dims, u32 num_hidden, u32 act_dim,
                 u32 batch_size);

// Rebuilds the batched training subgraph (batch_input -> batch_logits /
// batch_value) for a new minibatch size. The parameter tensors are shared, not
// re-created, so the model keeps its weights; only the fixed-size batch views
// and the training schedule are replaced. Returns false for a model that was not
// built by one of the builder functions above (a hand-built graph has no
// recorded architecture to rebuild from), and true immediately when the size already
// matches. Any pointer the caller held into batch_input / batch_logits /
// batch_value is invalidated by a rebuild.
b32 net_set_batch_size(Arena *arena, Net *model, u32 batch_size);

// Minibatch size the batched training graph is currently built for, or 0 when
// the model has no batched graph.
u32 net_batch_size(const Net *model);

// The kind of model a builder produced. Recorded in the graph so the batched
// subgraph can be rebuilt (net_set_batch_size) and so a checkpoint can prove it
// was written for the same kind of model.
typedef enum {
  NET_ARCH_NONE = 0,
  NET_ARCH_ACTOR,
  NET_ARCH_ACTOR_CRITIC,
  NET_ARCH_CONTINUOUS,
  NET_ARCH_Q,
  NET_ARCH_SAC_ACTOR,
  NET_ARCH_SAC_Q,
} NetArchKind;

// A read-only view of the architecture a builder recorded for `model`.
// `hidden_dims` points at `num_hidden` arena-owned widths; `batch_size` is the
// size the batched graph was last built for (0 before net_set_batch_size or
// when the builder was called with batch_size 0). Hand-built graphs have no
// record, and net_arch_info returns false for them.
typedef struct {
  NetArchKind kind;
  u32 obs_dim;
  u32 act_dim;
  u32 num_hidden;
  const u32 *hidden_dims;
  u32 batch_size;
} NetArchInfo;

b32 net_arch_info(const Net *model, NetArchInfo *out);

#endif // LAYERS_H
