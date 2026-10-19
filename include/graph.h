#ifndef GRAPH_H
#define GRAPH_H

#include "memory.h"
#include "base.h"
#include "linalg.h"

// Static computation graph over Mat values. Each Tensor is a node that holds a
// value, an optional gradient, and an operation tag. Forward and reverse
// interpretation both live in src/graph.c and dispatch on the tag in one place
// (no per-operation function-pointer tables).

typedef enum {
  TENSOR_PLAIN = 0,
  TENSOR_GRAD = (1 << 0),
  TENSOR_PARAM = (1 << 1),
} TensorFlags;

// Every operation takes at most two operands.
#define MAX_ARITY 2

typedef enum {
  OP_NONE = 0, // leaf: input, parameter, or constant
  OP_RELU,
  OP_TANH,
  OP_SOFTMAX,
  OP_ADD,
  OP_SUB,
  OP_MUL,
  OP_MM,
  OP_MSE,
  OP_REINFORCE,
  OP_SIGMOID,
  OP_EXP,
  OP_LOG,
  OP_DIV,
  OP_RESHAPE,   // same elements, aux_rows x aux_cols
  OP_BROADCAST, // 1 -> N replicas, aux_rows x aux_cols
  OP_SUM,       // reduce over `axis` (a ReduceMode)
  OP_MEAN,      // reduce over `axis`, divided by the reduced count
  OP_GATHER,    // per-column selection by a 1 x C index row
  OP_MIN,       // elementwise min of two same-shaped operands
  OP_MAX,       // elementwise max
  OP_CLIP,      // clamp to [aux_lo, aux_hi]
} OpKind;

typedef struct Tensor Tensor;

struct Tensor {
  u32 index;
  u32 flags;
  u8 arity;
  OpKind op;

  Mat *val;
  Mat *grad;

  Tensor *inputs[MAX_ARITY];

  // Result shape and parameters for ops whose output shape is not a function
  // of their operands alone: reshape/broadcast store the target in
  // aux_rows/aux_cols, sum/mean store a ReduceMode in axis, clip stores its
  // bounds in aux_lo/aux_hi. Zero for every other op.
  u32 aux_rows;
  u32 aux_cols;
  u32 axis;
  real aux_lo;
  real aux_hi;
};

// A topologically ordered list of nodes (inputs first).
typedef struct {
  Tensor **ops;
  u32 len;
} Schedule;

// Private architecture record owned by the builders in layers.c. `Net` only
// holds an opaque pointer to it, which keeps graph.h free of layer types while
// still letting a builder rebuild its batched subgraph for a new minibatch size
// (see net_set_batch_size).
typedef struct NetArchSpec NetArchSpec;

typedef struct {
  u32 n_tensors;

  // Parameters are registered as they are created. The table starts small and
  // doubles through the arena, so a model is not capped at an arbitrary count.
  Tensor **params;
  u32 n_params;
  u32 params_cap;

  // Set by actor_build / actor_critic_build; NULL for hand-built graphs.
  NetArchSpec *arch;

  Tensor *input;
  Tensor *output;
  Tensor *advantage;
  Tensor *cost;

  // Batched training and critic fields.
  Tensor *batch_input;
  Tensor *batch_logits;
  Tensor *batch_value;
  Tensor *value;

  // DQN only (q_net_build): plain, gradient-free target values for the batched
  // MSE loss, one column per sampled transition. The trainer refills it on
  // every gradient step; because the MSE backpropagates into its prediction
  // operand only, the network's own Q values are the sole gradient path.
  Tensor *target;

  // Continuous (diagonal-Gaussian) policy: the mean head and its batched view,
  // plus a learnable log standard deviation that is a parameter *outside* every
  // schedule (the Gaussian policy loss accumulates into its gradient directly).
  Tensor *mean;
  Tensor *batch_mean;
  Tensor *log_std;

  Schedule fwd;
  Schedule train;
  Schedule loss_plan;
} Net;

Tensor *tensor_create(Arena *arena, Net *model, u32 rows, u32 cols, u32 flags);

Tensor *tensor_relu(Arena *arena, Net *model, Tensor *input);
Tensor *tensor_tanh(Arena *arena, Net *model, Tensor *input);
Tensor *tensor_softmax(Arena *arena, Net *model, Tensor *input);

Tensor *tensor_add(Arena *arena, Net *model, Tensor *a, Tensor *b);
Tensor *tensor_sub(Arena *arena, Net *model, Tensor *a, Tensor *b);
Tensor *tensor_mul(Arena *arena, Net *model, Tensor *a, Tensor *b);
Tensor *tensor_div(Arena *arena, Net *model, Tensor *a, Tensor *b);
Tensor *tensor_sigmoid(Arena *arena, Net *model, Tensor *input);
Tensor *tensor_exp(Arena *arena, Net *model, Tensor *input);
Tensor *tensor_log(Arena *arena, Net *model, Tensor *input);
Tensor *tensor_mm(Arena *arena, Net *model, Tensor *a, Tensor *b);

Tensor *tensor_mse(Arena *arena, Net *model, Tensor *pred, Tensor *target);
Tensor *tensor_reinforce_loss(Arena *arena, Net *model, Tensor *probs,
                              Tensor *rt);

// Shape ops. `tensor_reshape` requires the same element count and copies in
// row-major order; `tensor_broadcast` accepts any input dimension that is 1 or
// already the target size; `tensor_sum`/`tensor_mean` reduce over a ReduceMode
// axis. Gradients flow through all of them: reshape copies back, broadcast sums
// the replicas, and a reduction broadcasts its seed.
Tensor *tensor_reshape(Arena *arena, Net *model, Tensor *input, u32 rows,
                       u32 cols);
Tensor *tensor_broadcast(Arena *arena, Net *model, Tensor *input, u32 rows,
                         u32 cols);
Tensor *tensor_sum(Arena *arena, Net *model, Tensor *input, ReduceMode mode);
Tensor *tensor_mean(Arena *arena, Net *model, Tensor *input, ReduceMode mode);

// Per-column gather. `values` is (R x C), `indices` is (1 x C) holding integral
// action indices in [0, R), and the result is (1 x C). The indices are data,
// not a differentiable input: the gradient scatters back into `values`.
Tensor *tensor_gather(Arena *arena, Net *model, Tensor *values,
                      Tensor *indices);

// Elementwise selection and clamping. min/max take two same-shaped operands
// (ties route the gradient to the first); clip passes gradient inside
// [lo, hi] and blocks it outside.
Tensor *tensor_min(Arena *arena, Net *model, Tensor *a, Tensor *b);
Tensor *tensor_max(Arena *arena, Net *model, Tensor *a, Tensor *b);
Tensor *tensor_clip(Arena *arena, Net *model, Tensor *input, real lo, real hi);

Schedule schedule_from(Arena *arena, Net *model, Tensor *out_var);

// Builds one schedule reachable from several roots. Needed when a model has
// more than one output (e.g. actor-critic with separate policy and value
// heads): a single synthetic root such as `logits + value` is wrong because the
// result of `add` is not the action distribution.
Schedule schedule_from_roots(Arena *arena, Net *model, Tensor **roots,
                              u32 num_roots);

void schedule_run(Schedule *graph);

// Zeroes every activation (non-parameter) gradient in the graph. Parameter
// gradients are left alone so they can accumulate across minibatches.
void schedule_zero_act(Schedule *graph);

// Reverse sweep only: runs each node's backward in reverse topological order
// without clearing or seeding anything. The caller must have cleared activation
// gradients and seeded the root gradient(s) first.
void schedule_grad_accum(Schedule *graph);

void schedule_grad(Schedule *graph);
void schedule_grad_from(Schedule *graph, Tensor *loss_var, b32 seed_one);

Net *net_new(Arena *arena);

#endif
