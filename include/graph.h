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
Tensor *tensor_mm(Arena *arena, Net *model, Tensor *a, Tensor *b);

Tensor *tensor_mse(Arena *arena, Net *model, Tensor *pred, Tensor *target);
Tensor *tensor_reinforce_loss(Arena *arena, Net *model, Tensor *probs,
                              Tensor *rt);

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
