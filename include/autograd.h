#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "arena.h"
#include "base.h"
#include "mat.h"

typedef enum {
  VAR_FLAG_NONE = 0,
  VAR_FLAG_REQUIRES_GRAD = (1 << 0),
  VAR_FLAG_PARAMETER = (1 << 1),
} VarFlags;

#define VAR_MAX_INPUTS 2

typedef struct Var Var;
typedef struct VarType VarType;
typedef void VarFunc(Var *var);
typedef b32 VarShapeFunc(Var *a, Var *b, u32 *rows, u32 *cols);

struct VarType {
  const char *name;
  u32 num_inputs;
  VarShapeFunc *shape;
  VarFunc *forward;
  VarFunc *backward;
};

struct Var {
  u32 index;
  u32 flags;

  matrix *val;
  matrix *grad;

  const VarType *type;
  Var *inputs[VAR_MAX_INPUTS];
};

extern const VarType VAR_TYPE_RELU;
extern const VarType VAR_TYPE_TANH;
extern const VarType VAR_TYPE_SOFTMAX;
extern const VarType VAR_TYPE_ADD;
extern const VarType VAR_TYPE_SUB;
extern const VarType VAR_TYPE_MUL;
extern const VarType VAR_TYPE_MATMUL;
extern const VarType VAR_TYPE_MSE;
extern const VarType VAR_TYPE_REINFORCE_LOSS;

typedef struct {
  Var **vars;
  u32 size;
} Graph;

#define MODEL_MAX_PARAMS 128

typedef struct {
  u32 num_vars;

  Var *params[MODEL_MAX_PARAMS];
  u32 num_params;

  Var *input;
  Var *output;
  Var *advantage;
  Var *cost;

  // Batched training and Critic fields:
  Var *batch_input;
  Var *batch_logits;
  Var *batch_value;
  Var *value;

  Graph forward_graph;
  Graph batch_graph;
  Graph cost_graph;
} model_state;

Var *var_create(mem_arena *arena, model_state *model, u32 rows, u32 cols,
                u32 flags);

Var *var_relu(mem_arena *arena, model_state *model, Var *input);
Var *var_tanh(mem_arena *arena, model_state *model, Var *input);
Var *var_softmax(mem_arena *arena, model_state *model, Var *input);

Var *var_add(mem_arena *arena, model_state *model, Var *a, Var *b);
Var *var_sub(mem_arena *arena, model_state *model, Var *a, Var *b);
Var *var_mul(mem_arena *arena, model_state *model, Var *a, Var *b);
Var *var_matmul(mem_arena *arena, model_state *model, Var *a, Var *b);

Var *var_mse(mem_arena *arena, model_state *model, Var *pred, Var *target);
Var *var_reinforce_loss(mem_arena *arena, model_state *model, Var *probs,
                        Var *rt);

Graph build_graph(mem_arena *arena, model_state *model, Var *out_var);

// Builds a single graph reachable from several roots. This is required when a
// model has more than one output (e.g. an actor-critic with separate policy and
// value heads): a single "fake" root such as `logits + value` is wrong because
// the result of `add` is not the softmax/action distribution.
Graph build_graph_multi(mem_arena *arena, model_state *model, Var **roots,
                        u32 num_roots);

void forward_pass(Graph *graph);

// Zeroes every activation (non-parameter) gradient in the graph. Parameter
// gradients are left untouched so they can accumulate across minibatches.
void graph_zero_activation_grads(Graph *graph);

// Reverse sweep only: runs every node's backward in reverse topological order
// without clearing or seeding anything. The caller must have cleared activation
// gradients and seeded the root gradient(s) first.
void backward_pass_accumulate(Graph *graph);

void backward_pass(Graph *graph);
void backward_pass_from(Graph *graph, Var *loss_var, b32 seed_one);

model_state *model_create(mem_arena *arena);

#endif
