#include "graph.h"

#include <string.h>

// ---------------------------------------------------------------------------
// Shape rules: decide the output shape of an op and reject mismatches up front.
// ---------------------------------------------------------------------------

typedef b32 ShapeRule(const Tensor *a, const Tensor *b, u32 *rows, u32 *cols);

static b32 shape_same(const Tensor *a, const Tensor *b, u32 *rows, u32 *cols) {
  if (a == NULL) {
    return false;
  }
  if (b != NULL && (a->val->rows != b->val->rows ||
                    a->val->cols != b->val->cols)) {
    return false;
  }
  *rows = a->val->rows;
  *cols = a->val->cols;
  return true;
}

static b32 shape_broadcast(const Tensor *a, const Tensor *b, u32 *rows,
                           u32 *cols) {
  if (a == NULL || b == NULL || a->val->rows != b->val->rows) {
    return false;
  }
  u32 ca = a->val->cols;
  u32 cb = b->val->cols;
  if (ca != cb && ca != 1u && cb != 1u) {
    return false;
  }
  *rows = a->val->rows;
  *cols = (ca == cb) ? ca : (ca == 1u ? cb : ca);
  return true;
}

static b32 shape_mm(const Tensor *a, const Tensor *b, u32 *rows, u32 *cols) {
  if (a == NULL || b == NULL || a->val->cols != b->val->rows) {
    return false;
  }
  *rows = a->val->rows;
  *cols = b->val->cols;
  return true;
}

static b32 shape_scalar(const Tensor *a, const Tensor *b, u32 *rows,
                        u32 *cols) {
  if (a == NULL || b == NULL || a->val->rows != b->val->rows ||
      a->val->cols != b->val->cols) {
    return false;
  }
  *rows = 1u;
  *cols = 1u;
  return true;
}

static u8 op_arity(OpKind op) {
  switch (op) {
  case OP_NONE:
    return 0u;
  case OP_RELU:
  case OP_TANH:
  case OP_SOFTMAX:
  case OP_SIGMOID:
  case OP_EXP:
  case OP_LOG:
  case OP_RESHAPE:
  case OP_BROADCAST:
  case OP_SUM:
  case OP_MEAN:
  case OP_CLIP:
    return 1u;
  default:
    return 2u;
  }
}

static ShapeRule *shape_rule(OpKind op) {
  switch (op) {
  case OP_RELU:
  case OP_TANH:
  case OP_SOFTMAX:
  case OP_MUL:
  case OP_REINFORCE:
  case OP_SIGMOID:
  case OP_EXP:
  case OP_LOG:
  case OP_DIV:
  case OP_MIN:
  case OP_MAX:
    return shape_same;
  case OP_ADD:
  case OP_SUB:
    return shape_broadcast;
  case OP_MM:
    return shape_mm;
  case OP_MSE:
    return shape_scalar;
  case OP_NONE:
  default:
    return NULL;
  }
}

// ---------------------------------------------------------------------------
// Node construction
// ---------------------------------------------------------------------------

// Adds a parameter to the model's registry, growing the table through the arena
// when it is full. Growth doubles the capacity, so registering N parameters
// costs O(N) total copying, and the old table is simply abandoned to the arena
// (no individual free, no cap).
static void register_param(Arena *arena, Net *model, Tensor *t) {
  if (model->n_params == model->params_cap) {
    const u32 cap = (model->params_cap > 0) ? model->params_cap * 2u : 16u;
    Tensor **grown = ARENA_ARR(arena, Tensor *, cap);
    for (u32 i = 0; i < model->n_params; i++) {
      grown[i] = model->params[i];
    }
    model->params = grown;
    model->params_cap = cap;
  }
  model->params[model->n_params++] = t;
}

Tensor *tensor_create(Arena *arena, Net *model, u32 rows, u32 cols,
                      u32 flags) {
  Tensor *out = ARENA_OBJ(arena, Tensor);
  RL_LIB_ASSERT(out != NULL, "Arena out of memory in tensor_create");

  out->index = model->n_tensors++;
  out->flags = flags;
  out->arity = 0u;
  out->op = OP_NONE;
  out->inputs[0] = NULL;
  out->inputs[1] = NULL;
  out->aux_rows = 0u;
  out->aux_cols = 0u;
  out->axis = 0u;
  out->aux_lo = (real)0.0;
  out->aux_hi = (real)0.0;
  out->val = mat_new(arena, rows, cols);
  out->grad = (flags & TENSOR_GRAD) ? mat_new(arena, rows, cols) : NULL;

  if (flags & TENSOR_PARAM) {
    register_param(arena, model, out);
  }

  return out;
}

static Tensor *make_node(Arena *arena, Net *model, OpKind op, Tensor *a,
                         Tensor *b, u32 rows, u32 cols, b32 wants_grad) {
  Tensor *out = tensor_create(arena, model, rows, cols,
                              wants_grad ? TENSOR_GRAD : TENSOR_PLAIN);
  out->op = op;
  out->arity = op_arity(op);
  out->inputs[0] = a;
  out->inputs[1] = b;
  out->aux_rows = rows;
  out->aux_cols = cols;
  return out;
}

static Tensor *make_op(Arena *arena, Net *model, OpKind op, Tensor *a,
                       Tensor *b) {
  ShapeRule *rule = shape_rule(op);
  RL_LIB_ASSERT(rule != NULL, "make_op: unknown operation");

  u32 rows = 0;
  u32 cols = 0;
  RL_LIB_ASSERT(rule(a, b, &rows, &cols), "make_op: operand shape mismatch");

  const b32 needs_grad = (a != NULL && (a->flags & TENSOR_GRAD)) ||
                         (b != NULL && (b->flags & TENSOR_GRAD));
  return make_node(arena, model, op, a, b, rows, cols, needs_grad);
}

Tensor *tensor_relu(Arena *arena, Net *model, Tensor *input) {
  return make_op(arena, model, OP_RELU, input, NULL);
}

Tensor *tensor_tanh(Arena *arena, Net *model, Tensor *input) {
  return make_op(arena, model, OP_TANH, input, NULL);
}

Tensor *tensor_sigmoid(Arena *arena, Net *model, Tensor *input) {
  return make_op(arena, model, OP_SIGMOID, input, NULL);
}

Tensor *tensor_exp(Arena *arena, Net *model, Tensor *input) {
  return make_op(arena, model, OP_EXP, input, NULL);
}

Tensor *tensor_log(Arena *arena, Net *model, Tensor *input) {
  return make_op(arena, model, OP_LOG, input, NULL);
}

Tensor *tensor_div(Arena *arena, Net *model, Tensor *a, Tensor *b) {
  return make_op(arena, model, OP_DIV, a, b);
}

Tensor *tensor_softmax(Arena *arena, Net *model, Tensor *input) {
  return make_op(arena, model, OP_SOFTMAX, input, NULL);
}

Tensor *tensor_add(Arena *arena, Net *model, Tensor *a, Tensor *b) {
  return make_op(arena, model, OP_ADD, a, b);
}

Tensor *tensor_sub(Arena *arena, Net *model, Tensor *a, Tensor *b) {
  return make_op(arena, model, OP_SUB, a, b);
}

Tensor *tensor_mul(Arena *arena, Net *model, Tensor *a, Tensor *b) {
  return make_op(arena, model, OP_MUL, a, b);
}

Tensor *tensor_mm(Arena *arena, Net *model, Tensor *a, Tensor *b) {
  return make_op(arena, model, OP_MM, a, b);
}

Tensor *tensor_mse(Arena *arena, Net *model, Tensor *pred, Tensor *target) {
  return make_op(arena, model, OP_MSE, pred, target);
}

Tensor *tensor_reinforce_loss(Arena *arena, Net *model, Tensor *probs,
                              Tensor *rt) {
  return make_op(arena, model, OP_REINFORCE, probs, rt);
}

Tensor *tensor_reshape(Arena *arena, Net *model, Tensor *input, u32 rows,
                       u32 cols) {
  RL_LIB_ASSERT(input != NULL, "tensor_reshape: NULL input");
  RL_LIB_ASSERT(rows > 0u && cols > 0u, "tensor_reshape: empty result shape");
  RL_LIB_ASSERT((u64)rows * (u64)cols ==
                    (u64)input->val->rows * (u64)input->val->cols,
                "tensor_reshape: element count mismatch");
  return make_node(arena, model, OP_RESHAPE, input, NULL, rows, cols,
                   (input->flags & TENSOR_GRAD) != 0u);
}

Tensor *tensor_broadcast(Arena *arena, Net *model, Tensor *input, u32 rows,
                         u32 cols) {
  RL_LIB_ASSERT(input != NULL, "tensor_broadcast: NULL input");
  RL_LIB_ASSERT(rows > 0u && cols > 0u, "tensor_broadcast: empty result shape");
  RL_LIB_ASSERT((input->val->rows == rows || input->val->rows == 1u) &&
                    (input->val->cols == cols || input->val->cols == 1u),
                "tensor_broadcast: input cannot be broadcast to that shape");
  return make_node(arena, model, OP_BROADCAST, input, NULL, rows, cols,
                   (input->flags & TENSOR_GRAD) != 0u);
}

static void reduced_shape(ReduceMode mode, const Tensor *input, u32 *rows,
                          u32 *cols) {
  switch (mode) {
  case REDUCE_ROWS:
    *rows = 1u;
    *cols = input->val->cols;
    break;
  case REDUCE_COLS:
    *rows = input->val->rows;
    *cols = 1u;
    break;
  case REDUCE_ALL:
    *rows = 1u;
    *cols = 1u;
    break;
  }
}

Tensor *tensor_sum(Arena *arena, Net *model, Tensor *input, ReduceMode mode) {
  RL_LIB_ASSERT(input != NULL, "tensor_sum: NULL input");
  RL_LIB_ASSERT(mode == REDUCE_ROWS || mode == REDUCE_COLS ||
                    mode == REDUCE_ALL,
                "tensor_sum: invalid reduction mode");
  u32 rows = 0u;
  u32 cols = 0u;
  reduced_shape(mode, input, &rows, &cols);
  Tensor *out = make_node(arena, model, OP_SUM, input, NULL, rows, cols,
                          (input->flags & TENSOR_GRAD) != 0u);
  out->axis = (u32)mode;
  return out;
}

Tensor *tensor_mean(Arena *arena, Net *model, Tensor *input, ReduceMode mode) {
  RL_LIB_ASSERT(input != NULL, "tensor_mean: NULL input");
  RL_LIB_ASSERT(mode == REDUCE_ROWS || mode == REDUCE_COLS ||
                    mode == REDUCE_ALL,
                "tensor_mean: invalid reduction mode");
  u32 rows = 0u;
  u32 cols = 0u;
  reduced_shape(mode, input, &rows, &cols);
  Tensor *out = make_node(arena, model, OP_MEAN, input, NULL, rows, cols,
                          (input->flags & TENSOR_GRAD) != 0u);
  out->axis = (u32)mode;
  return out;
}

Tensor *tensor_gather(Arena *arena, Net *model, Tensor *values,
                      Tensor *indices) {
  RL_LIB_ASSERT(values != NULL && indices != NULL,
                "tensor_gather: NULL input");
  RL_LIB_ASSERT(indices->val->rows == 1u &&
                    indices->val->cols == values->val->cols,
                "tensor_gather: indices must be 1 x values->cols");
  return make_node(arena, model, OP_GATHER, values, indices, 1u,
                   values->val->cols, (values->flags & TENSOR_GRAD) != 0u);
}

Tensor *tensor_min(Arena *arena, Net *model, Tensor *a, Tensor *b) {
  return make_op(arena, model, OP_MIN, a, b);
}

Tensor *tensor_max(Arena *arena, Net *model, Tensor *a, Tensor *b) {
  return make_op(arena, model, OP_MAX, a, b);
}

Tensor *tensor_clip(Arena *arena, Net *model, Tensor *input, real lo,
                    real hi) {
  RL_LIB_ASSERT(input != NULL, "tensor_clip: NULL input");
  RL_LIB_ASSERT(lo <= hi, "tensor_clip: lo must not exceed hi");
  Tensor *out = make_node(arena, model, OP_CLIP, input, NULL,
                          input->val->rows, input->val->cols,
                          (input->flags & TENSOR_GRAD) != 0u);
  out->aux_lo = lo;
  out->aux_hi = hi;
  return out;
}

// ---------------------------------------------------------------------------
// Topological ordering
// ---------------------------------------------------------------------------

// Iterative depth-first post-order walk from each root. A node is emitted only
// once, after every operand it depends on, which yields a valid left-to-right
// evaluation order. `mark` doubles as the visited set:
//   0 = unseen, 1 = on the stack, 2 = emitted.
typedef struct {
  Tensor *node;
  b32 expanded;
} Frame;

Schedule schedule_from_roots(Arena *arena, Net *model, Tensor **roots,
                             u32 num_roots) {
  Schedule sched = {0};
  if (arena == NULL || model == NULL || roots == NULL || num_roots == 0u ||
      model->n_tensors == 0u) {
    fprintf(stderr,
            "[RL_lib-in-C ERROR] schedule build: invalid arguments "
            "(arena=%p model=%p roots=%p num_roots=%u n_tensors=%u)\n",
            (void *)arena, (void *)model, (void *)roots, num_roots,
            model != NULL ? model->n_tensors : 0u);
    return sched;
  }

  const u32 total = model->n_tensors;
  u8 *mark = ARENA_ARR(arena, u8, total);
  Frame *stack = ARENA_ARR_RAW(arena, Frame, total);
  Tensor **order = ARENA_ARR_RAW(arena, Tensor *, total);
  RL_LIB_ASSERT(mark != NULL && stack != NULL && order != NULL,
             "Arena out of memory in schedule_from_roots");

  u32 emitted = 0u;

  for (u32 r = 0; r < num_roots; r++) {
    Tensor *root = roots[r];
    RL_LIB_ASSERT(root != NULL, "schedule_from_roots: NULL root");
    RL_LIB_ASSERT(root->index < total,
               "schedule_from_roots: root index out of range");
    if (mark[root->index] == 2u) {
      continue; // already reachable from an earlier root
    }

    u32 sp = 0u;
    stack[sp++] = (Frame){root, false};
    mark[root->index] = 1u;

    while (sp > 0u) {
      Frame *top = &stack[sp - 1u];

      if (!top->expanded) {
        top->expanded = true;
        Tensor *cur = top->node;
        for (u32 i = 0; i < cur->arity; i++) {
          Tensor *in = cur->inputs[i];
          if (in != NULL && mark[in->index] == 0u) {
            mark[in->index] = 1u;
            RL_LIB_ASSERT(sp < total, "schedule_from_roots: stack overflow");
            stack[sp++] = (Frame){in, false};
          }
        }
        continue;
      }

      mark[top->node->index] = 2u;
      RL_LIB_ASSERT(emitted < total, "schedule_from_roots: output overflow");
      order[emitted++] = top->node;
      sp--;
    }
  }

  RL_LIB_ASSERT(emitted > 0u, "schedule_from_roots: no reachable nodes");

  sched.ops = ARENA_ARR_RAW(arena, Tensor *, emitted);
  RL_LIB_ASSERT(sched.ops != NULL, "Arena out of memory in schedule_from_roots");
  memcpy(sched.ops, order, sizeof(Tensor *) * emitted);
  sched.len = emitted;

  return sched;
}

Schedule schedule_from(Arena *arena, Net *model, Tensor *out_var) {
  return schedule_from_roots(arena, model, &out_var, 1u);
}

// ---------------------------------------------------------------------------
// Forward interpretation
// ---------------------------------------------------------------------------

static void apply_forward(Tensor *t) {
  switch (t->op) {
  case OP_NONE:
    return;
  case OP_RELU:
    RL_LIB_CHECK(mat_relu(t->val, t->inputs[0]->val));
    return;
  case OP_TANH:
    RL_LIB_CHECK(mat_tanh(t->val, t->inputs[0]->val));
    return;
  case OP_SIGMOID:
    RL_LIB_CHECK(mat_sigmoid(t->val, t->inputs[0]->val));
    return;
  case OP_EXP:
    RL_LIB_CHECK(mat_exp(t->val, t->inputs[0]->val));
    return;
  case OP_LOG:
    RL_LIB_CHECK(mat_log(t->val, t->inputs[0]->val));
    return;
  case OP_DIV:
    RL_LIB_CHECK(mat_div(t->val, t->inputs[0]->val, t->inputs[1]->val));
    return;
  case OP_SOFTMAX:
    RL_LIB_CHECK(mat_softmax(t->val, t->inputs[0]->val));
    return;
  case OP_ADD:
    RL_LIB_CHECK(mat_add(t->val, t->inputs[0]->val, t->inputs[1]->val));
    return;
  case OP_SUB:
    RL_LIB_CHECK(mat_sub(t->val, t->inputs[0]->val, t->inputs[1]->val));
    return;
  case OP_MUL:
    RL_LIB_CHECK(mat_mul(t->val, t->inputs[0]->val, t->inputs[1]->val));
    return;
  case OP_MM:
    RL_LIB_CHECK(mat_mm(t->val, t->inputs[0]->val, t->inputs[1]->val, 1u, 0u, 0u));
    return;
  case OP_MSE:
    RL_LIB_CHECK(mat_mse_loss(t->val, t->inputs[0]->val, t->inputs[1]->val));
    return;
  case OP_REINFORCE:
    RL_LIB_CHECK(mat_reinforce_loss(t->val, t->inputs[0]->val,
                                 t->inputs[1]->val));
    return;
  case OP_RESHAPE:
    RL_LIB_CHECK(mat_reshape(t->val, t->inputs[0]->val));
    return;
  case OP_BROADCAST:
    RL_LIB_CHECK(mat_broadcast(t->val, t->inputs[0]->val));
    return;
  case OP_SUM:
    RL_LIB_CHECK(mat_sum(t->val, t->inputs[0]->val, (ReduceMode)t->axis));
    return;
  case OP_MEAN:
    RL_LIB_CHECK(mat_mean(t->val, t->inputs[0]->val, (ReduceMode)t->axis));
    return;
  case OP_GATHER:
    RL_LIB_CHECK(mat_gather(t->val, t->inputs[0]->val, t->inputs[1]->val));
    return;
  case OP_MIN:
    RL_LIB_CHECK(mat_min(t->val, t->inputs[0]->val, t->inputs[1]->val));
    return;
  case OP_MAX:
    RL_LIB_CHECK(mat_max(t->val, t->inputs[0]->val, t->inputs[1]->val));
    return;
  case OP_CLIP:
    RL_LIB_CHECK(mat_clip(t->val, t->inputs[0]->val, t->aux_lo, t->aux_hi));
    return;
  }
}

void schedule_run(Schedule *graph) {
  if (graph == NULL || graph->len == 0u) {
    return;
  }
  for (u32 i = 0; i < graph->len; i++) {
    apply_forward(graph->ops[i]);
  }
}

// ---------------------------------------------------------------------------
// Reverse interpretation
// ---------------------------------------------------------------------------

static b32 wants_grad(const Tensor *t) {
  return t != NULL && (t->flags & TENSOR_GRAD);
}

static void apply_backward(Tensor *t) {
  Tensor *a = t->inputs[0];
  Tensor *b = t->inputs[1];

  switch (t->op) {
  case OP_NONE:
    return;
  case OP_RELU:
    if (wants_grad(a)) {
      RL_LIB_CHECK(mat_relu_add_grad(a->grad, a->val, t->grad));
    }
    return;
  case OP_TANH:
    if (wants_grad(a)) {
      RL_LIB_CHECK(mat_tanh_add_grad(a->grad, t->val, t->grad));
    }
    return;
  case OP_SIGMOID:
    if (wants_grad(a)) {
      RL_LIB_CHECK(mat_sigmoid_add_grad(a->grad, t->val, t->grad));
    }
    return;
  case OP_EXP:
    if (wants_grad(a)) {
      RL_LIB_CHECK(mat_exp_add_grad(a->grad, t->val, t->grad));
    }
    return;
  case OP_LOG:
    if (wants_grad(a)) {
      RL_LIB_CHECK(mat_log_add_grad(a->grad, a->val, t->grad));
    }
    return;
  case OP_DIV:
    RL_LIB_CHECK(mat_div_add_grad(wants_grad(a) ? a->grad : NULL,
                                  wants_grad(b) ? b->grad : NULL, a->val,
                                  b->val, t->grad));
    return;
  case OP_SOFTMAX:
    if (wants_grad(a)) {
      RL_LIB_CHECK(mat_softmax_add_grad(a->grad, t->val, t->grad));
    }
    return;
  case OP_ADD:
    RL_LIB_CHECK(mat_add_broadcast_grad(wants_grad(a) ? a->grad : NULL,
                                     wants_grad(b) ? b->grad : NULL, t->grad));
    return;
  case OP_SUB:
    RL_LIB_CHECK(mat_sub_broadcast_grad(wants_grad(a) ? a->grad : NULL,
                                     wants_grad(b) ? b->grad : NULL, t->grad));
    return;
  case OP_MUL:
    RL_LIB_CHECK(mat_mul_add_grad(wants_grad(a) ? a->grad : NULL,
                               wants_grad(b) ? b->grad : NULL, a->val, b->val,
                               t->grad));
    return;
  case OP_MM:
    if (wants_grad(a)) {
      RL_LIB_CHECK(mat_mm(a->grad, t->grad, b->val, 0u, 0u, 1u));
    }
    if (wants_grad(b)) {
      RL_LIB_CHECK(mat_mm(b->grad, a->val, t->grad, 0u, 1u, 0u));
    }
    return;
  case OP_MSE:
    if (wants_grad(a)) {
      RL_LIB_CHECK(mat_mse_add_grad(a->grad, a->val, b->val, t->grad));
    }
    return;
  case OP_REINFORCE:
    if (wants_grad(a)) {
      RL_LIB_CHECK(mat_reinforce_add_grad(a->grad, a->val, b->val, t->grad));
    }
    return;
  case OP_RESHAPE:
    if (wants_grad(a)) {
      RL_LIB_CHECK(mat_reshape_add_grad(a->grad, t->grad));
    }
    return;
  case OP_BROADCAST:
    if (wants_grad(a)) {
      RL_LIB_CHECK(mat_broadcast_add_grad(a->grad, t->grad));
    }
    return;
  case OP_SUM:
    if (wants_grad(a)) {
      RL_LIB_CHECK(mat_sum_add_grad(a->grad, t->grad, (ReduceMode)t->axis));
    }
    return;
  case OP_MEAN:
    if (wants_grad(a)) {
      RL_LIB_CHECK(mat_mean_add_grad(a->grad, t->grad, (ReduceMode)t->axis));
    }
    return;
  case OP_GATHER:
    if (wants_grad(a)) {
      RL_LIB_CHECK(mat_gather_add_grad(a->grad, b->val, t->grad));
    }
    return;
  case OP_MIN:
    RL_LIB_CHECK(mat_min_add_grad(wants_grad(a) ? a->grad : NULL,
                                  wants_grad(b) ? b->grad : NULL, a->val,
                                  b->val, t->grad));
    return;
  case OP_MAX:
    RL_LIB_CHECK(mat_max_add_grad(wants_grad(a) ? a->grad : NULL,
                                  wants_grad(b) ? b->grad : NULL, a->val,
                                  b->val, t->grad));
    return;
  case OP_CLIP:
    if (wants_grad(a)) {
      RL_LIB_CHECK(mat_clip_add_grad(a->grad, a->val, t->aux_lo, t->aux_hi,
                                     t->grad));
    }
    return;
  }
}

void schedule_zero_act(Schedule *graph) {
  if (graph == NULL || graph->len == 0u) {
    return;
  }
  for (u32 i = 0; i < graph->len; i++) {
    Tensor *cur = graph->ops[i];
    if ((cur->flags & TENSOR_GRAD) == 0u || (cur->flags & TENSOR_PARAM)) {
      continue; // plain node, or a parameter whose grad accumulates
    }
    mat_zero(cur->grad);
  }
}

void schedule_grad_accum(Schedule *graph) {
  if (graph == NULL || graph->len == 0u) {
    return;
  }
  for (i32 i = (i32)graph->len - 1; i >= 0; i--) {
    Tensor *cur = graph->ops[i];
    if ((cur->flags & TENSOR_GRAD) == 0u) {
      continue;
    }
    apply_backward(cur);
  }
}

void schedule_grad_from(Schedule *graph, Tensor *loss_var, b32 seed_one) {
  if (graph == NULL || graph->len == 0u) {
    return;
  }

  // Wipe activation gradients, preserving parameters (accumulate) and — when
  // the caller already filled it in — the seed node.
  for (u32 i = 0; i < graph->len; i++) {
    Tensor *cur = graph->ops[i];
    if ((cur->flags & TENSOR_GRAD) == 0u || (cur->flags & TENSOR_PARAM)) {
      continue;
    }
    if (!seed_one && cur == loss_var) {
      continue;
    }
    mat_zero(cur->grad);
  }

  Tensor *root = (loss_var != NULL) ? loss_var : graph->ops[graph->len - 1u];

  b32 present = false;
  for (u32 i = 0; i < graph->len; i++) {
    if (graph->ops[i] == root) {
      present = true;
      break;
    }
  }
  RL_LIB_ASSERT(present, "schedule_grad_from: loss_var is not in the graph");

  if (seed_one && root->grad != NULL) {
    mat_fill(root->grad, (real)1.0);
  }

  schedule_grad_accum(graph);
}

void schedule_grad(Schedule *graph) {
  if (graph == NULL || graph->len == 0u) {
    return;
  }
  schedule_grad_from(graph, graph->ops[graph->len - 1u], true);
}

Net *net_new(Arena *arena) {
  return ARENA_OBJ(arena, Net);
}
