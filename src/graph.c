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

Tensor *tensor_create(Arena *arena, Net *model, u32 rows, u32 cols,
                      u32 flags) {
  Tensor *out = ARENA_OBJ(arena, Tensor);
  CRL_ASSERT(out != NULL, "Arena out of memory in tensor_create");

  out->index = model->n_tensors++;
  out->flags = flags;
  out->arity = 0u;
  out->op = OP_NONE;
  out->inputs[0] = NULL;
  out->inputs[1] = NULL;
  out->val = mat_new(arena, rows, cols);
  out->grad = (flags & TENSOR_GRAD) ? mat_new(arena, rows, cols) : NULL;

  if (flags & TENSOR_PARAM) {
    CRL_ASSERT(model->n_params < MAX_PARAMS, "Exceeded MAX_PARAMS");
    model->params[model->n_params++] = out;
  }

  return out;
}

static Tensor *make_op(Arena *arena, Net *model, OpKind op, Tensor *a,
                       Tensor *b) {
  ShapeRule *rule = shape_rule(op);
  CRL_ASSERT(rule != NULL, "make_op: unknown operation");

  u32 rows = 0;
  u32 cols = 0;
  CRL_ASSERT(rule(a, b, &rows, &cols), "make_op: operand shape mismatch");

  const b32 needs_grad = (a != NULL && (a->flags & TENSOR_GRAD)) ||
                         (b != NULL && (b->flags & TENSOR_GRAD));
  Tensor *out = tensor_create(arena, model, rows, cols,
                              needs_grad ? TENSOR_GRAD : TENSOR_PLAIN);
  out->op = op;
  out->arity = op_arity(op);
  out->inputs[0] = a;
  out->inputs[1] = b;
  return out;
}

Tensor *tensor_relu(Arena *arena, Net *model, Tensor *input) {
  return make_op(arena, model, OP_RELU, input, NULL);
}

Tensor *tensor_tanh(Arena *arena, Net *model, Tensor *input) {
  return make_op(arena, model, OP_TANH, input, NULL);
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
            "[cRL ERROR] schedule build: invalid arguments "
            "(arena=%p model=%p roots=%p num_roots=%u n_tensors=%u)\n",
            (void *)arena, (void *)model, (void *)roots, num_roots,
            model != NULL ? model->n_tensors : 0u);
    return sched;
  }

  const u32 total = model->n_tensors;
  u8 *mark = ARENA_ARR(arena, u8, total);
  Frame *stack = ARENA_ARR_RAW(arena, Frame, total);
  Tensor **order = ARENA_ARR_RAW(arena, Tensor *, total);
  CRL_ASSERT(mark != NULL && stack != NULL && order != NULL,
             "Arena out of memory in schedule_from_roots");

  u32 emitted = 0u;

  for (u32 r = 0; r < num_roots; r++) {
    Tensor *root = roots[r];
    CRL_ASSERT(root != NULL, "schedule_from_roots: NULL root");
    CRL_ASSERT(root->index < total,
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
            CRL_ASSERT(sp < total, "schedule_from_roots: stack overflow");
            stack[sp++] = (Frame){in, false};
          }
        }
        continue;
      }

      mark[top->node->index] = 2u;
      CRL_ASSERT(emitted < total, "schedule_from_roots: output overflow");
      order[emitted++] = top->node;
      sp--;
    }
  }

  CRL_ASSERT(emitted > 0u, "schedule_from_roots: no reachable nodes");

  sched.ops = ARENA_ARR_RAW(arena, Tensor *, emitted);
  CRL_ASSERT(sched.ops != NULL, "Arena out of memory in schedule_from_roots");
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
    CRL_CHECK(mat_relu(t->val, t->inputs[0]->val));
    return;
  case OP_TANH:
    CRL_CHECK(mat_tanh(t->val, t->inputs[0]->val));
    return;
  case OP_SOFTMAX:
    CRL_CHECK(mat_softmax(t->val, t->inputs[0]->val));
    return;
  case OP_ADD:
    CRL_CHECK(mat_add(t->val, t->inputs[0]->val, t->inputs[1]->val));
    return;
  case OP_SUB:
    CRL_CHECK(mat_sub(t->val, t->inputs[0]->val, t->inputs[1]->val));
    return;
  case OP_MUL:
    CRL_CHECK(mat_mul(t->val, t->inputs[0]->val, t->inputs[1]->val));
    return;
  case OP_MM:
    CRL_CHECK(mat_mm(t->val, t->inputs[0]->val, t->inputs[1]->val, 1u, 0u, 0u));
    return;
  case OP_MSE:
    CRL_CHECK(mat_mse_loss(t->val, t->inputs[0]->val, t->inputs[1]->val));
    return;
  case OP_REINFORCE:
    CRL_CHECK(mat_reinforce_loss(t->val, t->inputs[0]->val,
                                 t->inputs[1]->val));
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
      CRL_CHECK(mat_relu_add_grad(a->grad, a->val, t->grad));
    }
    return;
  case OP_TANH:
    if (wants_grad(a)) {
      CRL_CHECK(mat_tanh_add_grad(a->grad, t->val, t->grad));
    }
    return;
  case OP_SOFTMAX:
    if (wants_grad(a)) {
      CRL_CHECK(mat_softmax_add_grad(a->grad, t->val, t->grad));
    }
    return;
  case OP_ADD:
    CRL_CHECK(mat_add_broadcast_grad(wants_grad(a) ? a->grad : NULL,
                                     wants_grad(b) ? b->grad : NULL, t->grad));
    return;
  case OP_SUB:
    CRL_CHECK(mat_sub_broadcast_grad(wants_grad(a) ? a->grad : NULL,
                                     wants_grad(b) ? b->grad : NULL, t->grad));
    return;
  case OP_MUL:
    CRL_CHECK(mat_mul_add_grad(wants_grad(a) ? a->grad : NULL,
                               wants_grad(b) ? b->grad : NULL, a->val, b->val,
                               t->grad));
    return;
  case OP_MM:
    if (wants_grad(a)) {
      CRL_CHECK(mat_mm(a->grad, t->grad, b->val, 0u, 0u, 1u));
    }
    if (wants_grad(b)) {
      CRL_CHECK(mat_mm(b->grad, a->val, t->grad, 0u, 1u, 0u));
    }
    return;
  case OP_MSE:
    if (wants_grad(a)) {
      CRL_CHECK(mat_mse_add_grad(a->grad, a->val, b->val, t->grad));
    }
    return;
  case OP_REINFORCE:
    if (wants_grad(a)) {
      CRL_CHECK(mat_reinforce_add_grad(a->grad, a->val, b->val, t->grad));
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
  CRL_ASSERT(present, "schedule_grad_from: loss_var is not in the graph");

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
