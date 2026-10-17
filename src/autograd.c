#include "autograd.h"
#include <string.h>

static void var_relu_forward(Var *var);
static void var_relu_backward(Var *var);
static void var_tanh_forward(Var *var);
static void var_tanh_backward(Var *var);
static void var_softmax_forward(Var *var);
static void var_softmax_backward(Var *var);
static void var_add_forward(Var *var);
static void var_add_backward(Var *var);
static void var_sub_forward(Var *var);
static void var_sub_backward(Var *var);
static void var_mul_forward(Var *var);
static void var_mul_backward(Var *var);
static void var_matmul_forward(Var *var);
static void var_matmul_backward(Var *var);
static void var_mse_forward(Var *var);
static void var_mse_backward(Var *var);
static void var_reinforce_forward(Var *var);
static void var_reinforce_backward(Var *var);

static b32 var_shape_same(Var *a, Var *b, u32 *rows, u32 *cols);
static b32 var_shape_broadcast(Var *a, Var *b, u32 *rows, u32 *cols);
static b32 var_shape_matmul(Var *a, Var *b, u32 *rows, u32 *cols);
static b32 var_shape_scalar(Var *a, Var *b, u32 *rows, u32 *cols);

const VarType VAR_TYPE_RELU = {
    .name = "mat_relu",
    .num_inputs = 1,
    .shape = var_shape_same,
    .forward = var_relu_forward,
    .backward = var_relu_backward,
};

const VarType VAR_TYPE_TANH = {
    .name = "tanh",
    .num_inputs = 1,
    .shape = var_shape_same,
    .forward = var_tanh_forward,
    .backward = var_tanh_backward,
};

const VarType VAR_TYPE_SOFTMAX = {
    .name = "mat_softmax",
    .num_inputs = 1,
    .shape = var_shape_same,
    .forward = var_softmax_forward,
    .backward = var_softmax_backward,
};

const VarType VAR_TYPE_ADD = {
    .name = "mat_add",
    .num_inputs = 2,
    .shape = var_shape_broadcast,
    .forward = var_add_forward,
    .backward = var_add_backward,
};

const VarType VAR_TYPE_SUB = {
    .name = "mat_sub",
    .num_inputs = 2,
    .shape = var_shape_broadcast,
    .forward = var_sub_forward,
    .backward = var_sub_backward,
};

const VarType VAR_TYPE_MUL = {
    .name = "mat_mul",
    .num_inputs = 2,
    .shape = var_shape_same,
    .forward = var_mul_forward,
    .backward = var_mul_backward,
};

const VarType VAR_TYPE_MATMUL = {
    .name = "mat_matmul",
    .num_inputs = 2,
    .shape = var_shape_matmul,
    .forward = var_matmul_forward,
    .backward = var_matmul_backward,
};

const VarType VAR_TYPE_MSE = {
    .name = "mse",
    .num_inputs = 2,
    .shape = var_shape_scalar,
    .forward = var_mse_forward,
    .backward = var_mse_backward,
};

const VarType VAR_TYPE_REINFORCE_LOSS = {
    .name = "mat_reinforce_loss",
    .num_inputs = 2,
    .shape = var_shape_same,
    .forward = var_reinforce_forward,
    .backward = var_reinforce_backward,
};

Var *var_create(mem_arena *arena, model_state *model, u32 rows, u32 cols,
                u32 flags) {
  Var *out = PUSH_STRUCT(arena, Var);
  CRL_ASSERT(out != NULL, "Arena out of memory in var_create");

  out->index = model->num_vars++;
  out->flags = flags;
  out->type = NULL;
  out->val = mat_create(arena, rows, cols);

  if (flags & VAR_FLAG_REQUIRES_GRAD) {
    out->grad = mat_create(arena, rows, cols);
  }

  if (flags & VAR_FLAG_PARAMETER) {
    if (model->num_params < MODEL_MAX_PARAMS) {
      model->params[model->num_params++] = out;
    } else {
      CRL_ASSERT(false, "Exceeded MODEL_MAX_PARAMS");
    }
  }

  return out;
}

static b32 var_shape_same(Var *a, Var *b, u32 *rows, u32 *cols) {
  if (a == NULL) {
    return false;
  }

  if (b != NULL &&
      (a->val->rows != b->val->rows || a->val->cols != b->val->cols)) {
    return false;
  }

  *rows = a->val->rows;
  *cols = a->val->cols;

  return true;
}

static b32 var_shape_broadcast(Var *a, Var *b, u32 *rows, u32 *cols) {
  if (a == NULL || b == NULL) {
    return false;
  }

  if (a->val->rows != b->val->rows) {
    return false;
  }

  u32 cols_a = a->val->cols;
  u32 cols_b = b->val->cols;

  if (cols_a == cols_b) {
    *rows = a->val->rows;
    *cols = cols_a;
    return true;
  }

  if (cols_a == 1) {
    *rows = a->val->rows;
    *cols = cols_b;
    return true;
  }

  if (cols_b == 1) {
    *rows = a->val->rows;
    *cols = cols_a;
    return true;
  }

  return false;
}

static b32 var_shape_matmul(Var *a, Var *b, u32 *rows, u32 *cols) {
  if (a == NULL || b == NULL) {
    return false;
  }

  if (a->val->cols != b->val->rows) {
    return false;
  }

  *rows = a->val->rows;
  *cols = b->val->cols;

  return true;
}

static b32 var_shape_scalar(Var *a, Var *b, u32 *rows, u32 *cols) {
  if (a == NULL || b == NULL) {
    return false;
  }

  if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) {
    return false;
  }

  *rows = 1;
  *cols = 1;

  return true;
}

static Var *create_node(mem_arena *arena, model_state *model,
                        const VarType *type, Var *a, Var *b) {
  CRL_ASSERT(type != NULL && type->shape != NULL, "Invalid node type");

  u32 flags = VAR_FLAG_NONE;
  if ((a != NULL && (a->flags & VAR_FLAG_REQUIRES_GRAD)) ||
      (b != NULL && (b->flags & VAR_FLAG_REQUIRES_GRAD))) {
    flags = VAR_FLAG_REQUIRES_GRAD;
  }

  u32 rows = 0;
  u32 cols = 0;
  CRL_ASSERT(type->shape(a, b, &rows, &cols), "Shape error in create_node");

  Var *out = var_create(arena, model, rows, cols, flags);

  out->type = type;
  out->inputs[0] = a;
  if (type->num_inputs > 1) {
    out->inputs[1] = b;
  }

  return out;
}

Var *var_relu(mem_arena *arena, model_state *model, Var *input) {
  return create_node(arena, model, &VAR_TYPE_RELU, input, NULL);
}

Var *var_tanh(mem_arena *arena, model_state *model, Var *input) {
  return create_node(arena, model, &VAR_TYPE_TANH, input, NULL);
}

Var *var_softmax(mem_arena *arena, model_state *model, Var *input) {
  return create_node(arena, model, &VAR_TYPE_SOFTMAX, input, NULL);
}

Var *var_add(mem_arena *arena, model_state *model, Var *a, Var *b) {
  return create_node(arena, model, &VAR_TYPE_ADD, a, b);
}

Var *var_sub(mem_arena *arena, model_state *model, Var *a, Var *b) {
  return create_node(arena, model, &VAR_TYPE_SUB, a, b);
}

Var *var_mul(mem_arena *arena, model_state *model, Var *a, Var *b) {
  return create_node(arena, model, &VAR_TYPE_MUL, a, b);
}

Var *var_matmul(mem_arena *arena, model_state *model, Var *a, Var *b) {
  return create_node(arena, model, &VAR_TYPE_MATMUL, a, b);
}

Var *var_mse(mem_arena *arena, model_state *model, Var *pred, Var *target) {
  return create_node(arena, model, &VAR_TYPE_MSE, pred, target);
}

Var *var_reinforce_loss(mem_arena *arena, model_state *model, Var *probs,
                        Var *rt) {
  return create_node(arena, model, &VAR_TYPE_REINFORCE_LOSS, probs, rt);
}

// Iterative post-order DFS producing a topological order (every node appears
// after all of its inputs). One frame per node is enough because a node is
// pushed at most once, guarded by the `state` array.
typedef struct {
  Var *node;
  u32 next_input;
} DfsFrame;

Graph build_graph_multi(mem_arena *arena, model_state *model, Var **roots,
                        u32 num_roots) {
  Graph graph = {0};
  if (arena == NULL || model == NULL || roots == NULL || num_roots == 0 ||
      model->num_vars == 0) {
    fprintf(stderr,
            "[cRL ERROR] build_graph failed: arena=%p model=%p roots=%p "
            "num_roots=%u model->num_vars=%u\n",
            (void *)arena, (void *)model, (void *)roots, num_roots,
            model ? model->num_vars : 0);
    return graph;
  }

  u32 num_vars = model->num_vars;
  u8 *state = PUSH_ARRAY(arena, u8, num_vars); // 0=unvisited, 1=on stack, 2=done
  DfsFrame *stack = PUSH_ARRAY_UNINIT(arena, DfsFrame, num_vars);
  Var **out = PUSH_ARRAY_UNINIT(arena, Var *, num_vars);
  CRL_ASSERT(state != NULL && stack != NULL && out != NULL,
             "Arena out of memory in build_graph");

  u32 out_size = 0;

  for (u32 r = 0; r < num_roots; r++) {
    Var *root = roots[r];
    CRL_ASSERT(root != NULL, "build_graph: NULL root");
    CRL_ASSERT(root->index < num_vars, "build_graph: root index out of range");

    if (state[root->index] == 2) {
      continue; // already emitted from an earlier root
    }

    u32 sp = 0;
    stack[sp++] = (DfsFrame){root, 0};
    state[root->index] = 1;

    while (sp > 0) {
      DfsFrame *frame = &stack[sp - 1];
      Var *cur = frame->node;
      u32 num_inputs = (cur->type != NULL) ? cur->type->num_inputs : 0;

      if (frame->next_input < num_inputs) {
        Var *input = cur->inputs[frame->next_input++];
        if (input != NULL && state[input->index] == 0) {
          state[input->index] = 1;
          CRL_ASSERT(sp < num_vars, "build_graph: DFS stack overflow");
          stack[sp++] = (DfsFrame){input, 0};
        }
        continue;
      }

      state[cur->index] = 2;
      CRL_ASSERT(out_size < num_vars, "build_graph: output overflow");
      out[out_size++] = cur;
      sp--;
    }
  }

  CRL_ASSERT(out_size > 0, "build_graph: zero reachable nodes");

  graph.vars = PUSH_ARRAY_UNINIT(arena, Var *, out_size);
  CRL_ASSERT(graph.vars != NULL, "Arena out of memory in build_graph");
  memcpy(graph.vars, out, sizeof(Var *) * out_size);
  graph.size = out_size;

  return graph;
}

Graph build_graph(mem_arena *arena, model_state *model, Var *out_var) {
  return build_graph_multi(arena, model, &out_var, 1);
}

void forward_pass(Graph *graph) {
  if (graph == NULL || graph->size == 0) {
    return;
  }
  for (u32 i = 0; i < graph->size; i++) {
    Var *cur = graph->vars[i];

    if (cur->type != NULL && cur->type->forward != NULL) {
      cur->type->forward(cur);
    }
  }
}

void graph_zero_activation_grads(Graph *graph) {
  if (graph == NULL || graph->size == 0) {
    return;
  }
  for (u32 i = 0; i < graph->size; i++) {
    Var *cur = graph->vars[i];
    if ((cur->flags & VAR_FLAG_REQUIRES_GRAD) == 0) {
      continue;
    }
    if (cur->flags & VAR_FLAG_PARAMETER) {
      continue; // parameters accumulate, never cleared here
    }
    mat_clear(cur->grad);
  }
}

void backward_pass_accumulate(Graph *graph) {
  if (graph == NULL || graph->size == 0) {
    return;
  }
  for (i32 i = (i32)graph->size - 1; i >= 0; i--) {
    Var *cur = graph->vars[i];
    if ((cur->flags & VAR_FLAG_REQUIRES_GRAD) == 0) {
      continue;
    }
    if (cur->type != NULL && cur->type->backward != NULL) {
      cur->type->backward(cur);
    }
  }
}

void backward_pass_from(Graph *graph, Var *loss_var, b32 seed_one) {
  if (graph == NULL || graph->size == 0) {
    return;
  }

  // Clear every activation gradient except the seed node, whose gradient the
  // caller may have already populated.
  for (u32 i = 0; i < graph->size; i++) {
    Var *cur = graph->vars[i];
    if ((cur->flags & VAR_FLAG_REQUIRES_GRAD) == 0) {
      continue;
    }
    if (cur->flags & VAR_FLAG_PARAMETER) {
      continue;
    }
    if (!seed_one && cur == loss_var) {
      continue;
    }
    mat_clear(cur->grad);
  }

  Var *root = (loss_var != NULL) ? loss_var : graph->vars[graph->size - 1];

  b32 found = false;
  for (u32 i = 0; i < graph->size; i++) {
    if (graph->vars[i] == root) {
      found = true;
      break;
    }
  }
  CRL_ASSERT(found, "backward_pass_from: specified loss_var not in graph");

  if (seed_one && root->grad != NULL) {
    mat_fill(root->grad, 1.0f);
  }

  backward_pass_accumulate(graph);
}

void backward_pass(Graph *graph) {
  if (graph == NULL || graph->size == 0) {
    return;
  }
  backward_pass_from(graph, graph->vars[graph->size - 1], true);
}

static b32 var_requires_grad(Var *var) {
  return var != NULL && (var->flags & VAR_FLAG_REQUIRES_GRAD);
}

static void var_relu_forward(Var *var) {
  CRL_CHECK(mat_relu(var->val, var->inputs[0]->val));
}

static void var_relu_backward(Var *var) {
  Var *input = var->inputs[0];
  if (var_requires_grad(input)) {
    CRL_CHECK(mat_relu_add_grad(input->grad, input->val, var->grad));
  }
}

static void var_tanh_forward(Var *var) {
  CRL_CHECK(mat_tanh(var->val, var->inputs[0]->val));
}

static void var_tanh_backward(Var *var) {
  Var *input = var->inputs[0];
  if (var_requires_grad(input)) {
    CRL_CHECK(mat_tanh_add_grad(input->grad, var->val, var->grad));
  }
}

static void var_softmax_forward(Var *var) {
  CRL_CHECK(mat_softmax(var->val, var->inputs[0]->val));
}

static void var_softmax_backward(Var *var) {
  Var *input = var->inputs[0];
  if (var_requires_grad(input)) {
    CRL_CHECK(mat_softmax_add_grad(input->grad, var->val, var->grad));
  }
}

static void var_add_forward(Var *var) {
  CRL_CHECK(mat_add(var->val, var->inputs[0]->val, var->inputs[1]->val));
}

static void var_add_backward(Var *var) {
  Var *a = var->inputs[0];
  Var *b = var->inputs[1];
  CRL_CHECK(mat_add_broadcast_grad(var_requires_grad(a) ? a->grad : NULL,
                                   var_requires_grad(b) ? b->grad : NULL,
                                   var->grad));
}

static void var_sub_forward(Var *var) {
  CRL_CHECK(mat_sub(var->val, var->inputs[0]->val, var->inputs[1]->val));
}

static void var_sub_backward(Var *var) {
  Var *a = var->inputs[0];
  Var *b = var->inputs[1];
  CRL_CHECK(mat_sub_broadcast_grad(var_requires_grad(a) ? a->grad : NULL,
                                   var_requires_grad(b) ? b->grad : NULL,
                                   var->grad));
}

static void var_mul_forward(Var *var) {
  CRL_CHECK(mat_mul(var->val, var->inputs[0]->val, var->inputs[1]->val));
}

static void var_mul_backward(Var *var) {
  Var *a = var->inputs[0];
  Var *b = var->inputs[1];
  CRL_CHECK(mat_mul_add_grad(var_requires_grad(a) ? a->grad : NULL,
                             var_requires_grad(b) ? b->grad : NULL, a->val,
                             b->val, var->grad));
}

static void var_matmul_forward(Var *var) {
  CRL_CHECK(
      mat_matmul(var->val, var->inputs[0]->val, var->inputs[1]->val, 1, 0, 0));
}

static void var_matmul_backward(Var *var) {
  Var *a = var->inputs[0];
  Var *b = var->inputs[1];

  if (var_requires_grad(a)) {
    CRL_CHECK(mat_matmul(a->grad, var->grad, b->val, 0, 0, 1));
  }

  if (var_requires_grad(b)) {
    CRL_CHECK(mat_matmul(b->grad, a->val, var->grad, 0, 1, 0));
  }
}

static void var_mse_forward(Var *var) {
  CRL_CHECK(mat_mse_loss(var->val, var->inputs[0]->val, var->inputs[1]->val));
}

static void var_mse_backward(Var *var) {
  Var *pred = var->inputs[0];
  Var *target = var->inputs[1];
  if (var_requires_grad(pred)) {
    CRL_CHECK(mat_mse_add_grad(pred->grad, pred->val, target->val, var->grad));
  }
}

static void var_reinforce_forward(Var *var) {
  CRL_CHECK(mat_reinforce_loss(var->val, var->inputs[0]->val,
                               var->inputs[1]->val));
}

static void var_reinforce_backward(Var *var) {
  Var *probs = var->inputs[0];
  Var *rt = var->inputs[1];

  if (var_requires_grad(probs)) {
    CRL_CHECK(
        mat_reinforce_add_grad(probs->grad, probs->val, rt->val, var->grad));
  }
}

model_state *model_create(mem_arena *arena) {
  model_state *model = PUSH_STRUCT(arena, model_state);
  return model;
}
