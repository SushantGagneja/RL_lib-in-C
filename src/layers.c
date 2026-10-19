#include "layers.h"

#include <math.h>

// A trainable tensor: registered with the model and marked for gradient
// accumulation.
static Tensor *make_param(Arena *arena, Net *model, u32 rows, u32 cols) {
  return tensor_create(arena, model, rows, cols, TENSOR_PARAM | TENSOR_GRAD);
}

Dense dense_new(Arena *arena, Net *model, Rng *rng, u32 in_features,
                u32 out_features, b32 with_bias, real gain) {
  RL_LIB_ASSERT(in_features > 0 && out_features > 0,
             "dense_new: zero-sized layer");

  Dense layer = {
      .weight = NULL,
      .bias = NULL,
      .in_features = in_features,
      .out_features = out_features,
  };

  layer.weight = make_param(arena, model, out_features, in_features);
  const real limit = gain * real_sqrt(6.0f / (real)in_features);
  if (rng != NULL) {
    mat_rand_r(layer.weight->val, rng, -limit, limit);
  } else {
    mat_rand(layer.weight->val, -limit, limit);
  }

  if (with_bias) {
    layer.bias = make_param(arena, model, out_features, 1);
    mat_zero(layer.bias->val);
  }

  return layer;
}

Tensor *dense_apply(Arena *arena, Net *model, const Dense *layer, Tensor *in) {
  Tensor *out = tensor_mm(arena, model, layer->weight, in);
  if (layer->bias != NULL) {
    out = tensor_add(arena, model, out, layer->bias);
  }
  return out;
}

Tensor *mlp_new(Arena *arena, Net *model, Rng *rng, Tensor *in,
                const u32 *layer_dims, u32 num_layers) {
  Tensor *cur = in;
  u32 width = in->val->rows;

  for (u32 i = 0; i < num_layers; i++) {
    Dense layer = dense_new(arena, model, rng, width, layer_dims[i], true, 1.0f);
    cur = dense_apply(arena, model, &layer, cur);
    if (i + 1 < num_layers) {
      cur = tensor_relu(arena, model, cur);
    }
    width = layer_dims[i];
  }
  return cur;
}

// The architecture a builder used, kept so the batched training subgraph can be
// rebuilt for a different minibatch size without re-creating the parameters.
// Private to this file: graph.h only knows the opaque `NetArchSpec`.
struct NetArchSpec {
  NetArchKind kind;
  u32 obs_dim;
  u32 act_dim;
  u32 num_hidden;
  u32 *hidden_dims;       // arena copy of the caller's array
  Dense *trunk;           // num_hidden entries, shared with the rollout graph
  Dense head_a;           // policy head
  Dense head_b;           // value head (NET_ARCH_ACTOR_CRITIC only)
  u32 batch_size;         // current batch dimension of the batched graph
};

// Shared hidden trunk: applies `num_hidden` ReLU dense layers and stores the
// created layers in `layers` so the same trunk can be rebuilt for the batched
// graph.
static Tensor *build_trunk(Arena *arena, Net *model, Rng *rng, Tensor *in,
                           u32 in_dim, const u32 *hidden_dims, u32 num_hidden,
                           Dense *layers) {
  Tensor *cur = in;
  for (u32 i = 0; i < num_hidden; i++) {
    layers[i] = dense_new(arena, model, rng, in_dim, hidden_dims[i], true, 1.0f);
    cur = dense_apply(arena, model, &layers[i], cur);
    cur = tensor_relu(arena, model, cur);
    in_dim = hidden_dims[i];
  }
  return cur;
}

// Copies the caller's architecture description into the arena so it stays valid
// after the builder returns.
static NetArchSpec *record_arch(Arena *arena, Net *model, NetArchKind kind,
                                u32 obs_dim, u32 act_dim,
                                const u32 *hidden_dims, u32 num_hidden,
                                Dense *trunk, Dense head_a, Dense head_b) {
  NetArchSpec *arch = ARENA_OBJ(arena, NetArchSpec);
  arch->kind = kind;
  arch->obs_dim = obs_dim;
  arch->act_dim = act_dim;
  arch->num_hidden = num_hidden;
  arch->trunk = trunk;
  arch->head_a = head_a;
  arch->head_b = head_b;
  arch->batch_size = 0;
  arch->hidden_dims = ARENA_ARR(arena, u32, num_hidden > 0 ? num_hidden : 1u);
  for (u32 i = 0; i < num_hidden; i++) {
    arch->hidden_dims[i] = hidden_dims[i];
  }
  model->arch = arch;
  return arch;
}

// The batched training view: input (obs_dim x B) -> batch_logits / batch_mean
// (act_dim x B) and, for the critic-bearing kinds, batch_value (1 x B), sharing
// every parameter with the rollout graph. Called by the builders and, when the
// minibatch size changes, by net_set_batch_size.
static void build_batch_graph(Arena *arena, Net *model, NetArchSpec *arch,
                              u32 batch_size) {
  model->batch_input =
      tensor_create(arena, model, arch->obs_dim, batch_size, TENSOR_PLAIN);
  Tensor *b = model->batch_input;
  for (u32 i = 0; i < arch->num_hidden; i++) {
    b = dense_apply(arena, model, &arch->trunk[i], b);
    b = tensor_relu(arena, model, b);
  }

  if (arch->kind == NET_ARCH_ACTOR_CRITIC) {
    model->batch_logits = dense_apply(arena, model, &arch->head_a, b);
    model->batch_value = dense_apply(arena, model, &arch->head_b, b);
    Tensor *batch_roots[2] = {model->batch_logits, model->batch_value};
    model->train = schedule_from_roots(arena, model, batch_roots, 2);
  } else if (arch->kind == NET_ARCH_CONTINUOUS) {
    model->batch_mean = dense_apply(arena, model, &arch->head_a, b);
    model->batch_value = dense_apply(arena, model, &arch->head_b, b);
    Tensor *batch_roots[2] = {model->batch_mean, model->batch_value};
    model->train = schedule_from_roots(arena, model, batch_roots, 2);
  } else if (arch->kind == NET_ARCH_Q) {
    model->batch_logits = dense_apply(arena, model, &arch->head_a, b);
    model->target =
        tensor_create(arena, model, arch->act_dim, batch_size, TENSOR_PLAIN);
    model->cost = tensor_mse(arena, model, model->batch_logits, model->target);
    model->train = schedule_from(arena, model, model->batch_logits);
    model->loss_plan = schedule_from(arena, model, model->cost);
  } else {
    model->batch_logits = dense_apply(arena, model, &arch->head_a, b);
    model->train = schedule_from(arena, model, model->batch_logits);
  }
  arch->batch_size = batch_size;
}

b32 net_set_batch_size(Arena *arena, Net *model, u32 batch_size) {
  if (model == NULL || arena == NULL || batch_size == 0 || model->arch == NULL) {
    return false;
  }
  if (model->arch->batch_size == batch_size) {
    return true;
  }
  build_batch_graph(arena, model, model->arch, batch_size);
  return true;
}

u32 net_batch_size(const Net *model) {
  return (model != NULL && model->batch_input != NULL)
             ? model->batch_input->val->cols
             : 0u;
}

b32 net_arch_info(const Net *model, NetArchInfo *out) {
  if (model == NULL || out == NULL || model->arch == NULL) {
    return false;
  }
  out->kind = model->arch->kind;
  out->obs_dim = model->arch->obs_dim;
  out->act_dim = model->arch->act_dim;
  out->num_hidden = model->arch->num_hidden;
  out->hidden_dims = model->arch->hidden_dims;
  out->batch_size = model->arch->batch_size;
  return true;
}

void actor_build(Arena *arena, Net *model, Rng *rng, u32 obs_dim,
                 const u32 *hidden_dims, u32 num_hidden, u32 act_dim,
                 u32 batch_size) {
  model->input = tensor_create(arena, model, obs_dim, 1, TENSOR_PLAIN);

  Dense *trunk = ARENA_ARR(arena, Dense, num_hidden);
  Tensor *cur = build_trunk(arena, model, rng, model->input, obs_dim,
                            hidden_dims, num_hidden, trunk);
  const u32 trunk_dim = (num_hidden > 0) ? hidden_dims[num_hidden - 1] : obs_dim;

  // Policy head with a small gain so the initial policy is near-uniform.
  Dense head = dense_new(arena, model, rng, trunk_dim, act_dim, true, 0.01f);
  Tensor *logits = dense_apply(arena, model, &head, cur);
  model->output = tensor_softmax(arena, model, logits);

  model->advantage = tensor_create(arena, model, act_dim, 1, TENSOR_PLAIN);
  model->cost =
      tensor_reinforce_loss(arena, model, model->output, model->advantage);
  model->fwd = schedule_from(arena, model, model->output);

  NetArchSpec *arch =
      record_arch(arena, model, NET_ARCH_ACTOR, obs_dim, act_dim, hidden_dims,
                  num_hidden, trunk, head, (Dense){0});
  if (batch_size > 0) {
    build_batch_graph(arena, model, arch, batch_size);
  }
}

void actor_critic_build(Arena *arena, Net *model, Rng *rng, u32 obs_dim,
                        const u32 *hidden_dims, u32 num_hidden, u32 act_dim,
                        u32 batch_size) {
  model->input = tensor_create(arena, model, obs_dim, 1, TENSOR_PLAIN);

  Dense *trunk = ARENA_ARR(arena, Dense, num_hidden);
  Tensor *cur = build_trunk(arena, model, rng, model->input, obs_dim,
                            hidden_dims, num_hidden, trunk);
  const u32 trunk_dim = (num_hidden > 0) ? hidden_dims[num_hidden - 1] : obs_dim;

  // Actor head (small gain) and value head (unit gain).
  Dense actor = dense_new(arena, model, rng, trunk_dim, act_dim, true, 0.01f);
  Dense critic = dense_new(arena, model, rng, trunk_dim, 1, true, 1.0f);

  Tensor *logits = dense_apply(arena, model, &actor, cur);
  model->output = tensor_softmax(arena, model, logits);
  model->value = dense_apply(arena, model, &critic, cur);

  // The forward graph must reach both the softmax output and the value head.
  // Rooting it at a synthetic `logits + value` node both fails to build (shape
  // mismatch) and never runs the softmax.
  Tensor *forward_roots[2] = {model->output, model->value};
  model->fwd = schedule_from_roots(arena, model, forward_roots, 2);

  NetArchSpec *arch = record_arch(arena, model, NET_ARCH_ACTOR_CRITIC, obs_dim,
                                  act_dim, hidden_dims, num_hidden, trunk,
                                  actor, critic);
  if (batch_size > 0) {
    build_batch_graph(arena, model, arch, batch_size);
  }
}

void continuous_actor_critic_build(Arena *arena, Net *model, Rng *rng,
                                   u32 obs_dim, const u32 *hidden_dims,
                                   u32 num_hidden, u32 act_dim, u32 batch_size,
                                   real log_std_init) {
  RL_LIB_ASSERT(act_dim > 0, "continuous_actor_critic_build: act_dim == 0");

  model->input = tensor_create(arena, model, obs_dim, 1, TENSOR_PLAIN);

  Dense *trunk = ARENA_ARR(arena, Dense, num_hidden);
  Tensor *cur = build_trunk(arena, model, rng, model->input, obs_dim,
                            hidden_dims, num_hidden, trunk);
  const u32 trunk_dim = (num_hidden > 0) ? hidden_dims[num_hidden - 1] : obs_dim;

  Dense mean_head = dense_new(arena, model, rng, trunk_dim, act_dim, true, 0.01f);
  Dense critic = dense_new(arena, model, rng, trunk_dim, 1, true, 1.0f);

  model->mean = dense_apply(arena, model, &mean_head, cur);
  model->value = dense_apply(arena, model, &critic, cur);

  // One learnable log standard deviation per action dimension. This is a
  // parameter, so the optimiser, gradient clipping and checkpointing all cover
  // it automatically, but it is deliberately outside every schedule: no op
  // consumes it, and the Gaussian policy loss accumulates its gradient directly.
  model->log_std =
      tensor_create(arena, model, act_dim, 1, TENSOR_PARAM | TENSOR_GRAD);
  mat_fill(model->log_std->val, log_std_init);

  Tensor *forward_roots[2] = {model->mean, model->value};
  model->fwd = schedule_from_roots(arena, model, forward_roots, 2);

  NetArchSpec *arch = record_arch(arena, model, NET_ARCH_CONTINUOUS, obs_dim,
                                  act_dim, hidden_dims, num_hidden, trunk,
                                  mean_head, critic);
  if (batch_size > 0) {
    build_batch_graph(arena, model, arch, batch_size);
  }
}

void q_net_build(Arena *arena, Net *model, Rng *rng, u32 obs_dim,
                 const u32 *hidden_dims, u32 num_hidden, u32 act_dim,
                 u32 batch_size) {
  RL_LIB_ASSERT(act_dim > 0, "q_net_build: act_dim == 0");

  model->input = tensor_create(arena, model, obs_dim, 1, TENSOR_PLAIN);

  Dense *trunk = ARENA_ARR(arena, Dense, num_hidden);
  Tensor *cur = build_trunk(arena, model, rng, model->input, obs_dim,
                            hidden_dims, num_hidden, trunk);
  const u32 trunk_dim = (num_hidden > 0) ? hidden_dims[num_hidden - 1] : obs_dim;

  // The Q head is linear: one value per action, no softmax, unit gain like the
  // critic head. The regression target tensor and the MSE cost are created with
  // the batched graph (build_batch_graph), because every training step regresses
  // a whole minibatch at once.
  Dense head = dense_new(arena, model, rng, trunk_dim, act_dim, true, 1.0f);
  model->output = dense_apply(arena, model, &head, cur);

  model->fwd = schedule_from(arena, model, model->output);

  NetArchSpec *arch = record_arch(arena, model, NET_ARCH_Q, obs_dim, act_dim,
                                  hidden_dims, num_hidden, trunk, head,
                                  (Dense){0});
  if (batch_size > 0) {
    build_batch_graph(arena, model, arch, batch_size);
  }
}
