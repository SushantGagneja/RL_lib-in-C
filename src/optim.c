#include "optim.h"

#include <math.h>
#include <string.h>

static u64 param_count(const Tensor *p) {
  return (u64)p->val->rows * (u64)p->val->cols;
}

void zero_grad(Net *model) {
  if (model == NULL) {
    return;
  }
  for (u32 i = 0; i < model->n_params; i++) {
    Tensor *p = model->params[i];
    if (p != NULL && p->grad != NULL) {
      mat_zero(p->grad);
    }
  }
}

void sgd_step(Net *model, real lr, real grad_scale) {
  if (model == NULL) {
    return;
  }
  const real step = lr * grad_scale;
  for (u32 i = 0; i < model->n_params; i++) {
    Tensor *p = model->params[i];
    if (p == NULL || p->val == NULL || p->grad == NULL) {
      continue;
    }
    const u64 n = param_count(p);
    for (u64 j = 0; j < n; j++) {
      p->val->data[j] -= step * p->grad->data[j];
    }
  }
}

Adam *adam_new(Arena *arena, Net *model, real lr) {
  RL_LIB_ASSERT(model != NULL, "adam_new: NULL model");

  Adam *adam = ARENA_OBJ(arena, Adam);
  adam->lr = lr;
  adam->beta1 = 0.9f;
  adam->beta2 = 0.999f;
  adam->eps = 1e-8f;
  adam->step = 0;
  adam->n_params = model->n_params;

  adam->m1 = ARENA_ARR(arena, real *, model->n_params);
  adam->m2 = ARENA_ARR(arena, real *, model->n_params);

  for (u32 k = 0; k < model->n_params; k++) {
    Tensor *p = model->params[k];
    const u64 n = param_count(p);
    adam->m1[k] = ARENA_ARR(arena, real, n);
    adam->m2[k] = ARENA_ARR(arena, real, n);
    memset(adam->m1[k], 0, sizeof(real) * n);
    memset(adam->m2[k], 0, sizeof(real) * n);
  }
  return adam;
}

void adam_step(Adam *adam, Net *model) {
  if (adam == NULL || model == NULL) {
    return;
  }

  adam->step++;
  const real c1 = 1.0f - real_pow(adam->beta1, (real)adam->step);
  const real c2 = 1.0f - real_pow(adam->beta2, (real)adam->step);

  for (u32 k = 0; k < model->n_params; k++) {
    Tensor *p = model->params[k];
    if (p == NULL || p->val == NULL || p->grad == NULL) {
      continue;
    }

    const u64 n = param_count(p);
    real *m1 = adam->m1[k];
    real *m2 = adam->m2[k];

    for (u64 i = 0; i < n; i++) {
      const real g = p->grad->data[i];
      m1[i] = adam->beta1 * m1[i] + (1.0f - adam->beta1) * g;
      m2[i] = adam->beta2 * m2[i] + (1.0f - adam->beta2) * g * g;

      const real m_hat = m1[i] / c1;
      const real v_hat = m2[i] / c2;
      p->val->data[i] -= adam->lr * m_hat / (real_sqrt(v_hat) + adam->eps);
    }
  }
}

real clip_grad_norm(Net *model, real max_norm) {
  if (model == NULL || max_norm <= 0.0f) {
    return 0.0f;
  }

  real sum_sq = 0.0f;
  for (u32 k = 0; k < model->n_params; k++) {
    Tensor *p = model->params[k];
    if (p == NULL || p->grad == NULL) {
      continue;
    }
    const u64 n = param_count(p);
    for (u64 i = 0; i < n; i++) {
      const real g = p->grad->data[i];
      sum_sq += g * g;
    }
  }

  const real norm = real_sqrt(sum_sq);
  RL_LIB_ASSERT(isfinite(norm),
             "clip_grad_norm: non-finite gradient norm (NaN/Inf in grads)");

  if (norm > max_norm && norm > 1e-8f) {
    const real scale = max_norm / norm;
    for (u32 k = 0; k < model->n_params; k++) {
      Tensor *p = model->params[k];
      if (p == NULL || p->grad == NULL) {
        continue;
      }
      const u64 n = param_count(p);
      for (u64 i = 0; i < n; i++) {
        p->grad->data[i] *= scale;
      }
    }
  }

  return norm;
}
