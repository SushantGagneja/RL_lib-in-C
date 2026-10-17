#include "optim.h"
#include <math.h>
#include <string.h>

void optim_zero_grad(model_state *model) {
  if (model == NULL) {
    return;
  }
  for (u32 i = 0; i < model->num_params; i++) {
    Var *p = model->params[i];
    if (p != NULL && p->grad != NULL) {
      mat_clear(p->grad);
    }
  }
}

void optim_sgd_step(model_state *model, real lr, real grad_scale) {
  if (model == NULL) {
    return;
  }
  real step_size = lr * grad_scale;
  for (u32 i = 0; i < model->num_params; i++) {
    Var *p = model->params[i];
    if (p != NULL && p->val != NULL && p->grad != NULL) {
      u64 size = (u64)p->val->rows * p->val->cols;
      for (u64 j = 0; j < size; j++) {
        p->val->data[j] -= step_size * p->grad->data[j];
      }
    }
  }
}

AdamOptim *optim_adam_create(mem_arena *arena, model_state *model, real lr) {
  CRL_ASSERT(model != NULL, "Model cannot be NULL in optim_adam_create");
  AdamOptim *adam = PUSH_STRUCT(arena, AdamOptim);
  adam->lr = lr;
  adam->beta1 = 0.9f;
  adam->beta2 = 0.999f;
  adam->eps = 1e-8f;
  adam->t = 0;
  adam->num_params = model->num_params;

  adam->m = PUSH_ARRAY(arena, real *, model->num_params);
  adam->v = PUSH_ARRAY(arena, real *, model->num_params);

  for (u32 k = 0; k < model->num_params; k++) {
    Var *p = model->params[k];
    u64 n = (u64)p->val->rows * p->val->cols;
    adam->m[k] = PUSH_ARRAY(arena, real, n);
    adam->v[k] = PUSH_ARRAY(arena, real, n);
    memset(adam->m[k], 0, sizeof(real) * n);
    memset(adam->v[k], 0, sizeof(real) * n);
  }

  return adam;
}

void optim_adam_step(AdamOptim *adam, model_state *model) {
  if (adam == NULL || model == NULL) {
    return;
  }

  adam->t++;
  real c1 = 1.0f - real_pow(adam->beta1, (real)adam->t);
  real c2 = 1.0f - real_pow(adam->beta2, (real)adam->t);

  for (u32 k = 0; k < model->num_params; k++) {
    Var *p = model->params[k];
    if (p == NULL || p->val == NULL || p->grad == NULL) {
      continue;
    }

    u64 n = (u64)p->val->rows * p->val->cols;
    real *m = adam->m[k];
    real *v = adam->v[k];

    for (u64 i = 0; i < n; i++) {
      real g = p->grad->data[i];
      m[i] = adam->beta1 * m[i] + (1.0f - adam->beta1) * g;
      v[i] = adam->beta2 * v[i] + (1.0f - adam->beta2) * g * g;

      real m_hat = m[i] / c1;
      real v_hat = v[i] / c2;

      p->val->data[i] -= adam->lr * m_hat / (real_sqrt(v_hat) + adam->eps);
    }
  }
}

real optim_clip_grad_norm(model_state *model, real max_norm) {
  if (model == NULL || max_norm <= 0.0f) {
    return 0.0f;
  }

  real sum_sq = 0.0f;
  for (u32 k = 0; k < model->num_params; k++) {
    Var *p = model->params[k];
    if (p != NULL && p->grad != NULL) {
      u64 n = (u64)p->val->rows * p->val->cols;
      for (u64 i = 0; i < n; i++) {
        real g = p->grad->data[i];
        sum_sq += g * g;
      }
    }
  }

  real norm = real_sqrt(sum_sq);
  CRL_ASSERT(isfinite(norm),
             "Non-finite gradient norm (NaN/Inf in parameter gradients)");
  if (norm > max_norm && norm > 1e-8f) {
    real scale_factor = max_norm / norm;
    for (u32 k = 0; k < model->num_params; k++) {
      Var *p = model->params[k];
      if (p != NULL && p->grad != NULL) {
        u64 n = (u64)p->val->rows * p->val->cols;
        for (u64 i = 0; i < n; i++) {
          p->grad->data[i] *= scale_factor;
        }
      }
    }
  }

  return norm;
}
