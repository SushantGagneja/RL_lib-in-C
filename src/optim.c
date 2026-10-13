#include "optim.h"

void optim_zero_grad(model_state *model) {
  if (model == NULL) {
    return;
  }
  for (u32 i = 0; i < model->num_params; i++) {
    Var *p = model->params[i];
    if (p != NULL && p->grad != NULL) {
      clear(p->grad);
    }
  }
}

void optim_sgd_step(model_state *model, f32 lr, f32 scale) {
  if (model == NULL) {
    return;
  }
  f32 step_size = lr * scale;
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
