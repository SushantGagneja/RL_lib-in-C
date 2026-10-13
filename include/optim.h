#ifndef OPTIM_H
#define OPTIM_H

#include "arena.h"
#include "autograd.h"

typedef struct {
  f32 lr;
  f32 beta1;
  f32 beta2;
  f32 eps;
  u64 t;
  f32 **m; // 1st moment vector per parameter
  f32 **v; // 2nd moment vector per parameter
  u32 num_params;
} AdamOptim;

// Clears gradients for all registered parameters in the model state.
void optim_zero_grad(model_state *model);

// Applies an SGD parameter update: p->val -= (lr * scale) * p->grad
void optim_sgd_step(model_state *model, f32 lr, f32 scale);

// Creates an Adam optimizer state with m and v vectors allocated from arena.
// Default hyperparameters: beta1=0.9, beta2=0.999, eps=1e-8.
AdamOptim *optim_adam_create(mem_arena *arena, model_state *model, f32 lr);

// Applies an Adam optimizer step with bias corrections across all model parameters.
void optim_adam_step(AdamOptim *adam, model_state *model);

// Clips the global norm of all parameter gradients:
// If norm > max_norm, scales all parameter gradients by max_norm / norm.
// Returns the computed pre-clipping global norm.
f32 optim_clip_grad_norm(model_state *model, f32 max_norm);

#endif // OPTIM_H
