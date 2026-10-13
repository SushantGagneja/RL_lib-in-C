#ifndef OPTIM_H
#define OPTIM_H

#include "autograd.h"

// Clears gradients for all registered parameters in the model state.
void optim_zero_grad(model_state *model);

// Applies an SGD parameter update: p->val -= (lr * scale) * p->grad
// scale is typically 1.0f / sample_count.
void optim_sgd_step(model_state *model, f32 lr, f32 scale);

#endif // OPTIM_H
