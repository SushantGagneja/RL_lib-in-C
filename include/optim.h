#ifndef OPTIM_H
#define OPTIM_H

#include "memory.h"
#include "graph.h"

typedef struct {
  real lr;
  real beta1;
  real beta2;
  real eps;
  u64 step;
  real **m1; // first moment, one buffer per parameter
  real **m2; // second moment, one buffer per parameter
  u32 n_params;
} Adam;

// Zeroes every registered parameter gradient.
void zero_grad(Net *model);

// SGD update: p->val -= (lr * grad_scale) * p->grad
void sgd_step(Net *model, real lr, real grad_scale);

// Allocates Adam state (m1/m2 buffers) from the arena.
// Defaults: beta1 = 0.9, beta2 = 0.999, eps = 1e-8.
Adam *adam_new(Arena *arena, Net *model, real lr);

// One Adam step with bias correction over every parameter.
void adam_step(Adam *adam, Net *model);

// Clips the global gradient norm to max_norm (scaling every parameter gradient
// by max_norm / norm when exceeded). Returns the pre-clipping norm.
real clip_grad_norm(Net *model, real max_norm);

#endif // OPTIM_H
