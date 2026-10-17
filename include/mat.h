#ifndef MAT_H
#define MAT_H

#include "arena.h"
#include "base.h"
#include "prng.h"

typedef struct {
  u32 rows, cols;
  // Row-major: data[r * cols + c]
  real *data;
} matrix;

matrix *mat_create(mem_arena *arena, u32 rows, u32 cols);
void mat_clear(matrix *mat);
void mat_fill(matrix *mat, real x);
void mat_fill_rand(matrix *mat, real lower, real upper);
void mat_fill_rand_r(matrix *mat, prng_state *rng, real lower, real upper);
void mat_scale(matrix *mat, real scale_factor);

// Linear algebra & Elementwise
b32 mat_add(matrix *out, const matrix *a, const matrix *b);
b32 mat_sub(matrix *out, const matrix *a, const matrix *b);
b32 mat_mul(matrix *out, const matrix *a, const matrix *b);
b32 mat_matmul(matrix *out, const matrix *a, const matrix *b, b8 zero_out,
               b8 transpose_a, b8 transpose_b);

// Activations
b32 mat_relu(matrix *out, const matrix *in);
b32 mat_tanh(matrix *out, const matrix *in);
b32 mat_softmax(matrix *out, const matrix *in); // Per-column mat_softmax for batching

// Loss functions
b32 mat_mse_loss(matrix *out, const matrix *pred, const matrix *target);
b32 mat_reinforce_loss(matrix *out, const matrix *probs, const matrix *advantages);
b32 mat_fused_policy_loss(matrix *out, const matrix *logits, const u32 *actions,
                          const real *advantages, u32 batch_size);
// Mean per-column policy entropy H = -sum p log p (natural log), written to a
// 1x1 matrix. Used as an exploration bonus (minimise -coef * H).
b32 mat_entropy_loss(matrix *out, const matrix *logits, u32 batch_size);

// Gradients
b32 mat_add_broadcast_grad(matrix *a_grad, matrix *b_grad, const matrix *grad);
b32 mat_sub_broadcast_grad(matrix *a_grad, matrix *b_grad, const matrix *grad);
b32 mat_mul_add_grad(matrix *a_grad, matrix *b_grad, const matrix *a,
                     const matrix *b, const matrix *grad);
b32 mat_relu_add_grad(matrix *out, const matrix *in, const matrix *grad);
b32 mat_tanh_add_grad(matrix *out, const matrix *tanh_out, const matrix *grad);
b32 mat_softmax_add_grad(matrix *out, const matrix *softmax_out,
                         const matrix *grad); // Per-column mat_softmax backward
b32 mat_mse_add_grad(matrix *pred_grad, const matrix *pred, const matrix *target,
                     const matrix *grad);
b32 mat_reinforce_add_grad(matrix *prob_grads, const matrix *probs,
                           const matrix *rt, const matrix *grad);
b32 mat_fused_policy_add_grad(matrix *logits_grad, const matrix *logits,
                              const u32 *actions, const real *advantages,
                              u32 batch_size, const matrix *grad);
// Gradient of the mean entropy with respect to logits:
//   d(mean H)/dz_{i,b} = (1/B) * p_{i,b} * (-log p_{i,b} - H_b)
// scaled by grad->data[0], accumulated into logits_grad.
b32 mat_entropy_add_grad(matrix *logits_grad, const matrix *logits,
                         u32 batch_size, const matrix *grad);

#endif // MAT_H
