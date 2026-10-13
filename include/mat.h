#ifndef MAT_H
#define MAT_H

#include "arena.h"
#include "base.h"
#include "prng.h"

typedef struct {
  u32 rows, cols;
  // Row-major: data[r * cols + c]
  f32 *data;
} matrix;

matrix *mat_create(mem_arena *arena, u32 rows, u32 cols);
void mat_clear(matrix *mat);
void mat_fill(matrix *mat, f32 x);
void mat_fill_rand(matrix *mat, f32 lower, f32 upper);
void mat_fill_rand_r(matrix *mat, prng_state *rng, f32 lower, f32 upper);
void mat_scale(matrix *mat, f32 scale_factor);

// Linear algebra & Elementwise
b32 mat_add(matrix *out, const matrix *a, const matrix *b);
b32 mat_sub(matrix *out, const matrix *a, const matrix *b);
b32 mat_mul(matrix *out, const matrix *a, const matrix *b);
b32 mat_matmul(matrix *out, const matrix *a, const matrix *b, b8 zero_out,
               b8 transpose_a, b8 transpose_b);

// Activations
b32 mat_relu(matrix *out, const matrix *in);
b32 mat_tanh(matrix *out, const matrix *in);
b32 mat_softmax(matrix *out, const matrix *in); // Per-column softmax for batching

// Loss functions
b32 mat_mse_loss(matrix *out, const matrix *pred, const matrix *target);
b32 mat_reinforce_loss(matrix *out, const matrix *probs, const matrix *advantages);
b32 mat_fused_policy_loss(matrix *out, const matrix *logits, const u32 *actions,
                          const f32 *advantages, u32 batch_size);

// Gradients
b32 mat_add_broadcast_grad(matrix *a_grad, matrix *b_grad, const matrix *grad);
b32 mat_sub_broadcast_grad(matrix *a_grad, matrix *b_grad, const matrix *grad);
b32 mat_mul_add_grad(matrix *a_grad, matrix *b_grad, const matrix *a,
                     const matrix *b, const matrix *grad);
b32 mat_relu_add_grad(matrix *out, const matrix *in, const matrix *grad);
b32 mat_tanh_add_grad(matrix *out, const matrix *tanh_out, const matrix *grad);
b32 mat_softmax_add_grad(matrix *out, const matrix *softmax_out,
                         const matrix *grad); // Per-column softmax backward
b32 mat_mse_add_grad(matrix *pred_grad, const matrix *pred, const matrix *target,
                     const matrix *grad);
b32 mat_reinforce_add_grad(matrix *prob_grads, const matrix *probs,
                           const matrix *rt, const matrix *grad);
b32 mat_fused_policy_add_grad(matrix *logits_grad, const matrix *logits,
                              const u32 *actions, const f32 *advantages,
                              u32 batch_size, const matrix *grad);

// Compatibility macros for unprefixed short names
#ifndef CRL_NO_SHORTNAMES
#define create_matrix mat_create
#define clear mat_clear
#define fill mat_fill
#define fill_rand mat_fill_rand
#define fill_rand_r mat_fill_rand_r
#define scale mat_scale
#define add mat_add
#define sub mat_sub
#define mul mat_mul
#define matmul mat_matmul
#define relu mat_relu
#define tanh_act mat_tanh
#define softmax mat_softmax
#define mse_loss mat_mse_loss
#define reinforce_loss mat_reinforce_loss
#define fused_policy_loss mat_fused_policy_loss
#define add_broadcast_grad mat_add_broadcast_grad
#define sub_broadcast_grad mat_sub_broadcast_grad
#define mul_add_grad mat_mul_add_grad
#define relu_add_grad mat_relu_add_grad
#define tanh_add_grad mat_tanh_add_grad
#define softmax_add_grad mat_softmax_add_grad
#define mse_add_grad mat_mse_add_grad
#define reinforce_add_grad mat_reinforce_add_grad
#define fused_policy_add_grad mat_fused_policy_add_grad
#endif

#endif // MAT_H
