#ifndef MAT_H
#define MAT_H

#include "arena.h"
#include "base.h"

typedef struct {
  u32 rows, cols;
  // Row-major: data[r * cols + c]
  f32 *data;
} matrix;

matrix *create_matrix(mem_arena *arena, u32 rows, u32 cols);
void clear(matrix *mat);
void fill(matrix *mat, f32 x);
void fill_rand(matrix *mat, f32 lower, f32 upper);
void scale(matrix *mat, f32 scale);

// Linear algebra & Elementwise
b32 add(matrix *out, const matrix *a, const matrix *b);
b32 sub(matrix *out, const matrix *a, const matrix *b);
b32 mul(matrix *out, const matrix *a, const matrix *b);
b32 matmul(matrix *out, const matrix *a, const matrix *b, b8 zero_out,
           b8 transpose_a, b8 transpose_b);

// Activations
b32 relu(matrix *out, const matrix *in);
b32 tanh_act(matrix *out, const matrix *in);
b32 softmax(matrix *out, const matrix *in); // Per-column softmax for batching

// Loss functions
b32 mse_loss(matrix *out, const matrix *pred, const matrix *target);
b32 reinforce_loss(matrix *out, const matrix *probs, const matrix *advantages);

// Gradients
b32 add_broadcast_grad(matrix *a_grad, matrix *b_grad, const matrix *grad);
b32 sub_broadcast_grad(matrix *a_grad, matrix *b_grad, const matrix *grad);
b32 mul_add_grad(matrix *a_grad, matrix *b_grad, const matrix *a,
                 const matrix *b, const matrix *grad);
b32 relu_add_grad(matrix *out, const matrix *in, const matrix *grad);
b32 tanh_add_grad(matrix *out, const matrix *tanh_out, const matrix *grad);
b32 softmax_add_grad(matrix *out, const matrix *softmax_out,
                     const matrix *grad); // Per-column softmax backward
b32 mse_add_grad(matrix *pred_grad, const matrix *pred, const matrix *target,
                 const matrix *grad);
b32 reinforce_add_grad(matrix *prob_grads, const matrix *probs,
                       const matrix *rt, const matrix *grad);

#endif // MAT_H
