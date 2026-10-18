#ifndef LINALG_H
#define LINALG_H

#include "memory.h"
#include "base.h"
#include "random.h"

typedef struct {
  u32 rows, cols;
  // Row-major: data[r * cols + c]
  real *data;
} Mat;

Mat *mat_new(Arena *arena, u32 rows, u32 cols);
void mat_zero(Mat *mat);
void mat_fill(Mat *mat, real x);
void mat_rand(Mat *mat, real lower, real upper);
void mat_rand_r(Mat *mat, Rng *rng, real lower, real upper);
void mat_scale(Mat *mat, real scale_factor);

// Matrix algebra and elementwise ops
b32 mat_add(Mat *out, const Mat *a, const Mat *b);
b32 mat_sub(Mat *out, const Mat *a, const Mat *b);
b32 mat_mul(Mat *out, const Mat *a, const Mat *b);
b32 mat_mm(Mat *out, const Mat *a, const Mat *b, b8 zero_out,
               b8 transpose_a, b8 transpose_b);

// Portable (always scalar) matrix multiply with the same semantics as mat_mm.
// mat_mm() dispatches to the SIMD kernels when a backend is available and falls
// back to this; the tests use it as the reference the vector path must match.
b32 mat_mm_scalar(Mat *out, const Mat *a, const Mat *b, b8 zero_out,
                  b8 transpose_a, b8 transpose_b);

// Activations
b32 mat_relu(Mat *out, const Mat *in);
b32 mat_tanh(Mat *out, const Mat *in);
b32 mat_softmax(Mat *out, const Mat *in); // Per-column mat_softmax for batching

// Loss functions
b32 mat_mse_loss(Mat *out, const Mat *pred, const Mat *target);
b32 mat_reinforce_loss(Mat *out, const Mat *probs, const Mat *advantages);
b32 mat_fused_policy_loss(Mat *out, const Mat *logits, const u32 *actions,
                          const real *advantages, u32 batch_size);
// Mean per-column policy entropy H = -sum p log p (natural log), written to a
// 1x1 Mat. Used as an exploration bonus (minimise -coef * H).
b32 mat_entropy_loss(Mat *out, const Mat *logits, u32 batch_size);

// Gradients
b32 mat_add_broadcast_grad(Mat *a_grad, Mat *b_grad, const Mat *grad);
b32 mat_sub_broadcast_grad(Mat *a_grad, Mat *b_grad, const Mat *grad);
b32 mat_mul_add_grad(Mat *a_grad, Mat *b_grad, const Mat *a,
                     const Mat *b, const Mat *grad);
b32 mat_relu_add_grad(Mat *out, const Mat *in, const Mat *grad);
b32 mat_tanh_add_grad(Mat *out, const Mat *tanh_out, const Mat *grad);
b32 mat_softmax_add_grad(Mat *out, const Mat *softmax_out,
                         const Mat *grad); // Per-column mat_softmax backward
b32 mat_mse_add_grad(Mat *pred_grad, const Mat *pred, const Mat *target,
                     const Mat *grad);
b32 mat_reinforce_add_grad(Mat *prob_grads, const Mat *probs,
                           const Mat *rt, const Mat *grad);
b32 mat_fused_policy_add_grad(Mat *logits_grad, const Mat *logits,
                              const u32 *actions, const real *advantages,
                              u32 batch_size, const Mat *grad);
// Gradient of the mean entropy with respect to logits:
//   d(mean H)/dz_{i,b} = (1/B) * p_{i,b} * (-log p_{i,b} - H_b)
// scaled by grad->data[0], accumulated into logits_grad.
b32 mat_entropy_add_grad(Mat *logits_grad, const Mat *logits,
                         u32 batch_size, const Mat *grad);

// PPO clipped surrogate policy loss over columns (act_dim rows x batch cols):
//   L = -mean_b min( r_b * A_b, clamp(r_b, 1-eps, 1+eps) * A_b )
//   r_b = exp(log pi(a_b|s_b) - old_logp_b)
// where the log-probabilities are recovered from the pre-softmax logits.
// `old_logp[b]` are the behaviour-policy log-probabilities captured at rollout
// time. The gradient is accumulated into logits_grad and scaled by
// grad->data[0] (the loss front-end is already a mean over the batch).
b32 mat_ppo_policy_loss(Mat *out, const Mat *logits, const u32 *actions,
                        const real *advantages, const real *old_logp,
                        real clip_eps, u32 batch_size);
b32 mat_ppo_policy_add_grad(Mat *logits_grad, const Mat *logits,
                            const u32 *actions, const real *advantages,
                            const real *old_logp, real clip_eps,
                            u32 batch_size, const Mat *grad);

#endif // LINALG_H
