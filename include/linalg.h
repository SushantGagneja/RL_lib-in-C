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

// Activations and elementwise maps
b32 mat_relu(Mat *out, const Mat *in);
b32 mat_tanh(Mat *out, const Mat *in);
b32 mat_sigmoid(Mat *out, const Mat *in);
b32 mat_exp(Mat *out, const Mat *in);
// Natural log; the input must be strictly positive (the gradient divides by it).
b32 mat_log(Mat *out, const Mat *in);
b32 mat_div(Mat *out, const Mat *a, const Mat *b); // a / b, elementwise
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
b32 mat_sigmoid_add_grad(Mat *out, const Mat *sigmoid_out, const Mat *grad);
b32 mat_exp_add_grad(Mat *out, const Mat *exp_out, const Mat *grad);
b32 mat_log_add_grad(Mat *out, const Mat *in, const Mat *grad);
b32 mat_div_add_grad(Mat *a_grad, Mat *b_grad, const Mat *a, const Mat *b,
                     const Mat *grad);
b32 mat_softmax_add_grad(Mat *out, const Mat *softmax_out,
                         const Mat *grad); // Per-column mat_softmax backward
b32 mat_mse_add_grad(Mat *pred_grad, const Mat *pred, const Mat *target,
                     const Mat *grad);

// Value-loss gradient with the PPO-style clipped target: the unclipped branch
// (v - t) is used where |v - old_v| <= clip or where the unclipped error is
// already the larger of the two; the clipped branch contributes a constant, so
// outside the trust region its gradient is zero. `old_v` is the rollout's
// value estimate for the same row. `grad` carries the loss seed.
b32 mat_mse_clip_add_grad(Mat *pred_grad, const Mat *pred, const Mat *old_v,
                          const Mat *target, real clip, const Mat *grad);
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

// ── Continuous (squashed diagonal-Gaussian) policy ───────────────────
// The continuous policy is a diagonal Gaussian over a pre-squash coordinate,
// squashed into the environment's action box. The network supplies a mean
// (act_dim x 1, or act_dim x B when batched) and a learnable log standard
// deviation (act_dim x 1, shared across the batch):
//   u_d = mu_d + exp(log_std_d) * eps,   eps ~ N(0, 1)
//   a_d = center_d + half_d * tanh(u_d)
// with center_d/half_d the midpoint and half-width of [min_action_d,
// max_action_d]. The density carries the squash's log-determinant,
//   log pi(a) = log N(u | mu, sigma) - log(half_d) - log(sech^2(u_d)),
// the construction SAC and Stable-Baselines3 use. Three guards keep it finite
// on a bounded action space: the density is evaluated at the *sampled* u (not
// recovered from the action with atanh, which is infinite at the bounds), the
// mean is clamped to +-5 so the density cannot spike outside the box, and
// log_std is clamped on read to [-6, 2] so sigma stays away from zero and
// infinity. See src/linalg.c for the details and the failures each guard fixes.
//
// `min_action` and `max_action` each hold act_dim entries, one per action
// dimension (Env.action_min / Env.action_max).

// Half-width of the policy mean clamp. tanh(5) is within 1e-4 of 1, so the
// reachable actions still cover the whole box while the density's peak stays
// bounded. Exposed because a deterministic (greedy) action must apply exactly
// the same clamp as the sampler does.
#define RL_GAUSSIAN_MEAN_BOUND ((real)5)

// Read-time clamp on the learnable log standard deviation: sigma stays in
// roughly [2.5e-3, 7.4].
#define RL_GAUSSIAN_LOG_STD_MIN ((real)-6)
#define RL_GAUSSIAN_LOG_STD_MAX ((real)2)

// Samples one action (act_dim x 1) from the policy and writes the pre-squash
// coordinate it came from into `u_out` (also act_dim x 1). Every dimension is
// squashed into its own [min_action[d], max_action[d]] bound, so an environment
// with per-dimension limits (different joints, say) is supported. Keep `u_out`:
// mat_gaussian_logp and the PPO loss need it, and it cannot be recovered exactly
// from the action once an action saturates at a bound.
b32 mat_gaussian_sample(Mat *action_out, Mat *u_out, const Mat *raw_mean,
                        const Mat *log_std, const real *min_action,
                        const real *max_action, Rng *rng);

// Inverse of the squash for actions that were not produced by
// mat_gaussian_sample (a stored replay, say). Near the bounds this is lossy --
// atanh blows up at a = +-half -- so the training path carries u instead.
b32 mat_gaussian_unsquash(Mat *u_out, const Mat *actions,
                          const real *min_action, const real *max_action);

// log pi(a) per batch column (1 x batch_size) from the pre-squash sample block
// `u` (act_dim x batch_size); `raw_mean` may be act_dim x 1 and is then shared
// by every column.
b32 mat_gaussian_logp(Mat *out, const Mat *raw_mean, const Mat *log_std,
                      const Mat *u, const real *min_action,
                      const real *max_action, u32 batch_size);

// PPO clipped-surrogate loss for the continuous policy:
//   L = -mean_b min( r_b * A_b, clamp(r_b, 1-eps, 1+eps) * A_b ),
// the same objective as mat_ppo_policy_loss with the Gaussian log-probabilities.
b32 mat_gaussian_ppo_loss(Mat *out, const Mat *raw_mean, const Mat *log_std,
                          const Mat *u, const real *advantages,
                          const real *old_logp, const real *min_action,
                          const real *max_action, real clip_eps,
                          u32 batch_size);

// PPO clipped-surrogate gradient for the continuous policy. Accumulates into
// mean_grad (act_dim x batch_size) and log_std_grad (act_dim x 1); scaled by
// grad->data[0] / batch_size, matching the discrete loss.
b32 mat_gaussian_ppo_add_grad(Mat *mean_grad, Mat *log_std_grad,
                              const Mat *raw_mean, const Mat *log_std,
                              const Mat *u, const real *advantages,
                              const real *old_logp, const real *min_action,
                              const real *max_action, real clip_eps,
                              u32 batch_size, const Mat *grad);

// Mean entropy of the *pre-squash* Gaussian, H = sum_d log(sigma_d) +
// D/2 * log(2*pi*e). The squashed density's entropy has no closed form; this
// pre-squash value is the usual stand-in for the entropy bonus.
b32 mat_gaussian_entropy(Mat *out, const Mat *log_std);

// d(mean H)/d(log_std_d) = 1, accumulated and scaled by grad->data[0].
b32 mat_gaussian_entropy_add_grad(Mat *log_std_grad, const Mat *log_std,
                                  const Mat *grad);

#endif // LINALG_H
