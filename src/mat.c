#include "mat.h"
#include "prng.h"
#include <math.h>
#include <string.h>

matrix *mat_create(mem_arena *arena, u32 rows, u32 cols) {
  matrix *mat = PUSH_STRUCT(arena, matrix);
  CRL_ASSERT(mat != NULL, "Arena out of memory in mat_create");

  mat->rows = rows;
  mat->cols = cols;
  mat->data = PUSH_ARRAY(arena, real, (u64)rows * cols);
  CRL_ASSERT(mat->data != NULL, "Arena out of memory in mat_create data");

  return mat;
}

void mat_clear(matrix *mat) {
  memset(mat->data, 0, sizeof(real) * (u64)mat->rows * mat->cols);
}

void mat_fill(matrix *mat, real x) {
  u64 size = (u64)mat->rows * mat->cols;
  for (u64 i = 0; i < size; i++) {
    mat->data[i] = x;
  }
}

void mat_fill_rand(matrix *mat, real lower, real upper) {
  u64 size = (u64)mat->rows * mat->cols;
  for (u64 i = 0; i < size; i++) {
    mat->data[i] = prng_randf() * (upper - lower) + lower;
  }
}

void mat_fill_rand_r(matrix *mat, prng_state *rng, real lower, real upper) {
  u64 size = (u64)mat->rows * mat->cols;
  for (u64 i = 0; i < size; i++) {
    mat->data[i] = prng_randf_r(rng) * (upper - lower) + lower;
  }
}

void mat_scale(matrix *mat, real scale_factor) {
  u64 size = (u64)mat->rows * mat->cols;
  for (u64 i = 0; i < size; i++) {
    mat->data[i] *= scale_factor;
  }
}

b32 mat_add(matrix *out, const matrix *a, const matrix *b) {
  if (a->rows != b->rows) {
    return false;
  }

  u32 rows = a->rows;
  u32 cols_a = a->cols;
  u32 cols_b = b->cols;
  u32 max_cols = cols_a > cols_b ? cols_a : cols_b;

  if (out->rows != rows || out->cols != max_cols) {
    return false;
  }

  if (cols_a == cols_b) {
    u64 size = (u64)rows * max_cols;
    for (u64 i = 0; i < size; i++) {
      out->data[i] = a->data[i] + b->data[i];
    }
  } else if (cols_b == 1) {
    // Broadcast b along columns
    for (u32 r = 0; r < rows; r++) {
      real b_val = b->data[r];
      for (u32 c = 0; c < max_cols; c++) {
        out->data[r * max_cols + c] = a->data[r * max_cols + c] + b_val;
      }
    }
  } else if (cols_a == 1) {
    // Broadcast a along columns
    for (u32 r = 0; r < rows; r++) {
      real a_val = a->data[r];
      for (u32 c = 0; c < max_cols; c++) {
        out->data[r * max_cols + c] = a_val + b->data[r * max_cols + c];
      }
    }
  } else {
    return false;
  }

  return true;
}

b32 mat_add_broadcast_grad(matrix *a_grad, matrix *b_grad, const matrix *grad) {
  u32 rows = grad->rows;
  u32 cols = grad->cols;

  if (a_grad != NULL) {
    if (a_grad->rows != rows) return false;
    if (a_grad->cols == cols) {
      u64 size = (u64)rows * cols;
      for (u64 i = 0; i < size; i++) {
        a_grad->data[i] += grad->data[i];
      }
    } else if (a_grad->cols == 1) {
      // Sum across broadcasted columns
      for (u32 r = 0; r < rows; r++) {
        real sum = 0.0f;
        for (u32 c = 0; c < cols; c++) {
          sum += grad->data[r * cols + c];
        }
        a_grad->data[r] += sum;
      }
    } else {
      return false;
    }
  }

  if (b_grad != NULL) {
    if (b_grad->rows != rows) return false;
    if (b_grad->cols == cols) {
      u64 size = (u64)rows * cols;
      for (u64 i = 0; i < size; i++) {
        b_grad->data[i] += grad->data[i];
      }
    } else if (b_grad->cols == 1) {
      // Sum across broadcasted columns
      for (u32 r = 0; r < rows; r++) {
        real sum = 0.0f;
        for (u32 c = 0; c < cols; c++) {
          sum += grad->data[r * cols + c];
        }
        b_grad->data[r] += sum;
      }
    } else {
      return false;
    }
  }

  return true;
}

b32 mat_sub(matrix *out, const matrix *a, const matrix *b) {
  if (a->rows != b->rows) {
    return false;
  }

  u32 rows = a->rows;
  u32 cols_a = a->cols;
  u32 cols_b = b->cols;
  u32 max_cols = cols_a > cols_b ? cols_a : cols_b;

  if (out->rows != rows || out->cols != max_cols) {
    return false;
  }

  if (cols_a == cols_b) {
    u64 size = (u64)rows * max_cols;
    for (u64 i = 0; i < size; i++) {
      out->data[i] = a->data[i] - b->data[i];
    }
  } else if (cols_b == 1) {
    // Broadcast b along columns
    for (u32 r = 0; r < rows; r++) {
      real b_val = b->data[r];
      for (u32 c = 0; c < max_cols; c++) {
        out->data[r * max_cols + c] = a->data[r * max_cols + c] - b_val;
      }
    }
  } else if (cols_a == 1) {
    // Broadcast a along columns
    for (u32 r = 0; r < rows; r++) {
      real a_val = a->data[r];
      for (u32 c = 0; c < max_cols; c++) {
        out->data[r * max_cols + c] = a_val - b->data[r * max_cols + c];
      }
    }
  } else {
    return false;
  }

  return true;
}

b32 mat_sub_broadcast_grad(matrix *a_grad, matrix *b_grad, const matrix *grad) {
  u32 rows = grad->rows;
  u32 cols = grad->cols;

  if (a_grad != NULL) {
    if (a_grad->rows != rows) return false;
    if (a_grad->cols == cols) {
      u64 size = (u64)rows * cols;
      for (u64 i = 0; i < size; i++) {
        a_grad->data[i] += grad->data[i];
      }
    } else if (a_grad->cols == 1) {
      for (u32 r = 0; r < rows; r++) {
        real sum = 0.0f;
        for (u32 c = 0; c < cols; c++) {
          sum += grad->data[r * cols + c];
        }
        a_grad->data[r] += sum;
      }
    } else {
      return false;
    }
  }

  if (b_grad != NULL) {
    if (b_grad->rows != rows) return false;
    if (b_grad->cols == cols) {
      u64 size = (u64)rows * cols;
      for (u64 i = 0; i < size; i++) {
        b_grad->data[i] -= grad->data[i];
      }
    } else if (b_grad->cols == 1) {
      for (u32 r = 0; r < rows; r++) {
        real sum = 0.0f;
        for (u32 c = 0; c < cols; c++) {
          sum += grad->data[r * cols + c];
        }
        b_grad->data[r] -= sum;
      }
    } else {
      return false;
    }
  }

  return true;
}

b32 mat_mul(matrix *out, const matrix *a, const matrix *b) {
  if (a->rows != b->rows || a->cols != b->cols) {
    return false;
  }
  if (out->rows != a->rows || out->cols != a->cols) {
    return false;
  }

  u64 size = (u64)out->rows * out->cols;
  for (u64 i = 0; i < size; i++) {
    out->data[i] = a->data[i] * b->data[i];
  }
  return true;
}

b32 mat_mul_add_grad(matrix *a_grad, matrix *b_grad, const matrix *a,
                 const matrix *b, const matrix *grad) {
  if (a->rows != b->rows || a->cols != b->cols) return false;
  if (grad->rows != a->rows || grad->cols != a->cols) return false;

  u64 size = (u64)a->rows * a->cols;
  if (a_grad != NULL) {
    for (u64 i = 0; i < size; i++) {
      a_grad->data[i] += grad->data[i] * b->data[i];
    }
  }
  if (b_grad != NULL) {
    for (u64 i = 0; i < size; i++) {
      b_grad->data[i] += grad->data[i] * a->data[i];
    }
  }
  return true;
}

static void mat_mul_kernel_nn(matrix *out, const matrix *a, const matrix *b) {
  for (u64 i = 0; i < out->rows; i++) {
    for (u64 k = 0; k < a->cols; k++) {
      for (u64 j = 0; j < out->cols; j++) {
        out->data[j + i * out->cols] +=
            a->data[k + i * a->cols] * b->data[j + k * b->cols];
      }
    }
  }
}

static void mat_mul_kernel_nt(matrix *out, const matrix *a, const matrix *b) {
  for (u64 i = 0; i < out->rows; i++) {
    for (u64 k = 0; k < a->cols; k++) {
      for (u64 j = 0; j < out->cols; j++) {
        out->data[j + i * out->cols] +=
            a->data[k + i * a->cols] * b->data[k + j * b->cols];
      }
    }
  }
}

static void mat_mul_kernel_tn(matrix *out, const matrix *a, const matrix *b) {
  for (u64 i = 0; i < out->rows; i++) {
    for (u64 k = 0; k < a->rows; k++) {
      for (u64 j = 0; j < out->cols; j++) {
        out->data[j + i * out->cols] +=
            a->data[i + k * a->cols] * b->data[j + k * b->cols];
      }
    }
  }
}

static void mat_mul_kernel_tt(matrix *out, const matrix *a, const matrix *b) {
  for (u64 i = 0; i < out->rows; i++) {
    for (u64 k = 0; k < a->rows; k++) {
      for (u64 j = 0; j < out->cols; j++) {
        out->data[j + i * out->cols] +=
            a->data[i + k * a->cols] * b->data[k + j * b->cols];
      }
    }
  }
}

b32 mat_matmul(matrix *out, const matrix *a, const matrix *b, b8 zero_out,
           b8 transpose_a, b8 transpose_b) {
  u32 a_rows = transpose_a ? a->cols : a->rows;
  u32 a_cols = transpose_a ? a->rows : a->cols;
  u32 b_rows = transpose_b ? b->cols : b->rows;
  u32 b_cols = transpose_b ? b->rows : b->cols;

  if (a_cols != b_rows) {
    return false;
  }
  if (out->rows != a_rows || out->cols != b_cols) {
    return false;
  }

  if (zero_out) {
    mat_clear(out);
  }

  u32 mode = ((u32)transpose_a << 1) | (u32)transpose_b;
  switch (mode) {
  case 0:
    mat_mul_kernel_nn(out, a, b);
    break;
  case 1:
    mat_mul_kernel_nt(out, a, b);
    break;
  case 2:
    mat_mul_kernel_tn(out, a, b);
    break;
  case 3:
    mat_mul_kernel_tt(out, a, b);
    break;
  }

  return true;
}

b32 mat_relu(matrix *out, const matrix *in) {
  if (out->rows != in->rows || out->cols != in->cols) {
    return false;
  }

  u64 size = (u64)out->rows * out->cols;
  for (u64 i = 0; i < size; i++) {
    out->data[i] = MAX(0.0f, in->data[i]);
  }
  return true;
}

b32 mat_relu_add_grad(matrix *out, const matrix *in, const matrix *grad) {
  if (out->rows != in->rows || out->cols != in->cols) return false;
  if (out->rows != grad->rows || out->cols != grad->cols) return false;

  u64 size = (u64)out->rows * out->cols;
  for (u64 i = 0; i < size; i++) {
    out->data[i] += in->data[i] > 0.0f ? grad->data[i] : 0.0f;
  }
  return true;
}

b32 mat_tanh(matrix *out, const matrix *in) {
  if (out->rows != in->rows || out->cols != in->cols) return false;

  u64 size = (u64)out->rows * out->cols;
  for (u64 i = 0; i < size; i++) {
    out->data[i] = real_tanh(in->data[i]);
  }
  return true;
}

b32 mat_tanh_add_grad(matrix *out, const matrix *tanh_out, const matrix *grad) {
  if (out->rows != tanh_out->rows || out->cols != tanh_out->cols) return false;
  if (out->rows != grad->rows || out->cols != grad->cols) return false;

  u64 size = (u64)out->rows * out->cols;
  for (u64 i = 0; i < size; i++) {
    real y = tanh_out->data[i];
    out->data[i] += grad->data[i] * (1.0f - y * y);
  }
  return true;
}

// Per-column mat_softmax for batching (each column c is an independent distribution)
b32 mat_softmax(matrix *out, const matrix *in) {
  if (out->rows != in->rows || out->cols != in->cols) {
    return false;
  }

  u32 rows = in->rows;
  u32 cols = in->cols;

  for (u32 c = 0; c < cols; c++) {
    real max_val = in->data[0 * cols + c];
    for (u32 r = 1; r < rows; r++) {
      real val = in->data[r * cols + c];
      if (val > max_val) max_val = val;
    }

    real sum = 0.0f;
    for (u32 r = 0; r < rows; r++) {
      real e = real_exp(in->data[r * cols + c] - max_val);
      out->data[r * cols + c] = e;
      sum += e;
    }

    real inv_sum = 1.0f / sum;
    for (u32 r = 0; r < rows; r++) {
      out->data[r * cols + c] *= inv_sum;
    }
  }

  return true;
}

// Per-column mat_softmax backward
b32 mat_softmax_add_grad(matrix *out, const matrix *softmax_out,
                     const matrix *grad) {
  if (out->rows != softmax_out->rows || out->cols != softmax_out->cols) return false;
  if (grad->rows != softmax_out->rows || grad->cols != softmax_out->cols) return false;

  u32 rows = softmax_out->rows;
  u32 cols = softmax_out->cols;

  for (u32 c = 0; c < cols; c++) {
    real dot = 0.0f;
    for (u32 r = 0; r < rows; r++) {
      dot += grad->data[r * cols + c] * softmax_out->data[r * cols + c];
    }

    for (u32 r = 0; r < rows; r++) {
      out->data[r * cols + c] +=
          softmax_out->data[r * cols + c] * (grad->data[r * cols + c] - dot);
    }
  }

  return true;
}

b32 mat_mse_loss(matrix *out, const matrix *pred, const matrix *target) {
  if (pred->rows != target->rows || pred->cols != target->cols) return false;
  if (out->rows != 1 || out->cols != 1) return false;

  u64 size = (u64)pred->rows * pred->cols;
  real sum_sq = 0.0f;
  for (u64 i = 0; i < size; i++) {
    real diff = pred->data[i] - target->data[i];
    sum_sq += diff * diff;
  }
  out->data[0] = sum_sq / (real)size;
  return true;
}

b32 mat_mse_add_grad(matrix *pred_grad, const matrix *pred, const matrix *target,
                 const matrix *grad) {
  if (pred->rows != target->rows || pred->cols != target->cols) return false;
  if (pred_grad->rows != pred->rows || pred_grad->cols != pred->cols) return false;

  u64 size = (u64)pred->rows * pred->cols;
  real scale_factor = (2.0f / (real)size) * grad->data[0];
  for (u64 i = 0; i < size; i++) {
    pred_grad->data[i] += scale_factor * (pred->data[i] - target->data[i]);
  }
  return true;
}

b32 mat_reinforce_loss(matrix *out, const matrix *probs, const matrix *advantages) {
  if (probs->rows != advantages->rows || probs->cols != advantages->cols) return false;
  if (out->rows != advantages->rows || out->cols != advantages->cols) return false;

  u64 size = (u64)out->rows * out->cols;
  for (u64 i = 0; i < size; i++) {
    real p = MAX(probs->data[i], 1e-8f);
    out->data[i] = -real_log(p) * advantages->data[i];
  }
  return true;
}

b32 mat_reinforce_add_grad(matrix *prob_grads, const matrix *probs,
                       const matrix *rt, const matrix *grad) {
  if (probs->rows != rt->rows || probs->cols != rt->cols) return false;
  if (grad->rows != probs->rows || grad->cols != probs->cols) return false;
  if (prob_grads->rows != probs->rows || prob_grads->cols != probs->cols) return false;

  u64 size = (u64)probs->rows * probs->cols;
  for (u64 i = 0; i < size; i++) {
    real p = MAX(probs->data[i], 1e-8f);
    prob_grads->data[i] += grad->data[i] * (-rt->data[i] / p);
  }
  return true;
}

b32 mat_fused_policy_loss(matrix *out, const matrix *logits, const u32 *actions,
                      const real *advantages, u32 batch_size) {
  if (logits == NULL || actions == NULL || advantages == NULL || out == NULL) {
    return false;
  }
  if (out->rows != 1 || out->cols != 1) return false;
  if (logits->cols != batch_size) return false;

  u32 act_dim = logits->rows;
  real total_loss = 0.0f;

  for (u32 b = 0; b < batch_size; b++) {
    u32 a_chosen = actions[b];
    if (a_chosen >= act_dim) return false;

    // Log-sum-exp over column b
    real max_z = logits->data[0 * batch_size + b];
    for (u32 a = 1; a < act_dim; a++) {
      real z = logits->data[a * batch_size + b];
      if (z > max_z) max_z = z;
    }

    real sum_exp = 0.0f;
    for (u32 a = 0; a < act_dim; a++) {
      sum_exp += real_exp(logits->data[a * batch_size + b] - max_z);
    }
    real log_sum_exp = max_z + real_log(sum_exp);

    real log_prob_a = logits->data[a_chosen * batch_size + b] - log_sum_exp;
    total_loss += -advantages[b] * log_prob_a;
  }

  out->data[0] = total_loss / (real)batch_size;
  return true;
}

b32 mat_fused_policy_add_grad(matrix *logits_grad, const matrix *logits,
                          const u32 *actions, const real *advantages,
                          u32 batch_size, const matrix *grad) {
  if (logits == NULL || logits_grad == NULL || actions == NULL ||
      advantages == NULL || grad == NULL) {
    return false;
  }
  if (logits->cols != batch_size || logits_grad->cols != batch_size) return false;

  u32 act_dim = logits->rows;
  real g_scale = grad->data[0] / (real)batch_size;

  for (u32 b = 0; b < batch_size; b++) {
    u32 a_chosen = actions[b];

    // Softmax probabilities for column b
    real max_z = logits->data[0 * batch_size + b];
    for (u32 a = 1; a < act_dim; a++) {
      real z = logits->data[a * batch_size + b];
      if (z > max_z) max_z = z;
    }

    real sum_exp = 0.0f;
    for (u32 a = 0; a < act_dim; a++) {
      sum_exp += real_exp(logits->data[a * batch_size + b] - max_z);
    }
    real inv_sum = 1.0f / sum_exp;

    real adv_scale = advantages[b] * g_scale;

    for (u32 a = 0; a < act_dim; a++) {
      real prob = real_exp(logits->data[a * batch_size + b] - max_z) * inv_sum;
      real indicator = (a == a_chosen) ? 1.0f : 0.0f;
      // Gradient of -adv * log(p_a) is adv * (p - indicator)
      logits_grad->data[a * batch_size + b] += adv_scale * (prob - indicator);
    }
  }

  return true;
}

static void softmax_column(const matrix *logits, u32 b, real *probs_out,
                           real *entropy_out) {
  u32 act_dim = logits->rows;
  u32 batch_size = logits->cols;

  real max_z = logits->data[0 * batch_size + b];
  for (u32 a = 1; a < act_dim; a++) {
    real z = logits->data[a * batch_size + b];
    if (z > max_z) max_z = z;
  }

  real sum_exp = 0.0f;
  for (u32 a = 0; a < act_dim; a++) {
    real e = real_exp(logits->data[a * batch_size + b] - max_z);
    probs_out[a] = e;
    sum_exp += e;
  }

  real inv_sum = 1.0f / sum_exp;
  real entropy = 0.0f;
  for (u32 a = 0; a < act_dim; a++) {
    real p = probs_out[a] * inv_sum;
    probs_out[a] = p;
    if (p > 0.0f) {
      entropy -= p * real_log(p);
    }
  }
  *entropy_out = entropy;
}

b32 mat_entropy_loss(matrix *out, const matrix *logits, u32 batch_size) {
  if (out == NULL || logits == NULL) return false;
  if (out->rows != 1 || out->cols != 1) return false;
  if (logits->cols != batch_size || batch_size == 0) return false;

  real *probs = malloc(sizeof(real) * logits->rows);
  if (probs == NULL) return false;

  real total = 0.0f;
  for (u32 b = 0; b < batch_size; b++) {
    real h = 0.0f;
    softmax_column(logits, b, probs, &h);
    total += h;
  }
  free(probs);

  out->data[0] = total / (real)batch_size;
  return true;
}

b32 mat_entropy_add_grad(matrix *logits_grad, const matrix *logits,
                         u32 batch_size, const matrix *grad) {
  if (logits_grad == NULL || logits == NULL || grad == NULL) return false;
  if (logits->cols != batch_size || logits_grad->cols != batch_size) return false;
  if (logits->rows != logits_grad->rows) return false;

  u32 act_dim = logits->rows;
  real *probs = malloc(sizeof(real) * act_dim);
  if (probs == NULL) return false;

  real g_scale = grad->data[0] / (real)batch_size;

  for (u32 b = 0; b < batch_size; b++) {
    real h = 0.0f;
    softmax_column(logits, b, probs, &h);
    for (u32 a = 0; a < act_dim; a++) {
      real p = probs[a];
      real dH = -p * (real_log(p + 1e-30f) + h);
      logits_grad->data[a * batch_size + b] += g_scale * dH;
    }
  }

  free(probs);
  return true;
}
