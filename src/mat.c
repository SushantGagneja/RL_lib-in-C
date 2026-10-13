#include "mat.h"
#include "prng.h"
#include <math.h>
#include <string.h>

matrix *create_matrix(mem_arena *arena, u32 rows, u32 cols) {
  matrix *mat = PUSH_STRUCT(arena, matrix);
  assert(mat != NULL && "Arena out of memory in create_matrix");

  mat->rows = rows;
  mat->cols = cols;
  mat->data = PUSH_ARRAY(arena, f32, (u64)rows * cols);
  assert(mat->data != NULL && "Arena out of memory in create_matrix data");

  return mat;
}

void clear(matrix *mat) {
  memset(mat->data, 0, sizeof(f32) * (u64)mat->rows * mat->cols);
}

void fill(matrix *mat, f32 x) {
  u64 size = (u64)mat->rows * mat->cols;
  for (u64 i = 0; i < size; i++) {
    mat->data[i] = x;
  }
}

void fill_rand(matrix *mat, f32 lower, f32 upper) {
  u64 size = (u64)mat->rows * mat->cols;
  for (u64 i = 0; i < size; i++) {
    mat->data[i] = prng_randf() * (upper - lower) + lower;
  }
}

void scale(matrix *mat, f32 scale_factor) {
  u64 size = (u64)mat->rows * mat->cols;
  for (u64 i = 0; i < size; i++) {
    mat->data[i] *= scale_factor;
  }
}

b32 add(matrix *out, const matrix *a, const matrix *b) {
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
      f32 b_val = b->data[r];
      for (u32 c = 0; c < max_cols; c++) {
        out->data[r * max_cols + c] = a->data[r * max_cols + c] + b_val;
      }
    }
  } else if (cols_a == 1) {
    // Broadcast a along columns
    for (u32 r = 0; r < rows; r++) {
      f32 a_val = a->data[r];
      for (u32 c = 0; c < max_cols; c++) {
        out->data[r * max_cols + c] = a_val + b->data[r * max_cols + c];
      }
    }
  } else {
    return false;
  }

  return true;
}

b32 add_broadcast_grad(matrix *a_grad, matrix *b_grad, const matrix *grad) {
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
        f32 sum = 0.0f;
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
        f32 sum = 0.0f;
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

b32 sub(matrix *out, const matrix *a, const matrix *b) {
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
      f32 b_val = b->data[r];
      for (u32 c = 0; c < max_cols; c++) {
        out->data[r * max_cols + c] = a->data[r * max_cols + c] - b_val;
      }
    }
  } else if (cols_a == 1) {
    // Broadcast a along columns
    for (u32 r = 0; r < rows; r++) {
      f32 a_val = a->data[r];
      for (u32 c = 0; c < max_cols; c++) {
        out->data[r * max_cols + c] = a_val - b->data[r * max_cols + c];
      }
    }
  } else {
    return false;
  }

  return true;
}

b32 sub_broadcast_grad(matrix *a_grad, matrix *b_grad, const matrix *grad) {
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
        f32 sum = 0.0f;
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
        f32 sum = 0.0f;
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

b32 mul(matrix *out, const matrix *a, const matrix *b) {
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

b32 mul_add_grad(matrix *a_grad, matrix *b_grad, const matrix *a,
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

void _mat_mul_nn(matrix *out, const matrix *a, const matrix *b) {
  for (u64 i = 0; i < out->rows; i++) {
    for (u64 k = 0; k < a->cols; k++) {
      for (u64 j = 0; j < out->cols; j++) {
        out->data[j + i * out->cols] +=
            a->data[k + i * a->cols] * b->data[j + k * b->cols];
      }
    }
  }
}

void _mat_mul_nt(matrix *out, const matrix *a, const matrix *b) {
  for (u64 i = 0; i < out->rows; i++) {
    for (u64 k = 0; k < a->cols; k++) {
      for (u64 j = 0; j < out->cols; j++) {
        out->data[j + i * out->cols] +=
            a->data[k + i * a->cols] * b->data[k + j * b->cols];
      }
    }
  }
}

void _mat_mul_tn(matrix *out, const matrix *a, const matrix *b) {
  for (u64 i = 0; i < out->rows; i++) {
    for (u64 k = 0; k < a->rows; k++) {
      for (u64 j = 0; j < out->cols; j++) {
        out->data[j + i * out->cols] +=
            a->data[i + k * a->cols] * b->data[j + k * b->cols];
      }
    }
  }
}

void _mat_mul_tt(matrix *out, const matrix *a, const matrix *b) {
  for (u64 i = 0; i < out->rows; i++) {
    for (u64 k = 0; k < a->rows; k++) {
      for (u64 j = 0; j < out->cols; j++) {
        out->data[j + i * out->cols] +=
            a->data[i + k * a->cols] * b->data[k + j * b->cols];
      }
    }
  }
}

b32 matmul(matrix *out, const matrix *a, const matrix *b, b8 zero_out,
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
    clear(out);
  }

  u32 mode = (transpose_a << 1) | transpose_b;
  switch (mode) {
  case 0:
    _mat_mul_nn(out, a, b);
    break;
  case 1:
    _mat_mul_nt(out, a, b);
    break;
  case 2:
    _mat_mul_tn(out, a, b);
    break;
  case 3:
    _mat_mul_tt(out, a, b);
    break;
  }

  return true;
}

b32 relu(matrix *out, const matrix *in) {
  if (out->rows != in->rows || out->cols != in->cols) {
    return false;
  }

  u64 size = (u64)out->rows * out->cols;
  for (u64 i = 0; i < size; i++) {
    out->data[i] = MAX(0.0f, in->data[i]);
  }
  return true;
}

b32 relu_add_grad(matrix *out, const matrix *in, const matrix *grad) {
  if (out->rows != in->rows || out->cols != in->cols) return false;
  if (out->rows != grad->rows || out->cols != grad->cols) return false;

  u64 size = (u64)out->rows * out->cols;
  for (u64 i = 0; i < size; i++) {
    out->data[i] += in->data[i] > 0.0f ? grad->data[i] : 0.0f;
  }
  return true;
}

b32 tanh_act(matrix *out, const matrix *in) {
  if (out->rows != in->rows || out->cols != in->cols) return false;

  u64 size = (u64)out->rows * out->cols;
  for (u64 i = 0; i < size; i++) {
    out->data[i] = tanhf(in->data[i]);
  }
  return true;
}

b32 tanh_add_grad(matrix *out, const matrix *tanh_out, const matrix *grad) {
  if (out->rows != tanh_out->rows || out->cols != tanh_out->cols) return false;
  if (out->rows != grad->rows || out->cols != grad->cols) return false;

  u64 size = (u64)out->rows * out->cols;
  for (u64 i = 0; i < size; i++) {
    f32 y = tanh_out->data[i];
    out->data[i] += grad->data[i] * (1.0f - y * y);
  }
  return true;
}

// Per-column softmax for batching (each column c is an independent distribution)
b32 softmax(matrix *out, const matrix *in) {
  if (out->rows != in->rows || out->cols != in->cols) {
    return false;
  }

  u32 rows = in->rows;
  u32 cols = in->cols;

  for (u32 c = 0; c < cols; c++) {
    f32 max_val = in->data[0 * cols + c];
    for (u32 r = 1; r < rows; r++) {
      f32 val = in->data[r * cols + c];
      if (val > max_val) max_val = val;
    }

    f32 sum = 0.0f;
    for (u32 r = 0; r < rows; r++) {
      f32 e = expf(in->data[r * cols + c] - max_val);
      out->data[r * cols + c] = e;
      sum += e;
    }

    f32 inv_sum = 1.0f / sum;
    for (u32 r = 0; r < rows; r++) {
      out->data[r * cols + c] *= inv_sum;
    }
  }

  return true;
}

// Per-column softmax backward
b32 softmax_add_grad(matrix *out, const matrix *softmax_out,
                     const matrix *grad) {
  if (out->rows != softmax_out->rows || out->cols != softmax_out->cols) return false;
  if (grad->rows != softmax_out->rows || grad->cols != softmax_out->cols) return false;

  u32 rows = softmax_out->rows;
  u32 cols = softmax_out->cols;

  for (u32 c = 0; c < cols; c++) {
    f32 dot = 0.0f;
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

b32 mse_loss(matrix *out, const matrix *pred, const matrix *target) {
  if (pred->rows != target->rows || pred->cols != target->cols) return false;
  if (out->rows != 1 || out->cols != 1) return false;

  u64 size = (u64)pred->rows * pred->cols;
  f32 sum_sq = 0.0f;
  for (u64 i = 0; i < size; i++) {
    f32 diff = pred->data[i] - target->data[i];
    sum_sq += diff * diff;
  }
  out->data[0] = sum_sq / (f32)size;
  return true;
}

b32 mse_add_grad(matrix *pred_grad, const matrix *pred, const matrix *target,
                 const matrix *grad) {
  if (pred->rows != target->rows || pred->cols != target->cols) return false;
  if (pred_grad->rows != pred->rows || pred_grad->cols != pred->cols) return false;

  u64 size = (u64)pred->rows * pred->cols;
  f32 scale_factor = (2.0f / (f32)size) * grad->data[0];
  for (u64 i = 0; i < size; i++) {
    pred_grad->data[i] += scale_factor * (pred->data[i] - target->data[i]);
  }
  return true;
}

b32 reinforce_loss(matrix *out, const matrix *probs, const matrix *advantages) {
  if (probs->rows != advantages->rows || probs->cols != advantages->cols) return false;
  if (out->rows != advantages->rows || out->cols != advantages->cols) return false;

  u64 size = (u64)out->rows * out->cols;
  for (u64 i = 0; i < size; i++) {
    f32 p = MAX(probs->data[i], 1e-8f);
    out->data[i] = -logf(p) * advantages->data[i];
  }
  return true;
}

b32 reinforce_add_grad(matrix *prob_grads, const matrix *probs,
                       const matrix *rt, const matrix *grad) {
  if (probs->rows != rt->rows || probs->cols != rt->cols) return false;
  if (grad->rows != probs->rows || grad->cols != probs->cols) return false;
  if (prob_grads->rows != probs->rows || prob_grads->cols != probs->cols) return false;

  u64 size = (u64)probs->rows * probs->cols;
  for (u64 i = 0; i < size; i++) {
    f32 p = MAX(probs->data[i], 1e-8f);
    prob_grads->data[i] += grad->data[i] * (-rt->data[i] / p);
  }
  return true;
}
