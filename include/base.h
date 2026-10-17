#ifndef BASE_H
#define BASE_H

#define _CRT_SECURE_NO_WARNINGS

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef uint8_t b8;
typedef uint32_t b32;

typedef float f32;
typedef double f64;

// Scalar type for the math/autograd stack. Defaults to 32-bit `float`; build
// the whole library with -DCRL_USE_DOUBLE to switch to `double` (the gradient
// check uses this for a tight finite-difference tolerance). Use the real_*
// wrappers below instead of the libm float/double variants so the same source
// compiles under either mode; use REAL_C(x) for an exact-precision literal.
#ifdef CRL_USE_DOUBLE
typedef double real;
#define REAL_C(x) (x)
#define real_fabs fabs
#define real_sqrt sqrt
#define real_exp exp
#define real_log log
#define real_tanh tanh
#define real_sin sin
#define real_cos cos
#define real_pow pow
#define real_fmax fmax
#define real_fmin fmin
#define REAL_EPS DBL_EPSILON
#else
typedef float real;
#define REAL_C(x) x##f
#define real_fabs fabsf
#define real_sqrt sqrtf
#define real_exp expf
#define real_log logf
#define real_tanh tanhf
#define real_sin sinf
#define real_cos cosf
#define real_pow powf
#define real_fmax fmaxf
#define real_fmin fminf
#define REAL_EPS FLT_EPSILON
#endif

#define KiB(n) ((u64)(n) << 10)
#define MiB(n) ((u64)(n) << 20)
#define GiB(n) ((u64)(n) << 30)

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define ALIGN_UP_POW2(x, align) (((x) + (align) - 1) & ~((align) - 1))

// Fatal error: prints the expression and location, then aborts. Used to turn
// silent failures (shape errors, OOM, empty graphs) into immediate crashes.
#define CRL_ASSERT(expr, msg)                                                  \
  do {                                                                         \
    if (!(expr)) {                                                             \
      fprintf(stderr, "[cRL FATAL] %s (%s:%d: %s)\n", (msg), __FILE__,        \
              __LINE__, #expr);                                                \
      abort();                                                                 \
    }                                                                          \
  } while (0)

// Asserts that a boolean-returning library call succeeded. Use it to wrap
// mat_* calls whose `false` return signals a shape/shape mismatch.
#define CRL_CHECK(expr)                                                        \
  do {                                                                         \
    if (!(expr)) {                                                             \
      fprintf(stderr, "[cRL FATAL] check failed (%s:%d): %s\n", __FILE__,      \
              __LINE__, #expr);                                                \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#endif
