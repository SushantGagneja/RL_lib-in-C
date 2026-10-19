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
// the whole library with -DRL_LIB_USE_DOUBLE to switch to `double` (the gradient
// check uses this for a tight finite-difference tolerance). Use the real_*
// wrappers below instead of the libm float/double variants so the same source
// compiles under either mode; use REAL_C(x) for an exact-precision literal.
#ifdef RL_LIB_USE_DOUBLE
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
#define real_fmod fmod
#define real_atanh atanh
#define real_log1p log1p
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
#define real_fmod fmodf
#define real_atanh atanhf
#define real_log1p log1pf
#define REAL_EPS FLT_EPSILON
#endif

#define KiB(n) ((u64)(n) << 10)
#define MiB(n) ((u64)(n) << 20)
#define GiB(n) ((u64)(n) << 30)

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define ALIGN_UP_POW2(x, align) (((x) + (align) - 1) & ~((align) - 1))

// Fatal-error contract. RL_LIB_ASSERT/RL_LIB_CHECK guard programmer errors and
// route through rl_lib_fatal(), which formats one line and invokes the
// installed handler. The default handler prints to stderr and aborts; embedders
// can install their own with rl_lib_set_fatal_handler() (do it before spawning
// worker threads). A handler must not return -- rl_lib_fatal aborts if it does.
// Recoverable failures never come through here: those APIs return false and
// report through their own error text (e.g. checkpoint_last_error()).
typedef void (*rl_fatal_fn)(const char *msg);
rl_fatal_fn rl_lib_set_fatal_handler(rl_fatal_fn handler); // returns the old one
void rl_lib_fatal(const char *msg);

#define RL_LIB_ASSERT(expr, msg)                                               \
  do {                                                                         \
    if (!(expr)) {                                                             \
      char rl_lib_msg_[512];                                                   \
      snprintf(rl_lib_msg_, sizeof(rl_lib_msg_),                               \
               "[RL_lib-in-C FATAL] %s (%s:%d: %s)", (msg), __FILE__,          \
               __LINE__, #expr);                                               \
      rl_lib_fatal(rl_lib_msg_);                                               \
    }                                                                          \
  } while (0)

// Asserts that a boolean-returning library call succeeded. Use it to wrap
// mat_* calls whose `false` return signals a shape mismatch.
#define RL_LIB_CHECK(expr)                                                     \
  do {                                                                         \
    if (!(expr)) {                                                             \
      char rl_lib_msg_[512];                                                   \
      snprintf(rl_lib_msg_, sizeof(rl_lib_msg_),                               \
               "[RL_lib-in-C FATAL] check failed (%s:%d): %s", __FILE__,       \
               __LINE__, #expr);                                               \
      rl_lib_fatal(rl_lib_msg_);                                               \
    }                                                                          \
  } while (0)

#endif
