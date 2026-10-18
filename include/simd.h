#ifndef SIMD_H
#define SIMD_H

#include "base.h"

// Compile-time selection of the vectorised matrix-multiply backend.
//
//   RL_SIMD_NEON  — NEON: 4-wide f32, or 2-wide f64 on AArch64 (ARMv7 NEON
//                   loads 64-bit lanes but has no double-precision arithmetic).
//   RL_SIMD_AVX2  — AVX2+FMA on x86-64: 8-wide f32 and 4-wide f64, guarded by a
//                   runtime __builtin_cpu_supports() probe (the code is compiled
//                   with per-function target attributes, so the binary still runs
//                   on CPUs without AVX2).
//   RL_SIMD_NONE  — portable scalar fallback.
//
// The kernels cover all four (transpose_a, transpose_b) cases in both
// precisions; mat_mm_scalar() stays the reference every vector path is
// cross-checked against (tests/test_mat.c, in f32 and in the f64 suites).
// Define RL_SIMD_BACKEND=RL_SIMD_NONE (i.e. -DRL_SIMD_BACKEND=0) to force the
// scalar path on any build.

#define RL_SIMD_NONE 0
#define RL_SIMD_NEON 1
#define RL_SIMD_AVX2 2

#ifndef RL_SIMD_BACKEND
#if defined(RL_LIB_USE_DOUBLE)
#if defined(__aarch64__)
#define RL_SIMD_BACKEND RL_SIMD_NEON
#elif (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#define RL_SIMD_BACKEND RL_SIMD_AVX2
#else
#define RL_SIMD_BACKEND RL_SIMD_NONE
#endif
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#define RL_SIMD_BACKEND RL_SIMD_NEON
#elif (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#define RL_SIMD_BACKEND RL_SIMD_AVX2
#else
#define RL_SIMD_BACKEND RL_SIMD_NONE
#endif
#endif

#if RL_SIMD_BACKEND == RL_SIMD_NEON && defined(RL_LIB_USE_DOUBLE) && \
    !defined(__aarch64__)
#error "64-bit NEON kernels need AArch64; use RL_SIMD_BACKEND=RL_SIMD_NONE"
#endif

// Human-readable backend name ("scalar", "neon", or "avx2").
const char *simd_backend_name(void);

// True when a vectorised backend is compiled in *and* usable on this CPU (on
// x86-64 this performs a cached runtime AVX2/FMA probe).
b32 simd_accelerated(void);

#endif // SIMD_H
