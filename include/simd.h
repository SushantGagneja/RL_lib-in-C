#ifndef SIMD_H
#define SIMD_H

#include "base.h"

// Compile-time selection of the vectorised matrix-multiply backend.
//
//   RL_SIMD_NEON  — 4-wide f32 NEON. Baseline on AArch64, always available.
//   RL_SIMD_AVX2  — 8-wide f32 AVX2+FMA on x86-64, guarded by a runtime
//                   __builtin_cpu_supports() probe (the code is compiled with
//                   per-function target attributes, so the binary still runs on
//                   CPUs without AVX2).
//   RL_SIMD_NONE  — portable scalar fallback.
//
// The vector kernels are written for the default f32 build only; the f64 build
// (-DRL_LIB_USE_DOUBLE) uses the scalar reference, since it exists for the
// gradient check rather than for speed. Define RL_SIMD_BACKEND=RL_SIMD_NONE
// (i.e. -DRL_SIMD_BACKEND=0) to force the scalar path on any build; the test
// suite uses mat_mm_scalar() to cross-check the two implementations.

#define RL_SIMD_NONE 0
#define RL_SIMD_NEON 1
#define RL_SIMD_AVX2 2

#ifndef RL_SIMD_BACKEND
#if defined(RL_LIB_USE_DOUBLE)
#define RL_SIMD_BACKEND RL_SIMD_NONE
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#define RL_SIMD_BACKEND RL_SIMD_NEON
#elif (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#define RL_SIMD_BACKEND RL_SIMD_AVX2
#else
#define RL_SIMD_BACKEND RL_SIMD_NONE
#endif
#endif

#if RL_SIMD_BACKEND != RL_SIMD_NONE && defined(RL_LIB_USE_DOUBLE)
#error "The SIMD kernels are f32-only; build the f64 configuration with RL_SIMD_BACKEND=RL_SIMD_NONE"
#endif

// Human-readable backend name ("scalar", "neon", or "avx2").
const char *simd_backend_name(void);

// True when a vectorised backend is compiled in *and* usable on this CPU (on
// x86-64 this performs a cached runtime AVX2/FMA probe).
b32 simd_accelerated(void);

#endif // SIMD_H
