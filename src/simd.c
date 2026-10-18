#include "simd.h"

const char *simd_backend_name(void) {
#if RL_SIMD_BACKEND == RL_SIMD_NEON
  return "neon";
#elif RL_SIMD_BACKEND == RL_SIMD_AVX2
  return "avx2";
#else
  return "scalar";
#endif
}

b32 simd_accelerated(void) {
#if RL_SIMD_BACKEND == RL_SIMD_NEON
  return true;
#elif RL_SIMD_BACKEND == RL_SIMD_AVX2
  // Cached because mat_mm() consults this on every call. Benign data race: the
  // value written is the same on every thread.
  static int cached = -1;
  if (cached < 0) {
    cached = (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
                 ? 1
                 : 0;
  }
  return cached != 0;
#else
  return false;
#endif
}
