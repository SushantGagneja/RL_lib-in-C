#include "random.h"

// SplitMix64: expands a 64-bit seed into well-mixed 64-bit words. Used only to
// fill the xoshiro state, never to produce output directly.
static u64 splitmix64(u64 *x) {
  u64 z = (*x += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30u)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27u)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31u);
}

static u64 rotl64(u64 x, u32 k) { return (x << k) | (x >> (64u - k)); }

static void xoshiro_seed(Rng *rng, u64 seed) {
  u64 x = seed;
  for (u32 i = 0; i < 4u; i++) {
    rng->lane[i] = splitmix64(&x);
  }
}

void rng_seed(Rng *rng, u64 seed, u64 stream) {
  // Fold the stream tag into the seed so unrelated streams do not correlate.
  xoshiro_seed(rng, seed ^ (stream * 0x9E3779B97F4A7C15ULL));
}

u64 rng_u64(Rng *rng) {
  u64 *s = rng->lane;
  const u64 result = rotl64(s[1] * 5u, 7u) * 9u;
  const u64 t = s[1] << 17u;
  s[2] ^= s[0];
  s[3] ^= s[1];
  s[1] ^= s[2];
  s[0] ^= s[3];
  s[2] ^= t;
  s[3] = rotl64(s[3], 45u);
  return result;
}

u32 rng_u32(Rng *rng) { return (u32)(rng_u64(rng) >> 32u); }

u32 rng_below(Rng *rng, u32 range) {
  if (range == 0u) {
    return 0u;
  }
  u64 product = (u64)rng_u32(rng) * (u64)range;
  u32 low = (u32)product;
  if (low < range) {
    const u32 threshold = (u32)(0u - range) % range;
    while (low < threshold) {
      product = (u64)rng_u32(rng) * (u64)range;
      low = (u32)product;
    }
  }
  return (u32)(product >> 32u);
}

real rng_unit(Rng *rng) {
  return (real)(rng_u64(rng) >> 11u) * (real)(1.0 / 9007199254740992.0);
}

// Module-wide generator used by the convenience *_g entry points.
static Rng s_rng = {{0x853C49E6748FEA9BULL, 0xDA3E39CB94B95BDBULL,
                     0x1BADB002C0FFEE00ULL, 0x0DDC0FFEE0ADD1E5ULL}};

void rng_seed_g(u64 seed, u64 stream) { rng_seed(&s_rng, seed, stream); }
u32 rng_u32_g(void) { return rng_u32(&s_rng); }
u32 rng_below_g(u32 range) { return rng_below(&s_rng, range); }
real rng_unit_g(void) { return rng_unit(&s_rng); }
