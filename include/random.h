#ifndef RANDOM_H
#define RANDOM_H

#include "base.h"

// Pseudo-random numbers via xoshiro256** (Blackman & Vigna, released to the
// public domain / CC0). Seeds are expanded with SplitMix64. No third-party
// license obligations.
//
// Rng holds 256 bits of state and must not be all-zero. Use rng_seed() to
// initialise it from a 64-bit seed plus an independent 64-bit stream tag
// (so several Rng instances can share a seed and still diverge).

typedef struct {
  u64 lane[4];
} Rng;

void rng_seed(Rng *rng, u64 seed, u64 stream);
void rng_seed_g(u64 seed, u64 stream);

u64 rng_u64(Rng *rng);

u32 rng_u32(Rng *rng);
u32 rng_u32_g(void);

// Uniform integer in [0, range). Unbiased (Lemire rejection). range == 0
// returns 0.
u32 rng_below(Rng *rng, u32 range);
u32 rng_below_g(u32 range);

// Uniform real in [0, 1) using the top 53 bits.
real rng_unit(Rng *rng);
real rng_unit_g(void);

#endif
