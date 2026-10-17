#ifndef PRNG_H
#define PRNG_H

#include "base.h"

// Based on pcg random number generator (https://www.pcg-random.org)
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

typedef struct {
  u64 state;
  u64 inc;
} prng_state;

void prng_seed_r(prng_state *rng, u64 initstate, u64 initseq);
void prng_seed(u64 initstate, u64 initseq);

u32 prng_rand_r(prng_state *rng);
u32 prng_rand(void);

// Uniform integer in [0, range). Despite the old name it was never a normal
// distribution; renamed to say what it does.
u32 prng_range_r(prng_state *rng, u32 range);
u32 prng_range(u32 range);

real prng_randf_r(prng_state *rng);
real prng_randf(void);

#endif
