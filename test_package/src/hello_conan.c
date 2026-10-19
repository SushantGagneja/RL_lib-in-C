// The Conan test package's program: only installed headers and the imported
// target, no build-tree paths. It steps CartPole once and draws from the
// global RNG so a link against the library that is missing either the
// environment code or the PRNG fails here.
#include "envs/cartpole.h"
#include "memory.h"
#include "random.h"
#include "version.h"

#include <stdio.h>

int main(void) {
  printf("RL_lib-in-C %s (Conan test package)\n", RL_LIB_VERSION_STRING);

  Arena *arena = arena_new(MiB(16));
  Env *env = env_cartpole_create(arena);

  b32 done = 0;
  b32 truncated = 0;
  env->reset(env, 1234u);
  const real reward = env->step(env, 1u, &done, &truncated);

  rng_seed_g(1234u, 7u);
  printf("CartPole: obs_dim=%u act_dim=%u step reward=%.3f rng=%u\n",
         env->obs_dim, env->act_dim, (double)reward, rng_u32_g());

  arena_dispose(arena);
  return 0;
}
