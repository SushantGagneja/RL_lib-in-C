// hello_rl.c — the smallest end-to-end program you can write against an
// installed copy of the library: build an actor-critic for CartPole, take a few
// PPO updates, save and reload the policy, then score it.
//
// The `install-smoke` CI job builds exactly this file against the installed
// headers and `librl_lib` (via pkg-config) to prove the packaged library works
// outside the source tree.
#include "checkpoint.h"
#include "envs/cartpole.h"
#include "layers.h"
#include "memory.h"
#include "rl.h"
#include "version.h"

#include <stdio.h>

int main(void) {
  printf("RL_lib-in-C %s — hello_rl\n", RL_LIB_VERSION_STRING);

  Arena *arena = arena_new(MiB(64));
  Rng rng;
  rng_seed(&rng, 1234, 0);

  // The batched training graph is built for a fixed minibatch size, so
  // TrainCfg.batch_size has to match the batch_size passed to the builder.
  const u32 batch_size = 64;

  Env *env = env_cartpole_create(arena);
  Net *model = net_new(arena);
  const u32 hidden[2] = {16, 16};
  actor_critic_build(arena, model, &rng, env->obs_dim, hidden, 2, env->act_dim,
                     batch_size);

  const TrainCfg cfg = {
      .epochs = 5,
      .rollout_size = 512,
      .max_steps = 500,
      .learning_rate = 3e-4f,
      .gamma = 0.99f,
      .lambda = 0.95f,
      .vf_coef = 0.5f,
      .entropy_coef = 0.01f,
      .seed = 7,
      .verbose = 0,
      .optim = OPT_ADAM,
      .max_grad_norm = 0.5f,
      .batch_size = batch_size,
      .num_envs = 4,
      .ppo_epochs = 4,
      .ppo_clip = 0.2f,
  };
  train_ppo(env, model, &cfg, arena);

  const char *path = "hello_rl.ckpt";
  RL_LIB_CHECK(checkpoint_save(model, NULL, path));
  RL_LIB_CHECK(checkpoint_load(model, NULL, path));

  const EvalResult result = evaluate(env, model, &rng, 20, true);
  printf("greedy CartPole return over 20 episodes: %.1f\n",
         (double)result.avg_return);
  printf("policy saved to %s\n", path);

  remove(path);
  arena_dispose(arena);
  return 0;
}
