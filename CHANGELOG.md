# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) and the API only
breaks on a major bump. `RL_LIB_VERSION_STRING` in `include/version.h` is the
single source of truth — it is also stamped into `rl_lib.pc`.

## Unreleased

### Added
- **Soft actor-critic (`sac.h`).** Off-policy, maximum-entropy SAC for continuous
  action spaces: `sac_actor_build` (mean head + learnable `log_std`, no value
  head), `sac_q_build` (a critic whose first layer is split into observation and
  action weights, so the actor's cost can run through a graph-computed action
  without a concatenation op), `SacReplay` (a continuous replay ring with
  terminal and truncated flags kept apart), `sac_replay_gather`,
  `sac_soft_update` and `sac_td_target`. `train_sac` regresses twin critics onto
  a truncation-aware target built from the target critics and the next action's
  log-density, ascends the entropy-adjusted value through one reverse sweep of
  the actor cost, optionally learns the temperature (`auto_alpha`, Adam on a
  one-parameter `log_alpha` against `target_entropy`, default `-act_dim`), and
  soft-updates both targets each step. New `TrainCfg` knobs: `tau`, `alpha`,
  `auto_alpha`, `alpha_lr`, `target_entropy`; the actor/critic builders record
  the new `NET_ARCH_SAC_ACTOR` / `NET_ARCH_SAC_Q` kinds, so checkpoints are gated
  by architecture as for every other builder. `examples/train_sac.c`
  (`make sac`) solves Pendulum-v1; `test_sac` adds 77 checks (replay, gather,
  soft update, TD rule, builders, the actor cost against the fused kernels in
  value and gradient, checkpoint round trips, determinism and a Pendulum
  learning run).
- **General autograd shape ops.** `tensor_reshape`, `tensor_broadcast`,
  `tensor_sum` / `tensor_mean` (over rows, columns, or everything),
  `tensor_gather` (per-column selection with a scatter-add backward), and
  elementwise `tensor_min` / `tensor_max` / `tensor_clip` join the graph, each
  with an analytic backward pass and a finite-difference gradcheck.
  `test_gradcheck` grows from 12 to 16 suites (373 parameters), and `test_mat`
  gains direct edge-case coverage of the kernels (invalid shapes, out-of-range
  indices, gradient masks and accumulation).
- **Graph-based PPO losses.** The discrete and continuous PPO trainers now build
  their clipped surrogate, entropy bonus and value loss as tensor graphs and
  take one reverse sweep from a single scalar cost, instead of hand-seeding
  gradients on intermediate activations; `log_std` becomes an ordinary
  parameter inside the graph rather than a gradient accumulated by hand. New
  ops back this: stable `tensor_log_softmax`, `tensor_log_sech2` (finite in the
  tails where forming `tanh(u)^2` underflows), a straight-through
  `tensor_clip_straight` for the Gaussian mean bound, and reusable
  `tensor_gaussian_logp` / `tensor_gaussian_entropy` builders whose forward
  values are compared against the fused kernels and whose backward passes are
  finite-differenced. `test_gradcheck` grows from 16 to 19 suites (424
  parameters); training behaviour and the public API are unchanged.
- **Checkpoint format 4.** Builder-made models now record their architecture —
  kind, observation/action dimensions and hidden widths — in an ARCH block, and
  `checkpoint_save_ex` / `checkpoint_load_ex` carry a run's running
  observation-normalisation and reward-scaling statistics in a NORM block, so a
  normalised PPO run resumes with the statistics it trained with instead of
  restarting them. Loads validate the architecture and the statistics width
  before touching a parameter, so a mismatched model, a hand-built graph, or a
  file from v1–v3 is rejected with a clear message. The plain
  `checkpoint_save` / `checkpoint_load` pair is unchanged.
- **Checkpoint checksums.** Every file now ends in a CRC-32 trailer that is
  verified before parsing, so bit rot and half-written files are reported as
  corruption instead of turning into a confusing block-size error or silently
  wrong parameters. The format number moved to 4; older files are rejected up
  front, as with every earlier format change.
- **Transactional loads.** The reader walks a file twice — validate, then apply
  — so a load that fails *after* a block parsed cleanly can no longer leave the
  model half-updated. `test_checkpoint` grew from 61 to 616 checks (default
  build) with a 300-case mutation fuzz pass (bit flips, truncations, overwritten
  runs, a third of them with a recomputed checksum) asserting the no-crash and
  no-half-load invariants; the sanitizer job runs it under ASAN/UBSAN.
- **CI coverage and benchmark jobs.** The suites are also built with gcov
  (`make COVERAGE=1 test`) and summarised by `gcovr`, and a benchmark job runs
  the full multi-seed harness and uploads `bench_results.md` as an artifact.
- **Package-manager recipes.** A Conan 2 `conanfile.py` with a `test_package/`
  that consumes the freshly built package through `find_package(RL_lib-in-C)`
  (checked by a CI job), and a vcpkg reference port in `packaging/vcpkg/`.
- **Truncation-aware DQN transitions.** `replay_add_ex` records terminal and
  truncated steps separately, and `dqn_td_target` exposes the one-step target
  rule for tests and custom off-policy code. `replay_add` keeps its original
  signature as the terminal-only shorthand.
- **Overridable fatal-error handler.** `RL_LIB_ASSERT`/`RL_LIB_CHECK` now format
  a one-line message and hand it to `rl_lib_fatal`, whose handler is settable
  with `rl_lib_set_fatal_handler` (default: print to stderr and abort). An
  embedder can log and `longjmp` instead of losing the process; recoverable
  failures still return `false`. `test_base` covers the hook and the base
  helpers.

### Changed
- **Repository layout and contributor docs.** The multi-seed benchmark harness
  moved from `examples/bench.c` to `benchmarks/bench.c`; the `make bench*`
  targets are unchanged. Added `CONTRIBUTING.md` and `.editorconfig`, cleaned
  up `.gitignore`, and gave the README a contents list, badges and an updated
  tree. No library or API changes.

### Fixed
- DQN treated a time-limit truncation as a terminal transition, so the bootstrap
  was dropped exactly when CartPole reached its 500-step cap. Truncated
  transitions now bootstrap through `next_obs`, matching the truncation-aware
  GAE the policy-gradient trainers already use; `test_dqn` checks the target
  rule directly.
- The CartPole and Pendulum physics used `f`-suffixed literals even in the f64
  build, so constants such as 9.8 were rounded to float precision before being
  promoted to double. They now go through `REAL_C`, which is bit-identical for
  f32 and full precision for f64.
- Checkpoint saves overwrote the destination in place, so a crash or a failed
  write could destroy the previous checkpoint. Saves now write `<path>.tmp`,
  flush and sync it, and rename it over the destination; a failure leaves the
  old checkpoint untouched and removes the temporary.
- `make clean` did not remove `train_dqn` or its object file, and `.gitignore`
  did not cover the binary.
- A rejected `net_load`/`adam_load`/`checkpoint_load` could previously leave a
  model half-updated when the failure came from a block *after* the parameter
  block; loads are now validated end-to-end before anything is applied.
- The repository no longer tracks coverage instrumentation produced by a local
  `make COVERAGE=1 test` run. Those stale `.gcda` files broke the coverage CI
  job (a fresh GCC checkout refuses to merge counters written by Apple/LLVM
  clang, and `gcovr` then fails), so `.gitignore` now covers `*.gcda`/`*.gcno`
  output along with Python bytecode.

## 1.1.0 — 2026-10-19

### Added
- **Training recipe knobs.** `TrainCfg.lr_schedule`/`lr_end` decay the learning
  rate from `learning_rate` to `lr_end` over the run (linear or cosine), and
  `anneal_entropy`/`entropy_coef_end` do the same for the entropy bonus; every
  trainer honours them and refreshes Adam's stored rate each update. The PPO
  trainers additionally take `vf_clip` (clipped value loss against the
  rollout's own value estimate, `mat_mse_clip_add_grad`) and `target_kl`
  (trust-region early stop once the mean approximate KL — the k3 estimator —
  exceeds 1.5x it).
- **Observation normalization and reward scaling** for the PPO trainers
  (`TrainCfg.norm_obs`, `norm_reward`, `norm_clip`, `norm_reward_clip`,
  `norm`). The statistics are Welford-updated from the trainers' own rollout
  stream — folding happens in the serial collection loop, so worker threads
  never touch them and threaded stepping stays bit-identical to serial. Both
  transforms start from a standard-normal prior, so the first observations and
  returns pass through nearly unchanged. `NormStats` survives the run for
  evaluation: the new `evaluate_normed`/`evaluate_seeded_normed` (and their
  `_continuous` twins) apply the frozen observation transform and still report
  raw environment returns.
- `train_lr_at`/`train_entropy_coef_at` expose the trainers' schedules so logs
  and user code cannot drift from what the optimiser actually uses.
- **The five-seed Snake PPO protocol was re-run on the same binary** with the
  full recipe: **266.72 ± 10.12 greedy** (264.24 ± 12.83 stochastic) against
  87.90 ± 19.18 / 108.16 ± 13.62 for the original recipe, which `recipe 0`
  reproduces exactly — a 3.0x improvement, within 11% of the BFS planner, and
  no late-run collapse (seed 42 at 6000 updates: 266.6, where the original
  recipe fell to 54.1). README's Snake section carries the updated table and
  prose.
- `tests/test_rl.c`: schedule values, normalization statistics
  (hand-computed Welford/posterior/clip cases), knob-effect checks, and an
  end-to-end normalized-training suite (73 -> 103 checks).
- `examples/train_ppo_snake.c`: a `recipe` argument. Recipe 0 reproduces the
  1.0.0 Snake numbers bit for bit (verified against the previous binary);
  recipe 1 (the new default) enables the full recipe: cosine LR 3e-4 -> 3e-5,
  value clip 0.2, KL early stop 0.03, entropy 0.01 -> 0.001, observation
  normalization and reward scaling.
- **DQN** (`dqn.h`, `src/dqn.c`): off-policy deep Q-learning with a fixed-capacity
  ring replay buffer, a hard-synced target network, an epsilon schedule, and the
  DQN regression expressed as a *masked MSE* on the existing autograd engine
  (untaken actions keep the network's own value, so their error — and gradient —
  is exactly zero; no gather/scatter op needed). `q_net_build` in `layers.h`
  builds the Q-network's rollout and batched graphs, including the gradient-free
  `target` tensor. `TrainCfg` gained the DQN knobs (`replay_capacity`,
  `warmup_steps`, `train_every`, `target_sync`, `eps_start`, `eps_end`,
  `eps_decay_steps`). `tests/test_dqn.c` (33 checks: ring overwrite order,
  builder/rebuild, epsilon schedule, target sync, determinism, and a CartPole run
  reaching ~203 greedy return in 16000 steps), `examples/train_dqn.c` and
  `make dqn`.
- **New graph ops**: elementwise `sigmoid`, `exp`, `log` and `div`
  (`tensor_sigmoid`, `tensor_exp`, `tensor_log`, `tensor_div`, with matching
  `mat_*` forward and gradient kernels). The numerical gradient check grew from
  11 suites / 260 parameters to 12 suites / 280 parameters, green in f32 and
  f64.
- **Optional system BLAS** (`RL_LIB_USE_BLAS`, the Makefile's `BLAS=1`):
  `mat_mm` then goes through CBLAS for all four transpose cases — Accelerate on
  macOS, OpenBLAS or any CBLAS elsewhere — replacing the SIMD and scalar paths
  while `mat_mm_scalar` stays the reference the math suite cross-checks against.
  CMake gains an `RL_LIB_USE_BLAS` option, and a CI job builds and runs every
  suite against OpenBLAS.

### Changed
- The PPO trainers now take their learning rate and entropy coefficient from
  the schedules on every update. The defaults are constant, so pre-existing
  configurations train bit-identically to 1.0.0.

## 1.0.0 — 2026-10-18

First release. Everything below landed while preparing it, so this entry is the
whole history: the project was renamed to **RL_lib-in-C**, licensed under MIT,
independently reimplemented, and carried through the Phase 0–7 roadmap.

### Project renamed
- The project is now **RL_lib-in-C**: `CRL_*` macros and `[cRL ...]` messages
  became `RL_LIB_*` / `[RL_lib-in-C ...]`, and the benchmark binary is
  `rl_lib_bench`.
- Added an **MIT** `LICENSE`.

### Rewritten (independence pass)
- The engine core was reimplemented from scratch:
  - `graph.c` / `graph.h` — the autograd graph no longer uses per-op function
    pointer tables. Each node carries an `OpKind` tag and one `switch`
    interprets forward and backward. The topological build is a fresh iterative
    post-order walk.
  - `linalg.c` / `linalg.h` — row-pointer kernels with hoisted scalars; NN/NT/
    TN/TT matrix multiply, activations, losses, and their gradients re-derived
    (numerically identical: the f64 gradient check still passes, 260/260).
  - `memory.c` / `memory.h` — bump allocator reimplemented (`base`/`cap`/
    `offset`).
  - `random.c` / `random.h` — PCG32 replaced by **xoshiro256\*\*** seeded with
    SplitMix64 (public domain / CC0); unbiased `rng_below` via Lemire rejection.
- Types and files renamed to one consistent surface: `Arena`, `Mat`, `Tensor`,
  `Schedule`, `Net`, `Rng`, `Dense`, `Adam`, with `memory` / `random` /
  `linalg` / `graph` / `layers` headers. Public functions follow the same scheme
  (`arena_new`, `mat_new`, `mat_mm`, `schedule_from`, `schedule_grad`,
  `dense_new`, `train_a2c`, ...).

### Fixed
- **A2C build crash and dead gradients.** `actor_critic_build` no longer builds
  a fake `logits + value` root; `schedule_from_roots` builds one graph from
  multiple roots. Softmax is part of the forward graph, seeded gradients are no
  longer zeroed before backprop (`schedule_grad_from` clears activations first,
  then the caller seeds), and A2C accumulates over all minibatches and takes a
  single optimizer step per rollout.
- **A2C seeding/entropy.** Removed the double division by batch size; added an
  entropy bonus (`mat_entropy_loss` / `mat_entropy_add_grad`,
  `TrainCfg.entropy_coef`).
- **Snake.** Moving into the cell the tail vacates this step is now legal; each
  env owns its own `Rng` and `reset` takes a seed; the silent 180-degree reversal
  override is gone; grid size is validated; a win state exists when the board
  fills.
- **Autograd graph build.** Replaced the O(n^2) stack scan with an iterative
  post-order walk plus a visited-state array that asserts instead of silently
  dropping nodes.
- **REINFORCE/A2C batching.** The dropped `sample_count % batch_size` remainder
  is now a random shuffle of the sample order rather than a systematic tail.
- **Numerics.** `sample_action` uses strict `<`; the misleading `randn` name is
  gone (it was always a uniform range); Adam clipping guards against non-finite
  global norms.
- **Non-finite PPO gradients late in a run.** Both PPO surrogates now compute the
  importance ratio as `exp(clamp(new_logp - old_logp, ±20))`. `exp` overflows f32
  at ~88 nats, and a single overflowing sample poisoned a whole minibatch's
  gradient with `Inf` — seen as `clip_grad_norm: non-finite gradient norm` around
  update 800 of a Pendulum run. A ratio past `e^20` is far outside PPO's trust
  region (`ppo_clip` is 0.2), so the clamp only engages where the surrogate's own
  clip is already discounting the sample.
- **Failures are loud.** `RL_LIB_ASSERT` / `RL_LIB_CHECK` replace silent `false`
  returns, empty-graph no-ops, and unchecked arena pointers; `arena_alloc`
  aborts on OOM.
- **Arena.** 64-byte aligned backing store; `arena_mark` / `arena_rewind` for
  temp scopes.
- **Benchmarks.** Identical evaluation seeds across policies, f64 Welford
  sample standard deviation, both greedy and stochastic returns, best and worst
  seed, a BFS Snake baseline, and machine-generated `bench_results.md`.
- **Warnings.** `-Wall -Wextra -Wshadow -Wconversion -Wmissing-prototypes` are
  clean.
- **`make tsan` could not build on Linux.** The suites are compiled with
  `-fsanitize=undefined`, and because target-specific `CFLAGS` reach the shared
  library objects, `$(TARGET_PPO_CONT)` — which the same target builds — was
  linked against UBSan-instrumented objects without the UBSan runtime
  (`undefined reference to __ubsan_handle_*`), so the job died before running a
  single test. The flag is now the `UBSAN_FLAGS` variable, which the TSan target
  stands down, giving that target one consistent TSan-only configuration. It
  went unnoticed locally because Apple's `libclang_rt.tsan_osx_dynamic.dylib`
  also exports the `__ubsan_*` handlers (56 of them) and GCC's runtime does not.
- **On MSVC the static archive and the shared library's import library were both
  `rl_lib.lib`**, so one silently replaced the other in the build tree, and a
  consumer that linked the survivor needed `rl_lib.dll` without being told (that
  fails at load time, not link time — the first Windows CI runs died with exit
  code 127 on the consumer's last line). On top of that the shared target had no
  `ARCHIVE` install destination at all, so its import library was never
  installed and the exported `RL_lib-in-C::rl_lib_shared` carried no
  `IMPORTED_IMPLIB`, which makes a CMake consumer fail at configure time. The
  import library is `rl_lib_shared.lib` and it is installed now; the DLL keeps
  its name. The CI Windows job builds and runs a `find_package` consumer against
  both targets.

### Added
- **Checkpointing** (`checkpoint.c` / `checkpoint.h`): `net_save`, `net_load`,
  `adam_save`, `adam_load`, `checkpoint_save`, `checkpoint_load`, and
  `checkpoint_last_error`. A checkpoint is a versioned, endian-independent,
  block-structured binary file holding the parameters and the Adam moments, so a
  run can be stopped and resumed. Loads validate the scalar width, parameter
  count, payload size and every shape, and stage the payload before writing, so a
  rejected file leaves the model untouched. Covered by `tests/test_checkpoint.c`
  (61 checks), including a bit-exact Adam-state resume and a continuous-model round
  trip that covers the learnable `log_std`.
- **Packaging**: `make lib` builds `librl_lib.a` and a versioned shared library
  (`librl_lib.<major>.dylib` / `librl_lib.so.<version>` with a soname); `make
  install PREFIX=...` installs headers into `include/rl_lib/`, both libraries and
  a generated `rl_lib.pc`; `make uninstall` reverses it. A CI job installs into a
  scratch prefix and builds `examples/hello_rl.c` against only the installed
  headers and library.
- **`include/version.h`** with `RL_LIB_VERSION_MAJOR/MINOR/PATCH/STRING`, read by
  the Makefile when it stamps `rl_lib.pc`.
- **`docs/API.md`**: a complete reference for every public header, with the
  ownership, error-handling and precision conventions stated up front.
- **`examples/hello_rl.c`**: the smallest end-to-end program (build, train, save,
  load, evaluate), used as the packaging smoke test, plus `make hello`.
- **`make test-double`**: builds the f64 configuration into its own object tree
  and runs *every* suite in double precision, not just the gradient check.
- **SIMD MatMul** (`simd.c` / `simd.h` plus `linalg.c`): NEON and AVX2+FMA
  kernels for all four transpose cases in both precisions (4-wide f32 / 2-wide
  f64 on NEON, 8-wide f32 / 4-wide f64 on AVX2), a portable `mat_mm_scalar`
  reference, a compile-time backend selector (`-DRL_SIMD_BACKEND=0` forces
  scalar), and a runtime AVX2/FMA probe on x86-64. The NN/TN kernels process two
  output rows at once and keep the scalar reduction order, so they are
  element-wise identical to a single-row kernel while every element of B is
  loaded once for two rows instead of once per row. That roughly doubled the
  measured throughput: 2.1× at 256² and 2.6× at the 64×64×64 shape an RL layer
  uses over the auto-vectorized scalar loop (≈33/39 GFLOPS f32, ≈17/20 GFLOPS
  f64), against +11%/+67% for the single-row kernel before it. TT transposes a
  32×32 tile into the TN path instead of falling back to scalar. `test_mat`
  cross-checks every transpose case in both precisions, `make bench-throughput`
  measures the f32 kernels, `make bench-double` the f64 ones, and a CI job
  builds the scalar backend end to end.
- **Vectorized environments** (`vecenv.c` / `vecenv.h`): `VecEnv` steps N
  cloned envs in lockstep, auto-resets finished ones in place, and preserves
  each terminal observation (`final_obs`) for correct value bootstrapping.
  `Env` gained a `clone(arena, src)` function pointer.
- **PPO** (`train_ppo`, `mat_ppo_policy_loss` / `mat_ppo_policy_add_grad`): the
  clipped surrogate objective with multi-epoch shuffled minibatches, per-env
  GAE-λ, and truncation-aware bootstrapping. Covered by a new f64/f32 gradient
  check, CartPole learning and same-seed determinism tests, and
  `examples/train_ppo.c`.
- **Continuous actions** (`mat_gaussian_*` kernels, `envs/pendulum.c`,
  `train_ppo_continuous`): `continuous_actor_critic_build` adds a mean head plus a
  *learnable* `log_std` parameter (so the optimiser, the gradient clip and the
  checkpoint writer all cover it); actions are tanh-squashed into agent-supplied
  per-dimension bounds and the density carries the squash's log-determinant. The
  density is evaluated at the sampled pre-squash coordinate `u` — which
  `mat_gaussian_sample` returns next to the action — rather than recovered from
  the action with `atanh`, which is infinite at the bounds and, once a mean
  drifted outside the box, described samples that were never drawn. `Env` gained
  `continuous`, `action_min` / `action_max` and `step_continuous`; `VecEnv` gained
  `vecenv_step_continuous` / `vecenv_action_bounds`; Pendulum-v1 implements the
  Gymnasium dynamics (dt 0.05, torque ±2, 200-step truncation); and
  `train_ppo_continuous` / `evaluate_continuous[_seeded]` reuse the discrete
  trainer's rollout, GAE and minibatch machinery. Covered by new Pendulum and
  continuous-`VecEnv` tests, `log_std` gradient checks in f32 and f64, a
  log-density check against the textbook formula in double precision, a learning
  smoke test, a checkpoint round trip, and `make ppo-continuous`.
- `make ppo`, `make ppo-continuous`, `EXTRA_CFLAGS` Makefile support, and a
  scalar-backend CI job.
- **Threaded `VecEnv`** (`vecenv_set_threads`, `vecenv_threads`,
  `vecenv_shutdown`, `TrainCfg.num_threads`): a reusable worker pool spreads the
  per-env steps over threads and the calling thread does a slice too. The
  synchronisation is a barrier built from a mutex and a condition variable,
  because `pthread_barrier_t` is optional in POSIX and absent on macOS; the pool
  is sized for the workers that actually started, so a failed spawn degrades to
  fewer threads instead of deadlocking on a barrier nobody reaches. Stepping is
  bit-identical to the serial path — `tests/test_envs.c` compares every buffer
  for 260 steps on CartPole and Pendulum, `tests/test_rl.c` compares the trained
  weights — and it is visible end to end (`./train_ppo_continuous 42 1200 8`).
  `-pthread` is now part of the build, `THREAD_FLAGS=` plus
  `-DRL_LIB_NO_THREADS` removes it, and `rl_lib.pc` carries the flag — now via
  the `@LIBS_EXTRA@` token the CMake project stamps too, so a no-threads build
  no longer advertises `-pthread`. The
  benchmark measures the crossover with a synthetic compute-bound env:
  1.92×/3.2×/2.7× on 2/4/8 threads against the serial loop.
- `make gradcheck-double`: builds the library in f64 (`-DRL_LIB_USE_DOUBLE`, the
  `real` scalar typedef) so the gradient check runs with `h = 1e-5` and a
  `1e-6` tolerance. Input/activation gradients and an elementwise `mul` with
  both operands requiring grad are checked too.
- `tests/test_rl.c` (73 checks): GAE against a hand-computed trajectory, a single
  Adam step, gradient-norm clipping, actor-critic and continuous actor-critic
  construction, value-head learning, batched-vs-per-sample gradient equivalence,
  a parameter registry grown past 128 entries, rebuilding the batched graph for a
  new minibatch size, CartPole learning smoke tests for REINFORCE and PPO with
  same-seed determinism, a Pendulum continuous-PPO learning smoke test, and
  threaded rollouts reaching bit-identical weights in both trainers.
- `tests/test_envs.c` (191 checks) grew threaded-vs-serial stepping, compared
  field by field over 260 steps for a discrete and a continuous vecenv of 7 envs
  on 3 threads.
- `make asan` (ASAN + UBSAN) and `make tsan` (ThreadSanitizer over the threaded
  `VecEnv` suites), plus a GitHub Actions workflow running `make test`
  with warnings as errors on gcc/clang/macOS, the sanitizer targets, a
  scalar-backend build, the f64 suites, and an install-and-consume smoke test.
- `examples/train_a2c.c` and `examples/train_ppo.c`.
- `examples/train_ppo_snake.c` (`make ppo-snake`): 6x6 Snake with PPO over 8
  vectorized environments, plus an optional potential-based shaping wrapper
  (`r' = r + c(γΦ' − Φ)`, `Φ = -manhattan(head, food)`) that doubles as a worked
  example of composing environments — it clones its inner env and forwards
  `observe()`. Evaluation runs on the benchmark harness's seed list, so the
  README table is directly comparable with the other Snake baselines: 87.9 ± 19.2
  greedy over five seeds, against 39.7 for REINFORCE and 298.7 for the BFS
  planner. The example's budget check and shaping control are recorded honestly
  in the README rather than smoothed over: 4× the updates still misses parity and
  collapses in the last few hundred updates, and shaping affects that late-run
  stability but not the 1500-update score.
- **A CMake project** (`CMakeLists.txt`, `cmake/`), which is what Windows uses:
  static + shared libraries, every example, every suite, `ctest`, the same
  install layout as `make install` plus a `find_package(RL_lib-in-C)` package
  config, and a pkg-config file with a relocatable `${pcfiledir}` prefix. Its
  options mirror the Makefile knobs (`RL_LIB_USE_DOUBLE`, `RL_LIB_NO_THREADS`,
  `RL_SIMD_BACKEND`, `RL_LIB_BUILD_TESTS`, `RL_LIB_BUILD_EXAMPLES`).
- **Windows support.** The threaded `VecEnv` pool has a Win32 backend now
  (`CreateThread`, `CRITICAL_SECTION`, `CONDITION_VARIABLE` — the same pool and
  the same sense-reversing barrier, so stepping stays bit-identical),
  `arena_new` uses `_aligned_malloc`/`_aligned_free` where there is no
  `posix_memalign`, and the benchmark's wall clock is `QueryPerformanceCounter`.
  MSVC builds select the scalar SIMD reference, because it cannot compile the
  `target`-attributed AVX2 kernels; MinGW and clang-cl get AVX2.
- `RL_LIB_HAS_THREADS` (`vecenv.h`): 1 when a threading backend was compiled in.
  The threaded comparisons in `test_envs` / `test_rl` skip themselves — and say
  so — when it is 0, so the documented `-DRL_LIB_NO_THREADS` build now passes its
  suites instead of aborting on `vecenv_set_threads`.
- `make examples` compiles every example and the benchmark with the current
  flags; CI runs it with `ERR=1`, so example code is held to the same warning
  standard as the library. (The test binaries cannot do that: their
  target-specific `CFLAGS` give the shared library objects
  `-fsanitize=undefined`.)
- CI: a `windows-latest` job (MSVC configure / build / `ctest` / install) and a
  serial-only job for the no-threads configuration.

### Changed
- **`MAX_PARAMS` is gone.** The parameter registry (`Net.params`) now doubles
  through the arena, so models are no longer capped at 128 parameters.
  `MAX_ARITY` stays at 2, which is a property of the op set (every operation is
  unary or binary) rather than a capacity limit.
- **The minibatch size is a runtime knob.** `net_set_batch_size` /
  `net_batch_size` rebuild the batched subgraph from the architecture the
  builders now record, sharing the existing parameter tensors, and the trainers
  apply `TrainCfg.batch_size` before sizing anything from the graph.
- **Auto-reset seeds are per sub-environment.** `VecEnv` no longer carries one
  shared `Rng` that handed out reset seeds in stepping order — a sub-environment's
  trajectory then depended on how many of its siblings reset first, which both
  serialised the reset path and made parallel stepping change the results. Each
  env now keeps its own stream (`reset_base` + `reset_counts`, seeded by
  `vecenv_reset[_seeds]` and mixed once per auto-reset), so trajectories are
  order-independent, which is exactly the property the threaded pool relies on.
  Runs seeded before this change produce slightly different trajectories: the
  Pendulum table in the README moved by a few points, into the same regime.
- Removed the short-name macros from the linear-algebra header; everything uses
  the `mat_*` names.
- Extracted a `Dense` struct (`dense_new` / `dense_apply`) and deleted the three
  duplicated layer/init code paths and the weight-sharing hack.
- The matrix-multiply kernels are `static` (`mm_nn` / `mm_nt` / `mm_tn` /
  `mm_tt`); `mat_mm` dispatches to the SIMD kernels when a backend is available.
- The benchmark reports the active SIMD backend and precision, two MatMul shapes
  (large and RL-sized), `VecEnv` throughput, and the threaded `VecEnv` scaling on
  a synthetic compute-bound env. The scalar sections keep using CPU time
  (`clock()`); the threaded section uses a monotonic wall clock, because a
  speedup is invisible in CPU time by construction.

### Known limitations
See the "Known limitations" section of the README.

### License
MIT — see [`LICENSE`](LICENSE). The engine is an original implementation and
there is no bundled third-party code (the PRNG is public-domain xoshiro256\*\*).
