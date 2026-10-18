# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) and the API only
breaks on a major bump. `RL_LIB_VERSION_STRING` in `include/version.h` is the
single source of truth — it is also stamped into `rl_lib.pc`.

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
  `rl_lib.lib`**, so one silently replaced the other in the build tree and in
  the install prefix, and a consumer that linked the survivor needed
  `rl_lib.dll` without being told (it fails at load time, not link time). The
  import library is `rl_lib_shared.lib` now; the DLL keeps its name. The CI
  Windows job builds and runs a `find_package` consumer against both targets.

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
