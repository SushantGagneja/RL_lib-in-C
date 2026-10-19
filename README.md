# RL_lib-in-C: Compact Reinforcement Learning & Autograd Library in C

A small, readable C library for reverse-mode autograd, neural networks, and
reinforcement learning. No third-party dependencies — only the C standard
library, `libm`, and either POSIX `make` or CMake.

It is a learning project: the library sources are small enough to read end to
end (the tests and examples are the long tail), and the benchmarks below are
reported with their real weaknesses rather than tuned to flatter the
algorithms.

---

## License

MIT — see [`LICENSE`](LICENSE). Copyright (c) 2026 Sushant Gagneja.

The engine here is an original implementation written for this project:

- the arena allocator (`memory.c`),
- the row-major linear algebra (`linalg.c`), and
- the reverse-mode autograd graph (`graph.c`),

are new code. The pseudo-random number generator is **xoshiro256\*\*** with
SplitMix64 seeding (public domain / CC0). There is no bundled third-party code,
so there is nothing to attribute and no external license to carry.

---

## Features

- **Reverse-mode autograd engine**
  - Static computation graph; every node carries an operation tag, and forward
    and backward are interpreted by one `switch` dispatch (no vtable).
  - Ops: matrix multiply (NN/NT/TN/TT), ReLU, Tanh, numerically stable
    per-column Softmax, elementwise Mul, Add, Sub, MSE, REINFORCE loss, a fused
    policy cross-entropy loss with a mean-entropy bonus, and a PPO clipped
    surrogate loss.
  - Gradient accumulation across minibatches; `schedule_from_roots` builds a
    graph for models with more than one output (e.g. actor-critic).
- **Arena allocator** — 64-byte aligned bump allocator for graph nodes,
  matrices, parameters, and rollout buffers. A handful of small scratch buffers
  use `malloc`.
- **Seeded xoshiro256\*\* PRNG** (SplitMix64 seeding, public domain) — isolated
  `Rng` state per environment, weight initializer, and trainer.
- **Discrete actor / actor-critic builders** with a shared `Dense` struct and
  He-uniform initialization (tiny-gain policy head). The builders record the
  architecture, so `net_set_batch_size` can rebuild the batched training graph
  for a new minibatch size without touching the weights — `TrainCfg.batch_size`
  is a runtime knob, not a build-time constant.
- **Continuous (squashed Gaussian) policies** — `continuous_actor_critic_build`
  adds a mean head plus a *learnable* log standard deviation (a first-class
  parameter: the optimiser, gradient clipping and checkpointing all cover it).
  Samples are squashed into the environment's action box with tanh and the
  density carries the squash's log-determinant Jacobian, the construction SAC and
  Stable-Baselines3 use. Per-dimension action bounds are supported. The density
  is evaluated at the sampled pre-squash coordinate (returned by the sampler)
  rather than recovered from the action, which removes the `atanh` singularity at
  the bounds; the mean and `log_std` carry documented clamps so the density
  cannot spike outside the box or divide by a collapsed sigma.
- **Optimizers** — SGD and Adam with bias correction and global gradient-norm
  clipping (with a non-finite guard).
- **Algorithms** — REINFORCE with mean-baseline normalization, Advantage
  Actor-Critic (A2C) with GAE-λ and an entropy bonus, PPO with the clipped
  surrogate objective (multiple epochs of shuffled minibatches, plus optional
  value clipping, approximate-KL early stopping, LR/entropy schedules and
  running observation normalization / reward scaling), continuous PPO on the
  squashed Gaussian policy, and an off-policy **DQN** with experience replay, a
  hard-synced target network and an epsilon-greedy schedule.
- **Environments** — CartPole-like, grid Snake, and Pendulum-v1 behind a common
  `Env` interface (`reset(env, seed)`, `step` or `step_continuous`, `observe`,
  `clone`, plus the action bounds a continuous policy needs).
- **Vectorized environments** (`VecEnv`) — N independent cloned envs stepped in
  lockstep with per-env auto-reset, terminal-observation capture for
  bootstrapping, and deterministic seeding. Used by the PPO trainer.
- **SIMD MatMul kernels** — hand-written NEON (AArch64) and AVX2+FMA (x86-64,
  behind a runtime CPU probe) kernels for all four transpose cases in both
  precisions (4-wide f32 / 2-wide f64 on NEON, 8-wide f32 / 4-wide f64 on
  AVX2), with the portable scalar path as the reference. The f64 build uses
  them too; `-DRL_SIMD_BACKEND=0` forces scalar on any build.
- **Checkpointing** — a versioned, endian-independent binary format that saves
  and restores the parameters *and* the Adam moments, so a training run can be
  stopped and resumed. Every file ends in a CRC-32 trailer that is verified
  before parsing, loading is all-or-nothing (the reader validates the whole file
  before touching a parameter), and saving is atomic: the image is written to
  `<path>.tmp`, synced, and renamed over the destination, so a crash or a failed
  save leaves the previous checkpoint intact.
- **Installable and packageable** — `make install` puts the headers in
  `include/rl_lib/`, builds `librl_lib.a` plus a versioned shared library, and
  writes a `rl_lib.pc` pkg-config file; the CMake project installs the same
  layout (plus a `find_package` config) and is the path on Windows. There are
  package-manager recipes too — a Conan 2 `conanfile.py` (with a test package)
  and a vcpkg port. Either way, there is no build-time dependency beyond the
  compiler, `libm` and the build system itself.

---

## Install as a library

```bash
make install PREFIX=/usr/local     # or any prefix you own
```

That installs:

```
$PREFIX/include/rl_lib/*.h        # + envs/*.h
$PREFIX/lib/librl_lib.a
$PREFIX/lib/librl_lib.<major>.dylib   (Linux: librl_lib.so.<version>)
$PREFIX/lib/librl_lib.dylib           (Linux: librl_lib.so)
$PREFIX/lib/pkgconfig/rl_lib.pc
```

Then link as usual:

```bash
cc -O2 $(pkg-config --cflags rl_lib) my_program.c $(pkg-config --libs rl_lib) -o my_program
```

`PREFIX` is baked into the shared library's install name, so pass the same value
when building and installing (a single `make install PREFIX=...` does both).
`make uninstall PREFIX=...` removes exactly what was installed.

The CMake project installs the same layout plus a package config, so a CMake
consumer can `find_package(RL_lib-in-C)` and link `RL_lib-in-C::rl_lib` — see
[CMake](#cmake-required-on-windows-an-alternative-everywhere) below.

The [`docs/API.md`](docs/API.md) reference documents every public function, and
`examples/hello_rl.c` is a complete program that CI builds against the *installed*
library to prove the packaged copy works outside the source tree. The C ABI is
also all a foreign-function caller needs: `examples/python/ctypes_demo.py` drives
the installed shared library from CPython with nothing but `ctypes` — it runs
CartPole through the `Env` vtable and multiplies two matrices.

#### Conan and vcpkg

`conanfile.py` at the repository root is a Conan 2 recipe: `conan create .`
builds the CMake project inside Conan's toolchain, runs the ctest suites (skip
them with `-c tools.build:skip_test=True`) and packages the install tree above.
`test_package/` then consumes that package through `find_package(RL_lib-in-C)`,
which is how CI checks the recipe. The vcpkg side is a thin reference port in
`packaging/vcpkg/` — the two files vcpkg's `ports/rl-lib-in-c/` carries, wrapping
the same CMake install.

---

## Benchmarks & baselines

All numbers below are produced by `make bench` (5 seeds × 100 episodes per
policy, identical evaluation seeds for every policy, sample `n-1` standard
deviation computed in f64 with Welford's method) and written to
`bench_results.md`. They are from one machine; run `make bench` to reproduce
on yours.

### CartPole-v1 (REINFORCE + Adam, 50 epochs)

| Policy | Return (mean ± sample std) | Best seed | Worst seed |
|---|---|---|---|
| Random | 21.80 ± 0.82 | 22.38 | 20.48 |
| Greedy (PD controller) | 484.62 ± 3.10 | 488.80 | 481.16 |
| **Trained (greedy eval)** | **497.65 ± 2.68** | 500.00 | 494.51 |
| Trained (stochastic eval) | 487.86 ± 23.93 | 500.00 | 445.13 |

Stochastic evaluation is much noisier than greedy, which is expected: greedy
evaluation flatters a policy trained with only 50 updates. The physics run in
`real` (f32 by default) with `theta_threshold = 0.2095`, versus Gymnasium's
`12 * 2π / 360 = 0.20944` in f64, so read this as "CartPole-like", not exact
`CartPole-v1`.

### Snake 6×6 (REINFORCE + Adam 200 epochs; PPO 1500 updates)

| Policy | Return (mean ± sample std) | Best seed | Worst seed |
|---|---|---|---|
| Random | -5.32 ± 0.70 | -4.20 | -6.00 |
| Greedy (BFS to food, avoids body/walls) | 298.66 ± 3.83 | 305.40 | 296.30 |
| REINFORCE + Adam, greedy eval | 39.68 ± 7.33 | 50.70 | 30.90 |
| REINFORCE + Adam, stochastic eval | 41.96 ± 7.24 | 54.60 | 36.80 |
| PPO + `VecEnv`, original recipe, greedy eval | 87.90 ± 19.18 | 110.60 | 60.50 |
| PPO + `VecEnv`, original recipe, stochastic eval | 108.16 ± 13.62 | 122.50 | 85.70 |
| **PPO + `VecEnv`, full recipe, greedy eval** | **266.72 ± 10.12** | 281.70 | 253.30 |
| **PPO + `VecEnv`, full recipe, stochastic eval** | **264.24 ± 12.83** | 284.20 | 249.10 |

**The full training recipe closed most of the gap to the planner.** The two
recipe rows are a re-run of the identical protocol — same five seeds, same 1500
updates, same evaluation seed list, same binary — with `recipe 1`, which turns
on the standard PPO recipe: cosine learning-rate decay 3e-4 → 3e-5, value
clipping (0.2), approximate-KL early stop (0.03), entropy annealing
0.01 → 0.001, and running observation normalization plus reward scaling.
Evaluation applies the frozen observation statistics and reports raw rewards
(reward scaling is a training transform). The same budget now reaches
266.72 ± 10.12 greedy — **3.0× the original recipe** (87.90 ± 19.18, the
published rows, reproduced exactly by `recipe 0` on the current binary) — and
lands within 11% of the BFS planner: about 13 foods per episode against the
planner's ~15, versus 4-5 before. A planner that knows the rules still wins;
REINFORCE + Adam from scratch still reaches only ~40 after 200 epochs.

**No more late-run collapse.** Seed 42 run to 6000 updates (4× the rows above)
with the full recipe evaluates at 266.6 greedy / 266.8 stochastic — flat
against its 1500-update result instead of falling apart. With the original
recipe the same run peaked near 170 mid-training and then collapsed to
54.1/71.9: that was the fixed learning rate and the missing normalization/KL
stop showing through, and the recipe is what fixes it — not more compute.

`examples/train_ppo_snake.c` takes a `recipe` argument; `make ppo-snake` uses
recipe 1. Recipe 0 keeps the original numbers reachable bit for bit (verified
by diffing the full training output against the pre-recipe binary), which is
how the two sets of rows above were produced on one binary. The optional
potential-based shaping wrapper (Manhattan distance to the food,
`r' = r + c(γΦ' − Φ)`, Ng et al. 1999) is still available; at this budget it no
longer changes the outcome, so the default remains the raw sparse reward.

### Throughput (single core, Apple M4, NEON)

`make bench-throughput` measures `Env.step()` (random policy, excludes
`observe()`), the vectorized `vecenv_step` over 8 CartPoles, and the MatMul
kernel against a naive triple loop and against the scalar backend. The MatMul
rows are the NEON path on this machine; the CI x86-64 job publishes its own
AVX2 `bench_results.md`, so compare kernels within a row, not absolute GFLOPS
across CPUs.

| Benchmark | Result |
|---|---|
| CartPole `step()` | ≈ 24.8 M steps/sec |
| Snake `step()` | ≈ 83.7 M steps/sec |
| `vecenv_step` (×8 CartPole) | ≈ 35.6 M env-steps/sec |
| `mat_mm` 256×256×256 (`neon`, f32) | ≈ 33.1 GFLOPS |
| `mat_mm` 256×256×256 (scalar, f32) | ≈ 16.3 GFLOPS |
| Naive triple-loop 256×256×256 | ≈ 1.7 GFLOPS |
| `mat_mm` 64×64×64 (`neon`, f32) | ≈ 39.2 GFLOPS |
| `mat_mm` 64×64×64 (scalar, f32) | ≈ 15.2 GFLOPS |
| `mat_mm` 256×256×256 (`neon`, f64) | ≈ 16.7 GFLOPS |
| `mat_mm` 256×256×256 (scalar, f64) | ≈ 8.4 GFLOPS |
| `mat_mm` 64×64×64 (`neon`, f64) | ≈ 19.6 GFLOPS |
| `mat_mm` 64×64×64 (scalar, f64) | ≈ 10.7 GFLOPS |

The vector kernels are ~20× a naive triple loop and about 2× the *scalar*
backend: 2.1× at 256² and 2.6× at the 64×64×64 shape an RL layer actually uses
(2.0× and 1.8× in f64). That is the pay-off from processing two output rows at
once — every element of B is loaded once for two rows instead of once per row,
which roughly halves the kernel's dominant memory traffic. The scalar reference
the comparison runs against is untouched by that change, so the 2× is the
kernel's, not the baseline's. `make bench-double` runs the same measurement
against the f64 build, whose kernels are vectorised too (2-wide NEON, 4-wide
AVX2).

The kernels above are the default. A system BLAS is one flag away: `make BLAS=1
test` (or `-DRL_LIB_USE_BLAS` in any build) routes every `mat_mm` call through
CBLAS — Accelerate on macOS, OpenBLAS or any CBLAS elsewhere — for all four
transpose cases, while `mat_mm_scalar` stays the portable reference the math
suite cross-checks against. That trades the zero-dependency default for the
vendor kernel's cache blocking and threading when training wide layers in large
batches is the bottleneck; the numbers in the table keep measuring the shipped
kernels so the comparison stays apples to apples. Measured as an A/B within a
single session, Accelerate is ≈25× the shipped f32 kernel at 256² and ≈12× at
64³ (the gap narrows where the matrices already fit in cache) — the ratio, not
the absolute rate, is what transfers between machines.

#### Threaded `VecEnv`

Those env steps are far too cheap for a worker pool: its barrier costs tens of
microseconds per step, which is why `vecenv_set_threads` is opt-in and off by
default. `make bench-throughput` also runs it against a deliberately artificial
env whose step burns 4000 `sinf` calls — a known cost per step, so what is being
measured is the pool (8 envs, 200 lockstep steps, wall clock):

| Threads | Env steps/sec | Speedup |
|---|---|---|
| 1 | ≈ 21 500 | 1.00× |
| 2 | ≈ 41 300 | 1.92× |
| 4 | ≈ 69 000 | 3.2× |
| 8 | ≈ 58 000 | 2.7× |

This box has 4 performance cores (10 logical), so 4 threads scale near-linearly
and 8 threads oversubscribe: more barrier parties and efficiency cores. That is
the shape to expect — the pool removes the serial bottleneck, the hardware
still sets the ceiling.

### PPO (CartPole-like, 300 updates, `examples/train_ppo.c`)

| Policy | Greedy return |
|---|---|
| Random | ≈ 22 |
| PPO (8 envs, rollout 128, 10 epochs/update) | **500.0** (solved) |

PPO reaches the 500-step cap on CartPole well inside 300 updates — it is much
more sample-efficient than the REINFORCE baseline above, which is the expected
result of on-policy data reuse. On Snake the same trainer is wired up too (see
the Snake section): with the full recipe it reaches 3.0× its REINFORCE
baseline and sits within 11% of a BFS planner, and the section has the numbers.

### Pendulum-v1 (continuous PPO, 8 envs × 128 steps, 1200 updates)

`examples/train_ppo_continuous.c`, mean **undiscounted** episode return over 100
greedy episodes after 1200 updates:

| Seed | Greedy eval return | Last training update (stochastic) |
|---|---|---|
| 42 | -182.64 | -185.13 |
| 7 | -180.87 | -173.87 |
| 123 | -206.54 | -178.21 |

A random policy scores about -1200, and a policy that never applies torque
about -1000, so these are not "keep the pendulum near the bottom" scores. The
cost is θ² against upright, which makes a motionless hanging pendulum worth
about -1970 per episode; at -180..-207 (-0.9..-1.0 per step) every seed is
swinging up and balancing near the classic optimum for this task (reference
implementations report roughly -150..-250), just not at it. Read this as "nearly
solved", not "solved" — and definitely not as a hanging policy.

Two knobs decide whether this reproduces:

- **Learning rate.** Pendulum's good policies sit in a narrow basin and Adam
  moves every parameter by roughly `lr` per minibatch. At `lr = 3e-4` with 10
  epochs all three seeds peak near -150 around update 250 and then fall into the
  do-nothing basin (-1200); `2e-4` still collapses seed 42. At `1e-4` with 5
  epochs every seed stays between -130 and -240 for the whole run.
- **Exploration.** `log_std`'s gradient is mean-zero noise once the sampled
  actions stay inside the box, so `sigma` random-walks upward; a small entropy
  bonus (`0.01`) keeps it in place. `gamma = 0.9` (not 0.99) keeps the value
  targets small over the 200-step episodes, which the value head conditions on
  much better.

The same seed gives byte-identical evaluation returns at `-O2` and `-O3`.

---

## Build & test

### Prerequisites
- A C99 compiler — `gcc` or `clang` on POSIX, MSVC or MinGW on Windows
- POSIX `make`, or CMake 3.20+ (CMake is the only path on Windows)
- POSIX threads for the threaded `VecEnv`; optional (`-DRL_LIB_NO_THREADS`), and
  unnecessary on Windows, where the Win32 thread API is used instead

### Commands

```bash
make test               # build + run every test suite (incl. the f64 gradcheck)
make test-double        # run every suite in double precision as well
make release            # -O3 training binaries
make asan               # ASAN + UBSAN build, then run all tests
make tsan               # ThreadSanitizer over the threaded VecEnv suites
make ERR=1 test         # the strict warning set as errors (what CI runs)
make BLAS=1 test        # route MatMul through a system BLAS (see Throughput)

make lib                # librl_lib.a + the shared library
make install PREFIX=... # headers, libraries and pkg-config file
make uninstall PREFIX=...

make snake              # train Snake
make cartpole           # train CartPole
make a2c                # train CartPole with A2C
make ppo                # train CartPole with PPO
make ppo-continuous     # train Pendulum-v1 with continuous-action PPO
make ppo-snake          # train 6x6 Snake with PPO (sparse; optional shaping)
make dqn                # train CartPole with DQN (off-policy)
make hello              # the minimal end-to-end example

make bench              # multi-seed benchmarks -> bench_results.md
make bench-cartpole     # CartPole only
make bench-snake        # Snake only
make bench-throughput   # steps/sec and GEMM GFLOPS only
make bench-double       # the same GEMM measurement against the f64 build
make gradcheck-double   # reconfirm the gradient check in f64

make clean
```

### CMake (required on Windows, an alternative everywhere)

The same project configures, builds, tests and installs with CMake 3.20+ and no
POSIX tools:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/somewhere
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
cmake --install build --config Release
```

It builds the static and shared libraries, every example and every suite, and
installs what `make install` installs (flat headers under `include/rl_lib` with
an `envs/` subdirectory, plus a pkg-config file with a relocatable
`${pcfiledir}` prefix) and additionally a CMake package config, so a CMake
consumer can:

```cmake
find_package(RL_lib-in-C REQUIRED)
target_link_libraries(my_app PRIVATE RL_lib-in-C::rl_lib)
```

The Makefile knobs have CMake equivalents: `-DRL_LIB_USE_DOUBLE=ON`,
`-DRL_LIB_NO_THREADS=ON`, `-DRL_SIMD_BACKEND=0|1|2`, `-DRL_LIB_BUILD_TESTS=OFF`,
`-DRL_LIB_BUILD_EXAMPLES=OFF`.

Five platform notes:

- The threaded `VecEnv` exists there too: `src/vecenv.c` wraps the Win32 thread
  API (`CreateThread`, `CRITICAL_SECTION`, `CONDITION_VARIABLE`) behind the same
  pool and the same sense-reversing barrier, so both platforms step identically.
- `arena_new` uses `_aligned_malloc` instead of `posix_memalign`, and the
  benchmark's wall clock is `QueryPerformanceCounter` instead of `clock_gettime`.
- MSVC cannot compile the per-function `__attribute__((target("avx2,fma")))`
  kernels, so an MSVC build selects the scalar reference; MinGW (gcc) and
  clang-cl compile the AVX2 path like any other x86-64 GCC/Clang build. On ARM64
  Windows, clang-cl gets NEON.
- On Windows the shared build installs `bin/rl_lib.dll` with its import library
  `lib/rl_lib_shared.lib` — deliberately not `rl_lib.lib`, which stays the static
  archive (MSVC would otherwise have the two overwrite each other). A consumer
  that links `RL_lib-in-C::rl_lib_shared` needs the DLL next to its executable
  or on `PATH`; the static target needs nothing.
- On macOS the shared library records the configure-time
  `CMAKE_INSTALL_PREFIX` as its install name, exactly like the Makefile's
  `PREFIX`, so configure with the prefix you will install to rather than moving
  the tree afterwards with `cmake --install --prefix`.

A build with no threading backend (`RL_LIB_NO_THREADS`) reports the threaded
comparisons in `test_envs` / `test_rl` as `SKIP (no threading backend)` instead
of aborting; `RL_LIB_HAS_THREADS` in `vecenv.h` is the compile-time answer.

### Test suites

`make test` runs, in this order:

| Suite | What it covers |
|---|---|
| `test_gradcheck` (f64) | 12 suites / 280 parameters vs central differences, `h = 1e-5`, tolerance `1e-6`; includes the elementwise sigmoid/exp/log/div suite, the Gaussian PPO surrogate and its log-density against an independent double-precision formula |
| `test_mat` | 13 checks over MatMul (NN/NT/TN/TT), SIMD-vs-scalar agreement across two shape sets that cross the vector and TT-tile boundaries, softmax, ReLU, add/sub, broadcast add, mul/tanh/MSE |
| `test_gradcheck` (f32) | the same 12 suites in the default build (looser tolerance) |
| `test_envs` | 191 checks: CartPole bounds/termination/determinism; Snake reset, grid validation, wall/self collision, tail vacate, eating; Pendulum reset, hand-computed Euler dynamics, torque clamp, speed cap, determinism, clone independence and 200-step truncation; VecEnv shape, determinism, independent clones, auto-reset terminal-observation capture, vectorized continuous stepping with action bounds, and threaded-vs-serial stepping compared field by field over 260 steps |
| `test_rl` | 103 checks: GAE vs a hand-computed trajectory, one Adam step, grad clipping, actor-critic build, value-head learning, batched-vs-per-sample gradient equivalence, a parameter registry grown past 128 entries, rebuilding the batched graph for a new minibatch size (weights preserved, batch and single-sample inference agree), the continuous builder (per-dimension bounds, sampled `u`/squash round trip, `log_std` registered as a parameter and surviving a rebuild), CartPole learning smoke tests for REINFORCE and PPO with same-seed determinism, a Pendulum continuous-PPO learning smoke test plus determinism, the LR/entropy schedules against hand-computed values, normalization statistics (Welford/posterior/clip cases), PPO recipe-knob effects (each knob changes the run; the defaults do not), normalized-training determinism, and threaded rollouts reaching bit-identical weights in both trainers |
| `test_checkpoint` | 587 checks: byte-exact parameter round trips, bit-exact Adam-state resume, parameter-only and optimiser-only files, atomic replacement (no `.tmp` is left behind, and a save that cannot start leaves the previous checkpoint byte-identical and loadable), a continuous model round trip including `log_std`, a flipped byte caught by the CRC before the parser runs, a 300-case mutation fuzz pass (bit flips, truncations, overwritten runs, one third of them with a recomputed checksum), and rejection of missing, corrupt, truncated, wrong-width, wrongly-sized and wrongly-shaped files — each verified to leave the model untouched |
| `test_dqn` | 38 checks: replay-ring overwrite order (with terminal and truncated flags stored separately), the Q-network builder's rollout and batched views (including a rebuild that shares parameters), the epsilon schedule against hand-computed values, the one-step TD target (a terminal transition drops the bootstrap, a truncation keeps it), target-network sync (copies every parameter; untouched before its interval), same-seed determinism across two complete training runs, and a CartPole training run whose greedy evaluation clears 100 — it reaches ~140 in the sanitizer test build (~200 at `-O2`), against a random policy's ~22 |

`make test-double` then rebuilds the library with `-DRL_LIB_USE_DOUBLE` and runs
every one of those suites in f64, so the f64 path is not only exercised by the
gradient check.

Representative output:

```
=== Autograd gradient checks (f64) ===
  ...
  [gradcheck: Gaussian PPO surrogate (B=4, act_dim=2)]
  log-density vs textbook form (4 columns)... OK (0 bad)
  ...
---
Tests: 12 passed, 0 failed
Parameters checked: 280/280 passed within tolerance

--- test_mat ---
  SIMD path == scalar reference... (backend neon, max diff 2.38e-06) OK
Passed: 13  Failed: 0

--- test_envs ---
Passed: 191  Failed: 0

--- test_rl ---
Tests: 103 passed, 0 failed

--- test_checkpoint ---
  Adam state resumes a run exactly... OK
  Atomic save (temp file + rename)... OK
  continuous model round-trip (log_std included)... OK
Tests: 587 passed, 0 failed

--- test_dqn ---
Tests: 38 passed, 0 failed
```

---

## Architecture

```
.
├── include/
│   ├── base.h               # Primitive types, real/f32/f64, RL_LIB_ASSERT/RL_LIB_CHECK
│   ├── version.h            # RL_LIB_VERSION_* (also stamped into rl_lib.pc)
│   ├── memory.h             # 64-byte aligned bump allocator + temp scopes
│   ├── random.h             # xoshiro256** PRNG
│   ├── simd.h               # SIMD backend selection (NEON / AVX2 / scalar)
│   ├── linalg.h             # Matrix type, linear algebra, mat_* API
│   ├── graph.h              # Computation graph, Tensor nodes, reverse-mode AD
│   ├── layers.h             # Dense layer, MLP, discrete actor, actor-critic
│   ├── optim.h              # SGD, Adam, gradient clipping
│   ├── checkpoint.h         # Parameter + Adam state save/load
│   ├── env.h                # Env interface (reset(seed), step, observe, clone)
│   ├── vecenv.h             # Vectorized (batched) environment wrapper
│   ├── rl.h                 # Rollout buffer, REINFORCE, A2C, PPO, GAE, evaluation
│   └── dqn.h                # Replay buffer, epsilon schedule, DQN trainer
├── src/
│   ├── memory.c
│   ├── random.c
│   ├── simd.c               # Backend name + runtime AVX2/FMA probe
│   ├── linalg.c             # Matrix kernels (scalar + SIMD) and math ops
│   ├── graph.c              # Reverse-mode AD engine + op interpreter
│   ├── layers.c             # Builders and initialization
│   ├── optim.c              # Parameter updates
│   ├── checkpoint.c         # Versioned checkpoint format (LE, blocks + CRC-32)
│   ├── vecenv.c             # Synchronized env vector with auto-reset
│   ├── rl.c                 # Rollout, REINFORCE, A2C, PPO/GAE, evaluation
│   └── dqn.c                # Replay ring, target sync, DQN update loop
├── envs/
│   ├── snake.h / snake.c        # Grid Snake: 3-channel obs, 4 actions
│   ├── cartpole.h / cartpole.c  # CartPole-like physics
│   └── pendulum.h / pendulum.c  # Pendulum-v1: continuous torque, max 200 steps
├── examples/
│   ├── train_snake.c
│   ├── train_cartpole.c
│   ├── train_a2c.c
│   ├── train_ppo.c
│   ├── train_ppo_continuous.c   # Pendulum-v1 with a squashed Gaussian policy
│   ├── train_ppo_snake.c    # 6x6 Snake with PPO over VecEnv
│   ├── train_dqn.c          # CartPole with DQN and a replay buffer
│   ├── hello_rl.c           # Smallest end-to-end program (also used by CI)
│   ├── bench.c              # Multi-seed harness + throughput micro-benchmarks
│   └── python/
│       └── ctypes_demo.py   # Drives the installed shared library from CPython
├── tests/
│   ├── test_mat.c
│   ├── test_gradcheck.c     # f32 and f64 numerical gradient checks
│   ├── test_envs.c
│   ├── test_rl.c
│   ├── test_checkpoint.c    # Round trips, corrupt/truncated/fuzzed rejection
│   └── test_dqn.c           # Replay buffer, target sync, CartPole learning
├── docs/
│   └── API.md               # Full API reference
├── packaging/
│   └── vcpkg/               # Reference vcpkg port (vcpkg.json + portfile.cmake)
├── test_package/            # Conan test package: consumes the built package
├── conanfile.py             # Conan 2 recipe (conan create .)
├── .github/workflows/ci.yml # POSIX matrix + sanitizers + scalar + BLAS + f64 + coverage + bench + Windows/MSVC + install + Conan
├── cmake/
│   └── RL_lib-in-CConfig.cmake.in  # find_package() config template
├── rl_lib.pc.in             # pkg-config template (stamped from version.h)
├── CMakeLists.txt           # CMake project (POSIX and Windows)
├── CHANGELOG.md
├── Makefile
└── README.md
```

---

## Known limitations

- **Performance is modest and honest.** MatMul has hand-written NEON/AVX2
  kernels (≈33 GFLOPS at 256², ≈39 GFLOPS at the 64×64×64 shape an RL layer
  uses, f32), about 2× the compiler's own auto-vectorized scalar loop and ~20×
  a naive triple loop. That is a real but unremarkable number next to a tuned
  BLAS: MatMul is still single-threaded, there is no multi-level (i/j/k) cache
  tiling and no fused optimizer. "High performance" is not claimed — building
  with `make BLAS=1` swaps the kernels for the platform BLAS when that ceiling
  matters.
- **SIMD coverage is complete for the supported targets, and bounded elsewhere.**
  Both precisions are vectorised — 4-wide f32 / 2-wide f64 on NEON (f64 needs
  AArch64; ARMv7 NEON has no double-precision arithmetic) and 8-wide f32 /
  4-wide f64 on AVX2+FMA — and all four transpose cases have a vector path
  (TT transposes a 32×32 tile into the TN path). Everything else, including
  `-DRL_SIMD_BACKEND=0`, uses the scalar reference, which the tests cross-check
  in both precisions. AVX2 is chosen at runtime behind
  `__builtin_cpu_supports("avx2")`; NEON is assumed on AArch64. Note that the NT
  kernel reassociates the K reduction and TT adds a transpose pass, so those two
  can differ from the scalar reference by a few ULPs (the tests bound this).
  MSVC cannot compile `__attribute__((target(...)))` at all, so an MSVC build
  selects the scalar reference; MinGW and clang-cl get the AVX2 path.
- **Vectorized environments are synchronous, and threading them is opt-in and
  rarely worth it.** `vecenv_step` loops over the cloned sub-environments on one
  core; `vecenv_set_threads` spreads that loop over worker threads (pthreads on
  POSIX, the Win32 thread API on Windows) and is
  bit-identical to the serial run (`tests/test_envs.c` checks 260 steps field by
  field, `tests/test_rl.c` checks the trained weights), but the barrier
  round-trip costs about 30 µs per step while a CartPole or Pendulum step costs
  a fraction of a microsecond. Measured on the shipped Pendulum example, 1200
  updates take 8.0 s serial and 13.0 s with 8 threads. It pays off once a step
  costs more than that: on a synthetic compute-bound env the pool measures
  1.92× on 2 threads, 3.2× on 4 and 2.7× on 8 against the serial loop (the
  benchmark table above has the numbers and the machine's core layout).
- **Reproducibility is within a machine, not across machines.** Seeding makes
  runs deterministic for a given binary and libm, but `expf`/`logf` differ
  across libc implementations and `-O3` may reassociate, so bit-for-bit
  cross-machine reproduction is not guaranteed.
- **Precision is f32 by default.** The `real` typedef switches the whole
  library to f64 with `-DRL_LIB_USE_DOUBLE`; the gradient check uses this, and
  `make test-double` now runs every suite in f64 as well.
- **Snake is close to, but still short of, its planner.** With the full recipe
  PPO over 8 vectorized environments reaches 266.7 ± 10.1 greedy (five seeds)
  against the BFS planner's ~299 — within 11%, and with no late-run collapse;
  the original recipe scores 87.9 ± 19.2 at the same budget. REINFORCE + Adam
  plateaus at ~40. The numbers and the recipe are in the Snake section.
- **Continuous PPO is close to, but not at, the Pendulum optimum.** Runs
  converge to -130..-240 per 200-step episode, overlapping the band reference
  implementations report (roughly -150..-250), so these are genuine
  swing-up-and-balance policies (a random policy is about -1200, and hanging
  still is about -1970). The configuration is sensitive to the learning rate —
  see the Pendulum section. The squashed-Gaussian policy also clamps the mean
  to ±5 and `log_std` to [-6, 2] on read (documented constants in `linalg.h`),
  which the textbook construction does not do, and the entropy bonus uses the
  *pre-squash* Gaussian entropy because the squashed density's entropy has no
  closed form.
- **The arena is bump-only.** Nothing is reused within a run except through the
  `arena_mark`/`arena_rewind` temp scopes, and rebuilding a batched graph leaves
  its predecessor in the arena. `MAX_ARITY` is 2 because every operation in the
  set is unary or binary — a property of the op set, not a capacity cap.
- **Resizing applies to builder-made models.** `net_set_batch_size` works for
  graphs from `actor_build` / `actor_critic_build` (which record their
  architecture); a hand-built graph has no architecture to rebuild from, so its
  batch size stays fixed and the trainer asserts the match.
- **Checkpoints cover parameters and optimiser state, not the RNG.** A resumed
  run re-derives its generator from `TrainCfg.seed`, so it continues from the
  saved weights but is not bit-identical to a run that was never interrupted.
- **A few small heap allocations** remain (shuffle scratch, evaluation buffers,
  entropy scratch, checkpoint staging) alongside the arena.
- **Platform coverage is Linux, macOS and Windows, gcc/clang/MSVC.** The POSIX
  path is a Makefile; Windows builds with CMake (MSVC or MinGW) and has a Win32
  threading backend. MSVC gets the scalar SIMD reference, and the sanitizer
  targets and the benchmark's core-layout notes are POSIX-only (they are
  development aids, not part of the library).

---

## Roadmap

- [x] **Phase 0: Credibility**
  - [x] Matrix unit tests, numerical gradient checking, deterministic seeding,
        Random/Greedy baselines.
- [x] **Phase 1: Library structure**
  - [x] `Env` interface with separate `done` (terminal) and `truncated` flags.
  - [x] `include/` + `src/` split, arena-backed rollout buffer and parameter
        registry, network builders, decoupled trainer and optimizer.
- [x] **Phase 2: Autograd upgrades**
  - [x] Observation batching, per-column softmax, new operators (mul, tanh,
        MSE, fused policy loss, entropy), Adam with clipping.
- [x] **Phase 3: Batching & Snake overhaul**
  - [x] Batched GEMM training, Snake 3-channel observation, per-env RNG,
        `done` vs `truncated`, `schedule_grad_from` seeding, benchmark harness.
- [x] **Phase 4: Advanced RL algorithms**
  - [x] GAE-λ, actor-critic builder, A2C trainer. Verified by
        `tests/test_rl.c` (builds, one update, value-head loss decreases) and
        `examples/train_a2c.c` (CartPole greedy return ≈ 403 after 60 epochs).
- [x] **Phase 5: Performance & systems**
  - [x] SIMD (NEON/AVX2) MatMul kernels with a scalar reference and a
        runtime CPU probe; `test_mat` cross-checks the two paths.
  - [x] Vectorized environments (`VecEnv`): lockstep stepping, per-env
        auto-reset, terminal-observation capture, deterministic seeding.
  - [x] PPO with the clipped surrogate objective (multi-epoch shuffled
        minibatches, gradient-checked loss, `tests/test_rl.c`, `make ppo`).
  - [x] Optional threaded `VecEnv` (`vecenv_set_threads`, worker pool + a
        mutex/condition-variable barrier because macOS has no
        `pthread_barrier_t`). Verified bit-identical to the serial path at the
        environment level and at the trained-weights level; the pendulum example
        can exercise it with a third CLI argument.
  - [x] SIMD completeness: f64 kernels (2-wide NEON on AArch64, 4-wide AVX2),
        the TT case (a 32×32 transposed tile fed through the TN path) and
        two-row register blocking, which halved B's traffic and took the kernel
        from +11%/+67% over the scalar loop to ~2× (~20× a naive triple loop).
        Measured in f32 and f64 (`make bench-throughput`, `make bench-double`);
        `test_mat` cross-checks all four transpose cases in both precisions.
- [x] **Phase 6: Continuous actions**
  - [x] Squashed-Gaussian policy: mean head + learnable `log_std`, per-dimension
        action bounds, log-density from the sampled pre-squash coordinate, and a
        gradient-checked PPO surrogate (`test_gradcheck`, f32 + f64) that is also
        checked against the textbook density formula in double precision.
  - [x] Pendulum-v1 behind the continuous part of the `Env` interface
        (`step_continuous`, action bounds, 200-step truncation) plus
        `vecenv_step_continuous` / `vecenv_action_bounds`.
  - [x] `train_ppo_continuous` (truncation-aware GAE bootstrap, one optimiser step
        per minibatch), `evaluate_continuous[_seeded]`, a learning smoke test in
        `test_rl`, a checkpoint round trip, and `make ppo-continuous`
        (Pendulum -180..-207 across three seeds over 1200 updates).

- [x] **Phase 7: Parity attempt & adoption**
  - [x] Snake with PPO over `VecEnv` (first recipe): 2.2× the REINFORCE
        baseline (87.9 ± 19.2 greedy over five seeds vs 39.7) but still 3.4×
        short of the BFS planner, with long runs collapsing instead of closing
        the gap. Kept as the baseline the Phase 8 recipe supersedes; the
        comparison is in the Snake section. `examples/train_ppo_snake.c`,
        `make ppo-snake`.

- [x] **Phase 8: Recipes, off-policy and ops**
  - [x] Full PPO training recipe (`TrainCfg`): linear/cosine LR decay, entropy
        annealing, value clipping, approximate-KL early stop, running
        observation normalization and reward scaling with `*_normed`
        evaluation. Five-seed Snake re-run on the same protocol: **266.72 ±
        10.12 greedy vs 87.90 ± 19.18 with the original recipe** (reproduced
        exactly by `recipe 0`), 10.7% below the BFS planner, and no collapse at
        6000 updates (266.6).
  - [x] DQN (`dqn.h`): ring replay with separate terminal/truncated flags,
        target network, epsilon schedule, and the masked-MSE regression on the
        existing autograd engine; `make dqn` reaches ~194 greedy on CartPole in
        16000 steps (seed 42, 100 episodes); `test_dqn` covers it (38 checks).
  - [x] Graph ops: elementwise `sigmoid`, `exp`, `log` and `div` with matching
        kernels and numerical gradient-check coverage (12 suites / 280
        parameters, f32 + f64).

- [x] **Phase 9: Release and robustness**
  - [x] v1.1.0 tagged (recipe, DQN, new graph ops, optional system BLAS, ctypes
        demo): `version.h`, `docs/API.md` and the changelog move together, and
        the pkg-config stamp follows from `version.h`.
  - [x] Checkpoints carry a CRC-32 trailer (format 3), saves are atomic
        (`<path>.tmp` + fsync + rename, so a failed save cannot clobber the
        previous file), and loads are transactional, with a 300-case mutation
        fuzz pass in the suite and under the sanitizers.
  - [x] CI also reports coverage (`make COVERAGE=1 test` + `gcovr`) and
        publishes the benchmark artifact.
  - [x] Packaging: a Conan 2 recipe with a `test_package/` (verified by a CI
        job that builds, packages and consumes it) and a vcpkg reference port
        wrapping the same CMake install.

### Remaining

- [x] Windows/MSVC build via CMake: `CMakeLists.txt` configures, builds, tests
      and installs on POSIX and Windows, `src/vecenv.c` has a Win32 thread
      backend, `arena_new` uses `_aligned_malloc` there and the benchmark uses
      `QueryPerformanceCounter`. MSVC gets the scalar SIMD reference (it cannot
      compile the `target`-attributed AVX2 kernels), MinGW/clang-cl get AVX2.
      CI gained a `windows-latest` job (MSVC + ctest + install), a serial-only
      job, and every example is now compiled with `-Werror`.

Two things are deliberately not attempted, and said so rather than faked:
multi-machine benchmarks (this is a single-machine project — the numbers above
are one machine's, and they are not extrapolated) and bit-exact reproduction
across machines (that would need a pinned libm and compiler version; the
reproducibility limitation is stated instead).
