# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) and the API only
breaks on a major bump. `RL_LIB_VERSION_STRING` in `include/version.h` is the
single source of truth — it is also stamped into `rl_lib.pc`.

## 1.0.0 — 2026-10-18

First release. Everything below landed while preparing it, so this entry is the
whole history: the project was renamed to **RL_lib-in-C**, licensed under MIT,
independently reimplemented, and carried through the Phase 0–5 roadmap.

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
    (numerically identical: the f64 gradient check still passes, 248/248).
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

### Added
- **Checkpointing** (`checkpoint.c` / `checkpoint.h`): `net_save`, `net_load`,
  `adam_save`, `adam_load`, `checkpoint_save`, `checkpoint_load`, and
  `checkpoint_last_error`. A checkpoint is a versioned, endian-independent,
  block-structured binary file holding the parameters and the Adam moments, so a
  run can be stopped and resumed. Loads validate the scalar width, parameter
  count, payload size and every shape, and stage the payload before writing, so a
  rejected file leaves the model untouched. Covered by `tests/test_checkpoint.c`
  (53 checks), including a bit-exact Adam-state resume test.
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
- **SIMD MatMul** (`simd.c` / `simd.h` plus `linalg.c`): register-blocked NEON
  and AVX2+FMA kernels for the NN/NT/TN cases, a portable `mat_mm_scalar`
  reference, a compile-time backend selector (`-DRL_SIMD_BACKEND=0` forces
  scalar), and a runtime AVX2/FMA probe on x86-64. `test_mat` cross-checks the
  vector path against the scalar reference, and a CI job builds the scalar
  backend end to end.
- **Vectorized environments** (`vecenv.c` / `vecenv.h`): `VecEnv` steps N
  cloned envs in lockstep, auto-resets finished ones in place, and preserves
  each terminal observation (`final_obs`) for correct value bootstrapping.
  `Env` gained a `clone(arena, src)` function pointer.
- **PPO** (`train_ppo`, `mat_ppo_policy_loss` / `mat_ppo_policy_add_grad`): the
  clipped surrogate objective with multi-epoch shuffled minibatches, per-env
  GAE-λ, and truncation-aware bootstrapping. Covered by a new f64/f32 gradient
  check, CartPole learning and same-seed determinism tests, and
  `examples/train_ppo.c`.
- `make ppo`, `EXTRA_CFLAGS` Makefile support, and a scalar-backend CI job.
- `make gradcheck-double`: builds the library in f64 (`-DRL_LIB_USE_DOUBLE`, the
  `real` scalar typedef) so the gradient check runs with `h = 1e-5` and a
  `1e-6` tolerance. Input/activation gradients and an elementwise `mul` with
  both operands requiring grad are checked too.
- `tests/test_rl.c`: GAE against a hand-computed trajectory, a single Adam step,
  gradient-norm clipping, actor-critic construction and value-head learning,
  batched-vs-per-sample gradient equivalence, CartPole learning smoke tests for
  REINFORCE and PPO, and same-seed determinism for both.
- `make asan` (ASAN + UBSAN) and a GitHub Actions workflow running `make test`
  with warnings as errors on gcc/clang/macOS, plus the sanitizer target.
- `examples/train_a2c.c` and `examples/train_ppo.c`.

### Changed
- **`MAX_PARAMS` is gone.** The parameter registry (`Net.params`) now doubles
  through the arena, so models are no longer capped at 128 parameters.
  `MAX_ARITY` stays at 2, which is a property of the op set (every operation is
  unary or binary) rather than a capacity limit.
- **The minibatch size is a runtime knob.** `net_set_batch_size` /
  `net_batch_size` rebuild the batched subgraph from the architecture the
  builders now record, sharing the existing parameter tensors, and the trainers
  apply `TrainCfg.batch_size` before sizing anything from the graph.
- Removed the short-name macros from the linear-algebra header; everything uses
  the `mat_*` names.
- Extracted a `Dense` struct (`dense_new` / `dense_apply`) and deleted the three
  duplicated layer/init code paths and the weight-sharing hack.
- The matrix-multiply kernels are `static` (`mm_nn` / `mm_nt` / `mm_tn` /
  `mm_tt`); `mat_mm` dispatches to the SIMD kernels when a backend is available.
- The benchmark reports the active SIMD backend, two MatMul shapes (large and
  RL-sized), and `VecEnv` throughput.

### Known limitations
See the "Known limitations" section of the README.

### License
MIT — see [`LICENSE`](LICENSE). The engine is an original implementation and
there is no bundled third-party code (the PRNG is public-domain xoshiro256\*\*).
