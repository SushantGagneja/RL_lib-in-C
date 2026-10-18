# RL_lib-in-C: Compact Reinforcement Learning & Autograd Library in C

A small, readable C library for reverse-mode autograd, neural networks, and
reinforcement learning. No third-party dependencies — only the C standard
library, `libm`, and POSIX `make`.

It is a learning project: the code is meant to be read in one sitting, and the
benchmarks below are reported with their real weaknesses rather than tuned to
flatter the algorithms.

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
    per-column Softmax, elementwise Mul, Add, Sub, MSE, REINFORCE loss, and a
    fused policy cross-entropy loss with a mean-entropy bonus.
  - Gradient accumulation across minibatches; `schedule_from_roots` builds a
    graph for models with more than one output (e.g. actor-critic).
- **Arena allocator** — 64-byte aligned bump allocator for graph nodes,
  matrices, parameters, and rollout buffers. A handful of small scratch buffers
  use `malloc`.
- **Seeded xoshiro256\*\* PRNG** (SplitMix64 seeding, public domain) — isolated
  `Rng` state per environment, weight initializer, and trainer.
- **Discrete actor / actor-critic builders** with a shared `Dense` struct and
  He-uniform initialization (tiny-gain policy head).
- **Optimizers** — SGD and Adam with bias correction and global gradient-norm
  clipping (with a non-finite guard).
- **Algorithms** — REINFORCE with mean-baseline normalization, and Advantage
  Actor-Critic (A2C) with GAE-λ and an entropy bonus.
- **Environments** — CartPole-like and grid Snake behind a common `Env`
  interface (`reset(env, seed)`, `step`, `observe`).

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

### Snake 6×6 (REINFORCE + Adam, 200 epochs)

| Policy | Return (mean ± sample std) | Best seed | Worst seed |
|---|---|---|---|
| Random | -5.32 ± 0.70 | -4.20 | -6.00 |
| Greedy (BFS to food, avoids body/walls) | 298.66 ± 3.83 | 305.40 | 296.30 |
| Trained (greedy eval) | 39.68 ± 7.33 | 50.70 | 30.90 |
| Trained (stochastic eval) | 41.96 ± 7.24 | 54.60 | 36.80 |

**The trained policy loses badly to the baseline.** A BFS planner that knows
the rules scores ~299; REINFORCE + Adam from scratch reaches only ~40 after 200
epochs, and improving slowly. The earlier "beats greedy by 63%" claim used a
greedy agent that ignored walls and its own body — a strawman. With a real
baseline, the honest conclusion is that this Snake setup is not solved.

### Throughput (single core, one machine)

`make bench-throughput` measures `Env.step()` (random policy, excludes
`observe()`) and a 256×256 MatMul against a naive triple loop.

| Benchmark | Result |
|---|---|
| CartPole `step()` | ≈ 25 M steps/sec |
| Snake `step()` | ≈ 94 M steps/sec |
| `mat_mm` 256×256 (NN kernel) | ≈ 16.3 GFLOPS |
| Naive triple-loop 256×256 | ≈ 1.67 GFLOPS |

The SIMD-friendly thing to notice is that the kernel is ~10× a naive loop but
is still a scalar triple loop — this is "reasonable for a teaching library",
not "high-performance". See the roadmap.

---

## Build & test

### Prerequisites
- A C99 compiler (`gcc` or `clang`)
- POSIX `make`

### Commands

```bash
make test               # build + run every test suite (incl. the f64 gradcheck)
make release            # -O3 training binaries
make asan               # ASAN + UBSAN build, then run all tests
make ERR=1 test         # the strict warning set as errors (what CI runs)

make snake              # train Snake
make cartpole           # train CartPole
make a2c                # train CartPole with A2C

make bench              # multi-seed benchmarks -> bench_results.md
make bench-cartpole     # CartPole only
make bench-snake        # Snake only
make bench-throughput   # steps/sec and GEMM GFLOPS only
make gradcheck-double   # reconfirm the gradient check in f64

make clean
```

### Test suites

`make test` runs, in this order:

| Suite | What it covers |
|---|---|
| `test_gradcheck` (f64) | 9 suites / 236 parameters vs central differences, `h = 1e-5`, tolerance `1e-6` |
| `test_mat` | 12 checks over MatMul (NN/NT/TN/TT), softmax, ReLU, add/sub, broadcast add, mul/tanh/MSE |
| `test_gradcheck` (f32) | the same 9 suites in the default build (looser tolerance) |
| `test_envs` | 36 checks: CartPole bounds/termination/determinism; Snake reset, grid validation, wall/self collision, tail vacate, eating |
| `test_rl` | 23 checks: GAE vs a hand-computed trajectory, one Adam step, grad clipping, actor-critic build, value-head learning, batched-vs-per-sample gradient equivalence, a CartPole learning smoke test, and same-seed determinism |

Representative output:

```
=== Autograd gradient checks (f64) ===
  ...
---
Tests: 9 passed, 0 failed
Parameters checked: 236/236 passed within tolerance

--- test_mat ---
Passed: 12  Failed: 0

--- test_envs ---
Passed: 36  Failed: 0

--- test_rl ---
Tests: 23 passed, 0 failed
```

---

## Architecture

```
.
├── include/
│   ├── base.h               # Primitive types, real/f32/f64, RL_LIB_ASSERT/RL_LIB_CHECK
│   ├── memory.h             # 64-byte aligned bump allocator + temp scopes
│   ├── random.h             # xoshiro256** PRNG
│   ├── linalg.h             # Matrix type, linear algebra, mat_* API
│   ├── graph.h              # Computation graph, Tensor nodes, reverse-mode AD
│   ├── layers.h             # Dense layer, MLP, discrete actor, actor-critic
│   ├── optim.h              # SGD, Adam, gradient clipping
│   ├── env.h                # Env interface (reset(seed), step, observe)
│   └── rl.h                 # Rollout buffer, REINFORCE, A2C, GAE, evaluation
├── src/
│   ├── memory.c
│   ├── random.c
│   ├── linalg.c             # Matrix kernels and math ops
│   ├── graph.c              # Reverse-mode AD engine + op interpreter
│   ├── layers.c             # Builders and initialization
│   ├── optim.c              # Parameter updates
│   └── rl.c                 # Rollout, REINFORCE, A2C/GAE, evaluation
├── envs/
│   ├── snake.h / snake.c        # Grid Snake: 3-channel obs, 4 actions
│   └── cartpole.h / cartpole.c  # CartPole-like physics
├── examples/
│   ├── train_snake.c
│   ├── train_cartpole.c
│   ├── train_a2c.c
│   └── bench.c              # Multi-seed harness + throughput micro-benchmarks
├── tests/
│   ├── test_mat.c
│   ├── test_gradcheck.c     # f32 and f64 numerical gradient checks
│   ├── test_envs.c
│   └── test_rl.c
├── .github/workflows/ci.yml # gcc/clang/macOS + ASAN/UBSAN
├── CHANGELOG.md
├── Makefile
└── README.md
```

---

## Known limitations

- **Performance is modest and honest.** MatMul is a scalar triple loop
  (~16 GFLOPS at 256×256, ~10× a naive loop); there is no SIMD, no threading,
  and no vectorized environments. "High-performance" is not claimed.
- **Reproducibility is within a machine, not across machines.** Seeding makes
  runs deterministic for a given binary and libm, but `expf`/`logf` differ
  across libc implementations and `-O3` may reassociate, so bit-for-bit
  cross-machine reproduction is not guaranteed.
- **Precision is f32 by default.** The `real` typedef switches the whole
  library to f64 with `-DRL_LIB_USE_DOUBLE`; the gradient check uses this. The
  training pipeline is only exercised in f32.
- **Snake is not solved.** REINFORCE + Adam plateaus well below a BFS planner.
- **Fixed capacity limits.** `MAX_PARAMS` is 128 and `MAX_ARITY` is 2;
  exceeding them aborts. The arena is bump-only (no reuse within a run except
  the `arena_mark`/`arena_rewind` temp scopes).
- **One graph per batch size.** Inference (B=1) and batched training are
  separate graphs sharing parameter pointers; the batch size is frozen at build
  time. `TrainCfg.batch_size` is asserted against the graph, not used to resize
  it.
- **A few small heap allocations** remain (shuffle scratch, evaluation buffers,
  entropy scratch) alongside the arena.
- **Stack-based placement is untested.** Build/test coverage is Linux + macOS
  + clang/gcc only.

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
- [ ] **Phase 5: Performance & systems**
  - SIMD (NEON/AVX2) MatMul kernels, vectorized environments, and PPO with a
    clipped surrogate objective. Only after the correctness items above are
    locked down.
