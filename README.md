# cRL: Compact Reinforcement Learning & Autograd Library in C

A modular, high-performance C library for autograd, neural networks, and reinforcement learning with zero external dependencies (standard C runtime and `math.h` only).

---

## Credits & Attribution

> The core autograd engine, arena allocator, matrix layout, and computation graph
> in this library are derived from **[harshbhatt7585/cRL](https://github.com/harshbhatt7585/cRL)**
> by Harsh Bhatt. The original repository does not include an explicit license file.
>
> This fork adds modular library structure (`include/` + `src/`), seeded PRNG for
> reproducibility, numerical gradient checking, batched training, the CartPole-v1
> environment, fused policy cross-entropy loss, GAE / A2C, and multi-seed benchmarking.

---

## Features

- **Reverse-Mode Autograd Engine**:
  - Static computation graph with vtable-style operators (`VarType`).
  - Core ops: MatMul (NN/NT/TN/TT), ReLU, Tanh, numerically stable Softmax (per-column for batching), elementwise Mul, Add, Sub, MSE, and REINFORCE loss.
  - Fused policy cross-entropy loss with batched forward/backward.
  - Gradient accumulation designed for batch and minibatch optimization.
- **Arena Memory Management**:
  - Bump allocator with power-of-2 alignment; no fragmentation or dynamic reallocation during training.
- **Reproducible PCG32 PRNG**:
  - Deterministic seeding with isolated `prng_state` per environment, weight init, and trainer.
- **Verification & Rigorous Testing**:
  - Exact numerical gradient checks (central finite differences) across 8 suites covering 210 parameters.
  - 12 mathematical unit tests for all matrix operations and edge cases.
  - 25 environment interface assertions (CartPole and Snake).
- **Multi-Seed Benchmark Harness** (`make bench`):
  - Evaluates Random, Greedy-Heuristic, and Trained policies across 5 seeds × 100 episodes.
- **Batched Training**:
  - Observations packed as columns of $(obs\_dim \times B)$ matrices for GEMM-based forward/backward.
  - Weight sharing between rollout (1-sample inference) and batched training graphs.
- **Algorithms**:
  - REINFORCE with mean-baseline normalization.
  - Advantage Actor-Critic (A2C) with Generalized Advantage Estimation (GAE-λ).
- **Optimizers**:
  - SGD and Adam with bias correction and global gradient norm clipping.

---

## Benchmarks & Baselines

All numbers below are produced by `make bench` and can be reproduced exactly from this repo.

### CartPole-v1

Trained with REINFORCE + Adam for 50 epochs, evaluated greedy over 100 episodes per seed:

| Policy | Return (mean ± std) |
|---|---|
| **Trained (REINFORCE, 50 epochs)** | **499.97 ± 0.12** |
| Greedy (PD controller) | 485.65 ± 3.36 |
| Random | 22.37 ± 1.14 |

> CartPole-v1 is solved: **500.0 / 500.0** on 4 of 5 seeds, 499.9 on the 5th. 5-seed mean: **499.97 ± 0.12**.

### Snake 6×6

Trained with REINFORCE + Adam for 200 epochs:

| Policy | Return (mean ± std) |
|---|---|
| **Trained (REINFORCE, 200 epochs)** | **68.44 ± 13.71** |
| Greedy (food heuristic) | 41.96 ± 2.93 |
| Random | -5.12 ± 0.41 |

> Trained policy beats the greedy heuristic by **63%** on average. Best seed (777) reached **93.1** mean return.
> Run `make bench` to reproduce on your hardware.

---

## Build & Test

### Prerequisites
- Any C99-compliant compiler (`gcc` or `clang`)
- POSIX `make`

### Commands

```bash
# Build release binaries (train_snake, train_cartpole with -O3)
make release

# Run all test suites (mat ops, autograd gradcheck, environment tests)
make test

# Run multi-seed benchmarks (Random, Greedy, Trained)
make bench              # Both CartPole and Snake
make bench-cartpole     # CartPole only
make bench-snake        # Snake only

# Train Snake on the discrete grid
make snake

# Train CartPole-v1
make cartpole

# Clean build artifacts
make clean
```

### Test Suite Output

Running `make test` produces:
```
=== Running test_mat ===
  12 tests: matmul (NN/NT/TN/TT), softmax (sum-to-1, positivity, overflow,
            per-column batch), relu, add/sub, broadcast add, mul/tanh/mse
  Passed: 12  Failed: 0

=== Running test_gradcheck ===
  8 suites: MLP, softmax, sub, broadcast add, tanh·mul, MSE,
            fused policy loss, full actor→REINFORCE
  Parameters checked: 210/210 passed within tolerance

=== Running test_envs ===
  CartPole: reset & bounds, termination on fall
  Snake: reset & 3-channel observation, wall collision (done≠truncated),
         deterministic self-collision
  Passed: 25  Failed: 0
```

---

## Architecture Overview

```
.
├── include/
│   ├── base.h               # Primitive types, CRL_ASSERT, utility macros
│   ├── arena.h              # Bump allocator with power-of-2 alignment
│   ├── prng.h               # PCG32 pseudo-random number generator
│   ├── mat.h                # Matrix structure, linear algebra, mat_* API
│   ├── autograd.h           # Computation graph, Var nodes, reverse-mode AD
│   ├── nn.h                 # Neural network layers (linear, MLP, actor, actor-critic)
│   ├── optim.h              # Optimizers (SGD, Adam, gradient clipping)
│   ├── env.h                # Modular RL Environment interface (reset, step, observe)
│   └── rl.h                 # Rollout buffer, REINFORCE, A2C, GAE, evaluation
├── src/
│   ├── arena.c              # Arena implementation
│   ├── prng.c               # PRNG implementation
│   ├── mat.c                # Matrix kernels & math operations
│   ├── autograd.c           # Reverse-mode AD engine & node implementations
│   ├── nn.c                 # Model builders, weight sharing, Xavier/Glorot init
│   ├── optim.c              # Optimizer parameter updates
│   └── rl.c                 # Batched rollout, REINFORCE, A2C/GAE, evaluation
├── envs/
│   ├── snake.h / snake.c    # Grid Snake: 3-channel obs, 4 actions, body tracking
│   └── cartpole.h / cartpole.c # CartPole-v1 physics simulator
├── examples/
│   ├── train_snake.c        # Standalone Snake training demo
│   ├── train_cartpole.c     # Standalone CartPole-v1 training demo
│   └── bench.c              # Multi-seed benchmark harness (Random, Greedy, Trained)
├── tests/
│   ├── test_mat.c           # Math layer unit tests (12 checks)
│   ├── test_gradcheck.c     # Numerical gradient checks (8 suites, 210 params)
│   └── test_envs.c          # Environment unit tests (25 checks)
├── crl-improvement-plan.md  # Architectural roadmap and phased milestones
└── Makefile                 # Build, test, and benchmark orchestration
```

---

## Roadmap

- [x] **Phase 0: Credibility**
  - [x] Unit tests for all matrix operations (`test_mat`, 12 checks).
  - [x] Numerical gradient checking for autograd (`test_gradcheck`, 210 params).
  - [x] Isolated deterministic PRNG seeding & bit-for-bit reproducibility.
  - [x] Evaluation harness with Random & Greedy baselines over 5 seeds.
  - [x] Safe arena allocation and Makefile targets (`release`, `debug`, `test`).
- [x] **Phase 1: Real Library Structure**
  - [x] Standard `Env` interface with separate `done` (terminal) and `truncated` (time limit) flags.
  - [x] Decoupled `include/` headers and `src/` library implementation.
  - [x] Dynamic arena-backed `RolloutBuffer` and parameter registry in `model_state`.
  - [x] Neural network builders (`nn_linear`, `nn_mlp`, `nn_build_discrete_actor`).
  - [x] Fully decoupled trainer (`rl_train_reinforce`) and optimizer (`optim_sgd_step`).
  - [x] Multiple environments: CartPole-v1 and Snake.
  - [x] Lean example scripts: `train_snake.c` and `train_cartpole.c`.
- [x] **Phase 2: Autograd Upgrades**
  - [x] Observation batching support with broadcast bias add and per-column independent softmax.
  - [x] New operators: elementwise `mul`, `tanh`, `mse` loss, fused policy cross-entropy loss.
  - [x] Adam optimizer with 1st/2nd moment tracking, bias correction, and global gradient norm clipping.
  - [x] CartPole-v1 solved: achieves **500.0 / 500.0** mean return over 100 episodes in 50 epochs.
- [x] **Phase 3: Batching & Snake Overhaul**
  - [x] Real batched GEMM training: input is $(obs\_dim \times B)$, logits $(act\_dim \times B)$.
  - [x] Snake env: 3-channel Markovian observation (head/food/body grids), 4 directional actions.
  - [x] Body tracking with array-backed segments, proper self-collision detection.
  - [x] Seeded food respawn avoiding body positions.
  - [x] Correct `done` vs `truncated` semantics.
  - [x] `backward_pass_from()` for seeding gradients at arbitrary graph nodes.
  - [x] `nn_linear_with_params()` for weight sharing between inference and batch graphs.
  - [x] Multi-seed benchmark harness (`make bench`).
- [x] **Phase 4: Advanced RL Algorithms**
  - [x] Generalized Advantage Estimation (GAE-λ).
  - [x] Actor-Critic network builder with shared trunk (`nn_build_actor_critic`).
  - [x] Advantage Actor-Critic (A2C) trainer (`rl_train_a2c`).
- [ ] **Phase 5: Performance & Systems**
  - Vectorized environments.
  - ARM NEON / AVX2 SIMD matmul kernels.
  - PPO with clipped surrogate objective.
