# cRL: Compact Reinforcement Learning & Autograd Library in C

A minimalist, high-performance C library for autograd, neural networks, and reinforcement learning built from scratch with zero external dependencies (standard C runtime and `math.h` only).

---

## Features

- **Lightweight Autograd Engine (~350 LOC)**:
  - Reverse-mode automatic differentiation with a static computation graph.
  - Vtable-style operator representation (`VarType`) with shape verification and forward/backward passes.
  - Core operations: Matrix multiplication (all four transpose permutations: NN, NT, TN, TT), ReLU, numerically stable Softmax, Addition, Subtraction, and REINFORCE loss.
  - Gradient accumulation model designed for batch and minibatch optimization.
- **Arena Memory Management**:
  - Bump allocator with power-of-2 alignment avoiding fragmentation and dynamic reallocation during training.
- **Reproducible PCG32 PRNG**:
  - Deterministic pseudo-random number generation with isolated generator states per environment and trainer.
- **Verification & Rigorous Testing**:
  - Exact numerical gradient checks comparing analytical backward passes against central finite differences.
  - Full mathematical unit tests for matrix operations and numerical edge cases.
- **Benchmarks & Baseline Harness**:
  - Multi-seed evaluation framework comparing trained policies against random and hand-coded greedy heuristic baselines.

---

## Benchmarks & Baselines

Tested on the grid environment across 5 distinct random seeds (`42`, `100`, `2024`, `777`, `9999`), evaluated over 100 episodes per seed (500 total evaluation episodes):

| Policy | Foods Eaten (mean ± std) | Return (mean ± std) | Mean Steps / Episode | Description |
|---|---|---|---|---|
| **Greedy Heuristic Baseline** | **24.76 ± 0.18** | **495.12 ± 3.67** | **100.0** | Hand-coded heuristic stepping along the axis of maximal food displacement. |
| **Random Baseline** | 0.26 ± 0.01 | -4.72 ± 0.16 | 13.7 | Uniform random action selection over directional inputs. |
| **Trained Policy (REINFORCE)** | 0.10 ± 0.04 | -7.44 ± 1.02 | 9.9 | Standard REINFORCE with rollout-wide mean baseline. |

> **Note on Plain REINFORCE**: As analyzed in [`crl-improvement-plan.md`](./crl-improvement-plan.md), standard vanilla REINFORCE without an entropy bonus or learned per-state value baseline exhibits policy collapse on sparse grid rewards. Phase 1 & 2 introduce modular environment interfaces, learned critic baselines (A2C), and PPO to solve this.

---

## Build & Test

### Prerequisites
- Any C99-compliant compiler (`gcc` or `clang`)
- POSIX `make`

### Commands

```bash
# Build release binary (-O3)
make release

# Run mathematical unit tests and autograd gradient checks
make test

# Run default training with seed 42
make run

# Run multi-seed benchmark (5 seeds, 150 epochs)
make bench

# Clean build artifacts
make clean
```

### Test Suite Output

Running `make test`:
```
=== Running test_mat ===
[matmul variants]
  matmul NN... OK
  matmul NT... OK
  matmul TN... OK
  matmul TT... OK

[softmax]
  softmax sums to 1... OK
  softmax all positive... OK
  softmax large values (no overflow)... OK

[relu]
  relu... OK

[add/sub]
  add/sub... OK
---
Passed: 9  Failed: 0

=== Running test_gradcheck ===
[gradcheck: matmul → relu → matmul → add]
  W0 (3x4, 12 params)... OK (0 failures)
  W1 (2x3, 6 params)... OK (0 failures)
  b1 (2x1, 2 params)... OK (0 failures)

[gradcheck: matmul → softmax]
  W (5x3, 15 params)... OK (0 failures)
  b (5x1, 5 params)... OK (0 failures)

[gradcheck: matmul → sub]
  W (4x3, 12 params)... OK (0 failures)
  b (4x1, 4 params)... OK (0 failures)

[gradcheck: full actor → reinforce loss]
  W0 (6x8, 48 params)... OK (0 failures)
  b0 (6x1, 6 params)... OK (0 failures)
  W1 (4x6, 24 params)... OK (0 failures)
  b1 (4x1, 4 params)... OK (0 failures)
  W2 (3x4, 12 params)... OK (0 failures)
  b2 (3x1, 3 params)... OK (0 failures)
---
Tests: 4 passed, 0 failed
Parameters checked: 153/153 passed within tolerance
```

---

## Architecture Overview

```
.
├── arena.h / arena.c         # Memory arena allocator
├── prng.h / prng.c           # PCG32 PRNG with re-entrant state structs
├── mat.h / mat.c             # Matrix data structure and linear algebra
├── autograd.h / autograd.c   # Static computation graph and reverse-mode AD
├── model.h / model.c         # Neural network layer assembly
├── env.c                     # Environment, training loop, evaluation harness
├── tests/
│   ├── test_mat.c            # Math layer unit tests
│   └── test_gradcheck.c      # Finite-difference numerical gradient checks
├── crl-improvement-plan.md   # Architectural roadmap and phased milestones
└── Makefile                  # Build and test orchestration
```

---

## Roadmap

- [x] **Phase 0: Credibility**
  - [x] Unit tests for all matrix operations (`test_mat`).
  - [x] Numerical gradient checking for autograd (`test_gradcheck`).
  - [x] Isolated deterministic PRNG seeding & bit-for-bit reproducibility.
  - [x] Evaluation harness with Random & Greedy baselines over 5 seeds.
  - [x] Safe arena allocation and Makefile targets (`release`, `debug`, `test`, `bench`, `run`).
- [ ] **Phase 1: Library Structure**
  - Modular environment interface (`Env` struct with `reset`, `step`, `observe`).
  - Decoupled `RolloutBuffer`, neural network helpers (`nn_linear`, `nn_mlp`), and optimizer abstraction.
  - Separation of library code from examples (`train_snake.c`, `train_cartpole.c`).
- [ ] **Phase 2: Autograd Upgrades**
  - Observation batching $(obs\_dim \times B)$ with broadcast bias add and per-column softmax.
  - New operators: `mul`, `tanh`, `mse`, `entropy`, and fused clipped objectives.
  - Adam optimizer and global gradient norm clipping.
- [ ] **Phase 3: Standard Benchmarks**
  - Classic CartPole-v1 implementation.
  - Complete Snake with ring-buffer body and relative actions.
- [ ] **Phase 4: Advanced RL Algorithms**
  - REINFORCE with learned value baseline.
  - Advantage Actor-Critic (A2C) and Generalized Advantage Estimation (GAE).
  - Proximal Policy Optimization (PPO).
- [ ] **Phase 5: Performance & Systems**
  - Vectorized environments.
  - ARM NEON / AVX2 SIMD matmul kernels.
