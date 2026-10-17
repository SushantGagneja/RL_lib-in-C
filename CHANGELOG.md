# Changelog

All notable changes to cRL are recorded here. This project has no released
version yet; entries below are grouped by the review that prompted them.

## Unreleased

### Fixed
- **A2C build crash and dead gradients.** `nn_build_actor_critic` no longer
  builds a fake `logits + value` root; the graph is built from multiple roots
  via the new `build_graph_multi`. The softmax is now part of the forward graph,
  and seeded gradients are no longer zeroed before backprop (`backward_pass_from`
  clears activations first, then the caller seeds). A2C also accumulates over
  all minibatches and takes a single optimizer step per rollout.
- **A2C seeding/entropy.** Removed the double division by batch size; added an
  entropy bonus (`mat_entropy_loss` / `mat_entropy_add_grad`,
  `RLConfig.entropy_coef`).
- **Snake.** Moving into the cell the tail vacates this step is now legal; each
  env owns its own `prng_state` and `reset` takes a seed; the silent 180-degree
  reversal override is gone; grid size is validated; a win state exists when the
  board fills.
- **Autograd graph build.** Replaced the O(n^2) stack scan with an iterative
  post-order DFS plus a visited-state array that asserts instead of silently
  dropping nodes.
- **REINFORCE/A2C batching.** The dropped `sample_count % batch_size` remainder
  is now a random shuffle of the sample order rather than a systematic tail.
- **Numerics.** `sample_action` uses strict `<`; `prng_randn_r` renamed to
  `prng_range_r` (it was always a uniform range); Adam clipping guards against
  non-finite global norms.
- **Failures are loud.** `CRL_ASSERT`/`CRL_CHECK` replace silent `false`
  returns, empty-graph no-ops, and unchecked arena pointers; `arena_push` aborts
  on OOM.
- **Arena.** 64-byte aligned backing store; `arena_pos` / `arena_pop_to` for
  temp scopes.
- **Benchmarks.** Identical evaluation seeds across policies, f64 Welford
  sample standard deviation, both greedy and stochastic returns, best and worst
  seed, a BFS Snake baseline, and machine-generated `bench_results.md`.
- **Warnings.** `-Wall -Wextra -Wshadow -Wconversion -Wmissing-prototypes` are
  clean.

### Added
- `make gradcheck-double`: builds the library in f64 (`-DCRL_USE_DOUBLE`, the
  new `real` scalar typedef) so the gradient check runs with `h = 1e-5` and a
  `1e-6` tolerance. Input/activation gradients and an elementwise `mul` with
  both operands requiring grad are now checked too.
- `tests/test_rl.c`: GAE against a hand-computed trajectory, a single Adam step,
  gradient-norm clipping, actor-critic construction and value-head learning,
  batched-vs-per-sample gradient equivalence, a CartPole learning smoke test,
  and same-seed determinism.
- `make asan` (ASAN + UBSAN) and a GitHub Actions workflow running `make test`
  with warnings as errors on gcc/clang/macOS, plus the sanitizer target.
- `examples/train_a2c.c`.

### Changed
- Removed the short-name macros from `mat.h`; everything uses the `mat_*` names.
- Extracted a `Linear` struct (`nn_linear_create` / `nn_linear_apply`) and
  deleted the three duplicated layer/init code paths and the
  `nn_linear_with_params` weight-sharing hack.
- `_mat_mul_*` kernels are `static` and named `mat_mul_kernel_*`.

### Known limitations
See the "Known limitations" section of the README.

### Blocked
- **License.** The core engine is derived from an upstream repository that ships
  no license, so this repository currently has no LICENSE file. This must be
  resolved before the code can be redistributed. See the README.
