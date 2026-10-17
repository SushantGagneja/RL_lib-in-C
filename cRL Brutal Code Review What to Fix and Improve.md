# cRL Brutal Code Review: What to Fix and Improve

Oct 18, 2026 · @Sushant Gagneja!!

## Verdict

The REINFORCE path is solid, but A2C is broken three different ways and the README says it is done. This is a static read: I did not compile or run anything, so every claim below comes from reading the code.

**What is genuinely good:** a clean arena allocator, a reverse-mode engine small enough to read in one sitting, a seeded PCG32, a fused log-softmax policy loss (the numerically right way to do it), and a real gradient-check test. For a learning project that is above average.

**What is overclaimed:** "high-performance" (no timing numbers anywhere), "no dynamic allocation" (graph building uses malloc and calloc), "bit-for-bit reproducibility" (libm differs across machines), "A2C with GAE" (crashes on build), and the benchmark table (weak baselines, no learning curves, hand-typed numbers). Roughly half of Phase 4 is checked off without a single test touching it.

The pattern behind most problems: **failures are silent.** Shape errors return false and the caller ignores them, a failed `build_graph` returns an empty graph and training quietly does nothing, and `NULL` from the arena is dereferenced without a check.

## P0: A2C is broken in three independent ways

`nn_build_actor_critic` plus `rl_train_a2c` cannot work today. Each bug alone is fatal.

1. **It aborts at build time.** In `nn.c`, `var_add(logits, model->value)` adds an (act\_dim x 1) matrix to a (1 x 1) matrix. `var_shape_broadcast` requires equal row counts, so `create_node` hits `CRL_ASSERT` and the process dies for any act\_dim other than 1. The same mistake is repeated for `batch_logits + batch_value`. Fix: stop using a fake `add` node as a graph root. Build the graph from a list of roots, or give `Graph` a `build_graph_multi(roots[], n)`.
2. **Softmax is never computed.** Even if the add worked, `forward_graph` is rooted at that combined node, which depends on `logits` and `value` but not on `model->output` (the softmax node). `model->output->val` stays all zeros, so `sample_action` returns the last action almost every time. Same root cause, same fix.
3. **The seeded gradients are wiped.** In `rl_train_a2c` you seed `batch_logits->grad` and `batch_value->grad`, then call `backward_pass_from(&batch_graph, NULL, false)`. With `loss_var == NULL`, the `cur == loss_var` skip never fires, so the clearing loop zeroes both seeded gradients before backprop starts. The model would receive zero gradient on every step. REINFORCE avoids this only because it passes `batch_logits` as `loss_var`. Fix: add a mode to `backward_pass_from` that clears everything except explicitly listed seeded nodes, or clear first and seed after.

**Related A2C problems**

- The optimizer steps once per minibatch inside the loop over the same on-policy rollout. After the first step the data is off-policy, and there is no ratio or clipping. That is PPO's problem without PPO's fix.
- `rl_compute_gae` is only called when an episode ends. If `config->max_steps` is smaller than the env's limit, the loop exits with neither `done` nor `truncated`, GAE never runs, and the buffer holds advantages from the previous epoch.
- No entropy bonus, so the policy collapses early.

**Do not tick Phase 4 again until** a test builds the actor-critic, runs one update on a toy env, and checks that the value head's loss goes down.

## P1: Correctness bugs and silent failures

| Where | Problem | Fix |
| --- | --- | --- |
| `snake.c` `snake_step` | Self-collision is checked against the whole body, including the tail cell that moves away this step. In real Snake, moving into the vacating tail is legal. You are penalising correct moves with -10, which also skews the trained-vs-greedy comparison. | Check collision against `body[0..len-2]` unless the snake just ate. |
| `snake.c` `snake_reset` | Stores the caller's `prng_state *` inside the env. After `rl_train_*` returns, that pointer dangles. It only works because `reset` is always called again first. It also makes envs impossible to vectorise or copy. | Each env owns its own `prng_state`; `reset` takes a seed. |
| `snake.c` | Silently replaces a 180-degree reversal with the previous action. The agent sees one action and gets another. `MAX_SNAKE_LEN` is 256 but `grid_size` is never validated against it. No win state when the board fills. | Mask or penalise illegal moves explicitly; validate grid size; add a win condition. |
| `rl.c` REINFORCE batch loop | `sample_count / batch_size` drops the remainder. Samples are ordered per trajectory in reverse time, so what is dropped is systematically the last trajectory's early steps. | Pad the final batch with zero-weight samples, or shuffle first. |
| `rl.c` config | `RLConfig.batch_size`, `verbose` and (for REINFORCE) `lambda` are ignored. Batch size is hard-wired by `batch_input->cols` at graph-build time. The config field is a lie. | Assert `config->batch_size == batch_input->cols` or remove the field. |
| `rl.c` REINFORCE | Monte-Carlo return uses `G = 0` at the end of every episode, so time-limit truncation is treated as a real terminal state. The README advertises `done` vs `truncated`, but the main trainer ignores the distinction. | Bootstrap from a value estimate on truncation, or document the bias. |
| `rl.c` fallback path | If `batch_graph` is missing, the code uses `cost_graph`. If that is also empty (a forgotten `build_graph`), `forward_pass` and `backward_pass` return early and training silently does nothing, then Adam steps on zero gradients. | `CRL_ASSERT(graph.size > 0)` at the top of the trainer. |
| `rl.c` per epoch | `batch_acts` and `batch_advs` are pushed onto the arena every epoch and never released. It is small, but it contradicts the "no allocation during training" claim and will bite on long runs. | Allocate once before the epoch loop, or add arena temp scopes (save and restore `pos`). |
| Everywhere | Almost every `mat_*` return value (`b32` for shape errors) is ignored, e.g. `mat_fused_policy_loss` in `rl.c`. An action index out of range returns false and learning silently stops. | Wrap in `CRL_ASSERT` (or a `CRL_CHECK` macro) in debug builds. |
| Everywhere | `arena_push` returns `NULL` on OOM, but `PUSH_STRUCT` in `model_create`, `env_*_create`, `rl_buffer_create` and `optim_adam_create` is dereferenced unchecked. `arena_create(GiB(1))` in examples and bench is also unchecked and can fail with overcommit disabled. | Make `arena_push` abort on OOM, or check every caller. |
| `autograd.c` `build_graph` | Hand-rolled DFS with an O(n^2) "find and move to top" loop on a manual stack. Bounds guards (`stack_size < num_vars`) silently drop nodes instead of failing. Uses malloc and calloc in a library that claims to be arena-only. | Standard iterative post-order with a visited array; assert on overflow. |
| NaN handling | No finite check anywhere. A NaN gradient makes `norm > max_norm` false, so clipping is skipped and NaN goes straight into Adam's moments. | `isfinite` check on the global norm in debug mode. |
| `prng.c` / `rl.c` | `sample_action` uses `r <= cdf`, so r == 0 can pick a zero-probability action. Tiny, but it should be `<`. `prng_randn_r` is a uniform integer, not a normal; the name will mislead everyone. | Use `<`; rename to `prng_range_r`. |

## P1: The benchmark table does not prove what the README says

The headline numbers (499.97 on CartPole, +63% over greedy on Snake) are the first thing a reader sees, and they are the weakest part of the repo.

- **"Greedy (PD controller)" is not a weak baseline, and the Snake one is a strawman.** The CartPole controller scores 485.65, which is a hand-tuned solver. The Snake "greedy" agent walks straight at the food and ignores walls and its own body, so beating it at 6x6 means little. Add a wall-and-body-avoiding greedy or BFS agent before claiming a win.
- **50 optimizer steps to solve CartPole is suspicious.** In the batched path there is exactly one Adam step per epoch, after accumulating over all minibatches. 50 epochs means 50 updates. That might be real (greedy argmax evaluation flatters a mediocre policy, and the initial state noise is tiny), but I would not publish it without a learning curve. Logging prints only epoch 1 and epoch 50, so no curve exists.
- **Evaluation is deterministic argmax; training reports stochastic returns.** The table should show both, since the stochastic return is the one the algorithm actually optimised.
- **Standard deviation is computed in f32 with `E[x^2] - mean^2`.** For values near 500 that cancels catastrophically. The quoted "+/- 0.12" is within float noise. Use Welford's method in f64, and report the sample (n-1) std, not the population std, across only 5 seeds. Add confidence intervals.
- **Different RNG streams per policy.** Random and greedy share one stream sequentially, so their start states differ from the trained policy's (`seed + 5000`). Use identical eval seeds for every policy.
- **Cherry-picking.** "Best seed (777) reached 93.1" next to a +/-13.7 std says the method is high variance. Report the worst seed too.
- **Numbers are typed by hand.** The README says they are "produced by `make bench`", but nothing in the repo writes the table. Have `bench` emit markdown or CSV and paste that.
- **CartPole fidelity.** Your theta threshold is 0.2095 (Gymnasium uses 12 \* 2pi / 360 = 0.20944), and physics runs in f32 where Gymnasium uses f64. Small, but it means "CartPole-v1" is "CartPole-like". Say so or match it.
- **Unsubstantiated adjectives.** "High-performance", "bit-for-bit reproducible" and "zero fragmentation" have no evidence behind them. Either measure (steps per second, GEMM GFLOPS against a naive baseline) or delete the words. Bit-for-bit reproduction across machines will not survive differing `expf`/`logf` implementations or `-O3` reassociation.

## P1: The tests are good at the bottom and absent at the top

The counts in the README (12, 210, 25) are accurate, but they all test leaf operations. The code that actually fails is untested.

- **Gradcheck is looser than it looks.** The relative error is `|n - a| / (|n| + |a|)`. Because the denominator is the sum of both, a gradient that is 2% wrong passes a `1e-2` tolerance. It also runs in f32 with `h = 1e-2`, which can step across a ReLU kink. Compile the gradcheck in f64 (a `real` typedef), use `h = 1e-5`, and tighten the tolerance to `1e-4` or better.
- **Only parameter gradients are checked.** Input and activation gradients (the `mul` op with both sides requiring grad, for example) are never compared against finite differences.
- **The production loss path has no end-to-end gradcheck.** The fused policy loss is checked on a bare logits matrix. Nothing checks that the batched graph plus fused loss gives the same parameter gradient as summing per-sample gradients through the single-sample graph. That one test would have caught any batching bug.
- **Untested modules:** `nn.c` (all builders), `optim.c` (Adam, clipping), `rl.c` (trainers, GAE, evaluate). A GAE unit test against a hand-computed 3-step trajectory takes ten minutes to write.
- **No learning smoke test.** Add a fast CartPole run (20 epochs, fixed seed) asserting the return rises above a floor, and a determinism test that trains twice with the same seed and compares weight checksums. Neither is in `make test` today.
- **The self-collision test is fragile.** It casts `env->state` to a duplicate `SnakeTestState` struct that must match the private layout of `SnakeState` byte for byte. Reordering one field in `snake.c` makes the test write to garbage. Expose a `snake_test_set_body()` helper or move the test into the same translation unit.
- **Tests use `assert()` for setup**, which vanishes under `-DNDEBUG`. Use `CRL_ASSERT` everywhere.
- **No sanitizers, no CI.** Add `-fsanitize=address,undefined` to a `make asan` target and run it in a GitHub Actions workflow with `-Wall -Wextra -Wshadow -Wconversion -Wmissing-prototypes`. I expect that last flag alone to flag the non-static `_mat_mul_*` functions and the macro-shadowed names below.

## P2: Design and API problems

- **The short-name macros are a trap.** `mat.h` defines `clear`, `fill`, `scale`, `add`, `sub`, `mul`, `relu`, `softmax` and more as macros for every file that includes it, directly or through `autograd.h`, `nn.h`, `optim.h` or `rl.h`. Any variable or parameter called `scale` or `add` is silently renamed. This already happens: the `scale` parameter of `optim_sgd_step` in `optim.c` is rewritten to `mat_scale`, and it compiles only by accident. Worse, the opt-out `CRL_NO_SHORTNAMES` does not work, because `mat.c` and `autograd.c` define and call the short names. Define the functions under their `mat_` names, delete the macros, and sed the callers.
- **Reserved and exported names.** `_mat_mul_nn/nt/tn/tt` start with an underscore at file scope (reserved) and are non-static with no prototype. Make them `static`.
- **Three copies of the same layer-building code.** Xavier init and the W/b creation are pasted into `nn_linear`, `nn_build_discrete_actor` and `nn_build_actor_critic`. Introduce a `Linear { Var *W, *b; }` struct with `linear_create()` and `linear_apply(in)`. That also kills the `nn_linear_with_params` weight-sharing hack.
- **Two parallel graphs for one network.** Single-sample inference and batched training are separate graphs sharing parameter pointers, with the batch size frozen at build time. Either let a graph rebind its input shape (inference is just B=1), or keep one graph and run rollouts at batch width.
- **Fixed-size limits with no growth path.** `MODEL_MAX_PARAMS` is 128 and `VAR_MAX_INPUTS` is 2. Both fail with an abort or a silent drop. Fine for now; document it.
- **Env ownership.** `max_steps` lives in both `Env` and `RLConfig`, and they can disagree (see the GAE bug above). `observe` is separate from `step`, which forces a second call to get the next observation. `destroy` is a no-op in both envs. The env needs its own RNG and a `seed()` function before vectorisation (Phase 5) is possible.
- **Initialisation.** Xavier-uniform with ReLU is suboptimal (use He), and RL practice is a tiny-gain policy head (about 0.01) so the initial policy is near uniform. Cheap to try and likely to help Snake.
- **Arena alignment.** `ARENA_ALIGN` is `sizeof(void *)`, which is 8 bytes. Your Phase 5 plan is NEON and AVX2, which want 32 or 64. Do this now, before data layouts depend on it.
- **Matmul kernels.** The NT variant strides across `b` in the innermost loop, which is cache-hostile, and it is the one used for input gradients. Fine at 128x128, but the README calls the library high-performance. Benchmark first, then decide whether SIMD is worth it.
- **`base.h` includes everything** (stdio, stdlib, string, assert) in every translation unit. Trim it; headers should include only what they use.

## P0: Licensing and repo hygiene

- **No LICENSE, on a derived work.** The README says the core engine, arena and matrix layout come from another author's repository, which itself has no license. Without a license, that code is all-rights-reserved by default, and you cannot grant anyone permission to use your derivative. Public attribution does not fix this. Ask the original author for explicit permission (a GitHub issue works) or rewrite those parts yourself, then add a LICENSE. This is the one item on this list with real-world consequences.
- **PCG attribution.** `prng.c` carries a one-line note. Apache 2.0 expects the license text and notices to ship with the code; add the NOTICE.
- **The Makefile arrived as a binary file.** That usually means a BOM, UTF-16, or stray NUL bytes, and either will break `make` on Linux or macOS. I could not review it. Check it with `file Makefile` and re-save it as plain UTF-8 with LF line endings.
- **README drift.** The architecture tree lists `crl-improvement-plan.md`, which is not in the repository you gave me. "Zero dependencies" is true of third-party libraries but not libm or POSIX make. The roadmap ticks Phase 4.
- **Missing basics:** `.gitignore` (build outputs), CI config, a CHANGELOG, and a one-paragraph "known limitations" section in the README. Listing limitations yourself earns more trust than a reader discovering them.

## Fix order

**P0: do before anyone else reads the repo**

- [ ] Resolve the license situation with the upstream author, then add a LICENSE
- [ ] Untick Phase 4 in the README until A2C works
- [ ] Fix the three A2C bugs: shape-mismatched root `add`, softmax missing from the forward graph, seeded gradients wiped by `backward_pass_from(NULL)`
- [ ] Re-save the Makefile as plain UTF-8 and confirm `make test` works from a clean clone

**P1: this week**

- [ ] Snake tail-vacate collision fix, and give each env its own RNG
- [ ] Make failures loud: assert on `mat_*` return values, graph size, and arena OOM
- [ ] Float64 gradcheck with a tight tolerance, plus a batched-vs-per-sample gradient equality test
- [ ] Unit tests for GAE, Adam, grad clipping and the network builders
- [ ] A CartPole learning smoke test and a same-seed determinism test in `make test`
- [ ] Rework `bench`: identical eval seeds, f64 Welford std, sample std, both stochastic and greedy returns, learning curves, output written to a file
- [ ] Stronger Snake baseline (wall and body avoiding, or BFS)
- [ ] ASAN/UBSAN target and a CI workflow with strict warnings

**P2: when the above is green**

- [ ] Delete the short-name macros and rename internals to `mat_*`
- [ ] Extract a `Linear` struct and remove the three copies of the init code
- [ ] One graph with rebindable batch size instead of two parallel graphs
- [ ] Align the arena to 64 bytes, add temp scopes, measure steps per second and GEMM throughput
- [ ] Only then: vectorised envs, SIMD, PPO
