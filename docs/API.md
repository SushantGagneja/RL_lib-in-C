# RL_lib-in-C — API reference

Everything the library exposes lives in `include/` (plus the shipped environments
in `envs/`). There are no hidden translation units and no global state that a
program has to initialise: one arena, one random generator, and you are running.

```c
#include "layers.h"   // Dense / MLP / actor-critic builders
#include "rl.h"       // GAE, rollout buffers, REINFORCE / A2C / PPO, evaluation
#include "envs/cartpole.h"
```

## Contents

| Header | Provides |
|---|---|
| [`base.h`](#baseh) | scalar types (`real`), fixed-width ints, assertion macros |
| [`version.h`](#versionh) | `RL_LIB_VERSION_*` |
| [`memory.h`](#memoryh) | `Arena` bump allocator |
| [`random.h`](#randomh) | `Rng` (xoshiro256\*\*) |
| [`linalg.h`](#linalgh) | `Mat` plus every matrix / activation / loss / gradient kernel, including the squashed-Gaussian policy kernels |
| [`simd.h`](#simdh) | matrix-multiply backend selection and probing |
| [`graph.h`](#graphh) | `Tensor`, `Net`, reverse-mode autograd |
| [`layers.h`](#layersh) | `Dense`, `mlp_new`, `actor_build`, `actor_critic_build`, `continuous_actor_critic_build` |
| [`optim.h`](#optimh) | `zero_grad`, `sgd_step`, `Adam`, `clip_grad_norm` |
| [`checkpoint.h`](#checkpointh) | `net_save` / `net_load` / `checkpoint_save` / `checkpoint_load` |
| [`env.h`](#envh) | the `Env` interface every environment implements, discrete and continuous |
| [`vecenv.h`](#vecenvh) | `VecEnv`, synchronised parallel environments |
| [`rl.h`](#rlh) | `TrainCfg`, `Rollout`, `train_reinforce`, `train_a2c`, `train_ppo`, `train_ppo_continuous`, `evaluate`, `evaluate_continuous` |
| [`envs/`](#environments) | CartPole, Grid Snake, Pendulum-v1 |

Three conventions run through the whole API and are worth reading first:

* **Ownership.** Anything that allocates takes an `Arena *`. Objects live until
  the arena is disposed; nothing needs an individual `free`.
* **Errors.** Shape mistakes and out-of-memory conditions call
  `RL_LIB_ASSERT`, which prints `[RL_lib-in-C FATAL] ...` and aborts. Functions
  returning `b32` fail *softly* (`false`) for conditions a caller can legitimately
  handle — chiefly file I/O in `checkpoint.h`.
* **Precision.** `real` is `float` by default and `double` with
  `-DRL_LIB_USE_DOUBLE`. Use the `real_*` wrappers (`real_sqrt`, `real_exp`, …)
  and `REAL_C(x)` for literals instead of the libm float/double variants.

---

## `base.h`

```c
typedef int8_t i8;   typedef uint8_t  u8;   typedef float  f32;
typedef int16_t i16; typedef uint16_t u16;  typedef double f64;
typedef int32_t i32; typedef uint32_t u32;
typedef int64_t i64; typedef uint64_t u64;

typedef uint8_t  b8;   // byte-sized boolean
typedef uint32_t b32;  // word-sized boolean
```

`real` is the scalar type of the math stack: `float`, or `double` when the
library is built with `-DRL_LIB_USE_DOUBLE`. Alongside it come
`real_fabs/real_sqrt/real_exp/real_log/real_tanh/real_sin/real_cos/real_pow/real_fmax/real_fmin`,
the continuous-policy helpers `real_atanh/real_log1p/real_fmod`, `REAL_C(x)` (an
exactly-representable literal for the current mode) and `REAL_EPS`.

```c
#define KiB(n) ((u64)(n) << 10)   // also MiB(n), GiB(n)
#define MIN(a, b) ...             // also MAX(a, b)
#define ALIGN_UP_POW2(x, align) ...

RL_LIB_ASSERT(expr, msg);  // prints file/line and aborts when expr is false
RL_LIB_CHECK(expr);        // aborts when a b32-returning call fails
```

---

## `version.h`

```c
#define RL_LIB_VERSION_MAJOR 1
#define RL_LIB_VERSION_MINOR 0
#define RL_LIB_VERSION_PATCH 0
#define RL_LIB_VERSION_STRING "1.0.0"
```

`RL_LIB_VERSION_STRING` is also what the Makefile stamps into `rl_lib.pc`, so a
version can never drift between the headers and pkg-config. See
[CHANGELOG.md](../CHANGELOG.md) for the compatibility policy.

---

## `memory.h`

```c
#define ARENA_ALIGN 64   // every allocation is aligned to this

typedef struct { u8 *base; u64 cap; u64 offset; } Arena;

Arena *arena_new(u64 capacity);
void   arena_dispose(Arena *arena);
void  *arena_alloc(Arena *arena, u64 size, b32 zero);
u64    arena_mark(const Arena *arena);
void   arena_rewind(Arena *arena, u64 mark);
```

```c
#define ARENA_OBJ(arena, T)         (T *)arena_alloc((arena), sizeof(T), true)
#define ARENA_ARR(arena, T, n)      (T *)arena_alloc((arena), sizeof(T) * (n), true)
#define ARENA_ARR_RAW(arena, T, n)  (T *)arena_alloc((arena), sizeof(T) * (n), false)
```

A bump allocator. `arena_alloc` never returns `NULL`: exhausting an arena is
treated as unrecoverable and aborts with a message, because every other option
turns a size mistake into a crash somewhere less useful. Alignment is
`ARENA_ALIGN`, which is what lets `Mat` data be loaded with wide SIMD stores.

`arena_mark`/`arena_rewind` give you a temporary scratch scope inside an arena
that keeps its long-lived objects:

```c
u64 mark = arena_mark(arena);
/* ... throwaway graphs ... */
arena_rewind(arena, mark);
```

**Sizing.** One `arena_new(GiB(1))` comfortably holds a small-network training
setup including a rollout buffer; the examples use exactly that.

---

## `random.h`

```c
typedef struct { u64 lane[4]; } Rng;

void rng_seed(Rng *rng, u64 seed, u64 stream);
void rng_seed_g(u64 seed, u64 stream);            // seeds the global Rng
u64  rng_u64(Rng *rng);   u32 rng_u32(Rng *rng);
u32  rng_below(Rng *rng, u32 range);              // unbiased, [0, range)
real rng_unit(Rng *rng);                          // [0, 1), 53-bit
u64  rng_u64_g(void);     u32 rng_u32_g(void);
u32  rng_below_g(u32 range);
real rng_unit_g(void);
```

xoshiro256\*\* seeded through SplitMix64. State is 256 bits and must never be
all-zero, which `rng_seed` guarantees. `stream` is an independent tag, so two
generators seeded with the same value still diverge — this is how the vectorized
environment gives each sub-environment its own trajectory.

Determinism: a fixed seed produces identical runs on the same binary. Floating
point reduction order is fixed and machine-independent in the scalar kernels, but
SIMD kernels intentionally use different accumulation orders (see
[`simd.h`](#simdh)), so bit-exact cross-machine equality is not a goal.

The `*_g` variants touch a process-global generator. They exist for convenience
in tests and throwaway scripts; library code always takes an `Rng *`.

---

## `linalg.h`

```c
typedef struct { u32 rows, cols; real *data; } Mat;   // row-major
```

### Allocation and filling

```c
Mat *mat_new(Arena *arena, u32 rows, u32 cols);
void mat_zero(Mat *mat);
void mat_fill(Mat *mat, real x);
void mat_rand(Mat *mat, real lower, real upper);
void mat_rand_r(Mat *mat, Rng *rng, real lower, real upper);
void mat_scale(Mat *mat, real scale_factor);
```

### Algebra

```c
b32 mat_add(Mat *out, const Mat *a, const Mat *b);
b32 mat_sub(Mat *out, const Mat *a, const Mat *b);
b32 mat_mul(Mat *out, const Mat *a, const Mat *b);
b32 mat_mm(Mat *out, const Mat *a, const Mat *b, b8 zero_out, b8 transpose_a, b8 transpose_b);
b32 mat_mm_scalar(Mat *out, const Mat *a, const Mat *b, b8 zero_out, b8 transpose_a, b8 transpose_b);
```

`mat_mm` is the general matrix multiply: `zero_out` selects accumulation into
`out` versus overwriting it (which also matters for gradient accumulation), and
the transpose flags select among NN/NT/TN/TT. Shapes are validated; the return
value is `false` on a mismatch. `mat_add`/`mat_sub`/`mat_mul` broadcast a
`1 x n` or `m x 1` operand against the other argument, which is what makes
`y = Wx + b` a single `mat_add`.

`mat_mm_scalar` is the portable reference implementation, kept permanently so the
vectorized path can be cross-checked against it. `mat_mm` dispatches to SIMD when
the build and the CPU support it, and otherwise calls exactly this.

### Activations

```c
b32 mat_relu(Mat *out, const Mat *in);
b32 mat_tanh(Mat *out, const Mat *in);
b32 mat_softmax(Mat *out, const Mat *in);   // per column: softmax over rows
```

`mat_softmax` is column-wise: with a `(classes x batch)` input each column is an
independent distribution, which is what a batched policy needs.

### Losses

```c
b32 mat_mse_loss(Mat *out, const Mat *pred, const Mat *target);
b32 mat_reinforce_loss(Mat *out, const Mat *probs, const Mat *advantages);
b32 mat_fused_policy_loss(Mat *out, const Mat *logits, const u32 *actions,
                          const real *advantages, u32 batch_size);
b32 mat_entropy_loss(Mat *out, const Mat *logits, u32 batch_size);
b32 mat_ppo_policy_loss(Mat *out, const Mat *logits, const u32 *actions,
                        const real *advantages, const real *old_logp,
                        real clip_eps, u32 batch_size);
```

The fused policy losses take pre-softmax `logits` (`act_dim x batch`) plus the
chosen `actions` and per-sample `advantages`, and compute the mean loss over the
batch together with a numerically stable gradient — no intermediate `probs`
materialised, no `log(0)`. `mat_entropy_loss` returns the mean policy entropy
`H = -sum p log p` (natural log) into a `1 x 1` Mat, used as an exploration bonus.

### Gradients

Every gradient function *accumulates* into its output; none of them zero anything
first. That is deliberate: a minibatch loop zeroes once per step and accumulates
over the minibatch.

```c
b32 mat_add_broadcast_grad(Mat *a_grad, Mat *b_grad, const Mat *grad);
b32 mat_sub_broadcast_grad(Mat *a_grad, Mat *b_grad, const Mat *grad);
b32 mat_mul_add_grad(Mat *a_grad, Mat *b_grad, const Mat *a, const Mat *b, const Mat *grad);
b32 mat_relu_add_grad(Mat *out, const Mat *in, const Mat *grad);
b32 mat_tanh_add_grad(Mat *out, const Mat *tanh_out, const Mat *grad);
b32 mat_softmax_add_grad(Mat *out, const Mat *softmax_out, const Mat *grad);
b32 mat_mse_add_grad(Mat *pred_grad, const Mat *pred, const Mat *target, const Mat *grad);
b32 mat_reinforce_add_grad(Mat *prob_grads, const Mat *probs, const Mat *rt, const Mat *grad);
b32 mat_fused_policy_add_grad(Mat *logits_grad, const Mat *logits, const u32 *actions,
                              const real *advantages, u32 batch_size, const Mat *grad);
b32 mat_entropy_add_grad(Mat *logits_grad, const Mat *logits, u32 batch_size, const Mat *grad);
b32 mat_ppo_policy_add_grad(Mat *logits_grad, const Mat *logits, const u32 *actions,
                            const real *advantages, const real *old_logp, real clip_eps,
                            u32 batch_size, const Mat *grad);
```

`grad` is the upstream gradient (`dL/d(output)`) and is usually a `1 x 1` Mat
holding the loss weight. `mat_tanh_add_grad` takes the *forward output* rather
than the input, since `d tanh = 1 - tanh^2`.

The loss-specific entries are exact rather than numerically differentiated: their
gradients are cross-checked against finite differences in `tests/test_gradcheck.c`,
which is what keeps the closed forms honest.

### Continuous (squashed Gaussian) policy

The continuous policy is a diagonal Gaussian over a pre-squash coordinate `u`,
squashed into the environment's action box with `tanh`:

```
u_d = mean_d + exp(log_std_d) * eps,   eps ~ N(0, 1)
a_d = center_d + half_d * tanh(u_d)
log pi(a) = log N(u | mean, sigma) - log(half_d) - log(sech^2(u_d))
```

with `center_d` / `half_d` the midpoint and half-width of
`[action_min[d], action_max[d]]` (per dimension, so asymmetric boxes work). The
density carries the squash's log-determinant — the construction SAC and
Stable-Baselines3 use.

```c
#define RL_GAUSSIAN_MEAN_BOUND  ((real)5)    // mean clamp half-width
#define RL_GAUSSIAN_LOG_STD_MIN ((real)-6)   // sigma >= ~2.5e-3
#define RL_GAUSSIAN_LOG_STD_MAX ((real)2)    // sigma <= ~7.4

b32 mat_gaussian_sample(Mat *action_out, Mat *u_out, const Mat *raw_mean,
                        const Mat *log_std, const real *min_action,
                        const real *max_action, Rng *rng);
b32 mat_gaussian_unsquash(Mat *u_out, const Mat *actions, const real *min_action,
                          const real *max_action);
b32 mat_gaussian_logp(Mat *out, const Mat *raw_mean, const Mat *log_std,
                      const Mat *u, const real *min_action,
                      const real *max_action, u32 batch_size);
b32 mat_gaussian_ppo_loss(Mat *out, const Mat *raw_mean, const Mat *log_std,
                          const Mat *u, const real *advantages,
                          const real *old_logp, const real *min_action,
                          const real *max_action, real clip_eps, u32 batch_size);
b32 mat_gaussian_ppo_add_grad(Mat *mean_grad, Mat *log_std_grad,
                              const Mat *raw_mean, const Mat *log_std,
                              const Mat *u, const real *advantages,
                              const real *old_logp, const real *min_action,
                              const real *max_action, real clip_eps,
                              u32 batch_size, const Mat *grad);
b32 mat_gaussian_entropy(Mat *out, const Mat *log_std);
b32 mat_gaussian_entropy_add_grad(Mat *log_std_grad, const Mat *log_std,
                                  const Mat *grad);
```

**The sample is the input, not the action.** `mat_gaussian_sample` writes both
the squashed action *and* the `u` it came from, and the loss, the log-density and
the gradient all take `u`. Recovering `u` from the action instead needs `atanh`,
which is infinite at the bounds; once a mean drifts outside the box the recovered
value stops describing the sample that was actually drawn (this cost a training
run: log-probabilities off by hundreds of nats, then a non-finite gradient).
`mat_gaussian_unsquash` is provided for actions that were not sampled here (a
stored replay, say), and is documented as lossy near the bounds for that reason.
`raw_mean` may be `act_dim x 1` (broadcast over the batch) or `act_dim x batch`;
`log_std` is always `act_dim x 1`.

Three further guards keep this finite on a bounded action space:

* `log(sech^2(u)) = 2 * (log 2 - |u| - log1p(exp(-2|u|)))`, which stays finite
  for any `u`, unlike `log(1 - tanh(u)^2)` computed through `tanh`, which is
  `-inf` as soon as `tanh` saturates in f32 (`|u| > ~9`).
* The mean is clamped to `±RL_GAUSSIAN_MEAN_BOUND` on read, with the gradient
  passed straight through, so the density cannot spike outside the box (its peak
  grows like `exp(2|mean|)`) while the mean head never loses its gradient.
* `log_std` is clamped to `[RL_GAUSSIAN_LOG_STD_MIN, RL_GAUSSIAN_LOG_STD_MAX]` on
  read, which floors `sigma` (no division by zero, no collapsed exploration) and
  zeroes the gradient while the clamp is active.

The score gradients are the plain Gaussian ones, because the squash's Jacobian
depends on the sample but not on `mean` or `log_std`:
`d log pi / d mean = (u - mean) / sigma^2` and
`d log pi / d log_std = (u - mean)^2 / sigma^2 - 1`.

`mat_gaussian_entropy` returns the mean *pre-squash* Gaussian entropy
`H = sum_d log(sigma_d) + D/2 * log(2*pi*e)`, the usual stand-in for the entropy
bonus (the squashed density's entropy has no closed form).

---

## `simd.h`

```c
#define RL_SIMD_NONE 0
#define RL_SIMD_NEON 1
#define RL_SIMD_AVX2 2

const char *simd_backend_name(void);  // "scalar" | "neon" | "avx2"
b32         simd_accelerated(void);   // usable on *this* CPU?
```

The backend is chosen at compile time, per precision:

| Platform | Backend |
|---|---|
| AArch64 | `RL_SIMD_NEON` — 4-wide f32, or 2-wide f64 |
| x86-64 (GCC/Clang) | `RL_SIMD_AVX2` — 8-wide f32, or 4-wide f64, with per-function `target` attributes |
| anything else | `RL_SIMD_NONE` — the scalar reference |

Double precision on NEON needs AArch64: ARMv7 NEON loads 64-bit lanes but has no
double-precision arithmetic, so it selects the scalar path (and asking for
`RL_SIMD_NEON` there with `-DRL_LIB_USE_DOUBLE` is a compile-time `#error`).

Because the AVX2 kernels are compiled with per-function target attributes,
`simd_accelerated()` performs a cached runtime CPU probe: the binary still runs
on a CPU without AVX2 and simply falls back. Force the scalar path with
`-DRL_SIMD_BACKEND=0`.

All four transpose cases have a vector path. NN and TN process two output rows at
once, so every element of B is loaded once for two rows; each output element
still accumulates over k in ascending order, which makes them element-wise
identical to a single-row kernel. TT transposes a 32×32 tile of B into the TN
form, which costs one O(N·K) pass against the O(M·N·K) multiply. NT vectorises
the K reduction and adds the lane partials horizontally, so — like TT's extra
pass — those two can differ from the scalar reference by a few ulps; the tests
bound that instead of forbidding it, in both precisions.

---

## `graph.h`

```c
typedef enum { TENSOR_PLAIN = 0, TENSOR_GRAD = 1, TENSOR_PARAM = 2 } TensorFlags;
typedef enum { OP_NONE = 0, OP_RELU, OP_TANH, OP_SOFTMAX, OP_ADD, OP_SUB, OP_MUL,
               OP_MM, OP_MSE, OP_REINFORCE } OpKind;

struct Tensor {
  u32 index; u32 flags; u8 arity; OpKind op;
  Mat *val; Mat *grad;
  Tensor *inputs[MAX_ARITY];
};

typedef struct { Tensor **ops; u32 len; } Schedule;

typedef struct {
  u32 n_tensors;
  Tensor **params;       // grows through the arena; no fixed cap
  u32 n_params, params_cap;
  NetArchSpec *arch;     // private record of the builder's architecture
  Tensor *input, *output, *advantage, *cost;
  Tensor *batch_input, *batch_logits, *batch_value, *value;
  Schedule fwd, train, loss_plan;
} Net;
```

A `Tensor` is a node holding a value, an optional gradient, and an operation tag.
Forward and backward both live in `src/graph.c` and dispatch on that tag in one
place — there is no per-operation function-pointer table to keep in sync.

`Net` is a flat, topologically ordered registry of tensors plus the schedules
built from them. Parameter tensors carry `TENSOR_PARAM`; `grad` exists only on
tensors created with `TENSOR_GRAD`. `advantage` and `cost` are the slots the
trainers seed loss gradients from. The parameter table starts at 16 entries and
doubles through the arena as parameters are registered, so a model is not capped
at an arbitrary parameter count.

```c
Tensor *tensor_create(Arena *arena, Net *model, u32 rows, u32 cols, u32 flags);

Tensor *tensor_relu(Arena *arena, Net *model, Tensor *input);
Tensor *tensor_tanh(Arena *arena, Net *model, Tensor *input);
Tensor *tensor_softmax(Arena *arena, Net *model, Tensor *input);
Tensor *tensor_add(Arena *arena, Net *model, Tensor *a, Tensor *b);
Tensor *tensor_sub(Arena *arena, Net *model, Tensor *a, Tensor *b);
Tensor *tensor_mul(Arena *arena, Net *model, Tensor *a, Tensor *b);
Tensor *tensor_mm(Arena *arena, Net *model, Tensor *a, Tensor *b);
Tensor *tensor_mse(Arena *arena, Net *model, Tensor *pred, Tensor *target);
Tensor *tensor_reinforce_loss(Arena *arena, Net *model, Tensor *probs, Tensor *rt);

Schedule schedule_from(Arena *arena, Net *model, Tensor *out_var);
Schedule schedule_from_roots(Arena *arena, Net *model, Tensor **roots, u32 num_roots);
void schedule_run(Schedule *graph);
void schedule_grad_accum(Schedule *graph);
void schedule_grad(Schedule *graph);
void schedule_grad_from(Schedule *graph, Tensor *loss_var, b32 seed_one);
void schedule_zero_act(Schedule *graph);

Net *net_new(Arena *arena);
```

Two ways to build a schedule: `schedule_from` for a single output, and
`schedule_from_roots` when a model has several heads. The latter exists because
the obvious hack — a synthetic root such as `logits + value` — is *wrong*: it
makes the reverse sweep treat `add`'s result as the action distribution. Roots
are given in reverse execution order (outputs last).

`schedule_run` is the forward pass. `schedule_grad` = zero activation gradients,
seed the root with 1, and sweep backwards. `schedule_grad_from` seeds a specific
loss tensor instead, which is how the trainers attach their own loss heads.
`schedule_zero_act` clears activation gradients but deliberately leaves parameter
gradients alone, so they accumulate across minibatches. `MAX_ARITY` is 2 because
every operation in the set is unary or binary — it is a property of the op set,
not a capacity limit.

---

## `layers.h`

```c
typedef struct {
  Tensor *weight;   // (out_features x in_features)
  Tensor *bias;     // (out_features x 1) or NULL
  u32 in_features, out_features;
} Dense;

Dense dense_new(Arena *arena, Net *model, Rng *rng, u32 in_features,
                u32 out_features, b32 bias, real gain);
Tensor *dense_apply(Arena *arena, Net *model, const Dense *layer, Tensor *in);
Tensor *mlp_new(Arena *arena, Net *model, Rng *rng, Tensor *in,
                const u32 *layer_dims, u32 num_layers);
void actor_build(Arena *arena, Net *model, Rng *rng, u32 obs_dim,
                 const u32 *hidden_dims, u32 num_hidden, u32 act_dim, u32 batch_size);
void actor_critic_build(Arena *arena, Net *model, Rng *rng, u32 obs_dim,
                        const u32 *hidden_dims, u32 num_hidden, u32 act_dim, u32 batch_size);
void continuous_actor_critic_build(Arena *arena, Net *model, Rng *rng, u32 obs_dim,
                                   const u32 *hidden_dims, u32 num_hidden, u32 act_dim,
                                   u32 batch_size, real log_std_init);
```

`dense_new` allocates parameters with He-uniform initialisation
(`limit = gain * sqrt(6 / in_features)`) and a zero bias. Use `gain = 1` before a
ReLU and something small (`0.01`) for policy and value heads, so the initial
policy starts near uniform.

`dense_apply` creates a fresh `(matmul, optional add)` node sharing the layer's
parameter tensors, so one layer can be applied to the rollout input and the
batched input without duplicating weights.

`actor_critic_build` builds both views in one graph:

* rollout inference — `input (obs_dim x 1) -> output (probs, act_dim x 1)` and
  `value (1 x 1)`;
* batched training — `batch_input (obs_dim x B) -> batch_logits (act_dim x B)`
  and `batch_value (1 x B)`;

with a shared trunk and separate actor/value heads. `batch_size` is the
minibatch size the batched graph starts out built for; it is not a permanent
choice — see `net_set_batch_size` below.

### Continuous policies

```c
void continuous_actor_critic_build(Arena *arena, Net *model, Rng *rng, u32 obs_dim,
                                   const u32 *hidden_dims, u32 num_hidden,
                                   u32 act_dim, u32 batch_size, real log_std_init);
```

The continuous counterpart of `actor_critic_build`, recorded as
`NET_ARCH_CONTINUOUS`:

* rollout inference — `input (obs_dim x 1) -> mean (act_dim x 1)`,
  `log_std (act_dim x 1)`, `value (1 x 1)`;
* batched training — `batch_input (obs_dim x B) -> batch_mean (act_dim x B)` and
  `batch_value (1 x B)`, sharing `log_std`.

`log_std` is a *parameter* (`TENSOR_PARAM | TENSOR_GRAD`, every entry initialised
to `log_std_init`), not an activation, which is what makes the optimiser, the
gradient clip and the checkpoint writer cover it. `mean` / `batch_mean` /
`log_std` are the tensors a trainer needs; `model->output` (the discrete softmax
head) stays `NULL`, so the discrete trainers reject the model instead of
misreading the head. The action box itself is *not* part of the architecture: it
is read from `Env.action_min` / `Env.action_max` at training time, which is what
lets one policy be evaluated on any env with the same `act_dim`.

### Changing the minibatch size

```c
b32 net_set_batch_size(Arena *arena, Net *model, u32 batch_size);
u32 net_batch_size(const Net *model);
```

The builders record the architecture, so the batched subgraph
(`batch_input -> batch_logits` / `batch_value`) can be rebuilt for a different
minibatch size while sharing the same parameter tensors — the weights are
untouched, and the batch and single-sample graphs keep agreeing. This is what
lets `TrainCfg.batch_size` be a runtime choice: the trainers call
`net_set_batch_size` themselves before sizing any buffer from the graph, so
asking for 8 when the model was built for 64 just works. Returns `false` for a
model that was not built by `actor_build` / `actor_critic_build` /
`continuous_actor_critic_build` (there is no recorded architecture to rebuild
from), and `true` immediately when the size already matches. Pointers into
`batch_input` / `batch_logits` / `batch_value` / `batch_mean` are invalidated by
a rebuild.

---

## `optim.h`

```c
typedef struct {
  real lr, beta1, beta2, eps;
  u64 step;
  real **m1;  // first moment, one buffer per parameter
  real **m2;  // second moment
  u32 n_params;
} Adam;

void zero_grad(Net *model);
void sgd_step(Net *model, real lr, real grad_scale);
Adam *adam_new(Arena *arena, Net *model, real lr);
void adam_step(Adam *adam, Net *model);
real clip_grad_norm(Net *model, real max_norm);
```

`zero_grad` clears *parameter* gradients (activation gradients are handled by
`schedule_zero_act`). `sgd_step` applies `p -= lr * grad_scale * grad`, where
`grad_scale` is typically `1/minibatch_size`. `adam_new` defaults to
`beta1 = 0.9, beta2 = 0.999, eps = 1e-8` and `adam_step` applies bias correction
from `adam->step`.

`clip_grad_norm` scales every parameter gradient when the *global* norm exceeds
`max_norm`, and returns the pre-clipping norm — handy for logging and for
detecting divergence.

---

## `checkpoint.h`

```c
b32 net_save(const Net *model, const char *path);
b32 net_load(Net *model, const char *path);
b32 adam_save(const Adam *adam, const Net *model, const char *path);
b32 adam_load(Adam *adam, Net *model, const char *path);
b32 checkpoint_save(const Net *model, const Adam *adam, const char *path);
b32 checkpoint_load(Net *model, Adam *adam, const char *path);
const char *checkpoint_last_error(void);
```

A checkpoint is a small self-describing binary file (magic `RLLIBCK2`, little
endian, block-structured so unknown blocks can be skipped). `checkpoint_save`
writes the parameters plus the Adam moments; `checkpoint_load` restores both and
needs the model only for shape validation. Passing `adam = NULL` to either
function reduces it to parameters only.

Resuming a run:

```c
RL_LIB_CHECK(checkpoint_save(model, adam, "run.ckpt"));
/* ... later, in a fresh process ... */
Adam *adam = adam_new(arena, model, lr);
RL_LIB_CHECK(checkpoint_load(model, adam, "run.ckpt"));
```

Every function returns `false` instead of aborting, and the load is
all-or-nothing: shapes are validated and the payload staged *before* a single
parameter is written, so a rejected file leaves the model exactly as it was.
Failure reasons:

* bad magic or an unknown format version;
* a scalar width that does not match this build (an f32 checkpoint cannot be
  loaded into an f64 build — the values would be silently reinterpreted);
* a parameter count, payload size, or parameter shape that does not match the
  model (this is how a checkpoint from another architecture is caught);
* a truncated file, or an I/O error.

`checkpoint_last_error()` returns a human-readable reason for the most recent
failure — static storage, so copy it if you keep it, and read it on the thread
that made the call.

**Not saved:** the RNG state and the rollout buffer. A resumed run re-derives its
RNG from `TrainCfg.seed`, so it continues from the saved parameters but is not
bit-identical to a run that was never interrupted. In-process determinism *is*
guaranteed: identical seed and configuration produce identical weights.

---

## `env.h`

```c
typedef struct Env Env;

struct Env {
  u32 obs_dim, act_dim, max_steps;
  void *state;

  b32 continuous;              // actions are a real vector, not an index
  const real *action_min;      // act_dim entries, or NULL when discrete
  const real *action_max;

  void (*reset)(Env *e, u64 seed);
  real (*step)(Env *e, u32 action, b32 *done, b32 *truncated);
  real (*step_continuous)(Env *e, const real *action, b32 *done, b32 *truncated);
  void (*observe)(Env *e, real *obs_out);
  Env *(*clone)(Arena *arena, const Env *src);
  void (*destroy)(Env *e);
};
```

The whole environment contract: `reset` starts an episode from a seed, `step`
advances one action and returns the reward while setting `done` (terminal) and
`truncated` (time limit) separately, and `observe` copies the current
observation into a caller-owned `obs_dim`-length buffer. Separating `done` from
`truncated` is what lets a trainer bootstrap the value of a truncated episode
instead of treating the time limit as a real terminal state.

A continuous environment sets `continuous`, fills `action_min` / `action_max`
(one bound per action dimension, which is what the policy's squash needs) and
implements `step_continuous` instead of `step`; the trainer checks both rather
than assuming them. `step_continuous` reads an `act_dim`-length vector and
returns the same `done` / `truncated` pair. The squashed-Gaussian policy already
guarantees an action inside the box, but environments clamp defensively anyway
so a hand-written controller cannot corrupt the dynamics.

`clone` allocates a fresh `Env` with an independent copy of the state. It exists
so `VecEnv` can build N envs from a prototype without the caller knowing the
environment's internals. Discrete actions are `u32`: the interface reflects that
instead of hiding it behind a union.

Every `Env` owns its own `Rng` state, seeded through `reset`, so an env never
holds a pointer into a trainer's generator.

---

## `vecenv.h`

```c
typedef struct {
  u32 num_envs, obs_dim, act_dim;
  Env **envs;

  real *obs;        // num_envs * obs_dim, current observation
  real *final_obs;  // num_envs * obs_dim, the observation before an auto-reset
  real *rewards;    // num_envs
  b8 *dones;        // terminal OR truncated this step
  b8 *terminals;    // true terminal
  b8 *truncateds;   // time-limit truncation
  u64 *reset_base;  // num_envs, per-env auto-reset seed base
  u64 *reset_counts;// num_envs, auto-resets per env
  u64 step_count;
  VecEnvPool *pool; // NULL while stepping serially
} VecEnv;

VecEnv *vecenv_new(Arena *arena, const Env *prototype, u32 num_envs);
void vecenv_reset(VecEnv *v, u64 seed);
void vecenv_reset_seeds(VecEnv *v, const u64 *seeds);
void vecenv_step(VecEnv *v, const u32 *actions);
void vecenv_step_continuous(VecEnv *v, const real *actions);
void vecenv_sample_uniform(VecEnv *v, Rng *rng, u32 *actions_out);
void vecenv_action_bounds(const VecEnv *v, const real **min_out, const real **max_out);

void vecenv_set_threads(Arena *arena, VecEnv *v, u32 num_threads);
u32 vecenv_threads(const VecEnv *v);
void vecenv_shutdown(VecEnv *v);
```

`vecenv_new` clones the prototype `num_envs` times and resets every clone from a
per-index derived seed (or explicit seeds via `vecenv_reset_seeds`, which is how
two policies are compared on identical start states).

`vecenv_step` steps all sub-environments with one action each.
Gymnasium-style auto-reset happens *in place*: finished sub-environments restart
immediately, so `dones[i]` marks the step that ended an episode and `obs[i]`
already holds the post-reset observation. The pre-reset observation is preserved
in `final_obs[i]`, so a trainer can bootstrap correctly. `rewards[i]` is always
the reward of the step just taken.

`vecenv_step_continuous` is the same loop for real-vector actions: `actions`
holds `num_envs * act_dim` values in env-major order, and an env that does not
implement `step_continuous` is reported rather than silently stepped with a
garbage action. `vecenv_action_bounds` returns the sub-environments' bounds
(they are clones of one prototype, so the bounds are shared), which is how
`train_ppo_continuous` sizes a policy without reaching into the env structs.

### Auto-reset seeds

Each sub-environment owns its auto-reset stream: `vecenv_reset(v, seed)` starts
env `i` from `seed + i`, `vecenv_reset_seeds` from an explicit array, and every
auto-reset mixes that base with the number of resets *that env* has had. Nothing
is shared, so an env's trajectory does not depend on the order in which its
siblings are stepped — which is what makes the threaded path below exact.

### Threaded stepping

`vecenv_set_threads(arena, v, num_threads)` installs a worker pool that runs the
per-env steps on `num_threads` threads (0 or 1 restores the default serial path,
which does not allocate anything). The calling thread does a slice too, so
`num_threads` is the total number of participants. Threads are spawned once and
reused across steps; `vecenv_threads` reports the current count and
`vecenv_shutdown` joins them and returns the vecenv to serial (idempotent — call
it before the arena is disposed). Stepping is bit-identical to the serial path,
because the sub-environments share no state and each has its own PRNG;
`tests/test_envs.c` compares every buffer for 260 steps and `tests/test_rl.c`
compares trained weights.

The synchronisation is a barrier built from a mutex and a condition variable
(`pthread_barrier_t` is optional in POSIX and absent on macOS). It costs a
handful of microseconds per step, so threading only pays for environments whose
single step is more expensive than that — CartPole and Pendulum are far below
the crossover and get slower (8.0 s → 13.0 s for the 1200-update Pendulum run
with 8 threads). Builds without POSIX threads (`-DRL_LIB_NO_THREADS`, or
`_WIN32`) compile the serial path and refuse a count above 1 loudly.

---

## `rl.h`

```c
typedef enum { OPT_SGD = 0, OPT_ADAM = 1 } OptimKind;

typedef struct {
  u32 epochs, rollout_size, max_steps;
  real learning_rate, gamma, lambda, vf_coef, entropy_coef;
  u64 seed;
  b32 verbose;
  OptimKind optim;
  real max_grad_norm;
  u32 batch_size;

  u32 num_envs;    // parallel envs stepped per update (0 => 1)
  u32 ppo_epochs;  // inner optimisation passes per rollout (0 => 10)
  real ppo_clip;   // PPO clip epsilon (<= 0 => 0.2)
  u32 num_threads; // worker threads for the vecenv (0 or 1 => serial)
} TrainCfg;

typedef struct { real avg_return, avg_steps; } EvalResult;
```

`batch_size` is the minibatch size to train with. Zero means "use whatever the
graph is already built for"; a non-zero value is applied to the model through
`net_set_batch_size`, so it does not have to match the size the model was built
with. `epochs` is the number of updates for A2C and PPO and the number of passes
for REINFORCE. `lambda`, `vf_coef` and `entropy_coef` are ignored by REINFORCE;
`num_envs`, `ppo_epochs`, `ppo_clip` and `num_threads` are PPO-only (the
vectorised trainers). `num_threads` is forwarded to `vecenv_set_threads` and
cannot change the trained weights, only the wall time; see the note on when it
pays off under [`vecenv.h`](#vecenvh).

### Rollouts

```c
typedef struct {
  real *obs; u32 *actions; real *rewards, *returns, *values, *advantages;
  b8 *dones, *truncateds;
  u32 len;
} Episode;

typedef struct { Episode *trajectories; u32 count, max_steps, obs_dim; } Rollout;

Rollout *rollout_new(Arena *arena, u32 num_trajectories, u32 max_steps, u32 obs_dim);
void gae_compute(Episode *traj, real gamma, real lambda, real last_val);
```

`rollout_new` allocates a fixed-capacity buffer of trajectories and their
per-step arrays as separate flat buffers. `gae_compute` fills `advantages` and
`returns` in place; `last_val` is the bootstrap value for a truncated final step
(pass `0` for a true terminal).

### Training

```c
void train_reinforce(Env *env, Net *model, const TrainCfg *config, Arena *train_arena);
void train_a2c(Env *env, Net *model, const TrainCfg *config, Arena *train_arena);
void train_ppo(Env *env, Net *model, const TrainCfg *config, Arena *train_arena);
void train_ppo_continuous(Env *env, Net *model, const TrainCfg *config, Arena *train_arena);
```

All four mutate `model`'s parameters in place and allocate their rollout
buffers from `train_arena`. `train_reinforce` needs an `actor_build` model;
`train_a2c` and `train_ppo` need an `actor_critic_build` model. Each applies
`config->batch_size` to the model first, so the minibatch size is a runtime
choice — a hand-built graph, which cannot be resized, still has to match.

`train_ppo_continuous` is `train_ppo` for continuous environments: it needs a
`continuous_actor_critic_build` model and an env with `continuous` set,
`step_continuous` implemented and the action bounds filled in. The rollout stores
the sampled pre-squash coordinate `u` (see `linalg.h`), the surrogate is the
Gaussian clipped objective, and a truncated episode bootstraps from the value of
the env's final observation exactly as in the discrete trainer. Everything else —
the rollout/GAE layout, the advantage normalisation, the PPO-only `TrainCfg`
fields and the one-optimiser-step-per-minibatch rule — is shared with `train_ppo`.

| | REINFORCE | A2C | PPO | PPO (continuous) |
|---|---|---|---|---|
| Model | `actor_build` | `actor_critic_build` | `actor_critic_build` | `continuous_actor_critic_build` |
| Rollout | `rollout_size` episodes | 1 episode | `num_envs x rollout_size` steps | same |
| Advantage | discounted return | GAE(`lambda`) | GAE(`lambda`), normalised once | same |
| Updates/rollout | 1 pass | 1 | `ppo_epochs` passes, shuffled minibatches | same |
| Clipping | – | – | `ppo_clip` | `ppo_clip` |
| Policy | categorical | categorical | categorical | squashed Gaussian |
| Env | discrete | discrete | discrete | `step_continuous` + bounds |

`train_ppo` collects from a `VecEnv` built from `config->num_envs` and takes
*one* optimiser step per minibatch, not per rollout. Advantages are normalised
once per rollout, and truncation-aware bootstrapping is applied per
sub-environment.

### Evaluation

```c
EvalResult evaluate(Env *env, Net *model, Rng *rng, u32 episodes, b32 greedy);
EvalResult evaluate_seeded(Env *env, Net *model, const u64 *seeds,
                           u32 num_episodes, b32 greedy);
EvalResult evaluate_continuous(Env *env, Net *model, Rng *rng, u32 episodes, b32 greedy);
EvalResult evaluate_continuous_seeded(Env *env, Net *model, const u64 *seeds,
                                      u32 num_episodes, b32 greedy);
```

`greedy = true` selects `argmax` actions; otherwise they are sampled.
`evaluate` derives per-episode seeds from `rng`; `evaluate_seeded` uses the
caller's seeds so two policies can be compared on identical start states.

The `_continuous` variants step the environment through the policy's squash: a
greedy evaluation applies the same mean clamp the sampler does and then
`center + half * tanh(mean)`, so it really is the mode of the trained density,
and a stochastic evaluation draws a fresh `u` per step with the env's own RNG.

---

## Environments

### `envs/cartpole.h`

```c
Env *env_cartpole_create(Arena *arena);
```

CartPole-v1: `obs_dim 4` (`x, x_dot, theta, theta_dot`), `act_dim 2`, 500 steps.

### `envs/snake.h`

```c
Env *env_snake_create(Arena *arena, u32 grid_size);
void snake_test_set_body(Env *e, const i32 *xs, const i32 *ys, u32 len);
void snake_test_set_food(Env *e, i32 x, i32 y);
```

Grid Snake: `obs_dim 3 * grid_size` (head plane, food plane, body plane),
`act_dim 4` (`LEFT=0, RIGHT=1, UP=2, DOWN=3`), 100 steps. The two setters exist
so tests can construct a position without duplicating the private state layout.

Both environments treat running into a wall or their own body as a terminal
`done`, and the step limit as `truncated`.

### `envs/pendulum.h`

```c
Env *env_pendulum_create(Arena *arena);
void pendulum_test_set_state(Env *e, real theta, real theta_dot);
```

Pendulum-v1: `obs_dim 3` (`cos theta, sin theta, theta_dot`), `act_dim 1`,
200 steps, and the library's only *continuous* environment — it sets
`continuous`, implements `step_continuous`, and exposes `action_min = {-2}` /
`action_max = {2}` (torque). The dynamics are Gymnasium's: `dt = 0.05`,
`g = 10`, `m = l = 1`, `max_speed = 8`, torque clamped to the bounds, and the
reward is `-(angle_normalize(theta)^2 + 0.1 * theta_dot^2 + 0.001 * u^2)`.
Episodes end by truncation at 200 steps (`done` is never set), which is exactly
the case the trainer's bootstrap handles. `pendulum_test_set_state` is the
testing hook: it installs a state directly so the Euler step can be checked
against a hand computation.

Train it with `train_ppo_continuous`, which reads the bounds through
`vecenv_action_bounds` (see [`vecenv.h`](#vecenvh)).

---

## A complete program

`examples/hello_rl.c` is the smallest end-to-end program, and the one CI builds
against the *installed* library:

```c
#include "checkpoint.h"
#include "envs/cartpole.h"
#include "layers.h"
#include "memory.h"
#include "rl.h"
#include "version.h"

int main(void) {
  Arena *arena = arena_new(MiB(64));
  Rng rng; rng_seed(&rng, 1234, 0);

  const u32 batch_size = 64;
  Env *env = env_cartpole_create(arena);
  Net *model = net_new(arena);
  const u32 hidden[2] = {16, 16};
  actor_critic_build(arena, model, &rng, env->obs_dim, hidden, 2, env->act_dim,
                     batch_size);

  const TrainCfg cfg = {
      .epochs = 5, .rollout_size = 512, .max_steps = 500,
      .learning_rate = 3e-4f, .gamma = 0.99f, .lambda = 0.95f,
      .vf_coef = 0.5f, .entropy_coef = 0.01f, .seed = 7, .verbose = 0,
      .optim = OPT_ADAM, .max_grad_norm = 0.5f, .batch_size = batch_size,
      .num_envs = 4, .ppo_epochs = 4, .ppo_clip = 0.2f,
  };
  train_ppo(env, model, &cfg, arena);

  RL_LIB_CHECK(checkpoint_save(model, NULL, "hello_rl.ckpt"));
  EvalResult result = evaluate(env, model, &rng, 20, true);
  printf("greedy return: %.1f\n", (double)result.avg_return);

  arena_dispose(arena);
  return 0;
}
```

Build it against an installed copy:

```sh
make install PREFIX=/usr/local
cc -O2 $(pkg-config --cflags rl_lib) hello_rl.c $(pkg-config --libs rl_lib) -o hello_rl
```

See the [README](../README.md) for building from source and the
[CHANGELOG](../CHANGELOG.md) for what changed in each release.
