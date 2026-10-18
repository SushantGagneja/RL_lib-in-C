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
| [`linalg.h`](#linalgh) | `Mat` plus every matrix / activation / loss / gradient kernel |
| [`simd.h`](#simdh) | matrix-multiply backend selection and probing |
| [`graph.h`](#graphh) | `Tensor`, `Net`, reverse-mode autograd |
| [`layers.h`](#layersh) | `Dense`, `mlp_new`, `actor_build`, `actor_critic_build` |
| [`optim.h`](#optimh) | `zero_grad`, `sgd_step`, `Adam`, `clip_grad_norm` |
| [`checkpoint.h`](#checkpointh) | `net_save` / `net_load` / `checkpoint_save` / `checkpoint_load` |
| [`env.h`](#envh) | the `Env` interface every environment implements |
| [`vecenv.h`](#vecenvh) | `VecEnv`, synchronised parallel environments |
| [`rl.h`](#rlh) | `TrainCfg`, `Rollout`, `train_reinforce`, `train_a2c`, `train_ppo`, `evaluate` |
| [`envs/`](#environments) | CartPole, Grid Snake |

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
`REAL_C(x)` (an exactly-representable literal for the current mode) and
`REAL_EPS`.

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

---

## `simd.h`

```c
#define RL_SIMD_NONE 0
#define RL_SIMD_NEON 1
#define RL_SIMD_AVX2 2

const char *simd_backend_name(void);  // "scalar" | "neon" | "avx2"
b32         simd_accelerated(void);   // usable on *this* CPU?
```

The backend is chosen at compile time:

| Platform | Backend |
|---|---|
| AArch64 | `RL_SIMD_NEON` — 4-wide f32 NEON, always available on the baseline |
| x86-64 (GCC/Clang) | `RL_SIMD_AVX2` — 8-wide f32 AVX2+FMA, per-function `target` attributes |
| anything else, or `-DRL_LIB_USE_DOUBLE` | `RL_SIMD_NONE` — the scalar reference |

Because the AVX2 kernels are compiled with per-function target attributes,
`simd_accelerated()` performs a cached runtime CPU probe: the binary still runs
on a CPU without AVX2 and simply falls back. Force the scalar path with
`-DRL_SIMD_BACKEND=0`. The f64 build always uses the scalar path — it exists for
the gradient check, not for speed — and requesting otherwise is a compile-time
`#error`.

The vector kernels are used for NN/NT/TN matmuls and never for the rarely emitted
TT case. They hold four accumulator vectors in registers; where that changes the
reduction order (only for NT) the result can differ from the scalar path in the
last ulp, which the tests bound rather than forbid.

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
model that was not built by `actor_build` / `actor_critic_build` (there is no
recorded architecture to rebuild from), and `true` immediately when the size
already matches. Pointers into `batch_input` / `batch_logits` / `batch_value`
are invalidated by a rebuild.

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
  void (*reset)(Env *e, u64 seed);
  real (*step)(Env *e, u32 action, b32 *done, b32 *truncated);
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

`clone` allocates a fresh `Env` with an independent copy of the state. It exists
so `VecEnv` can build N envs from a prototype without the caller knowing the
environment's internals. Actions are `u32`: the shipped environments are
discrete, and the interface reflects that rather than hiding it behind a union.

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
  Rng rng;
  u64 step_count;
} VecEnv;

VecEnv *vecenv_new(Arena *arena, const Env *prototype, u32 num_envs);
void vecenv_reset(VecEnv *v, u64 seed);
void vecenv_reset_seeds(VecEnv *v, const u64 *seeds);
void vecenv_step(VecEnv *v, const u32 *actions);
void vecenv_sample_uniform(VecEnv *v, Rng *rng, u32 *actions_out);
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
} TrainCfg;

typedef struct { real avg_return, avg_steps; } EvalResult;
```

`batch_size` is the minibatch size to train with. Zero means "use whatever the
graph is already built for"; a non-zero value is applied to the model through
`net_set_batch_size`, so it does not have to match the size the model was built
with. `epochs` is the number of updates for A2C and PPO and the number of passes
for REINFORCE. `lambda`, `vf_coef` and `entropy_coef` are ignored by REINFORCE;
`num_envs`, `ppo_epochs` and `ppo_clip` are PPO-only.

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
```

All three mutate `model`'s parameters in place and allocate their rollout
buffers from `train_arena`. `train_reinforce` needs an `actor_build` model;
`train_a2c` and `train_ppo` need an `actor_critic_build` model. Each applies
`config->batch_size` to the model first, so the minibatch size is a runtime
choice — a hand-built graph, which cannot be resized, still has to match.

| | REINFORCE | A2C | PPO |
|---|---|---|---|
| Model | `actor_build` | `actor_critic_build` | `actor_critic_build` |
| Rollout | `rollout_size` episodes | 1 episode | `num_envs x rollout_size` steps |
| Advantage | discounted return | GAE(`lambda`) | GAE(`lambda`), normalised once |
| Updates/rollout | 1 pass | 1 | `ppo_epochs` passes, shuffled minibatches |
| Clipping | – | – | `ppo_clip` |

`train_ppo` collects from a `VecEnv` built from `config->num_envs` and takes
*one* optimiser step per minibatch, not per rollout. Advantages are normalised
once per rollout, and truncation-aware bootstrapping is applied per
sub-environment.

### Evaluation

```c
EvalResult evaluate(Env *env, Net *model, Rng *rng, u32 episodes, b32 greedy);
EvalResult evaluate_seeded(Env *env, Net *model, const u64 *seeds,
                           u32 num_episodes, b32 greedy);
```

`greedy = true` selects `argmax` actions; otherwise they are sampled.
`evaluate` derives per-episode seeds from `rng`; `evaluate_seeded` uses the
caller's seeds so two policies can be compared on identical start states.

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
