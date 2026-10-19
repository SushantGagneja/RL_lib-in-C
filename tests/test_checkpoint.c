// test_checkpoint.c — checkpointing: exact parameter restore, Adam-state
// resume equivalence, and rejection of corrupt, truncated, or mismatched files
// without mutating the model.
#include "base.h"
#include "checkpoint.h"
#include "graph.h"
#include "layers.h"
#include "memory.h"
#include "optim.h"
#include "random.h"

#include <stdio.h>
#include <string.h>

static u32 tests_passed = 0;
static u32 tests_failed = 0;

#define ASSERT_TRUE(cond, msg)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("  FAIL: %s (line %d)\n", msg, __LINE__);                         \
      tests_failed++;                                                          \
    } else {                                                                   \
      tests_passed++;                                                          \
    }                                                                          \
  } while (0)

static const char *SAVE_PATH = "checkpoint_test.bin";
static const char *JUNK_PATH = "checkpoint_test_junk.bin";

static Net *build_model(Arena *arena, Rng *rng, u32 act_dim, u32 batch_size) {
  Net *model = net_new(arena);
  const u32 hidden[2] = {8, 8};
  actor_critic_build(arena, model, rng, 4, hidden, 2, act_dim, batch_size);
  return model;
}

// Order-sensitive so a reordering of the parameters is caught as well.
static real param_checksum(const Net *model) {
  real sum = (real)0;
  for (u32 i = 0; i < model->n_params; i++) {
    const Mat *m = model->params[i]->val;
    for (u32 j = 0; j < m->rows * m->cols; j++) {
      sum += m->data[j] * (real)(j + 1u);
    }
  }
  return sum;
}

static b32 params_identical(const Net *a, const Net *b) {
  if (a->n_params != b->n_params) {
    return false;
  }
  for (u32 i = 0; i < a->n_params; i++) {
    const Mat *x = a->params[i]->val;
    const Mat *y = b->params[i]->val;
    if (x->rows != y->rows || x->cols != y->cols) {
      return false;
    }
    if (memcmp(x->data, y->data,
               (size_t)x->rows * (size_t)x->cols * sizeof(real)) != 0) {
      return false;
    }
  }
  return true;
}

// A deterministic, parameter-independent gradient, so optimizer steps are
// reproducible without running a full training loop.
static void seed_grads(Net *model, u32 salt) {
  for (u32 i = 0; i < model->n_params; i++) {
    Mat *g = model->params[i]->grad;
    for (u32 j = 0; j < g->rows * g->cols; j++) {
      const u32 n = (j * 37u + i * 101u + salt * 11u) % 97u;
      g->data[j] = ((real)n / (real)97) - REAL_C(0.5);
    }
  }
}

static void put_u32_le(FILE *f, u32 v) {
  u8 b[4];
  b[0] = (u8)(v & 0xffu);
  b[1] = (u8)((v >> 8) & 0xffu);
  b[2] = (u8)((v >> 16) & 0xffu);
  b[3] = (u8)((v >> 24) & 0xffu);
  fwrite(b, 1, 4, f);
}

// The reader verifies a CRC-32 trailer before it parses anything, and the
// hand-built fixtures below bypass net_save, so they seal themselves with the
// same reflected algorithm (poly 0xEDB88320, a bit at a time) - otherwise every
// fixture would be rejected for its checksum instead of reaching the validation
// it was written to exercise.
static u32 crc32_bytes(u32 crc, const u8 *data, size_t n) {
  for (size_t i = 0; i < n; i++) {
    crc ^= data[i];
    for (u32 bit = 0; bit < 8; bit++) {
      crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
  }
  return crc;
}

static void seal_junk_file(const char *path) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) {
    return;
  }
  u32 crc = 0;
  u8 buf[4096];
  size_t got;
  while ((got = fread(buf, 1, sizeof buf, f)) > 0) {
    crc = crc32_bytes(crc, buf, got);
  }
  if (fclose(f) != 0) {
    return;
  }
  f = fopen(path, "ab");
  if (f != NULL) {
    put_u32_le(f, crc);
    fclose(f);
  }
}

// ── Parameters round-trip exactly ────────────────────────────────────

static void test_param_round_trip(void) {
  printf("  parameters round-trip byte for byte... ");

  Arena *a1 = arena_new(MiB(8));
  Arena *a2 = arena_new(MiB(8));
  Rng rng;
  rng_seed(&rng, 7, 1);
  Net *src = build_model(a1, &rng, 3, 4);
  Net *dst = build_model(a2, &rng, 3, 4);

  ASSERT_TRUE(!params_identical(src, dst), "the two models start out different");
  ASSERT_TRUE(net_save(src, SAVE_PATH), "net_save succeeds");
  ASSERT_TRUE(net_load(dst, SAVE_PATH), "net_load succeeds");
  ASSERT_TRUE(params_identical(src, dst), "every parameter matches exactly");
  ASSERT_TRUE(checkpoint_last_error()[0] == '\0',
              "no error is reported after a successful load");

  // Saving the same model twice is byte-identical (no timestamps, no padding).
  ASSERT_TRUE(net_save(src, JUNK_PATH), "second save succeeds");
  FILE *f1 = fopen(SAVE_PATH, "rb");
  FILE *f2 = fopen(JUNK_PATH, "rb");
  b32 same = f1 != NULL && f2 != NULL;
  if (same) {
    for (;;) {
      const int c1 = fgetc(f1);
      const int c2 = fgetc(f2);
      if (c1 != c2) {
        same = false;
        break;
      }
      if (c1 == EOF) {
        break;
      }
    }
  }
  if (f1 != NULL) {
    fclose(f1);
  }
  if (f2 != NULL) {
    fclose(f2);
  }
  ASSERT_TRUE(same, "repeated saves produce identical bytes");

  arena_dispose(a1);
  arena_dispose(a2);
  remove(SAVE_PATH);
  remove(JUNK_PATH);
  printf("OK\n");
}

// ── Adam state resumes a run bit-exactly ─────────────────────────────

static void test_adam_resume(void) {
  printf("  Adam state resumes a run exactly... ");

  Arena *a1 = arena_new(MiB(8));
  Rng rng;
  rng_seed(&rng, 21, 2);
  Net *model = build_model(a1, &rng, 2, 4);
  Adam *adam = adam_new(a1, model, REAL_C(0.01));

  for (u32 s = 0; s < 5; s++) {
    seed_grads(model, s);
    adam_step(adam, model);
  }
  ASSERT_TRUE(checkpoint_save(model, adam, SAVE_PATH), "checkpoint_save succeeds");
  const real saved_checksum = param_checksum(model);

  for (u32 s = 5; s < 8; s++) {
    seed_grads(model, s);
    adam_step(adam, model);
  }
  const real expected = param_checksum(model);

  // A fresh arena and model: the checkpoint is the only thing carried over.
  Arena *a2 = arena_new(MiB(8));
  Rng rng2;
  rng_seed(&rng2, 999, 3);
  Net *restored = build_model(a2, &rng2, 2, 4);
  Adam *restored_adam = adam_new(a2, restored, REAL_C(1.0)); // wrong on purpose

  ASSERT_TRUE(checkpoint_load(restored, restored_adam, SAVE_PATH),
              "checkpoint_load succeeds");
  for (u32 s = 5; s < 8; s++) {
    seed_grads(restored, s);
    adam_step(restored_adam, restored);
  }

  ASSERT_TRUE(param_checksum(restored) == expected,
              "resumed weights equal the uninterrupted run");
  ASSERT_TRUE(restored_adam->step == 8, "optimiser step count restored");
  ASSERT_TRUE(restored_adam->lr == REAL_C(0.01), "learning rate restored");

  // net_load alone also accepts a combined checkpoint (it skips the ADAM block).
  Rng rng3;
  rng_seed(&rng3, 4, 4);
  Net *params_only = build_model(a2, &rng3, 2, 4);
  ASSERT_TRUE(net_load(params_only, SAVE_PATH), "net_load accepts a combined file");
  ASSERT_TRUE(param_checksum(params_only) == saved_checksum,
              "net_load restores the parameters of a combined file");

  arena_dispose(a1);
  arena_dispose(a2);
  remove(SAVE_PATH);
  printf("OK\n");
}

// ── Adam-only and parameter-only files ───────────────────────────────

static void test_partial_files(void) {
  printf("  parameter-only and optimiser-only files... ");

  Arena *arena = arena_new(MiB(8));
  Rng rng;
  rng_seed(&rng, 33, 6);
  Net *model = build_model(arena, &rng, 2, 4);
  Adam *adam = adam_new(arena, model, REAL_C(0.02));
  seed_grads(model, 3);
  adam_step(adam, model);

  ASSERT_TRUE(net_save(model, SAVE_PATH), "net_save (parameters only)");
  ASSERT_TRUE(!adam_load(adam, model, SAVE_PATH),
              "adam_load rejects a file with no optimiser block");
  ASSERT_TRUE(strstr(checkpoint_last_error(), "optimiser") != NULL,
              "the error names the missing block");

  ASSERT_TRUE(adam_save(adam, model, SAVE_PATH), "adam_save (optimiser only)");
  ASSERT_TRUE(!net_load(model, SAVE_PATH),
              "net_load rejects a file with no parameter block");

  // Restoring an Adam-only file leaves the parameters alone but restores state.
  Arena *a2 = arena_new(MiB(8));
  Rng rng2;
  rng_seed(&rng2, 12, 7);
  Net *fresh = build_model(a2, &rng2, 2, 4);
  const real before = param_checksum(fresh);
  Adam *fresh_adam = adam_new(a2, fresh, REAL_C(0.5));
  ASSERT_TRUE(adam_load(fresh_adam, fresh, SAVE_PATH), "adam_load succeeds");
  ASSERT_TRUE(param_checksum(fresh) == before, "parameters untouched by adam_load");
  ASSERT_TRUE(fresh_adam->step == adam->step, "step count restored");

  arena_dispose(arena);
  arena_dispose(a2);
  remove(SAVE_PATH);
  printf("OK\n");
}

// ── Corrupt, truncated and mismatched files ──────────────────────────

static void test_failure_modes(void) {
  printf("  corrupt and mismatched files are rejected safely... ");

  Arena *arena = arena_new(MiB(8));
  Rng rng;
  rng_seed(&rng, 5, 5);
  Net *model = build_model(arena, &rng, 2, 4);
  const real before = param_checksum(model);

  ASSERT_TRUE(!net_load(model, "checkpoint_test_missing.bin"),
              "a missing file is rejected");
  ASSERT_TRUE(checkpoint_last_error()[0] != '\0', "the failure has a message");
  ASSERT_TRUE(param_checksum(model) == before, "model untouched");

  const char *junk = "this is not a checkpoint";
  FILE *f = fopen(JUNK_PATH, "wb");
  ASSERT_TRUE(f != NULL, "junk file created");
  if (f != NULL) {
    fwrite(junk, 1, strlen(junk), f);
    fclose(f);
    ASSERT_TRUE(!net_load(model, JUNK_PATH), "a bad magic is rejected");
    const char *why = checkpoint_last_error();
    ASSERT_TRUE(strstr(why, "magic") != NULL || strstr(why, "checksum") != NULL,
                "the error explains why it is not a checkpoint");
    ASSERT_TRUE(param_checksum(model) == before, "model untouched");
  }

  // Wrong format version.
  f = fopen(JUNK_PATH, "wb");
  if (f != NULL) {
    fwrite("RLLIBCK3", 1, 8, f);
    put_u32_le(f, 99u);
    fclose(f);
    seal_junk_file(JUNK_PATH);
    ASSERT_TRUE(!net_load(model, JUNK_PATH), "an unknown version is rejected");
  }

  // Truncated: a parameter block header that promises far more than it delivers.
  f = fopen(JUNK_PATH, "wb");
  if (f != NULL) {
    fwrite("RLLIBCK3", 1, 8, f);
    put_u32_le(f, 3u);
    put_u32_le(f, 1u);    // BLOCK_PARAMS
    put_u32_le(f, 1000u); // payload bytes that are not there
    fclose(f);
    seal_junk_file(JUNK_PATH);
    ASSERT_TRUE(!net_load(model, JUNK_PATH), "a truncated block is rejected");
    ASSERT_TRUE(param_checksum(model) == before, "model untouched");
  }

  // A payload sized for a different model.
  f = fopen(JUNK_PATH, "wb");
  if (f != NULL) {
    fwrite("RLLIBCK3", 1, 8, f);
    put_u32_le(f, 3u);
    put_u32_le(f, 1u); // BLOCK_PARAMS
    put_u32_le(f, 8u);
    put_u32_le(f, (u32)sizeof(real));
    put_u32_le(f, model->n_params);
    for (u32 i = 0; i < model->n_params; i++) {
      put_u32_le(f, model->params[i]->val->rows);
      put_u32_le(f, model->params[i]->val->cols);
    }
    fclose(f); // no data, and a payload size that cannot be right
    seal_junk_file(JUNK_PATH);
    ASSERT_TRUE(!net_load(model, JUNK_PATH), "a wrong payload size is rejected");
    ASSERT_TRUE(param_checksum(model) == before, "model untouched");
  }

  // Scalar width: a file claiming the other precision must not be reinterpreted.
  {
    const u32 wrong_width = (sizeof(real) == 4) ? 8u : 4u;
    u64 n_scalars = 0;
    for (u32 i = 0; i < model->n_params; i++) {
      n_scalars += (u64)model->params[i]->val->rows *
                   (u64)model->params[i]->val->cols;
    }
    const u32 payload =
        (u32)(8u + 8u * (u64)model->n_params + n_scalars * sizeof(real));
    f = fopen(JUNK_PATH, "wb");
    if (f != NULL) {
      fwrite("RLLIBCK3", 1, 8, f);
      put_u32_le(f, 3u);
      put_u32_le(f, 1u);
      put_u32_le(f, payload);
      put_u32_le(f, wrong_width);
      put_u32_le(f, model->n_params);
      for (u32 i = 0; i < model->n_params; i++) {
        put_u32_le(f, model->params[i]->val->rows);
        put_u32_le(f, model->params[i]->val->cols);
      }
      fclose(f);
      seal_junk_file(JUNK_PATH);
      ASSERT_TRUE(!net_load(model, JUNK_PATH),
                  "a different scalar width is rejected");
      ASSERT_TRUE(strstr(checkpoint_last_error(), "scalar width") != NULL,
                  "the error explains the width mismatch");
      ASSERT_TRUE(param_checksum(model) == before, "model untouched");
    }
  }

  // A checkpoint from a model with different dimensions. The payload size check
  // fires first here, because every dimension change moves the total scalar count.
  Arena *other = arena_new(MiB(8));
  Rng rng2;
  rng_seed(&rng2, 6, 6);
  Net *wide = build_model(other, &rng2, 3, 4);
  ASSERT_TRUE(net_save(wide, SAVE_PATH), "save a differently shaped model");
  ASSERT_TRUE(!net_load(model, SAVE_PATH),
              "a checkpoint from another architecture is rejected");
  ASSERT_TRUE(strstr(checkpoint_last_error(), "size") != NULL,
              "the error names the size mismatch");
  ASSERT_TRUE(param_checksum(model) == before, "model untouched");

  // The shape table itself: hand-craft a file whose payload size is exactly
  // right but whose second parameter claims the wrong shape.
  {
    u64 n_scalars = 0;
    for (u32 i = 0; i < model->n_params; i++) {
      n_scalars += (u64)model->params[i]->val->rows *
                   (u64)model->params[i]->val->cols;
    }
    const u32 payload =
        (u32)(8u + 8u * (u64)model->n_params + n_scalars * sizeof(real));
    f = fopen(JUNK_PATH, "wb");
    if (f != NULL) {
      fwrite("RLLIBCK3", 1, 8, f);
      put_u32_le(f, 3u);
      put_u32_le(f, 1u);
      put_u32_le(f, payload);
      put_u32_le(f, (u32)sizeof(real));
      put_u32_le(f, model->n_params);
      for (u32 i = 0; i < model->n_params; i++) {
        const Mat *m = model->params[i]->val;
        put_u32_le(f, i == 1 ? m->rows + 1u : m->rows);
        put_u32_le(f, m->cols);
      }
      fclose(f);
      seal_junk_file(JUNK_PATH);
      ASSERT_TRUE(!net_load(model, JUNK_PATH),
                  "a file with a wrong parameter shape is rejected");
      ASSERT_TRUE(strstr(checkpoint_last_error(), "shapes") != NULL,
                  "the error names the shape mismatch");
      ASSERT_TRUE(param_checksum(model) == before, "model untouched");
    }
  }

  // A corrupt tail leaves the parameters alone only if validation happens first;
  // truncating a real checkpoint mid-data must fail loudly.
  ASSERT_TRUE(net_save(model, SAVE_PATH), "save a good checkpoint");
  ASSERT_TRUE(net_load(model, SAVE_PATH), "and load it back");
  FILE *src = fopen(SAVE_PATH, "rb");
  b32 truncated_written = false;
  if (src != NULL) {
    u8 buf[4096];
    const size_t n = fread(buf, 1, sizeof(buf), src);
    fclose(src);
    FILE *dst = fopen(JUNK_PATH, "wb");
    if (dst != NULL && n > 16) {
      fwrite(buf, 1, n - 9, dst); // drop the END block and part of the data
      fclose(dst);
      truncated_written = true;
    }
  }
  ASSERT_TRUE(truncated_written, "truncated copy written");
  if (truncated_written) {
    ASSERT_TRUE(!net_load(model, JUNK_PATH), "a truncated checkpoint is rejected");
  }

  arena_dispose(arena);
  arena_dispose(other);
  remove(SAVE_PATH);
  remove(JUNK_PATH);
  printf("OK\n");
}

// ── Checksum and fuzzed inputs ───────────────────────────────────────

// A single flipped byte in the middle of a real checkpoint must be reported as
// corruption before the parser can act on it, and the model must survive.
static void test_crc_detects_bit_rot(void) {
  printf("  a flipped byte is caught by the checksum... ");

  Arena *arena = arena_new(MiB(8));
  Rng rng;
  rng_seed(&rng, 7, 7);
  Net *model = build_model(arena, &rng, 2, 4);
  ASSERT_TRUE(net_save(model, SAVE_PATH), "a good checkpoint is written");

  FILE *f = fopen(SAVE_PATH, "rb");
  u8 buf[8192];
  size_t n = 0;
  if (f != NULL) {
    n = fread(buf, 1, sizeof buf, f);
    fclose(f);
  }
  ASSERT_TRUE(n > 32u, "the checkpoint fits the buffer");
  if (n > 32u) {
    const size_t at = 16u + (n - 32u) / 2u; // inside a payload, not the trailer
    buf[at] ^= 0x40u;
    f = fopen(JUNK_PATH, "wb");
    ASSERT_TRUE(f != NULL, "the mutated copy is written");
    if (f != NULL) {
      fwrite(buf, 1, n, f);
      fclose(f);
    }
  }

  const real before = param_checksum(model);
  ASSERT_TRUE(!net_load(model, JUNK_PATH), "the flipped checkpoint is rejected");
  ASSERT_TRUE(strstr(checkpoint_last_error(), "checksum") != NULL,
              "the error names the checksum");
  ASSERT_TRUE(param_checksum(model) == before, "model untouched");

  remove(SAVE_PATH);
  remove(JUNK_PATH);
  arena_dispose(arena);
  printf("OK\n");
}

// The reader must survive arbitrary bytes: random mutations of a real
// checkpoint - with and without a recomputed checksum - plus truncations and
// overwritten runs. Every input either loads cleanly or is rejected with a
// message, the model is only touched when the load reports success, and
// nothing crashes; the sanitizer job turns that last point into a hard check.
static void test_fuzzed_inputs(void) {
  printf("  fuzzed checkpoints never crash and never half-load... ");

  Arena *arena = arena_new(MiB(8));
  Rng rng;
  rng_seed(&rng, 21, 21);
  Net *model = build_model(arena, &rng, 2, 4);
  ASSERT_TRUE(net_save(model, SAVE_PATH), "a good checkpoint is written");

  FILE *f = fopen(SAVE_PATH, "rb");
  u8 base[8192];
  size_t base_n = 0;
  if (f != NULL) {
    base_n = fread(base, 1, sizeof base, f);
    fclose(f);
  }
  ASSERT_TRUE(base_n > 32u, "the checkpoint fits the buffer");

  u32 rejected = 0;
  u32 loaded = 0;
  for (u32 iter = 0; iter < 300u && base_n > 32u; iter++) {
    u8 buf[8192];
    memcpy(buf, base, base_n);
    size_t n = base_n;

    switch (rng_below(&rng, 3u)) {
    case 0u: { // flip a handful of bits
      const u32 flips = 1u + rng_below(&rng, 4u);
      for (u32 i = 0; i < flips; i++) {
        buf[rng_below(&rng, (u32)n)] ^= (u8)(1u << rng_below(&rng, 8u));
      }
      break;
    }
    case 1u: // truncate anywhere
      n = 1u + (size_t)rng_below(&rng, (u32)n);
      break;
    default: { // overwrite a run
      const u32 at = rng_below(&rng, (u32)n);
      const u32 len = 1u + rng_below(&rng, 16u);
      for (u32 i = 0; i < len && (size_t)at + i < n; i++) {
        buf[(size_t)at + i] = (u8)(rng_u32(&rng) & 0xffu);
      }
      break;
    }
    }

    f = fopen(JUNK_PATH, "wb");
    if (f == NULL) {
      break;
    }
    fwrite(buf, 1, n, f);
    fclose(f);
    if ((iter % 3u) == 0u) {
      seal_junk_file(JUNK_PATH); // structurally hostile, checksum consistent
    }

    const real before = param_checksum(model);
    if (net_load(model, JUNK_PATH)) {
      loaded++;
    } else {
      rejected++;
      ASSERT_TRUE(checkpoint_last_error()[0] != '\0',
                  "a rejected file explains itself");
      ASSERT_TRUE(param_checksum(model) == before,
                  "a rejected file leaves the model alone");
    }
  }

  ASSERT_TRUE(rejected > 200u, "almost every mutation is rejected");
  ASSERT_TRUE(loaded + rejected == 300u, "every mutation produced a verdict");
  printf("(%u rejected / %u loaded) ", rejected, loaded);

  remove(SAVE_PATH);
  remove(JUNK_PATH);
  arena_dispose(arena);
  printf("OK\n");
}

// ── Null handling ────────────────────────────────────────────────────

static void test_null_args(void) {
  printf("  null arguments are refused... ");
  Arena *arena = arena_new(MiB(4));
  Rng rng;
  rng_seed(&rng, 1, 1);
  Net *model = build_model(arena, &rng, 2, 4);

  ASSERT_TRUE(!net_save(NULL, SAVE_PATH), "net_save(NULL)");
  ASSERT_TRUE(!net_save(model, NULL), "net_save(path=NULL)");
  ASSERT_TRUE(!net_load(NULL, SAVE_PATH), "net_load(NULL)");
  ASSERT_TRUE(!checkpoint_save(model, NULL, NULL), "checkpoint_save(NULL path)");
  ASSERT_TRUE(!adam_load(NULL, model, SAVE_PATH), "adam_load(NULL)");

  arena_dispose(arena);
  printf("OK\n");
}

// ── Continuous models round-trip their learnable log_std as well ──────

static void test_continuous_round_trip(void) {
  printf("  continuous model round-trip (log_std included)... ");

  Arena *a1 = arena_new(MiB(8));
  Arena *a2 = arena_new(MiB(8));
  Rng rng;
  rng_seed(&rng, 21, 1);

  Net *src = net_new(a1);
  Net *dst = net_new(a2);
  const u32 hidden[2] = {8, 8};
  // Re-seed between the two builds so both models start from the same weights.
  rng_seed(&rng, 21, 1);
  continuous_actor_critic_build(a1, src, &rng, 3, hidden, 2, 2, 4, -1.25f);
  rng_seed(&rng, 21, 1);
  continuous_actor_critic_build(a2, dst, &rng, 3, hidden, 2, 2, 4, -1.25f);

  ASSERT_TRUE(src->n_params == dst->n_params, "matching parameter counts");
  ASSERT_TRUE(params_identical(src, dst),
              "identically seeded models start out identical");

  // Move the learned log standard deviation away from its initial value: the
  // point of this test is that it is a first-class parameter.
  src->log_std->val->data[0] = REAL_C(0.75);
  src->log_std->val->data[1] = REAL_C(-0.5);
  ASSERT_TRUE(!params_identical(src, dst), "log_std makes them differ");

  ASSERT_TRUE(net_save(src, SAVE_PATH), "net_save succeeds");
  ASSERT_TRUE(net_load(dst, SAVE_PATH), "net_load succeeds");
  ASSERT_TRUE(params_identical(src, dst),
              "every parameter matches, log_std included");
  ASSERT_TRUE(dst->log_std->val->data[0] == REAL_C(0.75),
              "log_std value restored");
  ASSERT_TRUE(dst->log_std->val->data[1] == REAL_C(-0.5),
              "second log_std component restored");

  remove(SAVE_PATH);
  arena_dispose(a1);
  arena_dispose(a2);
  printf("OK\n");
}

int main(void) {
  printf("\n=== Checkpoint tests ===\n\n");

  test_param_round_trip();
  test_adam_resume();
  test_partial_files();
  test_failure_modes();
  test_crc_detects_bit_rot();
  test_fuzzed_inputs();
  test_null_args();
  test_continuous_round_trip();

  printf("\n---\nTests: %u passed, %u failed\n\n", tests_passed, tests_failed);
  return tests_failed > 0 ? 1 : 0;
}
