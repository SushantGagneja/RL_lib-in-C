// checkpoint.c — versioned, endian-independent save/load for parameters, Adam
// state and observation-normalisation statistics.
//
// File layout (little-endian throughout, so a checkpoint moves between
// machines):
//
//   char magic[8]  "RLLIBCK4"
//   u32  version   4
//   block*
//   u32  crc32     checksum of every preceding byte
//
//   block := u32 kind | u32 payload_bytes | payload
//     ARCH(4):   u32 kind | u32 obs_dim | u32 act_dim | u32 num_hidden
//                | u32 hidden[num_hidden]      (builder-made models only)
//     PARAMS(1): u32 scalar_bytes | u32 n_params | u32 rows[n] | u32 cols[n]
//                | data[n_params][rows*cols]
//     ADAM(2):   u32 scalar_bytes | u32 n_params | u32 rows[n] | u32 cols[n]
//                | u64 step | f64 lr | f64 beta1 | f64 beta2 | f64 eps
//                | m1[...] | m2[...]
//     NORM(3):   u32 obs_dim | u64 count | u64 ret_count | f64 eps
//                | f64 clip_obs | f64 ret_mean | f64 ret_m2 | f64 gamma
//                | f64 mean[obs_dim] | f64 m2[obs_dim]
//     END(0):    empty
//
// The trailing CRC-32 (reflected, poly 0xEDB88320) is checked before a single
// block is parsed, so bit rot is reported as such rather than turning into a
// confusing block-size error or silently wrong parameters. Format 4 is format 3
// plus the ARCH and NORM blocks; like every earlier format change, older files
// are rejected up front rather than migrated.
//
// `payload_bytes` lets a reader skip blocks it does not understand, so new block
// kinds can be appended without breaking older readers that only need the
// blocks they know. `scalar_bytes` records sizeof(real) at save time, which is
// how an f32/f64 mismatch is caught before values get reinterpreted: floating
// point bit patterns are endian-independent here, but not width-independent.
// NORM values travel as f64, which represents both widths exactly, so the
// statistics alone do not pin a file to one precision - the parameter blocks
// still do.
//
// Reads validate the whole file before touching the model - payloads are
// staged, and the stream is walked twice (validate, then apply) - so a
// truncated, mismatched or corrupted file is rejected without mutating
// anything. Saves are the mirror image: the image is written to "<path>.tmp",
// synced, and renamed over <path>, so a crash or a failed write cannot destroy
// the previously saved checkpoint.
#include "checkpoint.h"
#include "layers.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

#define CHECKPOINT_FORMAT 4u

enum {
  BLOCK_END = 0,
  BLOCK_PARAMS = 1,
  BLOCK_ADAM = 2,
  BLOCK_NORM = 3,
  BLOCK_ARCH = 4,
};

#define ADAM_HYPER_BYTES 8u  // one u64 step
#define ADAM_HYPER_F64 4u    // lr, beta1, beta2, eps

// Sanity caps for untrusted files: a block that promises more than this is
// rejected before it can turn into a huge allocation.
#define MAX_NORM_OBS_DIM (1u << 20) // observation components in NORM
#define MAX_ARCH_HIDDEN 1024u       // hidden layers named in ARCH

static const char CHECKPOINT_MAGIC[8] = {'R', 'L', 'L', 'I', 'B', 'C', 'K', '4'};

static const char *g_last_error = "";

static void set_error(const char *msg) { g_last_error = msg; }

const char *checkpoint_last_error(void) { return g_last_error; }

// ── Little-endian primitives ─────────────────────────────────────────

static b32 put_u32(FILE *f, u32 v) {
  u8 b[4];
  b[0] = (u8)(v & 0xffu);
  b[1] = (u8)((v >> 8) & 0xffu);
  b[2] = (u8)((v >> 16) & 0xffu);
  b[3] = (u8)((v >> 24) & 0xffu);
  return fwrite(b, 1, 4, f) == 4;
}

static b32 get_u32(FILE *f, u32 *out) {
  u8 b[4];
  if (fread(b, 1, 4, f) != 4) {
    return false;
  }
  *out = (u32)b[0] | ((u32)b[1] << 8) | ((u32)b[2] << 16) | ((u32)b[3] << 24);
  return true;
}

static b32 put_u64(FILE *f, u64 v) {
  for (u32 i = 0; i < 8; i++) {
    u8 byte = (u8)((v >> (8u * i)) & 0xffu);
    if (fwrite(&byte, 1, 1, f) != 1) {
      return false;
    }
  }
  return true;
}

static b32 get_u64(FILE *f, u64 *out) {
  u64 v = 0;
  for (u32 i = 0; i < 8; i++) {
    u8 byte = 0;
    if (fread(&byte, 1, 1, f) != 1) {
      return false;
    }
    v |= (u64)byte << (8u * i);
  }
  *out = v;
  return true;
}

// Hyper-parameters are always stored as f64 so a checkpoint is readable by both
// scalar widths; only the tensor payload follows sizeof(real).
static b32 put_f64(FILE *f, f64 v) {
  u64 bits = 0;
  memcpy(&bits, &v, sizeof(bits));
  return put_u64(f, bits);
}

static b32 get_f64(FILE *f, f64 *out) {
  u64 bits = 0;
  if (!get_u64(f, &bits)) {
    return false;
  }
  memcpy(out, &bits, sizeof(bits));
  return true;
}

// real <-> little-endian bytes. Split by build mode rather than testing
// sizeof(real) at run time, so neither branch ever reads past a scalar.
static void put_real(u8 *out, real x) {
#if defined(RL_LIB_USE_DOUBLE)
  u64 bits = 0;
  memcpy(&bits, &x, sizeof(bits));
  for (u32 i = 0; i < 8; i++) {
    out[i] = (u8)((bits >> (8u * i)) & 0xffu);
  }
#else
  u32 bits = 0;
  memcpy(&bits, &x, sizeof(bits));
  for (u32 i = 0; i < 4; i++) {
    out[i] = (u8)((bits >> (8u * i)) & 0xffu);
  }
#endif
}

static real get_real(const u8 *in) {
#if defined(RL_LIB_USE_DOUBLE)
  u64 bits = 0;
  for (u32 i = 0; i < 8; i++) {
    bits |= (u64)in[i] << (8u * i);
  }
#else
  u32 bits = 0;
  for (u32 i = 0; i < 4; i++) {
    bits |= (u32)in[i] << (8u * i);
  }
#endif
  real x = (real)0;
  memcpy(&x, &bits, sizeof(x));
  return x;
}

// ── Parameter layout helpers ─────────────────────────────────────────

static u64 total_scalars(const Net *model) {
  u64 total = 0;
  for (u32 i = 0; i < model->n_params; i++) {
    total += (u64)model->params[i]->val->rows * (u64)model->params[i]->val->cols;
  }
  return total;
}

// scalar_bytes | n_params | rows[] | cols[]. Shared by the parameter and
// optimiser blocks so both are validated against the same model.
static b32 write_shape_table(FILE *f, const Net *model) {
  if (!put_u32(f, (u32)sizeof(real)) || !put_u32(f, model->n_params)) {
    return false;
  }
  for (u32 i = 0; i < model->n_params; i++) {
    const Mat *m = model->params[i]->val;
    if (!put_u32(f, m->rows) || !put_u32(f, m->cols)) {
      return false;
    }
  }
  return true;
}

static b32 read_shape_table(FILE *f, const Net *model, const char **why) {
  u32 scalar_bytes = 0;
  u32 n_params = 0;
  if (!get_u32(f, &scalar_bytes) || !get_u32(f, &n_params)) {
    *why = "truncated shape table";
    return false;
  }
  if (scalar_bytes != (u32)sizeof(real)) {
    *why = "checkpoint scalar width differs from this build (-DRL_LIB_USE_DOUBLE)";
    return false;
  }
  if (n_params != model->n_params) {
    *why = "checkpoint parameter count differs from this model";
    return false;
  }
  for (u32 i = 0; i < model->n_params; i++) {
    u32 rows = 0;
    u32 cols = 0;
    if (!get_u32(f, &rows) || !get_u32(f, &cols)) {
      *why = "truncated shape table";
      return false;
    }
    const Mat *m = model->params[i]->val;
    if (rows != m->rows || cols != m->cols) {
      *why = "checkpoint parameter shapes differ from this model";
      return false;
    }
  }
  return true;
}

// Reads `count` scalars of sizeof(real) bytes into a fresh buffer, or NULL if
// the stream ends early (the caller reports the error).
static u8 *read_scalars(FILE *f, u64 count) {
  u8 *buf = (u8 *)malloc((size_t)(count * sizeof(real)));
  if (buf == NULL) {
    return NULL;
  }
  if (fread(buf, sizeof(real), (size_t)count, f) != (size_t)count) {
    free(buf);
    return NULL;
  }
  return buf;
}

// Scatters a staged parameter payload into the model, in registration order.
static void apply_to_params(const u8 *buf, const Net *model) {
  u64 off = 0;
  for (u32 i = 0; i < model->n_params; i++) {
    Mat *m = model->params[i]->val;
    const u64 n = (u64)m->rows * (u64)m->cols;
    for (u64 j = 0; j < n; j++) {
      m->data[j] = get_real(buf + (off + j) * sizeof(real));
    }
    off += n;
  }
}

// Gives the staged parameter payload to Adam's moment buffers. `m2_off` is the
// scalar offset (not byte offset) at which the second moment data starts, so one
// staging buffer carries both moments.
static void apply_to_adam(const u8 *buf, u64 m2_off, const Net *model,
                          Adam *adam) {
  u64 off = 0;
  for (u32 i = 0; i < model->n_params; i++) {
    const Mat *m = model->params[i]->val;
    const u64 n = (u64)m->rows * (u64)m->cols;
    for (u64 j = 0; j < n; j++) {
      adam->m1[i][j] = get_real(buf + (off + j) * sizeof(real));
      adam->m2[i][j] = get_real(buf + (m2_off + off + j) * sizeof(real));
    }
    off += n;
  }
}

// ── Checksum ─────────────────────────────────────────────────────────

// Reflected CRC-32 (poly 0xEDB88320), a bit at a time: a checkpoint is a few
// hundred kilobytes at most, so the table-free loop costs nothing measurable
// and the format needs no lookup-table state.
static u32 crc32_update(u32 crc, const u8 *data, size_t n) {
  for (size_t i = 0; i < n; i++) {
    crc ^= data[i];
    for (u32 bit = 0; bit < 8; bit++) {
      crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
  }
  return crc;
}

// Appends the trailer: the CRC-32 of every byte already in `f`'s file. The
// write stream is flushed but stays open, so the bytes are re-read from disk
// (checkpoints are small) and the trailer lands at the current position.
static b32 write_crc_trailer(FILE *f, const char *path) {
  if (fflush(f) != 0) {
    set_error("failed to flush checkpoint");
    return false;
  }
  FILE *in = fopen(path, "rb");
  if (in == NULL) {
    set_error("cannot reopen checkpoint for checksum");
    return false;
  }
  u32 crc = 0;
  u8 buf[4096];
  size_t got;
  while ((got = fread(buf, 1, sizeof buf, in)) > 0) {
    crc = crc32_update(crc, buf, got);
  }
  const b32 read_ok = !ferror(in);
  fclose(in);
  if (!read_ok) {
    set_error("failed to read checkpoint for checksum");
    return false;
  }
  return put_u32(f, crc);
}

// Verifies the trailing CRC-32 before the parser touches a single block, and
// leaves the stream at the start of the header. The file size bounds the
// payload, so a truncated or hostile file is rejected without allocating.
static b32 verify_crc(FILE *f) {
  if (fseek(f, 0, SEEK_END) != 0) {
    set_error("cannot seek to checksum");
    return false;
  }
  const long size = ftell(f);
  if (size < (long)(8u + 4u + sizeof(u32))) {
    set_error("not a checkpoint file (too small)");
    return false;
  }
  const long payload = size - (long)sizeof(u32);
  if (fseek(f, 0, SEEK_SET) != 0) {
    set_error("cannot rewind to checksum");
    return false;
  }
  u32 crc = 0;
  u8 buf[4096];
  for (long remaining = payload; remaining > 0;) {
    const size_t want =
        remaining > (long)sizeof buf ? sizeof buf : (size_t)remaining;
    const size_t got = fread(buf, 1, want, f);
    if (got != want) {
      set_error("truncated checkpoint");
      return false;
    }
    crc = crc32_update(crc, buf, got);
    remaining -= (long)got;
  }
  u32 stored = 0;
  if (!get_u32(f, &stored)) {
    set_error("truncated checksum");
    return false;
  }
  if (stored != crc) {
    set_error("checksum mismatch (corrupted checkpoint)");
    return false;
  }
  if (fseek(f, 0, SEEK_SET) != 0) {
    set_error("cannot rewind checkpoint");
    return false;
  }
  return true;
}

static b32 write_header(FILE *f) {
  return fwrite(CHECKPOINT_MAGIC, 1, 8, f) == 8 &&
         put_u32(f, CHECKPOINT_FORMAT);
}

static b32 write_end_block(FILE *f) {
  return put_u32(f, BLOCK_END) && put_u32(f, 0);
}

// ── Block writers ────────────────────────────────────────────────────

static b32 write_params_block(FILE *f, const Net *model) {
  const u64 n_scalars = total_scalars(model);
  const u64 payload =
      8u + 8u * (u64)model->n_params + n_scalars * (u64)sizeof(real);
  if (!put_u32(f, BLOCK_PARAMS) || !put_u32(f, (u32)payload) ||
      !write_shape_table(f, model)) {
    return false;
  }
  if (n_scalars == 0) {
    return true;
  }

  u8 *staging = (u8 *)malloc((size_t)(n_scalars * sizeof(real)));
  if (staging == NULL) {
    set_error("out of memory");
    return false;
  }
  u64 off = 0;
  for (u32 i = 0; i < model->n_params; i++) {
    const Mat *m = model->params[i]->val;
    const u64 n = (u64)m->rows * (u64)m->cols;
    for (u64 j = 0; j < n; j++) {
      put_real(staging + (off + j) * sizeof(real), m->data[j]);
    }
    off += n;
  }
  const b32 ok = fwrite(staging, sizeof(real), (size_t)n_scalars, f) ==
                 (size_t)n_scalars;
  free(staging);
  return ok;
}

static b32 write_adam_block(FILE *f, const Adam *adam, const Net *model) {
  const u64 n_scalars = total_scalars(model);
  const u64 payload = 8u + 8u * (u64)model->n_params +
                      (u64)ADAM_HYPER_BYTES + 8u * (u64)ADAM_HYPER_F64 +
                      2u * n_scalars * (u64)sizeof(real);
  if (!put_u32(f, BLOCK_ADAM) || !put_u32(f, (u32)payload) ||
      !write_shape_table(f, model)) {
    return false;
  }
  if (!put_u64(f, adam->step) || !put_f64(f, (f64)adam->lr) ||
      !put_f64(f, (f64)adam->beta1) || !put_f64(f, (f64)adam->beta2) ||
      !put_f64(f, (f64)adam->eps)) {
    return false;
  }
  if (n_scalars == 0) {
    return true;
  }

  const u64 half = n_scalars * sizeof(real);
  u8 *staging = (u8 *)malloc((size_t)(2u * half));
  if (staging == NULL) {
    set_error("out of memory");
    return false;
  }
  u64 off = 0;
  for (u32 i = 0; i < model->n_params; i++) {
    const Mat *m = model->params[i]->val;
    const u64 n = (u64)m->rows * (u64)m->cols;
    for (u64 j = 0; j < n; j++) {
      put_real(staging + (off + j) * sizeof(real), adam->m1[i][j]);
      put_real(staging + half + (off + j) * sizeof(real), adam->m2[i][j]);
    }
    off += n;
  }
  const b32 ok = fwrite(staging, sizeof(real), (size_t)(2u * n_scalars), f) ==
                 (size_t)(2u * n_scalars);
  free(staging);
  return ok;
}

// Builder-made models record the architecture they were built with; a
// hand-built graph has no record, so it writes no block and loading is gated
// by the shape table alone.
static b32 write_arch_block(FILE *f, const Net *model) {
  NetArchInfo info;
  if (!net_arch_info(model, &info)) {
    return true;
  }
  if (info.num_hidden > MAX_ARCH_HIDDEN) {
    set_error("architecture names too many hidden layers to record");
    return false;
  }
  const u64 payload = 16u + 4u * (u64)info.num_hidden;
  if (!put_u32(f, BLOCK_ARCH) || !put_u32(f, (u32)payload) ||
      !put_u32(f, (u32)info.kind) || !put_u32(f, info.obs_dim) ||
      !put_u32(f, info.act_dim) || !put_u32(f, info.num_hidden)) {
    return false;
  }
  for (u32 i = 0; i < info.num_hidden; i++) {
    if (!put_u32(f, info.hidden_dims[i])) {
      return false;
    }
  }
  return true;
}

static u64 norm_payload_bytes(const NormStats *norm) {
  // obs_dim | count | ret_count | eps | clip_obs | ret_mean | ret_m2 | gamma
  // | mean[obs_dim] | m2[obs_dim]
  return 4u + 8u + 8u + 5u * 8u + 16u * (u64)norm->obs_dim;
}

// The running statistics of a normalised run. Scalars and arrays travel as f64
// so a value never changes width in transit; the f32/f64 anchor stays in the
// parameter blocks' scalar_bytes.
static b32 write_norm_block(FILE *f, const NormStats *norm) {
  if (norm->obs_dim > MAX_NORM_OBS_DIM) {
    set_error("normalisation width is unreasonably large");
    return false;
  }
  const u64 payload = norm_payload_bytes(norm);
  if (!put_u32(f, BLOCK_NORM) || !put_u32(f, (u32)payload) ||
      !put_u32(f, norm->obs_dim) || !put_u64(f, norm->count) ||
      !put_u64(f, norm->ret_count) || !put_f64(f, (f64)norm->eps) ||
      !put_f64(f, (f64)norm->clip_obs) || !put_f64(f, (f64)norm->ret_mean) ||
      !put_f64(f, (f64)norm->ret_m2) || !put_f64(f, (f64)norm->gamma)) {
    return false;
  }
  for (u32 i = 0; i < norm->obs_dim; i++) {
    if (!put_f64(f, (f64)norm->mean[i])) {
      return false;
    }
  }
  for (u32 i = 0; i < norm->obs_dim; i++) {
    if (!put_f64(f, (f64)norm->m2[i])) {
      return false;
    }
  }
  return true;
}

// ── Block readers ────────────────────────────────────────────────────

// Payload bytes a correctly sized parameter block must have for this model.
static u64 params_payload_bytes(const Net *model) {
  return 8u + 8u * (u64)model->n_params + total_scalars(model) *
                                              (u64)sizeof(real);
}

static u64 adam_payload_bytes(const Net *model) {
  return 8u + 8u * (u64)model->n_params + (u64)ADAM_HYPER_BYTES +
         8u * (u64)ADAM_HYPER_F64 + 2u * total_scalars(model) *
                                        (u64)sizeof(real);
}

static b32 skip_block(FILE *f, u32 bytes) {
  if (fseek(f, (long)bytes, SEEK_CUR) != 0) {
    set_error("seek failed while skipping a block");
    return false;
  }
  return true;
}

static b32 read_params_block(FILE *f, u32 bytes, Net *model, b32 dry) {
  if ((u64)bytes != params_payload_bytes(model)) {
    set_error("parameter block size does not match this model");
    return false;
  }
  const char *why = "";
  if (!read_shape_table(f, model, &why)) {
    set_error(why);
    return false;
  }

  const u64 n_scalars = total_scalars(model);
  if (n_scalars == 0) {
    return true;
  }
  if (dry) {
    return skip_block(f, (u32)(n_scalars * (u64)sizeof(real)));
  }
  u8 *staging = read_scalars(f, n_scalars);
  if (staging == NULL) {
    set_error("truncated parameter data");
    return false;
  }
  apply_to_params(staging, model);
  free(staging);
  return true;
}

static b32 read_adam_block(FILE *f, u32 bytes, Adam *adam, const Net *model,
                           b32 dry) {
  if (adam->n_params != model->n_params) {
    set_error("optimiser state does not match this model");
    return false;
  }
  if ((u64)bytes != adam_payload_bytes(model)) {
    set_error("optimiser block size does not match this model");
    return false;
  }
  const char *why = "";
  if (!read_shape_table(f, model, &why)) {
    set_error(why);
    return false;
  }

  u64 step = 0;
  f64 lr = 0;
  f64 beta1 = 0;
  f64 beta2 = 0;
  f64 eps = 0;
  if (!get_u64(f, &step) || !get_f64(f, &lr) || !get_f64(f, &beta1) ||
      !get_f64(f, &beta2) || !get_f64(f, &eps)) {
    set_error("truncated optimiser header");
    return false;
  }

  const u64 n_scalars = total_scalars(model);
  if (n_scalars > 0) {
    if (dry) {
      if (!skip_block(f, (u32)(2u * n_scalars * (u64)sizeof(real)))) {
        return false;
      }
    } else {
      u8 *staging = read_scalars(f, 2u * n_scalars);
      if (staging == NULL) {
        set_error("truncated optimiser data");
        return false;
      }
      apply_to_adam(staging, n_scalars, model, adam);
      free(staging);
    }
  }

  if (!dry) {
    adam->step = step;
    adam->lr = (real)lr;
    adam->beta1 = (real)beta1;
    adam->beta2 = (real)beta2;
    adam->eps = (real)eps;
  }
  return true;
}

// The ARCH block is informational on the way out and a gate on the way in: a
// file that records an architecture must match the target model's record
// exactly (kind, dimensions, hidden widths), and a hand-built target has no
// record to match, so it is rejected rather than guessed at. The payload's own
// size fixes num_hidden, so a hostile file cannot ask for a huge compare.
static b32 read_arch_block(FILE *f, u32 bytes, const Net *model, b32 dry) {
  if (bytes < 16u || (bytes - 16u) % 4u != 0u) {
    set_error("malformed architecture block");
    return false;
  }
  const u32 num_hidden = (bytes - 16u) / 4u;
  if (num_hidden > MAX_ARCH_HIDDEN) {
    set_error("architecture block names too many hidden layers");
    return false;
  }
  if (!dry) {
    return skip_block(f, bytes); // fully checked in the validating pass
  }

  u32 kind = 0;
  u32 obs_dim = 0;
  u32 act_dim = 0;
  u32 hidden_count = 0;
  if (!get_u32(f, &kind) || !get_u32(f, &obs_dim) || !get_u32(f, &act_dim) ||
      !get_u32(f, &hidden_count)) {
    set_error("truncated architecture block");
    return false;
  }

  NetArchInfo info;
  if (!net_arch_info(model, &info)) {
    set_error("checkpoint records an architecture; the model is hand-built");
    return false;
  }
  if (kind != (u32)info.kind) {
    set_error("checkpoint architecture kind differs from this model");
    return false;
  }
  if (obs_dim != info.obs_dim || act_dim != info.act_dim) {
    set_error("checkpoint architecture dimensions differ from this model");
    return false;
  }
  if (hidden_count != num_hidden || hidden_count != info.num_hidden) {
    set_error("checkpoint hidden-layer count differs from this model");
    return false;
  }
  for (u32 i = 0; i < num_hidden; i++) {
    u32 width = 0;
    if (!get_u32(f, &width)) {
      set_error("truncated architecture block");
      return false;
    }
    if (width != info.hidden_dims[i]) {
      set_error("checkpoint hidden-layer widths differ from this model");
      return false;
    }
  }
  return true;
}

// The NORM block's destination fixes the expected shape, so a file whose width
// disagrees is rejected before anything is allocated. Values are read into a
// temporary and copied only once the whole payload is in hand, keeping the
// all-or-nothing contract even if the stream somehow changes between passes.
static b32 read_norm_block(FILE *f, u32 bytes, NormStats *norm,
                           const Net *model, b32 dry) {
  if (norm->obs_dim > MAX_NORM_OBS_DIM ||
      (u64)bytes != norm_payload_bytes(norm)) {
    set_error("normalisation block size does not match the NormStats");
    return false;
  }

  u32 obs_dim = 0;
  u64 count = 0;
  u64 ret_count = 0;
  f64 scalars[5] = {0, 0, 0, 0, 0}; // eps, clip_obs, ret_mean, ret_m2, gamma
  if (!get_u32(f, &obs_dim) || !get_u64(f, &count) || !get_u64(f, &ret_count)) {
    set_error("truncated normalisation block");
    return false;
  }
  for (u32 i = 0; i < 5u; i++) {
    if (!get_f64(f, &scalars[i])) {
      set_error("truncated normalisation block");
      return false;
    }
  }
  if (obs_dim != norm->obs_dim) {
    set_error("checkpoint normalisation width differs from the NormStats");
    return false;
  }
  if (model->input != NULL && obs_dim != model->input->val->rows) {
    set_error("checkpoint normalisation width differs from the model input");
    return false;
  }

  if (dry) {
    return skip_block(f, 16u * obs_dim);
  }

  f64 *vals = (f64 *)malloc((size_t)(2u * obs_dim) * sizeof(f64));
  if (vals == NULL) {
    set_error("out of memory for normalisation data");
    return false;
  }
  for (u32 i = 0; i < 2u * obs_dim; i++) {
    if (!get_f64(f, &vals[i])) {
      free(vals);
      set_error("truncated normalisation data");
      return false;
    }
  }
  for (u32 i = 0; i < obs_dim; i++) {
    norm->mean[i] = (real)vals[i];
    norm->m2[i] = (real)vals[obs_dim + i];
  }
  free(vals);
  norm->count = count;
  norm->ret_count = ret_count;
  norm->eps = (real)scalars[0];
  norm->clip_obs = (real)scalars[1];
  norm->ret_mean = (real)scalars[2];
  norm->ret_m2 = (real)scalars[3];
  norm->gamma = (real)scalars[4];
  return true;
}

// ── Stream driver ────────────────────────────────────────────────────

typedef struct {
  Net *model;
  Adam *adam;      // optional: an ADAM block is skipped when NULL
  NormStats *norm; // optional: a NORM block is skipped when NULL
  b32 saw_params;
  b32 saw_adam;
  b32 dry; // validate without applying anything
} LoadState;

static b32 load_stream(FILE *f, LoadState *st) {
  char magic[8];
  u32 version = 0;
  if (fread(magic, 1, 8, f) != 8) {
    set_error("truncated header");
    return false;
  }
  if (memcmp(magic, CHECKPOINT_MAGIC, 8) != 0) {
    if (memcmp(magic, "RLLIBCK", 7) == 0 && magic[7] >= '1' &&
        magic[7] <= '3') {
      set_error("older checkpoint format; this library writes format 4");
    } else if (memcmp(magic, "RLLIBCK", 7) == 0) {
      set_error("unsupported checkpoint format");
    } else {
      set_error("not a checkpoint file (bad magic)");
    }
    return false;
  }
  if (!get_u32(f, &version)) {
    set_error("truncated header");
    return false;
  }
  if (version != CHECKPOINT_FORMAT) {
    set_error("unsupported checkpoint version");
    return false;
  }

  for (;;) {
    u32 kind = 0;
    u32 bytes = 0;
    if (!get_u32(f, &kind) || !get_u32(f, &bytes)) {
      set_error("truncated block header");
      return false;
    }
    if (kind == BLOCK_END) {
      return true;
    }
    if (kind == BLOCK_PARAMS) {
      if (!read_params_block(f, bytes, st->model, st->dry)) {
        return false;
      }
      st->saw_params = true;
      continue;
    }
    if (kind == BLOCK_ADAM) {
      if (st->adam != NULL) {
        if (!read_adam_block(f, bytes, st->adam, st->model, st->dry)) {
          return false;
        }
      } else if (!skip_block(f, bytes)) {
        return false;
      }
      st->saw_adam = true;
      continue;
    }
    if (kind == BLOCK_NORM) {
      if (st->norm != NULL) {
        if (!read_norm_block(f, bytes, st->norm, st->model, st->dry)) {
          return false;
        }
      } else if (!skip_block(f, bytes)) {
        return false;
      }
      continue;
    }
    if (kind == BLOCK_ARCH) {
      if (!read_arch_block(f, bytes, st->model, st->dry)) {
        return false;
      }
      continue;
    }
    if (!skip_block(f, bytes)) { // unknown kind: forward compatible
      return false;
    }
  }
}

// Opens `path`, applies every block it can, and insists on the blocks the
// caller asked for. `model` supplies the shapes every block is validated
// against; `adam` and `norm` are optional and only receive their block when
// non-NULL.
static b32 load_from_path(const char *path, Net *model, Adam *adam,
                          NormStats *norm, b32 need_params, b32 need_adam) {
  g_last_error = "";
  if (model == NULL || path == NULL) {
    set_error("null model or path");
    return false;
  }
  FILE *f = fopen(path, "rb");
  if (f == NULL) {
    set_error("cannot open checkpoint for reading");
    return false;
  }

  LoadState st;
  st.model = model;
  st.adam = adam;
  st.norm = norm;
  st.saw_params = false;
  st.saw_adam = false;
  st.dry = true;

  // Two passes over a file whose checksum already verified: the first validates
  // every block without touching the model, the second applies them. A load that
  // reports failure therefore leaves the model exactly as it was, even when the
  // damage sits after a block that parsed cleanly.
  b32 ok = verify_crc(f) && load_stream(f, &st);
  if (ok && need_params && !st.saw_params) {
    set_error("checkpoint has no parameter block");
    ok = false;
  }
  if (ok && need_adam && !st.saw_adam) {
    set_error("checkpoint has no optimiser block");
    ok = false;
  }
  if (ok && fseek(f, 0, SEEK_SET) != 0) {
    set_error("cannot rewind checkpoint");
    ok = false;
  }
  if (ok) {
    st.dry = false;
    st.saw_params = false;
    st.saw_adam = false;
    ok = load_stream(f, &st);
  }
  if (fclose(f) != 0 && ok) {
    set_error("failed to close checkpoint");
    ok = false;
  }
  return ok;
}

// ── Atomic replacement ───────────────────────────────────────────────
//
// Every save writes "<path>.tmp" and renames it over "<path>" once the bytes
// are on disk. The rename is atomic on POSIX and uses MoveFileEx on Windows, so
// at any instant the destination is either the old checkpoint or the complete
// new one - never a partially written file. A failure removes the temporary and
// leaves the old checkpoint in place.

static char *temp_path_for(const char *path) {
  const size_t n = strlen(path);
  char *tmp = (char *)malloc(n + 5u); // ".tmp" plus the terminator
  if (tmp == NULL) {
    set_error("out of memory for the temporary checkpoint path");
    return NULL;
  }
  memcpy(tmp, path, n);
  memcpy(tmp + n, ".tmp", 5u);
  return tmp;
}

static b32 finish_save(FILE *f, char *tmp, const char *path, b32 ok) {
  if (ok && fflush(f) != 0) {
    set_error("failed to flush checkpoint");
    ok = false;
  }
#if defined(_WIN32)
  if (ok && _commit(_fileno(f)) != 0) {
    set_error("failed to sync checkpoint to disk");
    ok = false;
  }
#else
  if (ok && fsync(fileno(f)) != 0) {
    set_error("failed to sync checkpoint to disk");
    ok = false;
  }
#endif
  if (fclose(f) != 0 && ok) {
    set_error("failed to close checkpoint");
    ok = false;
  }
  if (ok) {
#if defined(_WIN32)
    ok = MoveFileExA(tmp, path,
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    ok = rename(tmp, path) == 0;
#endif
    if (!ok) {
      set_error("cannot replace checkpoint (rename failed)");
    }
  }
  if (!ok) {
    remove(tmp);
  }
  free(tmp);
  return ok;
}

// ── Public API ───────────────────────────────────────────────────────

b32 net_save(const Net *model, const char *path) {
  g_last_error = "";
  if (model == NULL || path == NULL) {
    set_error("null model or path");
    return false;
  }
  char *tmp = temp_path_for(path);
  if (tmp == NULL) {
    return false;
  }
  FILE *f = fopen(tmp, "wb");
  if (f == NULL) {
    set_error("cannot open checkpoint for writing");
    free(tmp);
    return false;
  }
  b32 ok = write_header(f) && write_arch_block(f, model) &&
           write_params_block(f, model) && write_end_block(f);
  if (ok) {
    ok = write_crc_trailer(f, tmp);
  }
  return finish_save(f, tmp, path, ok);
}

b32 net_load(Net *model, const char *path) {
  return load_from_path(path, model, NULL, NULL, true, false);
}

b32 adam_save(const Adam *adam, const Net *model, const char *path) {
  g_last_error = "";
  if (adam == NULL || model == NULL || path == NULL) {
    set_error("null argument");
    return false;
  }
  if (adam->n_params != model->n_params) {
    set_error("optimiser state does not match this model");
    return false;
  }
  char *tmp = temp_path_for(path);
  if (tmp == NULL) {
    return false;
  }
  FILE *f = fopen(tmp, "wb");
  if (f == NULL) {
    set_error("cannot open checkpoint for writing");
    free(tmp);
    return false;
  }
  b32 ok = write_header(f) && write_arch_block(f, model) &&
           write_adam_block(f, adam, model) && write_end_block(f);
  if (ok) {
    ok = write_crc_trailer(f, tmp);
  }
  return finish_save(f, tmp, path, ok);
}

b32 adam_load(Adam *adam, Net *model, const char *path) {
  return load_from_path(path, model, adam, NULL, false, true);
}

b32 checkpoint_save(const Net *model, const Adam *adam, const char *path) {
  return checkpoint_save_ex(model, adam, NULL, path);
}

b32 checkpoint_save_ex(const Net *model, const Adam *adam,
                       const NormStats *norm, const char *path) {
  g_last_error = "";
  if (model == NULL || path == NULL) {
    set_error("null model or path");
    return false;
  }
  if (adam != NULL && adam->n_params != model->n_params) {
    set_error("optimiser state does not match this model");
    return false;
  }
  if (norm != NULL && model->input != NULL &&
      norm->obs_dim != model->input->val->rows) {
    set_error("normalisation width does not match the model input");
    return false;
  }
  char *tmp = temp_path_for(path);
  if (tmp == NULL) {
    return false;
  }
  FILE *f = fopen(tmp, "wb");
  if (f == NULL) {
    set_error("cannot open checkpoint for writing");
    free(tmp);
    return false;
  }
  b32 ok = write_header(f) && write_arch_block(f, model) &&
           write_params_block(f, model);
  ok = ok && (adam == NULL || write_adam_block(f, adam, model));
  ok = ok && (norm == NULL || write_norm_block(f, norm));
  ok = ok && write_end_block(f);
  if (ok) {
    ok = write_crc_trailer(f, tmp);
  }
  return finish_save(f, tmp, path, ok);
}

b32 checkpoint_load(Net *model, Adam *adam, const char *path) {
  return checkpoint_load_ex(model, adam, NULL, path);
}

b32 checkpoint_load_ex(Net *model, Adam *adam, NormStats *norm,
                       const char *path) {
  return load_from_path(path, model, adam, norm, true, adam != NULL);
}
