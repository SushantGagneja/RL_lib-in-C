#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "graph.h"
#include "optim.h"
#include "rl.h"

// Parameter and optimiser checkpointing. A checkpoint is a small, versioned,
// little-endian binary file that stores the parameter values of a Net and
// (optionally) the Adam moments and the observation-normalisation statistics,
// so a training run can be stopped and resumed.
//
//   net_save / net_load        — parameters only
//   adam_save / adam_load      — optimiser moments only
//   checkpoint_save / _load    — both, in one file (use this for resume)
//   checkpoint_save_ex / _load_ex — both plus NormStats (normalised runs)
//
// Both loads apply every block the file contains (so adam_load also restores the
// parameters of a combined checkpoint) and require the block the call implies.
//
// Files written for a builder-made model (actor_build and friends) also carry an
// architecture block: kind, observation/action dimensions and hidden widths.
// Loading it into a different architecture is rejected up front, before any
// parameter changes, so a checkpoint cannot be silently loaded into the wrong
// model even when the parameter counts happen to match.
//
// Every function returns false instead of aborting on any I/O, format, or shape
// error, including a mismatch between the file and the model's parameter shapes
// or scalar width (an f32 checkpoint cannot be loaded into an f64 build, because
// the values would be silently reinterpreted). Loading is all-or-nothing: the
// file's shapes are validated and its data staged before a single parameter is
// written, so a rejected file leaves the model exactly as it was.
// checkpoint_last_error() describes the most recent failure.
//
// Saves are atomic: the new image is written to "<path>.tmp", synced, and
// renamed over <path>, so a crash or a failed write leaves the previous
// checkpoint intact instead of truncating it.
//
// Not saved: the RNG state and the rollout buffer. A resumed run re-derives its
// RNG from TrainCfg.seed, so it continues from the saved parameters — it just is
// not bit-identical to a run that was never interrupted.
//
// The format number is 4; files written by earlier versions are rejected with a
// clear error before any block is parsed.

b32 net_save(const Net *model, const char *path);
b32 net_load(Net *model, const char *path);

b32 adam_save(const Adam *adam, const Net *model, const char *path);
b32 adam_load(Adam *adam, Net *model, const char *path);

b32 checkpoint_save(const Net *model, const Adam *adam, const char *path);
b32 checkpoint_load(Net *model, Adam *adam, const char *path);

// As above, with the running statistics of a normalised run. Saving with a
// non-NULL `norm` adds a normalisation block; loading with a non-NULL `norm`
// restores it when the file has one and leaves it untouched otherwise (so a
// file written without statistics is still a valid resume target).
b32 checkpoint_save_ex(const Net *model, const Adam *adam,
                       const NormStats *norm, const char *path);
b32 checkpoint_load_ex(Net *model, Adam *adam, NormStats *norm,
                       const char *path);

// Human-readable reason for the last failed save/load ("" after a success).
// Points at static storage: copy it if you need to keep it, and read it from the
// same thread that made the call.
const char *checkpoint_last_error(void);

#endif // CHECKPOINT_H
