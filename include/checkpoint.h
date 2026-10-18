#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "graph.h"
#include "optim.h"

// Parameter and optimiser checkpointing. A checkpoint is a small, versioned,
// little-endian binary file that stores the parameter values of a Net and
// (optionally) the Adam moments, so a training run can be stopped and resumed.
//
//   net_save / net_load        — parameters only
//   adam_save / adam_load      — optimiser moments only
//   checkpoint_save / _load    — both, in one file (use this for resume)
//
// Both loads apply every block the file contains (so adam_load also restores the
// parameters of a combined checkpoint) and require the block the call implies.
//
// Every function returns false instead of aborting on any I/O, format, or shape
// error, including a mismatch between the file and the model's parameter shapes
// or scalar width (an f32 checkpoint cannot be loaded into an f64 build, because
// the values would be silently reinterpreted). Loading is all-or-nothing: the
// file's shapes are validated and its data staged before a single parameter is
// written, so a rejected file leaves the model exactly as it was.
// checkpoint_last_error() describes the most recent failure.
//
// Not saved: the RNG state and the rollout buffer. A resumed run re-derives its
// RNG from TrainCfg.seed, so it continues from the saved parameters — it just is
// not bit-identical to a run that was never interrupted.

b32 net_save(const Net *model, const char *path);
b32 net_load(Net *model, const char *path);

b32 adam_save(const Adam *adam, const Net *model, const char *path);
b32 adam_load(Adam *adam, Net *model, const char *path);

b32 checkpoint_save(const Net *model, const Adam *adam, const char *path);
b32 checkpoint_load(Net *model, Adam *adam, const char *path);

// Human-readable reason for the last failed save/load ("" after a success).
// Points at static storage: copy it if you need to keep it, and read it from the
// same thread that made the call.
const char *checkpoint_last_error(void);

#endif // CHECKPOINT_H
