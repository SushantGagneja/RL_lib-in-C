#ifndef MEMORY_H
#define MEMORY_H

#include "base.h"

// Arena alignment. ARENA_ALIGN must be a power of two; 64 keeps Mat data
// aligned for wider SIMD loads, and arena_new() hands out backing memory that
// is already aligned to it, so every slot we return is genuinely aligned.
#define ARENA_ALIGN 64

// Linear (bump) allocator. Everything allocated from an Arena lives until the
// Arena is disposed, so ownership is trivial: no per-object free.
typedef struct {
  u8 *base;    // start of the backing block
  u64 cap;     // total bytes in the backing block
  u64 offset;  // bytes handed out so far
} Arena;

Arena *arena_new(u64 capacity);
void arena_dispose(Arena *arena);

// Allocates `size` bytes aligned to ARENA_ALIGN. Aborts on exhaustion: an
// out-of-memory arena is unrecoverable here, so we fail loudly rather than
// return NULL and let callers crash later.
void *arena_alloc(Arena *arena, u64 size, b32 zero);

// Mark/rewind for temporary scopes:
//   u64 mark = arena_mark(a);  ...;  arena_rewind(a, mark);
u64 arena_mark(const Arena *arena);
void arena_rewind(Arena *arena, u64 mark);

#define ARENA_OBJ(arena, T) (T *)arena_alloc((arena), sizeof(T), true)
#define ARENA_ARR(arena, T, n) (T *)arena_alloc((arena), sizeof(T) * (n), true)
#define ARENA_ARR_RAW(arena, T, n)                                         \
  (T *)arena_alloc((arena), sizeof(T) * (n), false)

#endif
