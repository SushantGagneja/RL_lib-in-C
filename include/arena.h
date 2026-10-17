#ifndef ARENA_H
#define ARENA_H

#include "base.h"

// Bump allocator alignment. 64 bytes so future SIMD (AVX2/NEON) loads from
// matrix data stay aligned. arena_create backs the memory with a 64-byte
// aligned allocation, so offsets aligned to ARENA_ALIGN are truly aligned.
#define ARENA_ALIGN 64

typedef struct {
  u8 *memory;
  u64 capacity;
  u64 pos;
} mem_arena;

mem_arena *arena_create(u64 capacity);
void arena_destroy(mem_arena *arena);

// Allocates `size` bytes, aligned to ARENA_ALIGN. Aborts on out-of-memory
// (there is no way to recover meaningfully from an exhausted arena here).
void *arena_push(mem_arena *arena, u64 size, b32 zero);

// Current bump position and a way to rewind to it. Use to create a temporary
// scope: `u64 mark = arena_pos(a); ...; arena_pop_to(a, mark);`
u64 arena_pos(const mem_arena *arena);
void arena_pop_to(mem_arena *arena, u64 pos);

#define PUSH_STRUCT(arena, T) (T *)arena_push((arena), sizeof(T), true)
#define PUSH_ARRAY(arena, T, n) (T *)arena_push((arena), sizeof(T) * (n), true)
#define PUSH_ARRAY_UNINIT(arena, T, n)                                         \
  (T *)arena_push((arena), sizeof(T) * (n), false)

#endif
