#include "arena.h"
#include <string.h>

mem_arena *arena_create(u64 capacity) {
  mem_arena *arena = malloc(sizeof(mem_arena));
  if (arena == NULL) {
    return NULL;
  }

  // 64-byte aligned backing store so ARENA_ALIGN offsets are truly aligned.
  void *memory = NULL;
  if (posix_memalign(&memory, ARENA_ALIGN, capacity) != 0) {
    free(arena);
    return NULL;
  }

  arena->memory = (u8 *)memory;
  arena->capacity = capacity;
  arena->pos = 0;

  return arena;
}

void arena_destroy(mem_arena *arena) {
  if (arena == NULL) {
    return;
  }
  free(arena->memory);
  free(arena);
}

void *arena_push(mem_arena *arena, u64 size, b32 zero) {
  CRL_ASSERT(arena != NULL, "arena_push: NULL arena");

  u64 pos_aligned = ALIGN_UP_POW2(arena->pos, (u64)ARENA_ALIGN);
  if (pos_aligned > arena->capacity || size > arena->capacity - pos_aligned) {
    fprintf(stderr,
            "[cRL FATAL] arena out of memory: need %llu bytes at %llu, "
            "capacity %llu\n",
            (unsigned long long)size, (unsigned long long)pos_aligned,
            (unsigned long long)arena->capacity);
    abort();
  }

  u64 new_pos = pos_aligned + size;
  u8 *out = arena->memory + pos_aligned;
  arena->pos = new_pos;

  if (zero) {
    memset(out, 0, size);
  }

  return out;
}

u64 arena_pos(const mem_arena *arena) {
  CRL_ASSERT(arena != NULL, "arena_pos: NULL arena");
  return arena->pos;
}

void arena_pop_to(mem_arena *arena, u64 pos) {
  CRL_ASSERT(arena != NULL, "arena_pop_to: NULL arena");
  CRL_ASSERT(pos <= arena->pos, "arena_pop_to: pos is ahead of the bump pointer");
  arena->pos = pos;
}
