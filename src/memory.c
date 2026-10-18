#include "memory.h"

#include <stdlib.h>
#include <string.h>

// ARENA_ALIGN is a power of two, so alignment is a mask add.
static u64 slot_start(u64 offset) {
  return ALIGN_UP_POW2(offset, (u64)ARENA_ALIGN);
}

Arena *arena_new(u64 capacity) {
  Arena *arena = (Arena *)malloc(sizeof *arena);
  if (arena == NULL) {
    return NULL;
  }

  void *block = NULL;
  if (posix_memalign(&block, ARENA_ALIGN, (size_t)capacity) != 0 ||
      block == NULL) {
    free(arena);
    return NULL;
  }

  arena->base = (u8 *)block;
  arena->cap = capacity;
  arena->offset = 0u;
  return arena;
}

void arena_dispose(Arena *arena) {
  if (arena != NULL) {
    free(arena->base);
    free(arena);
  }
}

void *arena_alloc(Arena *arena, u64 size, b32 zero) {
  CRL_ASSERT(arena != NULL, "arena_alloc: NULL arena");

  const u64 start = slot_start(arena->offset);
  if (start > arena->cap || size > arena->cap - start) {
    fprintf(stderr,
            "[cRL FATAL] arena exhausted: %llu bytes requested at offset "
            "%llu of %llu\n",
            (unsigned long long)size, (unsigned long long)start,
            (unsigned long long)arena->cap);
    abort();
  }

  u8 *slot = arena->base + start;
  arena->offset = start + size;

  if (zero) {
    memset(slot, 0, (size_t)size);
  }
  return slot;
}

u64 arena_mark(const Arena *arena) {
  CRL_ASSERT(arena != NULL, "arena_mark: NULL arena");
  return arena->offset;
}

void arena_rewind(Arena *arena, u64 mark) {
  CRL_ASSERT(arena != NULL, "arena_rewind: NULL arena");
  CRL_ASSERT(mark <= arena->offset,
             "arena_rewind: mark is ahead of the bump pointer");
  arena->offset = mark;
}
