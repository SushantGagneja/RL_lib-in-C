#include "memory.h"

#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <malloc.h> // _aligned_malloc / _aligned_free
#endif

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
#if defined(_WIN32)
  // Windows (MSVC and mingw alike) has no posix_memalign; _aligned_malloc is the
  // native allocator and is freed with _aligned_free, never with free().
  block = _aligned_malloc((size_t)capacity, (size_t)ARENA_ALIGN);
  if (block == NULL) {
    free(arena);
    return NULL;
  }
#else
  if (posix_memalign(&block, ARENA_ALIGN, (size_t)capacity) != 0 ||
      block == NULL) {
    free(arena);
    return NULL;
  }
#endif

  arena->base = (u8 *)block;
  arena->cap = capacity;
  arena->offset = 0u;
  return arena;
}

void arena_dispose(Arena *arena) {
  if (arena != NULL) {
#if defined(_WIN32)
    _aligned_free(arena->base);
#else
    free(arena->base);
#endif
    free(arena);
  }
}

void *arena_alloc(Arena *arena, u64 size, b32 zero) {
  RL_LIB_ASSERT(arena != NULL, "arena_alloc: NULL arena");

  const u64 start = slot_start(arena->offset);
  if (start > arena->cap || size > arena->cap - start) {
    fprintf(stderr,
            "[RL_lib-in-C FATAL] arena exhausted: %llu bytes requested at offset "
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
  RL_LIB_ASSERT(arena != NULL, "arena_mark: NULL arena");
  return arena->offset;
}

void arena_rewind(Arena *arena, u64 mark) {
  RL_LIB_ASSERT(arena != NULL, "arena_rewind: NULL arena");
  RL_LIB_ASSERT(mark <= arena->offset,
             "arena_rewind: mark is ahead of the bump pointer");
  arena->offset = mark;
}
