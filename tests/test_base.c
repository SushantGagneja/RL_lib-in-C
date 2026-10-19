// test_base.c — the base.h contract: the byte/size and min/max helpers, and the
// fatal-error hook (the default aborts; an installed handler can capture the
// message and longjmp back into the suite).
#include "base.h"

#include <setjmp.h>
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

static jmp_buf s_jmp;
static char s_fatal[256];

static void capture_fatal(const char *msg) {
  snprintf(s_fatal, sizeof(s_fatal), "%s", msg);
  longjmp(s_jmp, 1);
}

int main(void) {
  printf("\n=== base.h tests ===\n\n");

  printf("  size and arithmetic helpers... ");
  ASSERT_TRUE(KiB(1) == (u64)1024, "KiB");
  ASSERT_TRUE(MiB(1) == (u64)1048576, "MiB");
  ASSERT_TRUE(GiB(1) == (u64)1073741824, "GiB");
  ASSERT_TRUE(MIN(3, 7) == 3 && MAX(3, 7) == 7, "MIN/MAX");
  ASSERT_TRUE(ALIGN_UP_POW2(5u, 8u) == 8u, "ALIGN_UP_POW2 rounds up");
  ASSERT_TRUE(ALIGN_UP_POW2(8u, 8u) == 8u, "ALIGN_UP_POW2 keeps multiples");
  ASSERT_TRUE(ALIGN_UP_POW2(9u, 8u) == 16u, "ALIGN_UP_POW2 crosses to the next block");
  printf("OK\n");

  printf("  fatal-error hook... ");
  rl_fatal_fn previous = rl_lib_set_fatal_handler(capture_fatal);
  ASSERT_TRUE(previous != NULL, "set returns the previous handler");

  s_fatal[0] = '\0';
  if (setjmp(s_jmp) == 0) {
    RL_LIB_CHECK(1 == 2);
    ASSERT_TRUE(false, "RL_LIB_CHECK dispatched instead of returning");
  } else {
    ASSERT_TRUE(strstr(s_fatal, "check failed") != NULL,
                "the check message names the failure");
    ASSERT_TRUE(strstr(s_fatal, "1 == 2") != NULL,
                "the check message carries the expression");
    ASSERT_TRUE(strstr(s_fatal, "test_base.c") != NULL,
                "the check message carries the source location");
  }

  s_fatal[0] = '\0';
  if (setjmp(s_jmp) == 0) {
    RL_LIB_ASSERT(0, "custom failure text");
    ASSERT_TRUE(false, "RL_LIB_ASSERT dispatched instead of returning");
  } else {
    ASSERT_TRUE(strstr(s_fatal, "custom failure text") != NULL,
                "the assert message carries the caller's text");
  }

  rl_fatal_fn restored = rl_lib_set_fatal_handler(NULL);
  ASSERT_TRUE(restored == capture_fatal, "NULL restores the default handler");
  printf("OK\n");

  printf("\n---\nTests: %u passed, %u failed\n\n", tests_passed, tests_failed);
  return tests_failed > 0 ? 1 : 0;
}
