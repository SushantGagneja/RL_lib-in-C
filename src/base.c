// base.c — the fatal-error hook described in base.h. The handler pointer is a
// plain global: install one before worker threads start, or synchronise the
// setter yourself. The default prints the message and returns, after which
// rl_lib_fatal aborts; a custom handler is expected to longjmp or exit.
#include "base.h"

static void default_fatal(const char *msg) { fprintf(stderr, "%s\n", msg); }

static rl_fatal_fn g_fatal_handler = default_fatal;

rl_fatal_fn rl_lib_set_fatal_handler(rl_fatal_fn handler) {
  rl_fatal_fn previous = g_fatal_handler;
  g_fatal_handler = (handler != NULL) ? handler : default_fatal;
  return previous;
}

void rl_lib_fatal(const char *msg) {
  g_fatal_handler((msg != NULL) ? msg : "[RL_lib-in-C FATAL]");
  abort(); // handlers must not return; this is the backstop
}
