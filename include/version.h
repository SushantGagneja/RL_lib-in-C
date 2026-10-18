#ifndef VERSION_H
#define VERSION_H

// Single source of truth for the library version. The Makefile reads
// RL_LIB_VERSION_STRING from this file to stamp rl_lib.pc, the git tag matches
// it, and the API may only break on a MAJOR bump (see CHANGELOG.md).
#define RL_LIB_VERSION_MAJOR 1
#define RL_LIB_VERSION_MINOR 0
#define RL_LIB_VERSION_PATCH 0
#define RL_LIB_VERSION_STRING "1.0.0"

#endif // VERSION_H
