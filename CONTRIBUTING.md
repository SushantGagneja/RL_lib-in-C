# Contributing

Thanks for taking a look. RL_lib-in-C is a small, dependency-free C library, and
the bar for a change is simple: keep it readable, keep it warning-clean in both
precisions, and keep the suites green.

By contributing you agree that your work is licensed under the project's MIT
license.

## Development loop

Prerequisites are a C99 compiler, POSIX `make`, and `libm`; CMake 3.20+ is the
alternative (and the only path on Windows). From a clean checkout:

```bash
make ERR=1 test        # every suite, strict warnings as errors
make asan              # ASan + UBSan build of every suite
make test-double       # every suite again in f64
make tsan              # the threaded VecEnv suites
make bench-throughput  # steps/sec and GEMM GFLOPS
```

`make clean` between build modes: the suites and examples share an object tree,
and mixing sanitizer and non-sanitizer objects produces link errors.

CMake is the same project:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## What a change should include

- **Tests.** New behaviour goes in the closest `tests/test_*.c`, with
  deterministic seeds. If it touches the autograd graph, add it to
  `test_gradcheck` as well so the analytic gradient is compared with central
  differences.
- **Docs.** Public functions are documented in the header and in
  [`docs/API.md`](docs/API.md). User-visible changes get a line in
  [`CHANGELOG.md`](CHANGELOG.md) under `Unreleased`.
- **Both precisions.** The library compiles as f32 by default and as f64 with
  `-DRL_LIB_USE_DOUBLE`; `make test-double` is the check.
- **No new dependencies.** The library is intentionally standard C99 + `libm`
  (+ pthreads for the optional threaded `VecEnv`). Optional system BLAS is the
  one supported escape hatch.

## Style

Match the surrounding code: four-space indentation, braces on the same line,
`static` helpers where possible, and public symbols in the `rl_*` / `mat_*` /
`net_*` / `env_*` namespaces used by the existing modules. Comments explain
*why* a non-obvious choice was made, not what the next line does.

## Reporting bugs

Include the platform and compiler, the exact command, and the smallest input
that reproduces the problem. Training failures should say whether they happen
in the f32 or f64 build, and whether the sanitizer targets are clean.
