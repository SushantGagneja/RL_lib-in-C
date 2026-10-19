CC ?= cc
# Strict warning set. Pass ERR=1 (e.g. `make ERR=1 test`) to turn them into
# errors; CI uses that, local builds do not so other compilers stay buildable.
WARN = -Wall -Wextra -Wshadow -Wconversion -Wmissing-prototypes
ifdef ERR
WARN += -Werror
endif
# Extra flags for ad-hoc configurations, e.g.
#   make EXTRA_CFLAGS=-DRL_SIMD_BACKEND=0 test
#   make EXTRA_CFLAGS=-DRL_LIB_NO_THREADS THREAD_FLAGS= test   (no pthreads)
EXTRA_CFLAGS ?=
# The library uses POSIX threads for the optional threaded VecEnv; every target
# links with it, so the flag lives here where the link rules can see it. (Windows
# has no make toolchain here: the CMake project builds there, and the Win32
# thread API needs no flag.)
THREAD_FLAGS ?= -pthread
# Optional system BLAS backend: `make BLAS=1 ...` routes every MatMul through
# CBLAS instead of the hand-written SIMD/scalar kernels. macOS selects
# Accelerate automatically; elsewhere point at your BLAS, e.g.
#   make BLAS=1 BLAS_CFLAGS=-I/usr/include/openblas BLAS_LIBS=-lopenblas test
ifdef BLAS
EXTRA_CFLAGS += -DRL_LIB_USE_BLAS $(BLAS_CFLAGS)
ifeq ($(shell uname -s),Darwin)
BLAS_LIBS ?= -framework Accelerate
else
BLAS_LIBS ?= -lopenblas
endif
endif
# Coverage: `make COVERAGE=1 test` instruments the build (the link rules use
# CFLAGS, so --coverage reaches them too) and the CI job summarises the .gcda
# files with gcovr.
ifdef COVERAGE
EXTRA_CFLAGS += --coverage
endif
CFLAGS = $(WARN) -Iinclude -I. $(THREAD_FLAGS) $(EXTRA_CFLAGS)

LIB_SRCS = src/memory.c src/random.c src/simd.c src/linalg.c src/graph.c src/layers.c src/optim.c src/checkpoint.c src/rl.c src/dqn.c src/vecenv.c envs/snake.c envs/cartpole.c envs/pendulum.c
LIB_OBJS = $(LIB_SRCS:.c=.o)

TARGET_SNAKE = train_snake
TARGET_CARTPOLE = train_cartpole
TARGET_A2C = train_a2c
TARGET_PPO = train_ppo
TARGET_PPO_CONT = train_ppo_continuous
TARGET_PPO_SNAKE = train_ppo_snake
TARGET_DQN = train_dqn
TARGET_HELLO = hello_rl

# ── Version and install layout ──────────────────────────────────────
# The version comes from include/version.h so the headers, pkg-config file, and
# shared-library soname can never drift apart.
VERSION = $(shell sed -n 's/.*RL_LIB_VERSION_STRING "\(.*\)"/\1/p' include/version.h)
VERSION_MAJOR = $(word 1,$(subst ., ,$(VERSION)))

PREFIX ?= /usr/local
LIBDIR ?= $(PREFIX)/lib
INCLUDEDIR ?= $(PREFIX)/include
PKGCONFIGDIR ?= $(LIBDIR)/pkgconfig

# Extra link flags a consumer of the installed library needs, stamped into
# rl_lib.pc: libm, plus the thread flag unless this build dropped threads (the
# documented no-threads invocation sets THREAD_FLAGS= empty).
PC_LIBS_EXTRA = -lm $(BLAS_LIBS) $(THREAD_FLAGS)

LIB_STATIC = librl_lib.a

# Shared objects need -fPIC, so they get their own object tree and never mix
# with the objects the in-tree binaries link.
UNAME_S = $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
SHARED_REAL = librl_lib.$(VERSION_MAJOR).dylib
SHARED_LINK = librl_lib.dylib
SHARED_LDFLAGS = -dynamiclib -install_name $(LIBDIR)/$(SHARED_REAL)
SHARED_EXTRA_LINKS =
else
SHARED_REAL = librl_lib.so.$(VERSION)
SHARED_LINK = librl_lib.so
SHARED_SONAME = librl_lib.so.$(VERSION_MAJOR)
SHARED_LDFLAGS = -shared -Wl,-soname,$(SHARED_SONAME)
SHARED_EXTRA_LINKS = $(SHARED_SONAME)
endif

SHARED_DIR = build/shared
SHARED_OBJS = $(patsubst %.c,$(SHARED_DIR)/%.o,$(LIB_SRCS))
SHARED_CFLAGS = $(WARN) -Iinclude -I. -O2 -fPIC $(EXTRA_CFLAGS)

# ── Build modes ─────────────────────────────────────────────────────

all: release

release: CFLAGS += -O3
release: clean $(TARGET_SNAKE) $(TARGET_CARTPOLE) $(TARGET_A2C) $(TARGET_PPO) $(TARGET_PPO_CONT) $(TARGET_PPO_SNAKE) $(TARGET_DQN) $(TARGET_HELLO)

debug: CFLAGS += -g -fsanitize=undefined
debug: clean $(TARGET_SNAKE) $(TARGET_CARTPOLE) $(TARGET_A2C) $(TARGET_PPO) $(TARGET_PPO_CONT) $(TARGET_PPO_SNAKE) $(TARGET_DQN) $(TARGET_HELLO)

# Compile every example and the benchmark with the current flags. CI runs this
# with ERR=1 so example code is held to the same warning standard as the
# library; the test binaries cannot do it, because they build the library
# objects with -fsanitize=undefined.
examples: $(TARGET_SNAKE) $(TARGET_CARTPOLE) $(TARGET_A2C) $(TARGET_PPO) $(TARGET_PPO_CONT) $(TARGET_PPO_SNAKE) $(TARGET_DQN) $(TARGET_HELLO)

# AddressSanitizer + UndefinedBehaviorSanitizer build of everything.
asan: CFLAGS += -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer
asan: clean $(TARGET_SNAKE) $(TARGET_CARTPOLE) $(TARGET_A2C) $(TARGET_PPO) $(TARGET_PPO_CONT) $(TARGET_PPO_SNAKE) $(TARGET_DQN) test_mat test_gradcheck test_envs test_rl test_checkpoint test_dqn
	@echo ""
	@echo "=== ASAN/UBSAN: running all tests ==="
	./test_mat
	./test_gradcheck
	./test_envs
	./test_rl
	./test_checkpoint
	./test_dqn

# ThreadSanitizer. Only the suites that exercise the threaded VecEnv pool are
# run: test_gradcheck and the training examples are single-threaded by
# construction, and TSan's slowdown would buy nothing there.
# TSan is the only sanitizer the suites' UBSan cannot be combined with: the
# library objects are shared by everything in this tree, so the examples that
# this target builds would otherwise link UBSan-instrumented objects without the
# UBSan runtime (undefined reference to __ubsan_handle_*). Emptying UBSAN_FLAGS
# gives the whole target a consistent TSan-only configuration.
tsan: UBSAN_FLAGS =
tsan: CFLAGS += -g -O1 -fsanitize=thread -fno-omit-frame-pointer
tsan: clean test_envs test_rl $(TARGET_PPO_CONT)
	@echo ""
	@echo "=== TSAN: threaded VecEnv suites ==="
	./test_envs
	./test_rl

$(TARGET_SNAKE): $(LIB_OBJS) examples/train_snake.o
	$(CC) $(CFLAGS) -o $@ $^ -lm $(BLAS_LIBS)

$(TARGET_CARTPOLE): $(LIB_OBJS) examples/train_cartpole.o
	$(CC) $(CFLAGS) -o $@ $^ -lm $(BLAS_LIBS)

$(TARGET_A2C): $(LIB_OBJS) examples/train_a2c.o
	$(CC) $(CFLAGS) -o $@ $^ -lm $(BLAS_LIBS)

$(TARGET_PPO): $(LIB_OBJS) examples/train_ppo.o
	$(CC) $(CFLAGS) -o $@ $^ -lm $(BLAS_LIBS)

$(TARGET_PPO_CONT): $(LIB_OBJS) examples/train_ppo_continuous.o
	$(CC) $(CFLAGS) -o $@ $^ -lm $(BLAS_LIBS)

$(TARGET_PPO_SNAKE): $(LIB_OBJS) examples/train_ppo_snake.o
	$(CC) $(CFLAGS) -o $@ $^ -lm $(BLAS_LIBS)

$(TARGET_DQN): $(LIB_OBJS) examples/train_dqn.o
	$(CC) $(CFLAGS) -o $@ $^ -lm $(BLAS_LIBS)

$(TARGET_HELLO): $(LIB_OBJS) examples/hello_rl.o
	$(CC) $(CFLAGS) -o $@ $^ -lm $(BLAS_LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ── Library artifacts ───────────────────────────────────────────────

$(LIB_STATIC): $(LIB_OBJS)
	rm -f $@
	ar rcs $@ $(LIB_OBJS)

$(SHARED_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(SHARED_CFLAGS) -c $< -o $@

$(SHARED_REAL): $(SHARED_OBJS)
	$(CC) $(SHARED_LDFLAGS) -o $@ $^ -lm $(BLAS_LIBS)

$(SHARED_LINK): $(SHARED_REAL)
	ln -sf $(SHARED_REAL) $(SHARED_LINK)
	@for l in $(SHARED_EXTRA_LINKS); do ln -sf $(SHARED_REAL) $$l; done

lib: $(LIB_STATIC) $(SHARED_LINK)

rl_lib.pc: rl_lib.pc.in include/version.h
	sed -e 's|@PREFIX@|$(PREFIX)|g' -e 's|@LIBDIR@|$(LIBDIR)|g' \
	    -e 's|@INCLUDEDIR@|$(INCLUDEDIR)|g' -e 's|@VERSION@|$(VERSION)|g' \
	    -e 's|@LIBS_EXTRA@|$(PC_LIBS_EXTRA)|g' \
	    $< > $@

install: lib rl_lib.pc
	install -d $(DESTDIR)$(INCLUDEDIR)/rl_lib/envs
	install -d $(DESTDIR)$(LIBDIR) $(DESTDIR)$(PKGCONFIGDIR)
	install -m 644 include/*.h $(DESTDIR)$(INCLUDEDIR)/rl_lib/
	install -m 644 envs/*.h $(DESTDIR)$(INCLUDEDIR)/rl_lib/envs/
	install -m 644 $(LIB_STATIC) $(DESTDIR)$(LIBDIR)/
	install -m 755 $(SHARED_REAL) $(DESTDIR)$(LIBDIR)/
	ln -sf $(SHARED_REAL) $(DESTDIR)$(LIBDIR)/$(SHARED_LINK)
	@for l in $(SHARED_EXTRA_LINKS); do \
	  (cd $(DESTDIR)$(LIBDIR) && ln -sf $(SHARED_REAL) $$l); done
	install -m 644 rl_lib.pc $(DESTDIR)$(PKGCONFIGDIR)/
	@echo "installed RL_lib-in-C $(VERSION) into $(DESTDIR)$(PREFIX)"

uninstall:
	rm -rf $(DESTDIR)$(INCLUDEDIR)/rl_lib
	rm -f $(DESTDIR)$(LIBDIR)/$(LIB_STATIC) \
	      $(DESTDIR)$(LIBDIR)/$(SHARED_REAL) \
	      $(DESTDIR)$(LIBDIR)/$(SHARED_LINK) \
	      $(DESTDIR)$(PKGCONFIGDIR)/rl_lib.pc
	@for l in $(SHARED_EXTRA_LINKS); do rm -f $(DESTDIR)$(LIBDIR)/$$l; done

# ── Tests ───────────────────────────────────────────────────────────

TEST_MAT_BIN = test_mat
TEST_GRAD_BIN = test_gradcheck
TEST_ENVS_BIN = test_envs
TEST_RL_BIN = test_rl
TEST_CKPT_BIN = test_checkpoint
TEST_DQN_BIN = test_dqn
TEST_BINS = $(TEST_MAT_BIN) $(TEST_GRAD_BIN) $(TEST_ENVS_BIN) $(TEST_RL_BIN) $(TEST_CKPT_BIN) $(TEST_DQN_BIN)

# The suites are also run under UndefinedBehaviorSanitizer. This is a variable
# because target-specific CFLAGS reach the shared library objects as well, so a
# build mode that cannot link the UBSan runtime (the TSan target, which links
# $(TARGET_PPO_CONT) against the same objects) has to stand it down - otherwise
# the objects are instrumented but the link line has no -fsanitize=undefined.
UBSAN_FLAGS ?= -g -fsanitize=undefined

$(TEST_MAT_BIN): CFLAGS += $(UBSAN_FLAGS)
$(TEST_MAT_BIN): $(LIB_OBJS) tests/test_mat.c
	$(CC) $(CFLAGS) -o $@ $(LIB_OBJS) tests/test_mat.c -lm $(BLAS_LIBS)

$(TEST_GRAD_BIN): CFLAGS += $(UBSAN_FLAGS)
$(TEST_GRAD_BIN): $(LIB_OBJS) tests/test_gradcheck.c
	$(CC) $(CFLAGS) -o $@ $(LIB_OBJS) tests/test_gradcheck.c -lm $(BLAS_LIBS)

$(TEST_ENVS_BIN): CFLAGS += $(UBSAN_FLAGS)
$(TEST_ENVS_BIN): $(LIB_OBJS) tests/test_envs.c
	$(CC) $(CFLAGS) -o $@ $(LIB_OBJS) tests/test_envs.c -lm $(BLAS_LIBS)

$(TEST_RL_BIN): CFLAGS += $(UBSAN_FLAGS)
$(TEST_RL_BIN): $(LIB_OBJS) tests/test_rl.c
	$(CC) $(CFLAGS) -o $@ $(LIB_OBJS) tests/test_rl.c -lm $(BLAS_LIBS)

$(TEST_CKPT_BIN): CFLAGS += $(UBSAN_FLAGS)
$(TEST_CKPT_BIN): $(LIB_OBJS) tests/test_checkpoint.c
	$(CC) $(CFLAGS) -o $@ $(LIB_OBJS) tests/test_checkpoint.c -lm $(BLAS_LIBS)

$(TEST_DQN_BIN): CFLAGS += $(UBSAN_FLAGS)
$(TEST_DQN_BIN): $(LIB_OBJS) tests/test_dqn.c
	$(CC) $(CFLAGS) -o $@ $(LIB_OBJS) tests/test_dqn.c -lm $(BLAS_LIBS)

# ── Double-precision builds ─────────────────────────────────────────
# Builds the whole library in f64 (RL_LIB_USE_DOUBLE) into its own object tree so
# the default f32 objects are not clobbered. test_gradcheck uses this for a tight
# finite-difference tolerance; `make test-double` runs every other suite in f64
# as well, so the f64 path is not just exercised by the gradient check.
DOUBLE_DIR = build/double
DOUBLE_CFLAGS = $(WARN) -Iinclude -I. -g -O2 -DRL_LIB_USE_DOUBLE -fsanitize=undefined $(EXTRA_CFLAGS)
DOUBLE_LIB_OBJS = $(patsubst %.c,$(DOUBLE_DIR)/%.o,$(LIB_SRCS))
DOUBLE_OBJS = $(DOUBLE_LIB_OBJS) $(DOUBLE_DIR)/tests/test_gradcheck.o
TEST_GRAD_DOUBLE_BIN = test_gradcheck_double

$(DOUBLE_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(DOUBLE_CFLAGS) -c $< -o $@

$(TEST_GRAD_DOUBLE_BIN): $(DOUBLE_OBJS)
	$(CC) $(DOUBLE_CFLAGS) -o $@ $^ -lm $(BLAS_LIBS)

DOUBLE_TEST_BINS = $(addprefix $(DOUBLE_DIR)/,$(filter-out $(TEST_GRAD_BIN),$(TEST_BINS)))

$(DOUBLE_TEST_BINS): $(DOUBLE_DIR)/%: tests/%.c $(DOUBLE_LIB_OBJS)
	$(CC) $(DOUBLE_CFLAGS) -o $@ $(DOUBLE_LIB_OBJS) $< -lm $(BLAS_LIBS)

gradcheck-double: $(TEST_GRAD_DOUBLE_BIN)
	@echo ""
	@echo "=== Running test_gradcheck (double precision) ==="
	./$(TEST_GRAD_DOUBLE_BIN)

test-double: $(DOUBLE_TEST_BINS)
	@echo ""
	@echo "=== Running every suite in double precision ==="
	@for t in $(DOUBLE_TEST_BINS); do \
	  echo "--- $$t"; ./$$t || exit 1; done
	@echo "=== All f64 suites complete ==="

test: $(TEST_BINS) gradcheck-double
	@echo ""
	@echo "=== Running test_mat ==="
	./$(TEST_MAT_BIN)
	@echo "=== Running test_gradcheck ==="
	./$(TEST_GRAD_BIN)
	@echo "=== Running test_envs ==="
	./$(TEST_ENVS_BIN)
	@echo "=== Running test_rl ==="
	./$(TEST_RL_BIN)
	@echo "=== Running test_checkpoint ==="
	./$(TEST_CKPT_BIN)
	@echo "=== Running test_dqn ==="
	./$(TEST_DQN_BIN)
	@echo "=== All tests complete ==="

snake: release
	./$(TARGET_SNAKE)

cartpole: release
	./$(TARGET_CARTPOLE)

dqn: release
	./$(TARGET_DQN)

a2c: release
	./$(TARGET_A2C)

ppo: release
	./$(TARGET_PPO)

ppo-continuous: release
	./$(TARGET_PPO_CONT)

ppo-snake: release
	./$(TARGET_PPO_SNAKE)

hello: $(TARGET_HELLO)
	./$(TARGET_HELLO)

# ── Benchmark ───────────────────────────────────────────────────────

BENCH_BIN = rl_lib_bench

$(BENCH_BIN): CFLAGS += -O3
$(BENCH_BIN): $(LIB_OBJS) examples/bench.o
	$(CC) $(CFLAGS) -o $@ $^ -lm $(BLAS_LIBS)

bench: $(BENCH_BIN)
	./$(BENCH_BIN)

bench-cartpole: $(BENCH_BIN)
	./$(BENCH_BIN) cartpole

bench-snake: $(BENCH_BIN)
	./$(BENCH_BIN) snake

bench-throughput: $(BENCH_BIN)
	./$(BENCH_BIN) throughput

# The f64 kernels are vectorised too (2-wide NEON, 4-wide AVX2), so the matmul
# comparison is reproducible in double precision as well. This gets its own
# object tree and stays plain -O3: a benchmark must not inherit the sanitizer
# flags the f64 test suites build with.
BENCH_DOUBLE_DIR = build/bench-double
BENCH_DOUBLE_BIN = $(BENCH_DOUBLE_DIR)/rl_lib_bench
BENCH_DOUBLE_CFLAGS = $(WARN) -Iinclude -I. -O3 -DRL_LIB_USE_DOUBLE $(THREAD_FLAGS) $(EXTRA_CFLAGS)
BENCH_DOUBLE_OBJS = $(patsubst %.c,$(BENCH_DOUBLE_DIR)/%.o,$(LIB_SRCS)) $(BENCH_DOUBLE_DIR)/examples/bench.o

$(BENCH_DOUBLE_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(BENCH_DOUBLE_CFLAGS) -c $< -o $@

$(BENCH_DOUBLE_BIN): $(BENCH_DOUBLE_OBJS)
	$(CC) $(BENCH_DOUBLE_CFLAGS) -o $@ $^ -lm $(BLAS_LIBS)

bench-double: $(BENCH_DOUBLE_BIN)
	./$(BENCH_DOUBLE_BIN) throughput

# ── Cleanup ─────────────────────────────────────────────────────────

clean:
	rm -f $(LIB_OBJS) examples/train_snake.o examples/train_cartpole.o examples/train_a2c.o examples/train_ppo.o examples/train_ppo_continuous.o examples/train_ppo_snake.o examples/train_dqn.o examples/hello_rl.o examples/bench.o
	rm -f $(TARGET_SNAKE) $(TARGET_CARTPOLE) $(TARGET_A2C) $(TARGET_PPO) $(TARGET_PPO_CONT) $(TARGET_PPO_SNAKE) $(TARGET_DQN) $(TARGET_HELLO) $(TEST_BINS) $(BENCH_BIN)
	rm -f $(TEST_GRAD_DOUBLE_BIN) $(LIB_STATIC) $(SHARED_REAL) $(SHARED_LINK) $(SHARED_EXTRA_LINKS) rl_lib.pc
	rm -rf build

.PHONY: all release debug examples asan test test-double gradcheck-double clean lib install uninstall \
        snake cartpole a2c ppo ppo-continuous dqn hello bench bench-cartpole bench-snake bench-throughput \
        bench-double
