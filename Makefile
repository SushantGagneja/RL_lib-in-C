CC ?= cc
# Strict warning set. Pass ERR=1 (e.g. `make ERR=1 test`) to turn them into
# errors; CI uses that, local builds do not so other compilers stay buildable.
WARN = -Wall -Wextra -Wshadow -Wconversion -Wmissing-prototypes
ifdef ERR
WARN += -Werror
endif
# Extra flags for ad-hoc configurations, e.g.
#   make EXTRA_CFLAGS=-DRL_SIMD_BACKEND=0 test
EXTRA_CFLAGS ?=
CFLAGS = $(WARN) -Iinclude -I. $(EXTRA_CFLAGS)

LIB_SRCS = src/memory.c src/random.c src/simd.c src/linalg.c src/graph.c src/layers.c src/optim.c src/checkpoint.c src/rl.c src/vecenv.c envs/snake.c envs/cartpole.c envs/pendulum.c
LIB_OBJS = $(LIB_SRCS:.c=.o)

TARGET_SNAKE = train_snake
TARGET_CARTPOLE = train_cartpole
TARGET_A2C = train_a2c
TARGET_PPO = train_ppo
TARGET_PPO_CONT = train_ppo_continuous
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
release: clean $(TARGET_SNAKE) $(TARGET_CARTPOLE) $(TARGET_A2C) $(TARGET_PPO) $(TARGET_PPO_CONT) $(TARGET_HELLO)

debug: CFLAGS += -g -fsanitize=undefined
debug: clean $(TARGET_SNAKE) $(TARGET_CARTPOLE) $(TARGET_A2C) $(TARGET_PPO) $(TARGET_PPO_CONT) $(TARGET_HELLO)

# AddressSanitizer + UndefinedBehaviorSanitizer build of everything.
asan: CFLAGS += -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer
asan: clean $(TARGET_SNAKE) $(TARGET_CARTPOLE) $(TARGET_A2C) $(TARGET_PPO) $(TARGET_PPO_CONT) test_mat test_gradcheck test_envs test_rl test_checkpoint
	@echo ""
	@echo "=== ASAN/UBSAN: running all tests ==="
	./test_mat
	./test_gradcheck
	./test_envs
	./test_rl
	./test_checkpoint

$(TARGET_SNAKE): $(LIB_OBJS) examples/train_snake.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(TARGET_CARTPOLE): $(LIB_OBJS) examples/train_cartpole.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(TARGET_A2C): $(LIB_OBJS) examples/train_a2c.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(TARGET_PPO): $(LIB_OBJS) examples/train_ppo.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(TARGET_PPO_CONT): $(LIB_OBJS) examples/train_ppo_continuous.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(TARGET_HELLO): $(LIB_OBJS) examples/hello_rl.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

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
	$(CC) $(SHARED_LDFLAGS) -o $@ $^ -lm

$(SHARED_LINK): $(SHARED_REAL)
	ln -sf $(SHARED_REAL) $(SHARED_LINK)
	@for l in $(SHARED_EXTRA_LINKS); do ln -sf $(SHARED_REAL) $$l; done

lib: $(LIB_STATIC) $(SHARED_LINK)

rl_lib.pc: rl_lib.pc.in include/version.h
	sed -e 's|@PREFIX@|$(PREFIX)|g' -e 's|@LIBDIR@|$(LIBDIR)|g' \
	    -e 's|@INCLUDEDIR@|$(INCLUDEDIR)|g' -e 's|@VERSION@|$(VERSION)|g' \
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
TEST_BINS = $(TEST_MAT_BIN) $(TEST_GRAD_BIN) $(TEST_ENVS_BIN) $(TEST_RL_BIN) $(TEST_CKPT_BIN)

$(TEST_MAT_BIN): CFLAGS += -g -fsanitize=undefined
$(TEST_MAT_BIN): $(LIB_OBJS) tests/test_mat.c
	$(CC) $(CFLAGS) -o $@ $(LIB_OBJS) tests/test_mat.c -lm

$(TEST_GRAD_BIN): CFLAGS += -g -fsanitize=undefined
$(TEST_GRAD_BIN): $(LIB_OBJS) tests/test_gradcheck.c
	$(CC) $(CFLAGS) -o $@ $(LIB_OBJS) tests/test_gradcheck.c -lm

$(TEST_ENVS_BIN): CFLAGS += -g -fsanitize=undefined
$(TEST_ENVS_BIN): $(LIB_OBJS) tests/test_envs.c
	$(CC) $(CFLAGS) -o $@ $(LIB_OBJS) tests/test_envs.c -lm

$(TEST_RL_BIN): CFLAGS += -g -fsanitize=undefined
$(TEST_RL_BIN): $(LIB_OBJS) tests/test_rl.c
	$(CC) $(CFLAGS) -o $@ $(LIB_OBJS) tests/test_rl.c -lm

$(TEST_CKPT_BIN): CFLAGS += -g -fsanitize=undefined
$(TEST_CKPT_BIN): $(LIB_OBJS) tests/test_checkpoint.c
	$(CC) $(CFLAGS) -o $@ $(LIB_OBJS) tests/test_checkpoint.c -lm

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
	$(CC) $(DOUBLE_CFLAGS) -o $@ $^ -lm

DOUBLE_TEST_BINS = $(addprefix $(DOUBLE_DIR)/,$(filter-out $(TEST_GRAD_BIN),$(TEST_BINS)))

$(DOUBLE_TEST_BINS): $(DOUBLE_DIR)/%: tests/%.c $(DOUBLE_LIB_OBJS)
	$(CC) $(DOUBLE_CFLAGS) -o $@ $(DOUBLE_LIB_OBJS) $< -lm

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
	@echo "=== All tests complete ==="

snake: release
	./$(TARGET_SNAKE)

cartpole: release
	./$(TARGET_CARTPOLE)

a2c: release
	./$(TARGET_A2C)

ppo: release
	./$(TARGET_PPO)

ppo-continuous: release
	./$(TARGET_PPO_CONT)

hello: $(TARGET_HELLO)
	./$(TARGET_HELLO)

# ── Benchmark ───────────────────────────────────────────────────────

BENCH_BIN = rl_lib_bench

$(BENCH_BIN): CFLAGS += -O3
$(BENCH_BIN): $(LIB_OBJS) examples/bench.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

bench: $(BENCH_BIN)
	./$(BENCH_BIN)

bench-cartpole: $(BENCH_BIN)
	./$(BENCH_BIN) cartpole

bench-snake: $(BENCH_BIN)
	./$(BENCH_BIN) snake

bench-throughput: $(BENCH_BIN)
	./$(BENCH_BIN) throughput

# ── Cleanup ─────────────────────────────────────────────────────────

clean:
	rm -f $(LIB_OBJS) examples/train_snake.o examples/train_cartpole.o examples/train_a2c.o examples/train_ppo.o examples/train_ppo_continuous.o examples/hello_rl.o examples/bench.o
	rm -f $(TARGET_SNAKE) $(TARGET_CARTPOLE) $(TARGET_A2C) $(TARGET_PPO) $(TARGET_PPO_CONT) $(TARGET_HELLO) $(TEST_BINS) $(BENCH_BIN)
	rm -f $(TEST_GRAD_DOUBLE_BIN) $(LIB_STATIC) $(SHARED_REAL) $(SHARED_LINK) $(SHARED_EXTRA_LINKS) rl_lib.pc
	rm -rf build

.PHONY: all release debug asan test test-double gradcheck-double clean lib install uninstall \
        snake cartpole a2c ppo ppo-continuous hello bench bench-cartpole bench-snake bench-throughput
