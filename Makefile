CC ?= cc
# Strict warning set. Pass ERR=1 (e.g. `make ERR=1 test`) to turn them into
# errors; CI uses that, local builds do not so other compilers stay buildable.
WARN = -Wall -Wextra -Wshadow -Wconversion -Wmissing-prototypes
ifdef ERR
WARN += -Werror
endif
CFLAGS = $(WARN) -Iinclude -I.

LIB_SRCS = src/memory.c src/random.c src/linalg.c src/graph.c src/layers.c src/optim.c src/rl.c envs/snake.c envs/cartpole.c
LIB_OBJS = $(LIB_SRCS:.c=.o)

TARGET_SNAKE = train_snake
TARGET_CARTPOLE = train_cartpole
TARGET_A2C = train_a2c

# ── Build modes ─────────────────────────────────────────────────────

all: release

release: CFLAGS += -O3
release: clean $(TARGET_SNAKE) $(TARGET_CARTPOLE) $(TARGET_A2C)

debug: CFLAGS += -g -fsanitize=undefined
debug: clean $(TARGET_SNAKE) $(TARGET_CARTPOLE) $(TARGET_A2C)

# AddressSanitizer + UndefinedBehaviorSanitizer build of everything.
asan: CFLAGS += -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer
asan: clean $(TARGET_SNAKE) $(TARGET_CARTPOLE) $(TARGET_A2C) test_mat test_gradcheck test_envs test_rl
	@echo ""
	@echo "=== ASAN/UBSAN: running all tests ==="
	./test_mat
	./test_gradcheck
	./test_envs
	./test_rl

$(TARGET_SNAKE): $(LIB_OBJS) examples/train_snake.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(TARGET_CARTPOLE): $(LIB_OBJS) examples/train_cartpole.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(TARGET_A2C): $(LIB_OBJS) examples/train_a2c.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ── Tests ───────────────────────────────────────────────────────────

TEST_MAT_BIN = test_mat
TEST_GRAD_BIN = test_gradcheck
TEST_ENVS_BIN = test_envs
TEST_RL_BIN = test_rl
TEST_BINS = $(TEST_MAT_BIN) $(TEST_GRAD_BIN) $(TEST_ENVS_BIN) $(TEST_RL_BIN)

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

# ── Double-precision gradient check ─────────────────────────────────
# Rebuilds the library in f64 (CRL_USE_DOUBLE) into its own object tree so the
# default f32 objects are not clobbered. Enables h=1e-5 and a 1e-6 tolerance.
DOUBLE_DIR = build/double
DOUBLE_CFLAGS = $(WARN) -Iinclude -I. -g -O2 -DCRL_USE_DOUBLE -fsanitize=undefined
DOUBLE_OBJS = $(patsubst %.c,$(DOUBLE_DIR)/%.o,$(LIB_SRCS)) \
              $(DOUBLE_DIR)/tests/test_gradcheck.o
TEST_GRAD_DOUBLE_BIN = test_gradcheck_double

$(DOUBLE_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(DOUBLE_CFLAGS) -c $< -o $@

$(TEST_GRAD_DOUBLE_BIN): $(DOUBLE_OBJS)
	$(CC) $(DOUBLE_CFLAGS) -o $@ $^ -lm

gradcheck-double: $(TEST_GRAD_DOUBLE_BIN)
	@echo ""
	@echo "=== Running test_gradcheck (double precision) ==="
	./$(TEST_GRAD_DOUBLE_BIN)

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
	@echo "=== All tests complete ==="

snake: release
	./$(TARGET_SNAKE)

cartpole: release
	./$(TARGET_CARTPOLE)

a2c: release
	./$(TARGET_A2C)

# ── Benchmark ───────────────────────────────────────────────────────

BENCH_BIN = crl_bench

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
	rm -f $(LIB_OBJS) examples/train_snake.o examples/train_cartpole.o examples/train_a2c.o examples/bench.o
	rm -f $(TARGET_SNAKE) $(TARGET_CARTPOLE) $(TARGET_A2C) $(TEST_BINS) $(BENCH_BIN)
	rm -f $(TEST_GRAD_DOUBLE_BIN)
	rm -rf build

.PHONY: all release debug asan test gradcheck-double clean snake cartpole a2c bench bench-cartpole bench-snake bench-throughput
