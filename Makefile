CC = gcc
CFLAGS = -Wall -Wextra -Wshadow -Iinclude -I.

LIB_SRCS = src/arena.c src/prng.c src/mat.c src/autograd.c src/nn.c src/optim.c src/rl.c envs/snake.c envs/cartpole.c
LIB_OBJS = $(LIB_SRCS:.c=.o)

TARGET_SNAKE = train_snake
TARGET_CARTPOLE = train_cartpole

# ── Build modes ─────────────────────────────────────────────────────

all: release

release: CFLAGS += -O3
release: clean $(TARGET_SNAKE) $(TARGET_CARTPOLE)

debug: CFLAGS += -g -fsanitize=undefined
debug: clean $(TARGET_SNAKE) $(TARGET_CARTPOLE)

$(TARGET_SNAKE): $(LIB_OBJS) examples/train_snake.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(TARGET_CARTPOLE): $(LIB_OBJS) examples/train_cartpole.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ── Tests ───────────────────────────────────────────────────────────

TEST_MAT_BIN = test_mat
TEST_GRAD_BIN = test_gradcheck
TEST_ENVS_BIN = test_envs

$(TEST_MAT_BIN): CFLAGS += -g -fsanitize=undefined
$(TEST_MAT_BIN): $(LIB_OBJS) tests/test_mat.c
	$(CC) $(CFLAGS) -o $@ $(LIB_OBJS) tests/test_mat.c -lm

$(TEST_GRAD_BIN): CFLAGS += -g -fsanitize=undefined
$(TEST_GRAD_BIN): $(LIB_OBJS) tests/test_gradcheck.c
	$(CC) $(CFLAGS) -o $@ $(LIB_OBJS) tests/test_gradcheck.c -lm

$(TEST_ENVS_BIN): CFLAGS += -g -fsanitize=undefined
$(TEST_ENVS_BIN): $(LIB_OBJS) tests/test_envs.c
	$(CC) $(CFLAGS) -o $@ $(LIB_OBJS) tests/test_envs.c -lm

test: $(TEST_MAT_BIN) $(TEST_GRAD_BIN) $(TEST_ENVS_BIN)
	@echo ""
	@echo "=== Running test_mat ==="
	./$(TEST_MAT_BIN)
	@echo "=== Running test_gradcheck ==="
	./$(TEST_GRAD_BIN)
	@echo "=== Running test_envs ==="
	./$(TEST_ENVS_BIN)
	@echo "=== All tests complete ==="

snake: release
	./$(TARGET_SNAKE)

cartpole: release
	./$(TARGET_CARTPOLE)

# ── Benchmark ───────────────────────────────────────────────────────

BENCH_BIN = crl_bench

$(BENCH_BIN): CFLAGS += -O3
$(BENCH_BIN): $(LIB_OBJS) examples/bench.c
	$(CC) $(CFLAGS) -o $@ $(LIB_OBJS) examples/bench.c -lm

bench: $(BENCH_BIN)
	./$(BENCH_BIN)

bench-cartpole: $(BENCH_BIN)
	./$(BENCH_BIN) cartpole

bench-snake: $(BENCH_BIN)
	./$(BENCH_BIN) snake

# ── Cleanup ─────────────────────────────────────────────────────────

clean:
	rm -f $(LIB_OBJS) examples/train_snake.o examples/train_cartpole.o examples/bench.o $(TARGET_SNAKE) $(TARGET_CARTPOLE) $(TEST_MAT_BIN) $(TEST_GRAD_BIN) $(TEST_ENVS_BIN) $(BENCH_BIN) cRL

.PHONY: all release debug test clean snake cartpole bench bench-cartpole bench-snake
