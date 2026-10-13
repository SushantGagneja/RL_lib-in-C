CC = gcc
CFLAGS = -Wall -Wextra -Wshadow

LIB_SRCS = arena.c prng.c mat.c autograd.c model.c
LIB_OBJS = $(LIB_SRCS:.c=.o)

APP_SRC = env.c
APP_OBJ = $(APP_SRC:.c=.o)

TARGET = cRL

# ── Build modes ─────────────────────────────────────────────────────

all: release

release: CFLAGS += -O3
release: clean $(TARGET)

debug: CFLAGS += -g -fsanitize=undefined
debug: clean $(TARGET)

$(TARGET): $(LIB_OBJS) $(APP_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ── Tests ───────────────────────────────────────────────────────────

TEST_MAT_SRC = tests/test_mat.c
TEST_GRAD_SRC = tests/test_gradcheck.c

test_mat: CFLAGS += -g -fsanitize=undefined
test_mat: $(LIB_OBJS) $(TEST_MAT_SRC)
	$(CC) $(CFLAGS) -I. -o $@ $(LIB_OBJS) $(TEST_MAT_SRC) -lm

test_gradcheck: CFLAGS += -g -fsanitize=undefined
test_gradcheck: $(LIB_OBJS) $(TEST_GRAD_SRC)
	$(CC) $(CFLAGS) -I. -o $@ $(LIB_OBJS) $(TEST_GRAD_SRC) -lm

test: test_mat test_gradcheck
	@echo ""
	@echo "=== Running test_mat ==="
	./test_mat
	@echo "=== Running test_gradcheck ==="
	./test_gradcheck
	@echo "=== All tests complete ==="

run: release
	./$(TARGET)

bench: release
	./$(TARGET) --benchmark 5 150

# ── Cleanup ─────────────────────────────────────────────────────────

clean:
	rm -f $(LIB_OBJS) $(APP_OBJ) $(TARGET) test_mat test_gradcheck

.PHONY: all release debug test clean run bench
