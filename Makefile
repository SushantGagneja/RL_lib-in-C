CC = gcc
CFLAGS = -Wall -Wextra -Wshadow

SRCS = arena.c prng.c mat.c autograd.c model.c env.c
OBJS = $(SRCS:.c=.o)
TARGET = cRL

all: release

release: CFLAGS += -O3
release: clean $(TARGET)

debug: CFLAGS += -g -fsanitize=address,undefined
debug: clean $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
