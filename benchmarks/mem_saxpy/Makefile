CC ?= cc
CFLAGS ?= -O3 -march=native

all: mem_saxpy

mem_saxpy: main.c
	$(CC) $(CFLAGS) -o $@ main.c

clean:
	rm -f mem_saxpy