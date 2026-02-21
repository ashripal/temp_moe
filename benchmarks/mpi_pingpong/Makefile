MPICC ?= mpicc
CFLAGS ?= -O3 -march=native

all: mpi_pingpong

mpi_pingpong: main.c
	$(MPICC) $(CFLAGS) -o $@ main.c

clean:
	rm -f mpi_pingpong