# CC ?= cc
# CFLAGS ?= -O3 -march=native
CC       := /opt/homebrew/opt/llvm/bin/clang
SDKROOT := $(shell xcrun --show-sdk-path 2>/dev/null)
CFLAGS += -isysroot $(SDKROOT)
LDFLAGS  := -L/opt/homebrew/opt/llvm/lib -lomp

# Try common OpenMP flags. On macOS you often need libomp installed.
# If OpenMP not available, build will fail and pytest will skip this benchmark.
OMPFLAGS_CLANG = -Xpreprocessor -fopenmp
OMPLIB_CLANG = -lomp
OMPFLAGS_GCC = -fopenmp
OMPLIB_GCC =

all: omp_imbalance

omp_imbalance: main.c
	@echo "Building OpenMP benchmark..."
	@$(CC) $(CFLAGS) $(OMPFLAGS_GCC) -o $@ main.c $(OMPLIB_GCC) 2>/dev/null || \
	$(CC) $(CFLAGS) $(OMPFLAGS_CLANG) -o $@ main.c $(OMPLIB_CLANG)

clean:
	rm -f omp_imbalance