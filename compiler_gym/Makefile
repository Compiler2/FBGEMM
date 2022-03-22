.PHONY: clean all 

GEMM_M ?= 64
GEMM_N ?= 800
GEMM_K ?= 320

CC := clang++
CUR_DIR := $(shell pwd)
IFLAGS := -I$(CUR_DIR)/../include
LFLAGS := -L$(CUR_DIR)/../build -lfbgemm -L$(CUR_DIR)/../build/cpuinfo -lcpuinfo -L$(CUR_DIR)/../build/cpuinfo/deps/clog -lclog -L$(CUR_DIR)/../build/asmjit -lasmjit -lpthread -lrt
CFLAGS := -m64 -mavx2 -mfma -masm=intel -g

BINS := gemm

all: bins

bins: $(BINS)

$(BINS): % : %.cc
		$(CC) $^ $(IFLAGS) $(LFLAGS) $(CFLAGS) -DGEMM_M=$(GEMM_M) -DGEMM_N=$(GEMM_N) -DGEMM_K=$(GEMM_K) -o $@

clean:
	rm -rf $(BINS)
