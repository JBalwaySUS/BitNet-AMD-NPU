# Makefile for BitNet AIE kernels
#
# This makefile compiles the int8 x int2 BitLinear kernel for AMD AIE
#
# Usage:
#   make              - Build default kernel (32x32 tiles)
#   make DIM_M=64     - Build with custom M dimension
#   make clean        - Remove build artifacts

# Default tile dimensions
DIM_M ?= 32
DIM_K ?= 32
NUM_WEIGHT_GROUPS ?= 4

# AIE compiler and tools
# These should be set by the MLIR-AIE environment
AIECC ?= aiecc.py
AIE_CLANG ?= $(VITIS_AIETOOLS_DIR)/bin/aie2-clang

# Compiler flags
AIE_FLAGS = -O2 -DDIM_M=$(DIM_M) -DDIM_K=$(DIM_K) -DNUM_WEIGHT_GROUPS=$(NUM_WEIGHT_GROUPS)

# Include paths for AIE API
AIE_INCLUDES = -I$(VITIS_AIETOOLS_DIR)/include

# Source files
KERNEL_SRC = bitlinear_int8xint2.cc

# Output object file
KERNEL_OBJ = bitlinear_$(DIM_M)x$(DIM_K).o

# Default target
all: $(KERNEL_OBJ)

# Compile kernel to object file
$(KERNEL_OBJ): $(KERNEL_SRC)
	$(AIE_CLANG) $(AIE_FLAGS) $(AIE_INCLUDES) -c $< -o $@

# Alternative: use xchesscc for compilation
kernel_chess: $(KERNEL_SRC)
	xchesscc -p me -P $(VITIS_AIETOOLS_DIR)/data/aie_ml/lib \
		-c $(AIE_FLAGS) $(KERNEL_SRC) -o $(KERNEL_OBJ)

# Build multiple tile sizes
tile_sizes: 
	$(MAKE) DIM_M=32 DIM_K=32
	$(MAKE) DIM_M=32 DIM_K=64
	$(MAKE) DIM_M=64 DIM_K=32
	$(MAKE) DIM_M=64 DIM_K=64

# Clean build artifacts
clean:
	rm -f *.o *.elf *.map

# Print configuration
info:
	@echo "DIM_M = $(DIM_M)"
	@echo "DIM_K = $(DIM_K)"
	@echo "NUM_WEIGHT_GROUPS = $(NUM_WEIGHT_GROUPS)"
	@echo "Output: $(KERNEL_OBJ)"

.PHONY: all clean tile_sizes info kernel_chess

