##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
# Copyright (C) 2025
# 
##===----------------------------------------------------------------------===##

# BitLinear kernel for int8 x int2 matrix-vector multiplication
# Based on mlir-aie/programming_examples/basic/matrix_multiplication/matrix_vector

# Default tile dimensions (must match design file)
m ?= 32
k ?= 32

# Number of weight groups for per-group scaling
num_weight_groups ?= 4

# Kernel output name
kernels = bitlinear_${m}x${k}

# Include common makefile settings
SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
include ${SELF_DIR}../makefile-common

# Kernel source directory
kernels_dir = ${SELF_DIR}

# Kernel defines
KERNEL_DEFINES = -DDIM_M=${m} -DDIM_K=${k} -DNUM_WEIGHT_GROUPS=${num_weight_groups}

# Default target
all: build/${kernels}.o

# Build directory
build:
	mkdir -p build

# Compile kernel using xchesscc_wrapper
build/${kernels}.o: ${kernels_dir}/bitlinear.cc | build
	cd build && ${KERNEL_CC} ${KERNEL_CFLAGS} ${KERNEL_DEFINES} -c ../$< -o ${@F}

# Compile all common tile sizes
tile_sizes: build
	$(MAKE) m=32 k=32
	$(MAKE) m=32 k=64
	$(MAKE) m=64 k=32
	$(MAKE) m=64 k=64

# Clean build artifacts
clean:
	rm -rf build

# Print configuration
info:
	@echo "BitLinear Kernel Configuration"
	@echo "=============================="
	@echo "m (tile rows)      = ${m}"
	@echo "k (tile cols)      = ${k}"
	@echo "num_weight_groups  = ${num_weight_groups}"
	@echo "Output             = build/${kernels}.o"
	@echo "KERNEL_CC          = ${KERNEL_CC}"
	@echo "KERNEL_CFLAGS      = ${KERNEL_CFLAGS}"

.PHONY: all clean tile_sizes info
