##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
# Copyright (C) 2025
#
# BitNet NPU Makefile
# Based on mlir-aie/programming_examples/basic/matrix_multiplication/matrix_vector
# 
##===----------------------------------------------------------------------===##

subdir = bitlinear

# Matrix dimensions (M=output, K=input)
M ?= 288
K ?= 288

# Tile dimensions (must divide M and K evenly)
m ?= 32
k ?= 32

# Target names include dimensions
targetname = bitlinear_${M}x${K}
kernels = bitlinear_${m}x${k}
use_iron ?= 1
num_weight_groups ?= 4

# Checkpoint directory for inference
CKPT_DIR ?= checkpoints

# Include common makefile settings
SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
include ${SELF_DIR}makefile-common

# Kernel source and build directories
kernels_dir = ${SELF_DIR}kernels
designs_dir = ${SELF_DIR}designs

# Kernel defines
KERNEL_DEFINES = -DDIM_M=${m} -DDIM_K=${k} -DNUM_WEIGHT_GROUPS=${num_weight_groups}

#===============================================================================
# Default target
#===============================================================================

.PHONY: all
all: ${BUILD_DIR}/${kernels}.o ${BUILD_DIR}/${targetname}.mlir

#===============================================================================
# Build directory
#===============================================================================

${BUILD_DIR}:
	mkdir -p ${BUILD_DIR}

#===============================================================================
# Kernel compilation
#===============================================================================

.PHONY: kernel kernel-all

${BUILD_DIR}/${kernels}.o: ${kernels_dir}/bitlinear.cc ${kernels_dir}/zero.cc | ${BUILD_DIR}
	cd ${BUILD_DIR} && ${KERNEL_CC} ${KERNEL_CFLAGS} ${KERNEL_DEFINES} -c ../$< -o ${@F}

kernel: ${BUILD_DIR}/${kernels}.o

kernel-all: ${BUILD_DIR}
	$(MAKE) m=32 k=32 kernel
	$(MAKE) m=32 k=64 kernel
	$(MAKE) m=64 k=32 kernel
	$(MAKE) m=64 k=64 kernel

#===============================================================================
# Design generation
#===============================================================================

.PHONY: design design-all

${BUILD_DIR}/${targetname}.mlir: ${designs_dir}/bitlinear_npu.py | ${BUILD_DIR}
	${PYTHON} $< --dev ${DEVICE} -M ${M} -K ${K} -m ${m} -k ${k} > $@

design: ${BUILD_DIR}/${targetname}.mlir

#===============================================================================
# XCLBIN compilation
#===============================================================================

.PHONY: compile

${BUILD_DIR}/${targetname}.xclbin: ${BUILD_DIR}/${targetname}.mlir ${BUILD_DIR}/${kernels}.o
	mkdir -p ${BUILD_DIR}
	cd ${BUILD_DIR} && aiecc.py --no-compile-host \
		--aie-generate-xclbin \
		--xclbin-name=${targetname}.xclbin \
		--aie-generate-npu-insts \
		--npu-insts-name=${targetname}.insts.txt \
		${targetname}.mlir

compile: ${BUILD_DIR}/${targetname}.xclbin

#===============================================================================
# Test targets
#===============================================================================

.PHONY: test test-cpu test-npu

test: test-cpu

test-cpu:
	${PYTHON} test_bitlinear.py --cpu-only -v

test-npu: compile
	${PYTHON} test_bitlinear.py \
		--xclbin ${BUILD_DIR}/${targetname}.xclbin \
		--instr ${BUILD_DIR}/${targetname}.insts.bin \
		-M ${M} -K ${K} -v

#===============================================================================
# Run targets
#===============================================================================

.PHONY: run run-interactive run-sample

run:
	${PYTHON} generate.py ${CKPT_DIR} --xclbin_dir ${BUILD_DIR}

run-interactive:
	${PYTHON} generate.py ${CKPT_DIR} --interactive --xclbin_dir ${BUILD_DIR}

run-sample:
	${PYTHON} generate.py ${CKPT_DIR} --sampling --xclbin_dir ${BUILD_DIR}

#===============================================================================
# Info targets
#===============================================================================

.PHONY: info layer-info

info:
	@echo "BitNet NPU Configuration"
	@echo "========================"
	@echo "M (output dim)     = ${M}"
	@echo "K (input dim)      = ${K}"
	@echo "m (tile rows)      = ${m}"
	@echo "k (tile cols)      = ${k}"
	@echo "num_weight_groups  = ${num_weight_groups}"
	@echo "DEVICE             = ${DEVICE}"
	@echo "BUILD_DIR          = ${BUILD_DIR}"
	@echo "KERNEL_CC          = ${KERNEL_CC}"
	@echo "Kernel object      = ${BUILD_DIR}/${kernels}.o"
	@echo "Design MLIR        = ${BUILD_DIR}/${targetname}.mlir"

layer-info:
	@echo "BitNet 3B Layer Dimensions"
	@echo "=========================="
	@echo "wqkv: M=3840, K=2560"
	@echo "wo:   M=2560, K=2560"
	@echo "w13:  M=13824, K=2560"
	@echo "w2:   M=2560, K=6912"

#===============================================================================
# Clean targets
#===============================================================================

.PHONY: clean clean-all

clean:
	rm -rf ${BUILD_DIR}

clean-all: clean
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

#===============================================================================
# Help
#===============================================================================

.PHONY: help

help:
	@echo "BitNet NPU Makefile"
	@echo "==================="
	@echo ""
	@echo "Build targets:"
	@echo "  all              - Build kernel and design"
	@echo "  kernel           - Build AIE kernel"
	@echo "  kernel-all       - Build all kernel sizes"
	@echo "  design           - Generate MLIR design"
	@echo "  compile          - Compile MLIR to XCLBIN"
	@echo ""
	@echo "Test targets:"
	@echo "  test             - Run CPU reference tests"
	@echo "  test-npu         - Run NPU hardware tests"
	@echo ""
	@echo "Run targets:"
	@echo "  run              - Generate text"
	@echo "  run-interactive  - Interactive generation"
	@echo "  run-sample       - Generation with sampling"
	@echo ""
	@echo "Other:"
	@echo "  info             - Print configuration"
	@echo "  layer-info       - Print layer dimensions"
	@echo "  clean            - Clean build artifacts"
	@echo "  help             - Show this help"
	@echo ""
	@echo "Variables:"
	@echo "  CKPT_DIR=<path>  - Model checkpoint directory"
	@echo "  M=<n>            - Output dimension"
	@echo "  K=<n>            - Input dimension"
	@echo "  m=<n>            - Tile rows"
	@echo "  k=<n>            - Tile columns"
