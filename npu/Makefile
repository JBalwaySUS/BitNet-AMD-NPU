# BitNet NPU Makefile
#
# Build and run BitNet inference on AMD NPU.
#
# Usage:
#   make kernel       - Build AIE kernel
#   make design       - Generate MLIR design
#   make compile      - Compile MLIR to XCLBIN
#   make test         - Run tests
#   make all          - Build everything
#   make clean        - Clean build artifacts

# Configuration
DIM_M ?= 32
DIM_K ?= 32
TILE_M ?= 32
TILE_K ?= 32
N_CORES ?= 1
DEVICE ?= npu
CKPT_DIR ?= checkpoints

# Paths
BUILD_DIR = build
KERNEL_DIR = kernels
DESIGN_DIR = designs

# MLIR-AIE tools (set by environment)
AIECC ?= aiecc.py
PYTHON ?= python3

# BitNet 3B layer dimensions
WQKV_DIMS = -M 3840 -K 2560
WO_DIMS = -M 2560 -K 2560
W13_DIMS = -M 13824 -K 2560
W2_DIMS = -M 2560 -K 6912

# Default target
all: kernel design

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

#===============================================================================
# Kernel targets
#===============================================================================

.PHONY: kernel kernel-all

kernel: $(BUILD_DIR)
	$(MAKE) -C $(KERNEL_DIR) DIM_M=$(DIM_M) DIM_K=$(DIM_K)
	cp $(KERNEL_DIR)/bitlinear_$(DIM_M)x$(DIM_K).o $(BUILD_DIR)/

kernel-all: $(BUILD_DIR)
	$(MAKE) -C $(KERNEL_DIR) tile_sizes
	cp $(KERNEL_DIR)/*.o $(BUILD_DIR)/

#===============================================================================
# Design targets
#===============================================================================

.PHONY: design design-all

design: $(BUILD_DIR)
	$(PYTHON) $(DESIGN_DIR)/bitlinear_npu.py \
		-d $(DEVICE) \
		-M $(DIM_M) -K $(DIM_K) \
		-m $(TILE_M) -k $(TILE_K) \
		-c $(N_CORES) \
		-o $(BUILD_DIR)/bitlinear_$(DIM_M)x$(DIM_K).mlir

design-all: $(BUILD_DIR)
	$(PYTHON) $(DESIGN_DIR)/bitlinear_npu.py -d $(DEVICE) $(WQKV_DIMS) \
		-o $(BUILD_DIR)/bitlinear_wqkv.mlir
	$(PYTHON) $(DESIGN_DIR)/bitlinear_npu.py -d $(DEVICE) $(WO_DIMS) \
		-o $(BUILD_DIR)/bitlinear_wo.mlir
	$(PYTHON) $(DESIGN_DIR)/bitlinear_npu.py -d $(DEVICE) $(W13_DIMS) \
		-o $(BUILD_DIR)/bitlinear_w13.mlir
	$(PYTHON) $(DESIGN_DIR)/bitlinear_npu.py -d $(DEVICE) $(W2_DIMS) \
		-o $(BUILD_DIR)/bitlinear_w2.mlir

#===============================================================================
# Compile targets
#===============================================================================

.PHONY: compile compile-all

compile: design kernel
	cd $(BUILD_DIR) && $(AIECC) \
		--aie-generate-xclbin \
		--aie-generate-npu-insts \
		--xclbin-name=bitlinear_$(DIM_M)x$(DIM_K).xclbin \
		--insts-name=bitlinear_$(DIM_M)x$(DIM_K).insts.bin \
		bitlinear_$(DIM_M)x$(DIM_K).mlir

compile-all: design-all kernel-all
	@for mlir in $(BUILD_DIR)/*.mlir; do \
		base=$$(basename $$mlir .mlir); \
		echo "Compiling $$base..."; \
		cd $(BUILD_DIR) && $(AIECC) \
			--aie-generate-xclbin \
			--aie-generate-npu-insts \
			--xclbin-name=$$base.xclbin \
			--insts-name=$$base.insts.bin \
			$$mlir; \
	done

#===============================================================================
# Test targets
#===============================================================================

.PHONY: test test-cpu test-npu

test: test-cpu

test-cpu:
	$(PYTHON) test_bitlinear.py --cpu-only -v

test-npu: compile
	$(PYTHON) test_bitlinear.py \
		--xclbin $(BUILD_DIR)/bitlinear_$(DIM_M)x$(DIM_K).xclbin \
		--instr $(BUILD_DIR)/bitlinear_$(DIM_M)x$(DIM_K).insts.bin \
		-M $(DIM_M) -K $(DIM_K) -v

#===============================================================================
# Run targets
#===============================================================================

.PHONY: run run-interactive

run:
	$(PYTHON) generate.py $(CKPT_DIR) --xclbin_dir $(BUILD_DIR)

run-interactive:
	$(PYTHON) generate.py $(CKPT_DIR) --interactive --xclbin_dir $(BUILD_DIR)

run-sample:
	$(PYTHON) generate.py $(CKPT_DIR) --sampling --xclbin_dir $(BUILD_DIR)

#===============================================================================
# Info targets
#===============================================================================

.PHONY: info layer-info

info:
	@echo "BitNet NPU Configuration"
	@echo "========================"
	@echo "DIM_M      = $(DIM_M)"
	@echo "DIM_K      = $(DIM_K)"
	@echo "TILE_M     = $(TILE_M)"
	@echo "TILE_K     = $(TILE_K)"
	@echo "N_CORES    = $(N_CORES)"
	@echo "DEVICE     = $(DEVICE)"
	@echo "BUILD_DIR  = $(BUILD_DIR)"
	@echo "CKPT_DIR   = $(CKPT_DIR)"

layer-info:
	$(PYTHON) $(DESIGN_DIR)/transformer_block.py --info

#===============================================================================
# Clean targets
#===============================================================================

.PHONY: clean clean-all

clean:
	rm -rf $(BUILD_DIR)
	$(MAKE) -C $(KERNEL_DIR) clean

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
	@echo "  kernel         - Build AIE kernel"
	@echo "  kernel-all     - Build all kernel sizes"
	@echo "  design         - Generate MLIR design"
	@echo "  design-all     - Generate all layer designs"
	@echo "  compile        - Compile MLIR to XCLBIN"
	@echo "  compile-all    - Compile all designs"
	@echo ""
	@echo "Test targets:"
	@echo "  test           - Run CPU reference tests"
	@echo "  test-npu       - Run NPU hardware tests"
	@echo ""
	@echo "Run targets:"
	@echo "  run            - Generate text"
	@echo "  run-interactive - Interactive generation"
	@echo "  run-sample     - Generation with sampling"
	@echo ""
	@echo "Other:"
	@echo "  info           - Print configuration"
	@echo "  layer-info     - Print layer dimensions"
	@echo "  clean          - Clean build artifacts"
	@echo "  help           - Show this help"
	@echo ""
	@echo "Variables:"
	@echo "  CKPT_DIR=<path>  - Model checkpoint directory"
	@echo "  DIM_M=<n>        - Output dimension"
	@echo "  DIM_K=<n>        - Input dimension"
