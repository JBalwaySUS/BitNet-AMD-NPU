# BitNet NPU Inference

Efficient BitNet inference on AMD NPU using MLIR-AIE, optimized for W2A8 (2-bit weights, 8-bit activations) computation. Tailored for [BitNet-b1.58-2B-4T](https://arxiv.org/abs/2504.12285).

## Features

- W2A8 (2-bit weight × 8-bit activation) matrix-vector multiplication
- Custom AIE kernels for AMD NPU (Ryzen AI)
- IRON-based design using MLIR-AIE
- Automatic CPU fallback when NPU unavailable

## Directory Structure

```
npu/
├── model.py              # BitLinear NPU layers & Transformer
├── generate.py           # Text generation
├── kernels/
│   ├── bitlinear_int8xint2.cc   # AIE C++ kernel
│   └── Makefile
├── designs/
│   ├── bitlinear_npu.py         # IRON design
│   └── transformer_block.py     # Layer configurations
├── test_bitlinear.py     # Test suite
└── Makefile              # Build system
```

## Prerequisites

1. **MLIR-AIE**: Install from [mlir-aie](https://github.com/Xilinx/mlir-aie)
2. **XRT**: Install AMD XRT runtime
3. **Python**:
   ```bash
   pip install -r requirements.txt
   ```

## Building NPU Kernels

```bash
# Build AIE kernel
make kernel

# Generate MLIR design
make design

# Compile to XCLBIN
make compile

# Or build everything at once
make all

# Build all BitNet layer sizes
make design-all
make compile-all
```

## Usage

### Model Conversion

```bash
# Download and convert the BitNet model
mkdir checkpoints
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-bf16 --local-dir ./checkpoints/bitnet-b1.58-2B-4T-bf16
python convert_safetensors.py --safetensors_file ./checkpoints/bitnet-b1.58-2B-4T-bf16/model.safetensors --output checkpoints/model_state.pt --model_name 2B
python convert_checkpoint.py --input ./checkpoints/model_state.pt
rm ./checkpoints/model_state.pt
```

### Inference

```bash
# Basic generation
python generate.py ./checkpoints/

# Interactive mode
python generate.py ./checkpoints/ --interactive

# With chat format
python generate.py ./checkpoints/ --interactive --chat_format

# With sampling
python generate.py ./checkpoints/ --sampling --temperature 0.7

# Specify XCLBIN directory
python generate.py ./checkpoints/ --xclbin_dir build
```

## Testing

```bash
# CPU reference tests
make test

# NPU hardware tests (requires compiled kernels)
make test-npu
```

## API

```python
from model import ModelArgs, Transformer, make_cache

# Create model with NPU acceleration
args = ModelArgs(use_kernel=True, xclbin_dir="build")
model = Transformer(args)

# Load weights
model.load_state_dict(torch.load("model_state_int2.pt"))

# Create KV cache
cache = make_cache(args, length=1024)

# Forward pass
logits = model(tokens, cache, start_pos=0)
```

## BitNet Layer Dimensions

| Layer | Output (M) | Input (K) | Packed Size |
|-------|------------|-----------|-------------|
| wqkv  | 3840       | 2560      | 2.4 MB      |
| wo    | 2560       | 2560      | 1.6 MB      |
| w13   | 13824      | 2560      | 8.8 MB      |
| w2    | 2560       | 6912      | 4.4 MB      |

## Implementation Details

### Weight Packing

Weights are packed as int2 (4 values per byte):
- Ternary values {-1, 0, 1} stored as {0, 1, 2}
- 4× memory reduction vs int8

### Quantization

- **Activations**: Quantized to int8 on-the-fly with per-tensor scaling
- **Weights**: Pre-quantized int2 with per-group scaling
- **Output**: bfloat16 after scaling

### Execution Modes

- **Prefill**: Uses `BitLinearPrefill` with fake quantization for batch efficiency
- **Decode**: Uses `BitLinear` with NPU-accelerated int8×int2 kernels

## Notes

- Falls back to CPU reference implementation if NPU/XCLBIN unavailable
- Requires AMD Ryzen AI hardware for NPU acceleration
- See `kernels/` for AIE kernel implementation details

## References

- [MLIR-AIE Programming Guide](../../mlir-aie/programming_guide/)
- [BitNet Paper](https://arxiv.org/abs/2310.11453)
- [BitNet-b1.58-2B-4T](https://arxiv.org/abs/2504.12285)
