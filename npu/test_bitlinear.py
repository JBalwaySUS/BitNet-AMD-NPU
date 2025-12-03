#!/usr/bin/env python3
"""
Test script for BitLinear NPU kernel.

This script tests the int8 x int2 BitLinear kernel implementation
by comparing NPU results against a CPU reference implementation.

Can also compare against GPU implementation if available.

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
Copyright (C) 2025
"""

import numpy as np
import argparse
import time
from pathlib import Path

# Try to import ml_dtypes for bfloat16
try:
    from ml_dtypes import bfloat16
    HAS_BFLOAT16 = True
except ImportError:
    HAS_BFLOAT16 = False
    bfloat16 = np.float16

# Try to import XRT for NPU execution
try:
    import pyxrt as xrt
    HAS_XRT = True
except ImportError:
    HAS_XRT = False

# Try to import torch for GPU comparison
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def unpack_int2_to_int8(packed: np.ndarray) -> np.ndarray:
    """
    Unpack int2 values from packed int8 array.
    
    Each byte contains 4 int2 values:
    - bits [1:0] -> value 0
    - bits [3:2] -> value 1
    - bits [5:4] -> value 2
    - bits [7:6] -> value 3
    
    Values are stored as 0, 1, 2 representing -1, 0, 1.
    """
    # Ensure input is uint8 for bit operations
    packed_u8 = packed.astype(np.uint8)
    
    # Extract each 2-bit value
    v0 = (packed_u8 & 0x03).astype(np.int8) - 1
    v1 = ((packed_u8 >> 2) & 0x03).astype(np.int8) - 1
    v2 = ((packed_u8 >> 4) & 0x03).astype(np.int8) - 1
    v3 = ((packed_u8 >> 6) & 0x03).astype(np.int8) - 1
    
    # Interleave to get unpacked array
    # Result shape: (M, K) where K = packed.shape[1] * 4
    if packed.ndim == 1:
        unpacked = np.empty(packed.shape[0] * 4, dtype=np.int8)
        unpacked[0::4] = v0
        unpacked[1::4] = v1
        unpacked[2::4] = v2
        unpacked[3::4] = v3
    else:
        M, K_packed = packed.shape
        unpacked = np.empty((M, K_packed * 4), dtype=np.int8)
        unpacked[:, 0::4] = v0
        unpacked[:, 1::4] = v1
        unpacked[:, 2::4] = v2
        unpacked[:, 3::4] = v3
    
    return unpacked


def pack_int2(unpacked: np.ndarray) -> np.ndarray:
    """
    Pack int8 values (-1, 0, 1) into int2 packed format.
    
    Values are mapped: -1 -> 0, 0 -> 1, 1 -> 2
    """
    # Map -1, 0, 1 to 0, 1, 2
    mapped = (unpacked + 1).astype(np.uint8)
    
    if unpacked.ndim == 1:
        K = unpacked.shape[0]
        assert K % 4 == 0, "K must be divisible by 4"
        K_packed = K // 4
        
        packed = np.zeros(K_packed, dtype=np.uint8)
        packed |= mapped[0::4]
        packed |= mapped[1::4] << 2
        packed |= mapped[2::4] << 4
        packed |= mapped[3::4] << 6
    else:
        M, K = unpacked.shape
        assert K % 4 == 0, "K must be divisible by 4"
        K_packed = K // 4
        
        packed = np.zeros((M, K_packed), dtype=np.uint8)
        packed |= mapped[:, 0::4]
        packed |= mapped[:, 1::4] << 2
        packed |= mapped[:, 2::4] << 4
        packed |= mapped[:, 3::4] << 6
    
    return packed.astype(np.int8)


def bitlinear_ref(
    input: np.ndarray,           # [K] int8 activations
    weights_packed: np.ndarray,  # [M, K/4] packed int2 weights
    act_scale: float,            # Activation scale
    weight_scales: np.ndarray,   # [num_groups] Weight scales
) -> np.ndarray:
    """
    CPU reference implementation of BitLinear int8 x int2 matvec.
    
    output = (input @ weights.T / act_scale) * weight_scales
    """
    M, K_packed = weights_packed.shape
    K = K_packed * 4
    num_groups = len(weight_scales)
    rows_per_group = M // num_groups
    
    assert input.shape[0] == K, f"Input size mismatch: {input.shape[0]} vs {K}"
    
    # Unpack weights
    weights = unpack_int2_to_int8(weights_packed)
    
    # Compute matvec with int32 accumulation
    output = np.zeros(M, dtype=np.float32)
    inv_act_scale = 1.0 / act_scale
    
    for row in range(M):
        acc = np.dot(input.astype(np.int32), weights[row].astype(np.int32))
        
        group_idx = min(row // rows_per_group, num_groups - 1)
        ws = weight_scales[group_idx]
        
        output[row] = float(acc) * inv_act_scale * ws
    
    return output


def quantize_input(x: np.ndarray) -> tuple:
    """
    Quantize bfloat16/float input to int8.
    
    Returns (quantized, scale) where:
    - quantized = round(x * scale).clip(-128, 127).astype(int8)
    - scale = 127 / max(abs(x))
    """
    x_float = x.astype(np.float32)
    scale = 127.0 / np.maximum(np.abs(x_float).max(), 1e-5)
    quantized = np.round(x_float * scale).clip(-128, 127).astype(np.int8)
    return quantized, scale


def generate_test_data(M: int, K: int, num_groups: int = 4, seed: int = 42):
    """Generate random test data for BitLinear."""
    np.random.seed(seed)
    
    # Random int8 input
    input_int8 = np.random.randint(-127, 128, size=K, dtype=np.int8)
    
    # Random ternary weights (-1, 0, 1)
    weights_ternary = np.random.choice([-1, 0, 1], size=(M, K)).astype(np.int8)
    weights_packed = pack_int2(weights_ternary)
    
    # Random scales
    act_scale = 127.0
    weight_scales = np.random.uniform(0.1, 2.0, size=num_groups).astype(np.float32)
    
    return {
        'input': input_int8,
        'weights_packed': weights_packed,
        'weights_ternary': weights_ternary,
        'act_scale': act_scale,
        'weight_scales': weight_scales,
    }


def test_packing():
    """Test int2 packing/unpacking roundtrip."""
    print("Testing int2 packing/unpacking...")
    
    # Generate random ternary values
    M, K = 32, 128
    original = np.random.choice([-1, 0, 1], size=(M, K)).astype(np.int8)
    
    # Pack and unpack
    packed = pack_int2(original)
    unpacked = unpack_int2_to_int8(packed)
    
    # Verify
    if np.array_equal(original, unpacked):
        print("  PASS: Pack/unpack roundtrip")
        return True
    else:
        print("  FAIL: Pack/unpack mismatch")
        diff_count = np.sum(original != unpacked)
        print(f"  Differences: {diff_count} / {M * K}")
        return False


def test_reference(M: int = 32, K: int = 64, num_groups: int = 4):
    """Test CPU reference implementation."""
    print(f"Testing CPU reference (M={M}, K={K})...")
    
    data = generate_test_data(M, K, num_groups)
    
    # Run reference
    output = bitlinear_ref(
        data['input'],
        data['weights_packed'],
        data['act_scale'],
        data['weight_scales'],
    )
    
    # Verify with manual computation
    weights = data['weights_ternary'].astype(np.float32)
    input_f = data['input'].astype(np.float32)
    inv_scale = 1.0 / data['act_scale']
    
    expected = np.zeros(M, dtype=np.float32)
    rows_per_group = M // num_groups
    
    for row in range(M):
        acc = np.dot(input_f, weights[row])
        group_idx = min(row // rows_per_group, num_groups - 1)
        expected[row] = acc * inv_scale * data['weight_scales'][group_idx]
    
    # Check results
    max_diff = np.abs(output - expected).max()
    if max_diff < 1e-5:
        print(f"  PASS: Max diff = {max_diff:.2e}")
        return True
    else:
        print(f"  FAIL: Max diff = {max_diff:.2e}")
        return False


def test_vs_gpu(M: int = 2560, K: int = 2560, num_groups: int = 4):
    """Test reference against GPU implementation (if available)."""
    if not HAS_TORCH or not torch.cuda.is_available():
        print("Skipping GPU comparison (torch/CUDA not available)")
        return True
    
    print(f"Testing vs GPU (M={M}, K={K})...")
    
    # Try to import the GPU kernel
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent / 'bitnet_kernels'))
        import ctypes
        
        bitnet_lib = ctypes.CDLL(str(Path(__file__).parent / 'bitnet_kernels/libbitnet.so'))
    except Exception as e:
        print(f"  Skipping: Could not load GPU kernel ({e})")
        return True
    
    # Generate test data
    data = generate_test_data(M, K, num_groups)
    
    # Run CPU reference
    cpu_output = bitlinear_ref(
        data['input'],
        data['weights_packed'],
        data['act_scale'],
        data['weight_scales'],
    )
    
    # TODO: Run GPU kernel and compare
    print("  GPU comparison not yet implemented")
    return True


def run_npu_test(
    xclbin_path: str,
    instr_path: str,
    M: int = 32,
    K: int = 32,
    num_groups: int = 4,
    n_iters: int = 1,
    verbose: bool = True,
):
    """Run test on NPU using XRT."""
    if not HAS_XRT:
        print("XRT not available, skipping NPU test")
        return None
    
    print(f"Running NPU test (M={M}, K={K}, iters={n_iters})...")
    
    # Generate test data
    data = generate_test_data(M, K, num_groups)
    
    # Compute reference
    ref_output = bitlinear_ref(
        data['input'],
        data['weights_packed'],
        data['act_scale'],
        data['weight_scales'],
    )
    
    # Initialize XRT
    device = xrt.device(0)
    xclbin = xrt.xclbin(xclbin_path)
    device.register_xclbin(xclbin)
    
    # Get kernel
    kernels = xclbin.get_kernels()
    kernel_name = kernels[0].get_name()
    context = xrt.hw_context(device, xclbin.get_uuid())
    kernel = xrt.kernel(context, kernel_name)
    
    # Load instructions
    with open(instr_path, 'rb') as f:
        instr_data = np.frombuffer(f.read(), dtype=np.uint32)
    
    # Create buffers
    K_packed = K // 4
    
    bo_instr = xrt.bo(device, instr_data.nbytes, xrt.bo.cacheable, kernel.group_id(1))
    bo_input = xrt.bo(device, K, xrt.bo.host_only, kernel.group_id(3))
    bo_weights = xrt.bo(device, M * K_packed, xrt.bo.host_only, kernel.group_id(4))
    bo_output = xrt.bo(device, M * 2, xrt.bo.host_only, kernel.group_id(5))  # bfloat16
    bo_act_scale = xrt.bo(device, 2, xrt.bo.host_only, kernel.group_id(6))  # bfloat16
    bo_weight_scales = xrt.bo(device, num_groups * 2, xrt.bo.host_only, kernel.group_id(7))
    
    # Copy input data
    np.copyto(np.frombuffer(bo_instr.map(), dtype=np.uint32), instr_data)
    np.copyto(np.frombuffer(bo_input.map(), dtype=np.int8), data['input'])
    np.copyto(np.frombuffer(bo_weights.map(), dtype=np.int8), data['weights_packed'].flatten())
    
    # Copy scales (as bfloat16)
    if HAS_BFLOAT16:
        act_scale_bf16 = np.array([data['act_scale']], dtype=bfloat16)
        weight_scales_bf16 = data['weight_scales'].astype(bfloat16)
    else:
        act_scale_bf16 = np.array([data['act_scale']], dtype=np.float16)
        weight_scales_bf16 = data['weight_scales'].astype(np.float16)
    
    np.copyto(np.frombuffer(bo_act_scale.map(), dtype=act_scale_bf16.dtype), act_scale_bf16)
    np.copyto(np.frombuffer(bo_weight_scales.map(), dtype=weight_scales_bf16.dtype), weight_scales_bf16)
    
    # Sync to device
    bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_input.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_weights.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_act_scale.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_weight_scales.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    
    # Run kernel
    times = []
    for i in range(n_iters):
        start = time.perf_counter()
        run = kernel(3, bo_instr, len(instr_data), bo_input, bo_weights, 
                    bo_output, bo_act_scale, bo_weight_scales)
        run.wait()
        times.append(time.perf_counter() - start)
    
    # Sync output
    bo_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    
    if HAS_BFLOAT16:
        npu_output = np.frombuffer(bo_output.map(), dtype=bfloat16).astype(np.float32)
    else:
        npu_output = np.frombuffer(bo_output.map(), dtype=np.float16).astype(np.float32)
    
    # Compare results
    abs_diff = np.abs(npu_output - ref_output)
    max_diff = abs_diff.max()
    avg_diff = abs_diff.mean()
    
    # Calculate throughput
    avg_time = np.mean(times)
    macs = M * K
    gmacs_per_sec = (macs / 1e9) / avg_time
    
    if verbose:
        print(f"  Avg time: {avg_time * 1e6:.1f} us")
        print(f"  Throughput: {gmacs_per_sec:.2f} GMACs/s")
        print(f"  Max diff: {max_diff:.2e}")
        print(f"  Avg diff: {avg_diff:.2e}")
    
    # Check tolerance
    abs_tol = 1e-2
    rel_tol = 1e-2
    errors = np.sum((abs_diff > abs_tol) & (abs_diff / (np.abs(ref_output) + 1e-10) > rel_tol))
    
    if errors == 0:
        print("  PASS")
        return True
    else:
        print(f"  FAIL ({errors} errors)")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test BitLinear NPU kernel",
    )
    parser.add_argument(
        "--xclbin", type=str, default="build/final.xclbin",
        help="Path to XCLBIN file",
    )
    parser.add_argument(
        "--instr", type=str, default="build/insts.bin",
        help="Path to instruction binary",
    )
    parser.add_argument(
        "-M", "--output-dim", type=int, default=32,
        help="Output dimension M",
    )
    parser.add_argument(
        "-K", "--input-dim", type=int, default=32,
        help="Input dimension K",
    )
    parser.add_argument(
        "-n", "--iters", type=int, default=1,
        help="Number of iterations",
    )
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="Run CPU tests only",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    all_passed = True
    
    # Run unit tests
    print("\n=== Unit Tests ===")
    all_passed &= test_packing()
    all_passed &= test_reference()
    all_passed &= test_reference(M=64, K=128)
    all_passed &= test_reference(M=256, K=256)
    
    # GPU comparison
    if HAS_TORCH:
        print("\n=== GPU Comparison ===")
        all_passed &= test_vs_gpu()
    
    # NPU test
    if not args.cpu_only:
        print("\n=== NPU Test ===")
        xclbin = Path(args.xclbin)
        instr = Path(args.instr)
        
        if xclbin.exists() and instr.exists():
            result = run_npu_test(
                str(xclbin),
                str(instr),
                M=args.output_dim,
                K=args.input_dim,
                n_iters=args.iters,
                verbose=args.verbose,
            )
            if result is not None:
                all_passed &= result
        else:
            print(f"Skipping NPU test (files not found)")
            print(f"  XCLBIN: {xclbin} (exists: {xclbin.exists()})")
            print(f"  INSTR: {instr} (exists: {instr.exists()})")
    
    # Summary
    print("\n" + "=" * 40)
    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    exit(main())

