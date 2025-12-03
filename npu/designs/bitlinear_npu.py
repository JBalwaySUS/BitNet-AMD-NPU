#
# BitLinear NPU Design using IRON API
#
# This design implements int8 x int2 matrix-vector multiplication for BitNet
# inference on AMD NPU (AIE-ML).
#
# The design follows the MLIR-AIE IRON programming model with:
# - ObjectFifos for data movement
# - Workers for compute tasks
# - Runtime sequence for host-device data transfer
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copyright (C) 2025
#

import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D
from aie.helpers.taplib.tap import TensorAccessPattern

# Try to import ml_dtypes for bfloat16 support
try:
    from ml_dtypes import bfloat16
except ImportError:
    # Fallback: use float16 as proxy
    bfloat16 = np.float16


def bitlinear_design(
    dev,
    M: int = 2560,  # Output dimension (rows of weight matrix)
    K: int = 2560,  # Input dimension (columns of weight matrix)
    m: int = 32,    # Tile size for M
    k: int = 32,    # Tile size for K
    n_cores: int = 1,
    trace_size: int = 0,
):
    """
    Create an IRON design for BitLinear int8 x int2 matrix-vector multiplication.
    
    The operation computes: output = (input @ weights.T) * scales
    where:
    - input: int8 vector of size K (quantized activations)
    - weights: int2 packed matrix of size M x K (stored as M x K/4 bytes)
    - output: bfloat16 vector of size M
    - scales: per-group weight scales
    
    Args:
        dev: Device type (NPU1 or NPU2)
        M: Output dimension
        K: Input dimension
        m: Tile size for M dimension
        k: Tile size for K dimension
        n_cores: Number of AIE cores to use
        trace_size: Size of trace buffer (0 to disable)
    
    Returns:
        MLIR module string
    """
    
    enable_trace = trace_size > 0
    
    # Validate dimensions
    assert M % m == 0, f"M ({M}) must be divisible by tile size m ({m})"
    assert K % k == 0, f"K ({K}) must be divisible by tile size k ({k})"
    assert K % 4 == 0, f"K ({K}) must be divisible by 4 (int2 packing)"
    assert M % n_cores == 0, f"M ({M}) must be divisible by n_cores ({n_cores})"
    
    # Compute derived dimensions
    M_div_m = M // m
    K_div_k = K // k
    M_per_core = M // n_cores
    M_div_m_per_core = M_div_m // n_cores
    
    # Packed weight size (4 int2 values per byte)
    k_packed = k // 4  # Packed tile size
    K_packed = K // 4  # Total packed K dimension
    
    # Number of weight groups for scaling (typically 4)
    num_weight_groups = 4
    
    # Define data types
    # Input: int8 activations
    input_tile_ty = np.ndarray[(k,), np.dtype[np.int8]]
    input_full_ty = np.ndarray[(K,), np.dtype[np.int8]]
    
    # Weights: int8 storing packed int2 values (4 per byte)
    weight_tile_ty = np.ndarray[(m, k_packed), np.dtype[np.int8]]
    weight_full_ty = np.ndarray[(M, K_packed), np.dtype[np.int8]]
    
    # Output: bfloat16 
    output_tile_ty = np.ndarray[(m,), np.dtype[bfloat16]]
    output_full_ty = np.ndarray[(M,), np.dtype[bfloat16]]
    
    # Accumulator: int32 for intermediate results
    acc_tile_ty = np.ndarray[(m,), np.dtype[np.int32]]
    
    # Scales: bfloat16
    act_scale_ty = np.ndarray[(1,), np.dtype[bfloat16]]
    weight_scale_ty = np.ndarray[(num_weight_groups,), np.dtype[bfloat16]]
    
    # Define AIE kernels
    # Kernel for accumulating partial products (no scaling)
    bitlinear_acc = Kernel(
        "bitlinear_acc_scalar",
        f"bitlinear_{m}x{k}.o",
        [input_tile_ty, weight_tile_ty, acc_tile_ty],
    )
    
    # Kernel for applying scales to accumulated results
    bitlinear_scale = Kernel(
        "bitlinear_scale",
        f"bitlinear_{m}x{k}.o",
        [acc_tile_ty, output_tile_ty, act_scale_ty, weight_scale_ty],
    )
    
    # Kernel for zeroing accumulator
    zero_acc = Kernel(
        "zero_i32",
        f"bitlinear_{m}x{k}.o",
        [acc_tile_ty],
    )
    
    # Full kernel that does accumulate + scale in one pass
    bitlinear_full = Kernel(
        "bitlinear_full",
        f"bitlinear_{m}x{k}.o",
        [input_tile_ty, weight_tile_ty, output_tile_ty, act_scale_ty, weight_scale_ty],
    )
    
    # Define ObjectFifos for data movement
    # Input activations (broadcast to all cores processing same K)
    of_input = ObjectFifo(input_tile_ty, name="input", depth=2)
    
    # Per-core ObjectFifos
    of_weights = []
    of_outputs = []
    
    for i in range(n_cores):
        of_weights.append(ObjectFifo(weight_tile_ty, name=f"weights_{i}", depth=2))
        of_outputs.append(ObjectFifo(output_tile_ty, name=f"output_{i}", depth=2))
    
    # Scales (shared across all cores)
    of_act_scale = ObjectFifo(act_scale_ty, name="act_scale", depth=1)
    of_weight_scales = ObjectFifo(weight_scale_ty, name="weight_scales", depth=1)
    
    # Define worker function for compute cores
    def core_fn(of_in, of_weights, of_out, of_act_scale, of_weight_scales, 
                bitlinear_full, M_div_m_per_core, K_div_k):
        """
        Worker function that processes M_div_m_per_core output tiles.
        Each tile requires K_div_k iterations over the K dimension.
        """
        # Acquire scales once (they stay constant)
        act_scale = of_act_scale.acquire(1)
        weight_scales = of_weight_scales.acquire(1)
        
        # Process each output tile assigned to this core
        for _ in range_(M_div_m_per_core):
            # Acquire output buffer
            elem_out = of_out.acquire(1)
            
            # For simple version: use full kernel per tile
            # This assumes K_div_k = 1 for now
            # For larger K, need to accumulate across tiles
            
            for _ in range_(K_div_k):
                # Acquire input and weight tiles
                elem_in = of_in.acquire(1)
                elem_weights = of_weights.acquire(1)
                
                # Compute bitlinear for this tile
                bitlinear_full(elem_in, elem_weights, elem_out, 
                              act_scale, weight_scales)
                
                # Release input and weight tiles
                of_in.release(1)
                of_weights.release(1)
            
            # Release output tile
            of_out.release(1)
        
        # Release scales
        of_act_scale.release(1)
        of_weight_scales.release(1)
    
    # Create workers for each core
    workers = []
    for i in range(n_cores):
        w = Worker(
            core_fn,
            fn_args=[
                of_input.cons(),
                of_weights[i].cons(),
                of_outputs[i].prod(),
                of_act_scale.cons(),
                of_weight_scales.cons(),
                bitlinear_full,
                M_div_m_per_core,
                K_div_k,
            ],
        )
        workers.append(w)
    
    # Define tensor access patterns for tiled data movement
    # Input: broadcast same input tiles to all cores K_div_k times per output tile
    input_tap = TensorAccessPattern(
        tensor_dims=(K,),
        offset=0,
        sizes=[M_div_m_per_core, K_div_k, 1, k],
        strides=[0, k, 0, 1],  # Repeat input for each M tile
    )
    
    # Weight taps for each core
    weight_taps = []
    for core_idx in range(n_cores):
        # Each core gets M_per_core / m tiles of weights
        # Weights are stored as M x (K/4) packed
        core_offset = core_idx * M_per_core * K_packed
        tap = TensorAccessPattern(
            tensor_dims=(M, K_packed),
            offset=core_offset,
            sizes=[M_div_m_per_core, K_div_k, m, k_packed],
            strides=[K_packed * m, k_packed, K_packed, 1],
        )
        weight_taps.append(tap)
    
    # Output taps for each core
    output_taps = []
    for core_idx in range(n_cores):
        core_offset = core_idx * M_per_core
        tap = TensorAccessPattern(
            tensor_dims=(M,),
            offset=core_offset,
            sizes=[1, 1, 1, M_per_core],
            strides=[0, 0, 0, 1],
        )
        output_taps.append(tap)
    
    # Define runtime sequence
    rt = Runtime()
    with rt.sequence(
        input_full_ty,      # Input activations
        weight_full_ty,     # Packed weights
        output_full_ty,     # Output
        act_scale_ty,       # Activation scale
        weight_scale_ty,    # Weight scales
    ) as (input_buf, weight_buf, output_buf, act_scale_buf, weight_scales_buf):
        
        # Start all workers
        rt.start(*workers)
        
        # Fill scales (once)
        rt.fill(of_act_scale.prod(), act_scale_buf)
        rt.fill(of_weight_scales.prod(), weight_scales_buf)
        
        # Fill input (broadcast to all cores)
        rt.fill(of_input.prod(), input_buf, input_tap)
        
        # Fill weights for each core
        for i in range(n_cores):
            rt.fill(of_weights[i].prod(), weight_buf, weight_taps[i])
        
        # Drain outputs from each core
        for i in range(n_cores):
            rt.drain(of_outputs[i].cons(), output_buf, output_taps[i], wait=True)
    
    # Create program and resolve placement
    my_program = Program(dev, rt)
    module = my_program.resolve_program(SequentialPlacer())
    
    return module


def main():
    """Main entry point for generating MLIR from command line."""
    parser = argparse.ArgumentParser(
        prog="BitLinear NPU Design",
        description="Generate MLIR for BitLinear int8 x int2 matvec on AMD NPU",
    )
    
    parser.add_argument(
        "-d", "--dev",
        type=str,
        choices=["npu", "npu2"],
        default="npu",
        help="Target device (npu or npu2)",
    )
    parser.add_argument(
        "-M", "--output-dim",
        type=int,
        default=2560,
        help="Output dimension M",
    )
    parser.add_argument(
        "-K", "--input-dim",
        type=int,
        default=2560,
        help="Input dimension K",
    )
    parser.add_argument(
        "-m", "--tile-m",
        type=int,
        default=32,
        help="Tile size for M dimension",
    )
    parser.add_argument(
        "-k", "--tile-k",
        type=int,
        default=32,
        help="Tile size for K dimension",
    )
    parser.add_argument(
        "-c", "--cores",
        type=int,
        default=1,
        help="Number of AIE cores to use",
    )
    parser.add_argument(
        "-t", "--trace-size",
        type=int,
        default=0,
        help="Trace buffer size (0 to disable)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file (default: stdout)",
    )
    
    args = parser.parse_args()
    
    # Select device
    if args.dev == "npu":
        dev = NPU1()
    else:
        dev = NPU2()
    
    # Generate design
    module = bitlinear_design(
        dev=dev,
        M=args.output_dim,
        K=args.input_dim,
        m=args.tile_m,
        k=args.tile_k,
        n_cores=args.cores,
        trace_size=args.trace_size,
    )
    
    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(str(module))
    else:
        print(module)


if __name__ == "__main__":
    main()

