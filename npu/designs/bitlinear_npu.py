#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025
#
# BitLinear int8 x int2 matrix-vector multiplication for AMD NPU
# Based on mlir-aie/programming_examples/basic/matrix_multiplication/matrix_vector/matrix_vector_iron.py

import numpy as np
import argparse

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D


def bitlinear_design(dev, M=288, K=288, m=32, k=32):
    """
    Create IRON design for BitLinear int8 x int2 matrix-vector multiplication.
    
    Operation: output = input @ weights.T (int32 accumulator)
    - input: int8 vector [K] (quantized activations)  
    - weights: int2 packed matrix [M, K/4] (4 int2 per byte)
    - output: int32 vector [M] (host applies scaling to get bfloat16)
    
    Note: Scaling is done on host to keep the NPU design simple.
    
    Args:
        dev: Device type ("npu" or "npu2")
        M: Output dimension (rows of weight matrix)
        K: Input dimension (columns of weight matrix)
        m: Tile size for M dimension
        k: Tile size for K dimension
    """
    # Hardware DMA limit: each dimension size must be <= 64
    DMA_LIMIT = 64
    
    # Validate dimensions
    assert M % m == 0, f"M ({M}) must be divisible by m ({m})"
    assert K % k == 0, f"K ({K}) must be divisible by k ({k})"
    assert K % 4 == 0, f"K ({K}) must be divisible by 4 (int2 packing)"
    
    M_div_m = M // m
    K_div_k = K // k
    
    # Check DMA limits and provide helpful error messages
    if K_div_k > DMA_LIMIT:
        # Find minimum k that works
        min_k = k
        while K // min_k > DMA_LIMIT:
            min_k *= 2
        raise ValueError(
            f"K/k = {K_div_k} exceeds DMA limit of {DMA_LIMIT}. "
            f"Use k={min_k} instead: make compile M={M} K={K} m={m} k={min_k}"
        )
    
    # Calculate minimum cores needed to keep M iterations under limit
    n_cores = 1
    while M_div_m // n_cores > DMA_LIMIT:
        n_cores += 1
        if n_cores > 4:
            # Find minimum m that works with 4 cores
            min_m = m
            while M // min_m > DMA_LIMIT * 4:
                min_m *= 2
            raise ValueError(
                f"M={M} with m={m} requires more than 4 cores. "
                f"Use m={min_m} instead: make compile M={M} K={K} m={min_m} k={k}"
            )
    
    M_div_n_cores = M // n_cores
    M_div_m_div_n_cores = M // (m * n_cores)
    
    print(f"BitLinear design: M={M}, K={K}, m={m}, k={k}, n_cores={n_cores}, M_iters={M_div_m_div_n_cores}, K_iters={K_div_k}", file=__import__('sys').stderr)

    # Packed sizes (4 int2 per byte)
    k_packed = k // 4
    K_packed = K // 4

    # Whether to use vectorized kernel
    vectorized = False

    # Define types - following matrix_vector_iron.py pattern
    dtype_in = np.dtype[np.int8]
    dtype_in_str = "i8"
    dtype_weight = np.dtype[np.int8]  # Packed int2
    dtype_weight_str = "i2"
    dtype_out = np.dtype[np.int32]    # Output is int32 (scaling done on host)
    dtype_out_str = "i32"

    # Full tensor types
    A_ty = np.ndarray[(M, K_packed), dtype_weight]  # Packed weights [M, K/4]
    B_ty = np.ndarray[(1, K), dtype_in]              # Input activations [1, K]
    C_ty = np.ndarray[(1, M), dtype_out]             # Output [1, M] - int32

    # Tile types
    inA_ty = np.ndarray[(m, k_packed), dtype_weight]  # Weight tile [m, k/4]
    inB_ty = np.ndarray[(k,), dtype_in]               # Input tile [k]
    outC_ty = np.ndarray[(m,), dtype_out]             # Output tile [m] - int32

    # AIE kernel declarations - match matrix_vector pattern
    func_type = "vectorized" if vectorized else "scalar"
    
    # Zero kernel for int32 output
    zero = Kernel(
        f"zero_{func_type}_{dtype_out_str}",
        f"bitlinear_{m}x{k}.o",
        [outC_ty],
    )
    
    # BitLinear matvec kernel: int8 input x int2 weights -> int32 output
    bitlinear = Kernel(
        f"bitlinear_{func_type}_{dtype_in_str}_{dtype_weight_str}_{dtype_out_str}",
        f"bitlinear_{m}x{k}.o",
        [inB_ty, inA_ty, outC_ty],
    )

    # Define the work each core will do - exactly like matrix_vector_iron.py
    def core_fn(of_a, of_b, of_c, zero, bitlinear):
        elem_out = of_c.acquire(1)
        zero(elem_out)
        for _ in range_(K_div_k):
            elem_in_a = of_a.acquire(1)
            elem_in_b = of_b.acquire(1)
            bitlinear(elem_in_b, elem_in_a, elem_out)
            of_a.release(1)
            of_b.release(1)
        of_c.release(1)

    # Create object fifos and workers for each core
    memA_fifos = []
    coreA_fifos = []
    outC_fifos = []
    workers = []
    B_fifo = ObjectFifo(inB_ty)
    
    for i in range(n_cores):
        a_fifo = ObjectFifo(inA_ty, name=f"memA{i}")
        memA_fifos.append(a_fifo)
        coreA_fifos.append(a_fifo.cons().forward())
        outC_fifos.append(ObjectFifo(outC_ty, name=f"outC{i}"))
        w = Worker(
            core_fn,
            [coreA_fifos[i].cons(), B_fifo.cons(), outC_fifos[i].prod(), zero, bitlinear],
        )
        workers.append(w)

    # Define the tiling access patterns for input and output tensors
    A_taps = TensorTiler2D.group_tiler(
        (M, K_packed), (m, k_packed), (M_div_m_div_n_cores, K_div_k)
    )
    C_taps = TensorTiler2D.simple_tiler((1, M), (1, M_div_n_cores))
    b_tap = TensorTiler2D.simple_tiler(
        (1, K), pattern_repeat=M_div_m_div_n_cores
    )[0]

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (a_in, b_in, c_out):
        rt.start(*workers)

        # there is only one b tile
        rt.fill(B_fifo.prod(), b_in, b_tap)

        for i, (a_tap, c_tap) in enumerate(zip(A_taps, C_taps)):
            rt.fill(memA_fifos[i].prod(), a_in, a_tap)
            rt.drain(outC_fifos[i].cons(), c_out, c_tap, wait=True)

    # Create the program from the device type and runtime
    if dev == "npu":
        dev_ty = NPU1()
    else:
        dev_ty = NPU2()
    my_program = Program(dev_ty, rt)

    # Place components and generate MLIR module
    module = my_program.resolve_program(SequentialPlacer())

    return module


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        prog="BitLinear NPU Design",
        description="Generate MLIR for BitLinear int8 x int2 matvec on AMD NPU",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("-M", type=int, default=288, help="Output dimension")
    argparser.add_argument("-K", type=int, default=288, help="Input dimension")
    argparser.add_argument("-m", type=int, default=64, help="Tile size for M (use 64 for large M)")
    argparser.add_argument("-k", type=int, default=64, help="Tile size for K (use 64 for large K)")
    args, _ = argparser.parse_known_args()

    module = bitlinear_design(args.dev, M=args.M, K=args.K, m=args.m, k=args.k)
    print(module)
