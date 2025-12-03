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

# bfloat16 support
try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = np.float16


def bitlinear_design(dev):
    """
    Create IRON design for BitLinear int8 x int2 matrix-vector multiplication.
    
    Operation: output = (input @ weights.T) * scales
    - input: int8 vector [K] (quantized activations)  
    - weights: int2 packed matrix [M, K/4] (4 int2 per byte)
    - output: bfloat16 vector [M]
    """
    # Dimensions
    M = 288  # Output dimension (rows of weight matrix)
    K = 288  # Input dimension (columns of weight matrix)
    m = 32   # Tile size for M
    k = 32   # Tile size for K

    n_cores = 1  # TODO: increase for parallelism
    M_div_n_cores = M // n_cores
    M_div_m_div_n_cores = M // (m * n_cores)
    K_div_k = K // k

    # Packed sizes (4 int2 per byte)
    k_packed = k // 4
    K_packed = K // 4

    # Number of weight groups for scaling
    num_weight_groups = 4

    # Whether to use vectorized kernel
    vectorized = False

    # Define types
    dtype_in = np.dtype[np.int8]
    dtype_in_str = "i8"
    dtype_weight = np.dtype[np.int8]  # Packed int2
    dtype_weight_str = "i2"
    dtype_acc = np.dtype[np.int32]
    dtype_acc_str = "i32"
    dtype_out = np.dtype[bfloat16]
    dtype_out_str = "bf16"

    # Full tensor types
    A_ty = np.ndarray[(M, K_packed), dtype_weight]  # Packed weights [M, K/4]
    B_ty = np.ndarray[(1, K), dtype_in]              # Input activations [1, K]
    C_ty = np.ndarray[(1, M), dtype_out]             # Output [1, M]
    S_ty = np.ndarray[(1,), dtype_out]               # Activation scale
    WS_ty = np.ndarray[(num_weight_groups,), dtype_out]  # Weight scales

    # Tile types
    inA_ty = np.ndarray[(m, k_packed), dtype_weight]  # Weight tile [m, k/4]
    inB_ty = np.ndarray[(k,), dtype_in]               # Input tile [k]
    outC_ty = np.ndarray[(m,), dtype_out]             # Output tile [m]
    accC_ty = np.ndarray[(m,), dtype_acc]             # Accumulator tile [m]

    # AIE kernel declarations
    func_type = "vectorized" if vectorized else "scalar"
    
    zero = Kernel(
        f"zero_{func_type}_{dtype_acc_str}",
        f"bitlinear_{m}x{k}.o",
        [accC_ty],
    )
    
    bitlinear = Kernel(
        f"bitlinear_{func_type}_{dtype_in_str}_{dtype_weight_str}_{dtype_acc_str}",
        f"bitlinear_{m}x{k}.o",
        [inB_ty, inA_ty, accC_ty],
    )
    
    scale = Kernel(
        f"bitlinear_scale_{dtype_acc_str}_{dtype_out_str}",
        f"bitlinear_{m}x{k}.o",
        [accC_ty, outC_ty, S_ty, WS_ty],
    )

    # Worker function: accumulate then scale
    def core_fn(of_a, of_b, of_c, of_s, of_ws, zero, bitlinear, scale):
        # Get scales (constant across all tiles)
        elem_s = of_s.acquire(1)
        elem_ws = of_ws.acquire(1)

        for _ in range_(M_div_m_div_n_cores):
            # Acquire output buffer and zero it
            elem_out = of_c.acquire(1)

            # Accumulate across K dimension
            for _ in range_(K_div_k):
                elem_in_a = of_a.acquire(1)  # Weight tile
                elem_in_b = of_b.acquire(1)  # Input tile
                bitlinear(elem_in_b, elem_in_a, elem_out)
                of_a.release(1)
                of_b.release(1)

            # Apply scaling
            scale(elem_out, elem_out, elem_s, elem_ws)
            of_c.release(1)

        of_s.release(1)
        of_ws.release(1)

    # Create object fifos and workers
    memA_fifos = []
    coreA_fifos = []
    outC_fifos = []
    workers = []
    
    B_fifo = ObjectFifo(inB_ty, name="inB")
    S_fifo = ObjectFifo(S_ty, name="scale")
    WS_fifo = ObjectFifo(WS_ty, name="weight_scales")

    for i in range(n_cores):
        a_fifo = ObjectFifo(inA_ty, name=f"memA{i}")
        memA_fifos.append(a_fifo)
        coreA_fifos.append(a_fifo.cons().forward())
        outC_fifos.append(ObjectFifo(outC_ty, name=f"outC{i}"))
        
        w = Worker(
            core_fn,
            [
                coreA_fifos[i].cons(),
                B_fifo.cons(),
                outC_fifos[i].prod(),
                S_fifo.cons(),
                WS_fifo.cons(),
                zero,
                bitlinear,
                scale,
            ],
        )
        workers.append(w)

    # Tensor access patterns
    A_taps = TensorTiler2D.group_tiler(
        (M, K_packed), (m, k_packed), (M_div_m_div_n_cores, K_div_k)
    )
    C_taps = TensorTiler2D.simple_tiler((1, M), (1, M_div_n_cores))
    b_tap = TensorTiler2D.simple_tiler(
        (1, K), pattern_repeat=M_div_m_div_n_cores
    )[0]

    # Runtime sequence
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty, S_ty, WS_ty) as (a_in, b_in, c_out, s_in, ws_in):
        rt.start(*workers)

        # Fill scales
        rt.fill(S_fifo.prod(), s_in)
        rt.fill(WS_fifo.prod(), ws_in)

        # Fill input (broadcast to all cores)
        rt.fill(B_fifo.prod(), b_in, b_tap)

        # Fill weights and drain outputs per core
        for i, (a_tap, c_tap) in enumerate(zip(A_taps, C_taps)):
            rt.fill(memA_fifos[i].prod(), a_in, a_tap)
            rt.drain(outC_fifos[i].cons(), c_out, c_tap, wait=True)

    # Create program
    if dev == "npu":
        dev_ty = NPU1()
    else:
        dev_ty = NPU2()
    
    my_program = Program(dev_ty, rt)
    module = my_program.resolve_program(SequentialPlacer())

    return module


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        prog="BitLinear NPU Design",
        description="Generate MLIR for BitLinear int8 x int2 matvec on AMD NPU",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    args, _ = argparser.parse_known_args()

    module = bitlinear_design(args.dev)
    print(module)
