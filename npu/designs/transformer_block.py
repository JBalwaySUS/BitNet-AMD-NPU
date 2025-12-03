#
# BitNet Transformer Block NPU Design
#
# This design implements a full transformer block for BitNet inference,
# combining attention and feedforward layers on AMD NPU.
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
from aie.helpers.taplib.tap import TensorAccessPattern

try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = np.float16


# BitNet 3B model dimensions
class BitNetConfig:
    """Configuration for BitNet 3B model."""
    dim: int = 2560          # Model dimension
    n_heads: int = 20        # Number of attention heads
    n_kv_heads: int = 5      # Number of KV heads (GQA)
    ffn_dim: int = 6912      # FFN intermediate dimension
    head_dim: int = 128      # dim // n_heads
    
    # Derived dimensions for linear layers
    qkv_dim: int = (20 + 5 + 5) * 128  # 3840 = n_heads + 2*n_kv_heads) * head_dim
    
    # Tile sizes for NPU (must divide layer dimensions evenly)
    tile_m: int = 32
    tile_k: int = 32


def create_bitlinear_layer_design(
    dev,
    name: str,
    input_dim: int,
    output_dim: int,
    tile_m: int = 32,
    tile_k: int = 32,
    n_cores: int = 1,
):
    """
    Create design for a single BitLinear layer.
    
    This is a helper to generate designs for specific layer dimensions.
    """
    # Validate dimensions
    assert output_dim % tile_m == 0
    assert input_dim % tile_k == 0
    assert input_dim % 4 == 0  # int2 packing
    
    # Packed weight dimensions
    k_packed = tile_k // 4
    K_packed = input_dim // 4
    
    # Tile counts
    M_tiles = output_dim // tile_m
    K_tiles = input_dim // tile_k
    
    # Data types
    input_tile_ty = np.ndarray[(tile_k,), np.dtype[np.int8]]
    weight_tile_ty = np.ndarray[(tile_m, k_packed), np.dtype[np.int8]]
    output_tile_ty = np.ndarray[(tile_m,), np.dtype[bfloat16]]
    acc_tile_ty = np.ndarray[(tile_m,), np.dtype[np.int32]]
    
    # Full tensor types
    input_ty = np.ndarray[(input_dim,), np.dtype[np.int8]]
    weight_ty = np.ndarray[(output_dim, K_packed), np.dtype[np.int8]]
    output_ty = np.ndarray[(output_dim,), np.dtype[bfloat16]]
    scale_ty = np.ndarray[(1,), np.dtype[bfloat16]]
    weight_scale_ty = np.ndarray[(4,), np.dtype[bfloat16]]
    
    # Kernels
    bitlinear_acc = Kernel(
        "bitlinear_acc_scalar",
        f"bitlinear_{tile_m}x{tile_k}.o",
        [input_tile_ty, weight_tile_ty, acc_tile_ty],
    )
    
    bitlinear_scale = Kernel(
        "bitlinear_scale",
        f"bitlinear_{tile_m}x{tile_k}.o",
        [acc_tile_ty, output_tile_ty, scale_ty, weight_scale_ty],
    )
    
    zero_acc = Kernel(
        "zero_i32",
        f"bitlinear_{tile_m}x{tile_k}.o",
        [acc_tile_ty],
    )
    
    return {
        "name": name,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "tile_m": tile_m,
        "tile_k": tile_k,
        "M_tiles": M_tiles,
        "K_tiles": K_tiles,
        "kernels": {
            "acc": bitlinear_acc,
            "scale": bitlinear_scale,
            "zero": zero_acc,
        },
        "types": {
            "input_tile": input_tile_ty,
            "weight_tile": weight_tile_ty,
            "output_tile": output_tile_ty,
            "acc_tile": acc_tile_ty,
            "input": input_ty,
            "weight": weight_ty,
            "output": output_ty,
            "scale": scale_ty,
            "weight_scale": weight_scale_ty,
        },
    }


def transformer_block_design(dev, config: BitNetConfig = None, n_cores: int = 4):
    """
    Create an IRON design for a full BitNet transformer block.
    
    A transformer block consists of:
    1. RMSNorm (attention)
    2. Attention: wqkv projection (dim -> qkv_dim)
    3. Attention: output projection (dim -> dim)
    4. RMSNorm (FFN)
    5. FFN: w13 projection (dim -> 2*ffn_dim)
    6. FFN: w2 projection (ffn_dim -> dim)
    
    For NPU inference, we focus on the BitLinear layers first,
    with RMSNorm and attention compute potentially on host.
    """
    if config is None:
        config = BitNetConfig()
    
    # Layer configurations for BitNet transformer block
    layers = {
        # Attention projections
        "wqkv": {
            "input_dim": config.dim,      # 2560
            "output_dim": config.qkv_dim,  # 3840 (Q + K + V combined)
        },
        "wo": {
            "input_dim": config.dim,       # 2560
            "output_dim": config.dim,      # 2560
        },
        # FFN projections  
        "w13": {
            "input_dim": config.dim,       # 2560
            "output_dim": 2 * config.ffn_dim,  # 13824 (gate + up combined)
        },
        "w2": {
            "input_dim": config.ffn_dim,   # 6912
            "output_dim": config.dim,      # 2560
        },
    }
    
    # Generate layer designs
    layer_designs = {}
    for name, dims in layers.items():
        layer_designs[name] = create_bitlinear_layer_design(
            dev=dev,
            name=name,
            input_dim=dims["input_dim"],
            output_dim=dims["output_dim"],
            tile_m=config.tile_m,
            tile_k=config.tile_k,
            n_cores=n_cores,
        )
    
    return layer_designs


def generate_single_layer_mlir(
    dev,
    layer_name: str,
    input_dim: int,
    output_dim: int,
    tile_m: int = 32,
    tile_k: int = 32,
    n_cores: int = 1,
):
    """
    Generate MLIR for a single BitLinear layer.
    
    This creates a complete Program that can be compiled and run.
    """
    # Import here to avoid circular dependency
    from bitlinear_npu import bitlinear_design
    
    return bitlinear_design(
        dev=dev,
        M=output_dim,
        K=input_dim,
        m=tile_m,
        k=tile_k,
        n_cores=n_cores,
    )


def print_layer_info(layer_designs: dict):
    """Print information about transformer block layers."""
    print("BitNet Transformer Block Layer Information")
    print("=" * 60)
    
    for name, design in layer_designs.items():
        in_dim = design["input_dim"]
        out_dim = design["output_dim"]
        K_packed = in_dim // 4
        
        # Calculate sizes
        weight_bytes = out_dim * K_packed
        input_bytes = in_dim  # int8
        output_bytes = out_dim * 2  # bfloat16
        
        print(f"\nLayer: {name}")
        print(f"  Dimensions: {out_dim} x {in_dim} (output x input)")
        print(f"  Tiles: {design['M_tiles']} x {design['K_tiles']}")
        print(f"  Weight size: {weight_bytes / 1024:.1f} KB (packed int2)")
        print(f"  Input size: {input_bytes} bytes (int8)")
        print(f"  Output size: {output_bytes} bytes (bfloat16)")
        
        # Compute operations
        macs = out_dim * in_dim
        print(f"  MACs: {macs / 1e6:.2f} M")


def main():
    """Generate transformer block designs."""
    parser = argparse.ArgumentParser(
        prog="BitNet Transformer Block Design",
        description="Generate MLIR designs for BitNet transformer block layers",
    )
    
    parser.add_argument(
        "-d", "--dev",
        type=str,
        choices=["npu", "npu2"],
        default="npu",
        help="Target device",
    )
    parser.add_argument(
        "-l", "--layer",
        type=str,
        choices=["wqkv", "wo", "w13", "w2", "all"],
        default="all",
        help="Layer to generate (or 'all' for info)",
    )
    parser.add_argument(
        "-c", "--cores",
        type=int,
        default=1,
        help="Number of cores",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print layer information only",
    )
    
    args = parser.parse_args()
    
    # Select device
    if args.dev == "npu":
        dev = NPU1()
    else:
        dev = NPU2()
    
    config = BitNetConfig()
    layer_designs = transformer_block_design(dev, config, args.cores)
    
    if args.info or args.layer == "all":
        print_layer_info(layer_designs)
    else:
        # Generate MLIR for specific layer
        design = layer_designs[args.layer]
        module = generate_single_layer_mlir(
            dev=dev,
            layer_name=args.layer,
            input_dim=design["input_dim"],
            output_dim=design["output_dim"],
            tile_m=config.tile_m,
            tile_k=config.tile_k,
            n_cores=args.cores,
        )
        print(module)


if __name__ == "__main__":
    main()

