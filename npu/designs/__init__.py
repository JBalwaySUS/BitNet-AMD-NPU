"""
BitNet NPU Design Files

This package contains IRON (Interface Representation for hands-ON) designs
for BitNet inference on AMD NPU.

Modules:
- bitlinear_npu: Single BitLinear layer design
- transformer_block: Full transformer block design
"""

from .bitlinear_npu import bitlinear_design
from .transformer_block import transformer_block_design, BitNetConfig

__all__ = [
    'bitlinear_design',
    'transformer_block_design',
    'BitNetConfig',
]

