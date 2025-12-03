# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
#
# AMD NPU inference implementation for BitNet.

from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict
from pathlib import Path
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# NPU dependencies (optional - falls back to CPU if not available)
try:
    import pyxrt as xrt
    HAS_XRT = True
except ImportError:
    HAS_XRT = False
    xrt = None

# bfloat16 for numpy
try:
    from ml_dtypes import bfloat16 as np_bfloat16
    HAS_BFLOAT16 = True
except ImportError:
    HAS_BFLOAT16 = False
    np_bfloat16 = np.float16


#==============================================================================
# NPU Runtime
#==============================================================================

class NPURuntime:
    """
    Singleton class to manage NPU device and resources via XRT.
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.device = None
        self.context = None
        self.kernels: Dict[str, 'xrt.kernel'] = {}
        self.xclbins: Dict[str, 'xrt.xclbin'] = {}
        
    def initialize(self, device_index: int = 0):
        """Initialize NPU device."""
        if not HAS_XRT:
            raise RuntimeError("XRT not available. Install pyxrt to use NPU.")
        if self.device is not None:
            return
        self.device = xrt.device(device_index)
        print(f"Initialized NPU device {device_index}")
    
    def load_xclbin(self, xclbin_path: str, name: str = "default") -> 'xrt.xclbin':
        """Load XCLBIN and register with device."""
        if not HAS_XRT:
            raise RuntimeError("XRT not available")
        if self.device is None:
            self.initialize()
        if name in self.xclbins:
            return self.xclbins[name]
        
        xclbin = xrt.xclbin(xclbin_path)
        self.device.register_xclbin(xclbin)
        self.xclbins[name] = xclbin
        return xclbin
    
    def get_kernel(self, xclbin_name: str, kernel_name: str = None) -> 'xrt.kernel':
        """Get kernel from loaded XCLBIN."""
        if xclbin_name not in self.xclbins:
            raise ValueError(f"XCLBIN '{xclbin_name}' not loaded")
        
        xclbin = self.xclbins[xclbin_name]
        if kernel_name is None:
            kernel_name = xclbin.get_kernels()[0].get_name()
        
        cache_key = f"{xclbin_name}:{kernel_name}"
        if cache_key in self.kernels:
            return self.kernels[cache_key]
        
        if self.context is None:
            self.context = xrt.hw_context(self.device, xclbin.get_uuid())
        
        kernel = xrt.kernel(self.context, kernel_name)
        self.kernels[cache_key] = kernel
        return kernel


#==============================================================================
# CPU Reference Implementation for BitLinear
#==============================================================================

def bitnet_int8xint2_linear_cpu(input_int8, weights_packed, act_scale, weight_scales):
    """
    CPU reference implementation of int8 x int2 BitLinear.
    
    Args:
        input_int8: Quantized int8 activations [K] or [batch, K]
        weights_packed: Packed int2 weights [M, K/4]
        act_scale: Activation scale (scalar tensor)
        weight_scales: Per-group weight scales [num_groups]
    
    Returns:
        bfloat16 output tensor
    """
    M, K_packed = weights_packed.shape
    K = K_packed * 4
    num_groups = len(weight_scales)
    rows_per_group = M // num_groups
    
    # Unpack int2 weights to int8
    weights_u8 = weights_packed.cpu().numpy().astype(np.uint8)
    weights_unpacked = np.zeros((M, K), dtype=np.int8)
    weights_unpacked[:, 0::4] = ((weights_u8 & 0x03) - 1).astype(np.int8)
    weights_unpacked[:, 1::4] = (((weights_u8 >> 2) & 0x03) - 1).astype(np.int8)
    weights_unpacked[:, 2::4] = (((weights_u8 >> 4) & 0x03) - 1).astype(np.int8)
    weights_unpacked[:, 3::4] = (((weights_u8 >> 6) & 0x03) - 1).astype(np.int8)
    
    # Compute matrix-vector multiplication
    input_np = input_int8.cpu().numpy().astype(np.int32)
    weights_np = weights_unpacked.astype(np.int32)
    
    if input_np.ndim == 1:
        output = np.dot(weights_np, input_np)
    else:
        output = np.dot(input_np, weights_np.T)
    
    # Apply scaling
    inv_act_scale = 1.0 / act_scale.item()
    weight_scales_np = weight_scales.cpu().numpy().astype(np.float32)
    
    output_f = output.astype(np.float32) * inv_act_scale
    
    # Apply per-group weight scales
    for g in range(num_groups):
        start = g * rows_per_group
        end = (g + 1) * rows_per_group if g < num_groups - 1 else M
        if output_f.ndim == 1:
            output_f[start:end] *= weight_scales_np[g]
        else:
            output_f[..., start:end] *= weight_scales_np[g]
    
    return torch.from_numpy(output_f).to(dtype=torch.bfloat16, device=input_int8.device)


#==============================================================================
# Model Arguments
#==============================================================================

@dataclass
class ModelArgs:
    dim: int = 2560
    n_layers: int = 30
    n_heads: int = 20
    n_kv_heads: int = 5
    vocab_size: int = 128256
    ffn_dim: int = 6912
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    
    # NPU settings
    use_kernel: bool = False      # Use optimized NPU kernel
    xclbin_dir: str = "build"     # Directory with compiled NPU kernels
    num_weight_groups: int = 4    # Weight scale groups


LayerCache = Tuple[torch.Tensor, torch.Tensor]


#==============================================================================
# RMSNorm
#==============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


#==============================================================================
# BitLinear NPU Implementation
#==============================================================================

class BitLinear(nn.Module):
    """
    BitLinear layer for NPU inference.
    
    Performs int8 x int2 matrix-vector multiplication:
    - Activations are quantized to int8 on-the-fly
    - Weights are stored as packed int2 (4 values per byte)
    - Output is bfloat16 after scaling
    
    Falls back to CPU reference implementation if NPU is not available.
    """
    in_features: int
    out_features: int
    weight: torch.Tensor
    weight_scale: torch.Tensor
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = False,
        num_weight_groups: int = 4,
        xclbin_dir: str = "build",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_weight_groups = num_weight_groups
        
        # Packed int2 weights (4 values per byte)
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features // 4, dtype=torch.int8),
            requires_grad=False
        )
        self.weight_scale = nn.Parameter(
            torch.zeros(num_weight_groups, dtype=torch.bfloat16),
            requires_grad=False
        )
        
        # NPU resources (initialized lazily)
        self._npu_runtime = None
        self._kernel = None
        self._buffers_initialized = False
        self._use_cpu_fallback = not HAS_XRT
        
        # Paths to compiled kernels
        self.xclbin_path = str(Path(xclbin_dir) / f"bitlinear_{out_features}x{in_features}.xclbin")
        self.instr_path = str(Path(xclbin_dir) / f"bitlinear_{out_features}x{in_features}.insts.bin")
    
    def _init_npu(self):
        """Initialize NPU resources lazily on first forward pass."""
        if self._kernel is not None or self._use_cpu_fallback:
            return
        
        if not Path(self.xclbin_path).exists():
            print(f"Info: XCLBIN not found at {self.xclbin_path}, using CPU")
            self._use_cpu_fallback = True
            return
        
        try:
            self._npu_runtime = NPURuntime()
            self._npu_runtime.initialize()
            
            layer_name = f"bitlinear_{self.out_features}x{self.in_features}"
            self._npu_runtime.load_xclbin(self.xclbin_path, layer_name)
            self._kernel = self._npu_runtime.get_kernel(layer_name)
            
            print(f"Initialized NPU for BitLinear({self.in_features}, {self.out_features})")
        except Exception as e:
            print(f"Info: NPU init failed ({e}), using CPU")
            self._use_cpu_fallback = True
    
    @torch.compile
    def quant_input(self, input):
        """Quantize input activations to int8."""
        s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (input * s).round().clamp(-128, 127).to(torch.int8), s
    
    def forward(self, input):
        """
        Forward pass through BitLinear layer.
        
        Args:
            input: bfloat16 tensor [..., in_features]
        
        Returns:
            bfloat16 tensor [..., out_features]
        """
        # Quantize input to int8
        input_q, act_scale = self.quant_input(input)
        
        # Initialize NPU on first call
        if self._kernel is None and not self._use_cpu_fallback:
            self._init_npu()
        
        # TODO: Add NPU kernel execution when XCLBIN is compiled
        # For now, use CPU reference implementation
        return bitnet_int8xint2_linear_cpu(
            input_q, self.weight, act_scale.squeeze(), self.weight_scale
        )


class BitLinearPrefill(nn.Linear):
    """
    BitLinear for prefill phase using standard PyTorch operations.
    
    Uses fake quantization for better numerical accuracy during
    prompt processing where batch efficiency is important.
    """
    
    @torch.compile
    def quant_input(self, input):
        """Fake quantization: quantize and dequantize."""
        s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (input * s).round().clamp(-128, 127) / s

    def forward(self, input):
        input = self.quant_input(input)
        return F.linear(input, self.weight)


def get_linear_class(use_kernel: bool):
    """Get appropriate linear layer class."""
    if use_kernel:
        return BitLinear
    else:
        return BitLinearPrefill


#==============================================================================
# Rotary Position Embedding
#==============================================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex exponentials for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to Q and K."""
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    xq_complex = torch.view_as_complex(xq_)
    xk_complex = torch.view_as_complex(xk_)
    
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


#==============================================================================
# Attention
#==============================================================================

class Attention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA) support."""
    
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        norm_eps: float,
        use_kernel: bool,
        xclbin_dir: str = "build",
    ):
        super().__init__()

        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.n_local_heads = n_heads
        self.n_local_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads

        Linear = get_linear_class(use_kernel)
        
        linear_kwargs = {}
        if use_kernel:
            linear_kwargs['xclbin_dir'] = xclbin_dir

        self.wqkv = Linear(
            dim,
            (self.n_local_heads + 2 * self.n_local_kv_heads) * head_dim,
            bias=False,
            **linear_kwargs,
        )
        self.wo = Linear(
            self.n_local_heads * head_dim,
            dim,
            bias=False,
            **linear_kwargs,
        )

        self.attn_sub_norm = RMSNorm(dim, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[LayerCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # QKV projection
        xqkv = self.wqkv(x)
        
        q_dim = self.n_local_heads * self.head_dim
        kv_dim = self.n_local_kv_heads * self.head_dim
        
        xq = xqkv[..., :q_dim]
        xk = xqkv[..., q_dim:q_dim + kv_dim]
        xv = xqkv[..., q_dim + kv_dim:]

        # Reshape for attention
        xq = xq.view(batch, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(batch, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(batch, seq_len, self.n_local_kv_heads, self.head_dim)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Update KV cache if provided
        if cache is not None:
            cache_k, cache_v = cache
            cache_k[:, start_pos:start_pos + seq_len] = xk
            cache_v[:, start_pos:start_pos + seq_len] = xv
            xk = cache_k[:, :start_pos + seq_len]
            xv = cache_v[:, :start_pos + seq_len]

        # Expand KV for GQA
        if self.n_rep > 1:
            xk = xk.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1)
            xk = xk.reshape(batch, -1, self.n_local_heads, self.head_dim)
            xv = xv.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1)
            xv = xv.reshape(batch, -1, self.n_local_heads, self.head_dim)

        # Transpose for attention: [batch, heads, seq, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(xq, xk.transpose(-2, -1)) * scale

        # Causal mask
        if seq_len > 1:
            mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=x.device),
                diagonal=1
            )
            if cache is not None:
                kv_len = start_pos + seq_len
                full_mask = torch.zeros((seq_len, kv_len), device=x.device)
                full_mask[:, -seq_len:] = mask
                mask = full_mask
            attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, xv)

        # Reshape back
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, -1)

        # Sub-norm and output projection
        output = self.attn_sub_norm(output)
        output = self.wo(output)

        return output


#==============================================================================
# FeedForward
#==============================================================================

@torch.compile
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    """Squared ReLU activation."""
    return F.relu(x) ** 2


class FeedForward(nn.Module):
    """Feed-forward network with gated squared ReLU."""
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        norm_eps: float,
        use_kernel: bool,
        xclbin_dir: str = "build",
    ):
        super().__init__()

        Linear = get_linear_class(use_kernel)
        
        linear_kwargs = {}
        if use_kernel:
            linear_kwargs['xclbin_dir'] = xclbin_dir

        self.w13 = Linear(
            dim,
            2 * hidden_dim,
            bias=False,
            **linear_kwargs,
        )
        self.w2 = Linear(
            hidden_dim,
            dim,
            bias=False,
            **linear_kwargs,
        )
        self.ffn_sub_norm = RMSNorm(hidden_dim, norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x13 = self.w13(x)
        x1, x3 = x13.chunk(2, -1)
        inner = self.ffn_sub_norm(squared_relu(x1) * x3)
        output = self.w2(inner)
        return output


#==============================================================================
# Transformer Block and Full Model
#==============================================================================

class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward."""
    
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.dim % args.n_heads == 0
        head_dim = args.dim // args.n_heads
        n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads

        assert args.n_heads % n_kv_heads == 0

        self.attention = Attention(
            dim=args.dim,
            head_dim=head_dim,
            n_heads=args.n_heads,
            n_kv_heads=n_kv_heads,
            rope_theta=args.rope_theta,
            norm_eps=args.norm_eps,
            use_kernel=args.use_kernel,
            xclbin_dir=args.xclbin_dir,
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.ffn_dim,
            norm_eps=args.norm_eps,
            use_kernel=args.use_kernel,
            xclbin_dir=args.xclbin_dir,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[LayerCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        h = x + self.attention(
            self.attention_norm(x),
            freqs_cis,
            cache,
            start_pos,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """BitNet Transformer model for NPU inference."""
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size > 0
        self.args = args

        self.tok_embeddings = nn.Embedding(
            num_embeddings=args.vocab_size,
            embedding_dim=args.dim,
        )

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(
            args.dim,
            args.vocab_size,
            bias=False,
        )
        
        # Precompute RoPE frequencies
        head_dim = args.dim // args.n_heads
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, 4096, args.rope_theta),
            persistent=False
        )

    @torch.no_grad()
    def forward(
        self,
        token_values: torch.Tensor,
        cache: list[LayerCache],
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            token_values: Input token IDs [batch, seq_len]
            cache: List of KV caches for each layer
            start_pos: Starting position for RoPE and cache
        
        Returns:
            Logits tensor [batch, seq_len, vocab_size]
        """
        batch, seq_len = token_values.shape
        
        h = self.tok_embeddings(token_values)
        
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len]

        for i, layer in enumerate(self.layers):
            h = layer(h, freqs_cis, cache[i], start_pos)

        logits = self.output(self.norm(h))
        return logits.float()


#==============================================================================
# Cache Utilities
#==============================================================================

def make_cache(
    args: ModelArgs,
    length: int,
    device: Optional[Union[str, torch.device]] = None,
    n_layers: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> list[LayerCache]:
    """
    Allocate KV cache for the transformer.

    Args:
        args: Model configuration
        length: Cache length (max_batch * max_seq)
        device: Device for cache tensors
        n_layers: Number of layers (defaults to model setting)
        dtype: Data type for cache

    Returns:
        List of (k_cache, v_cache) tuples for each layer
    """
    if device is None:
        device = "cpu"
    if dtype is None:
        dtype = torch.bfloat16

    head_dim = args.dim // args.n_heads
    n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads

    if n_layers is None:
        n_layers = args.n_layers

    shape = (1, length, n_kv_heads, head_dim)
    return [
        (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )
        for _ in range(n_layers)
    ]


def cache_prefix(cache: list[LayerCache], length: int) -> list[LayerCache]:
    """Take a prefix view of a larger cache."""
    if len(cache) > 0:
        assert cache[0][0].shape[1] >= length
    return [(ck[:, :length], cv[:, :length]) for ck, cv in cache]
