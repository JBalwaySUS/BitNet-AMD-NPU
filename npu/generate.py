# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
#
# AMD NPU text generation for BitNet.

import os
import readline  # noqa: enables input history
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import fire
import model as fast
import torch
from stats import Stats
from tokenizer import Tokenizer, ChatFormat
import sample_utils


@dataclass
class GenArgs:
    """Generation arguments."""
    gen_length: int = 128
    gen_bsz: int = 1
    prompt_length: int = 64
    
    use_sampling: bool = False
    temperature: float = 0.7
    top_p: float = 0.95


class NPUGenerator:
    """
    Text generator using BitNet on AMD NPU.
    
    Manages model loading, KV cache, and token generation.
    """
    tokenizer: Tokenizer

    @staticmethod
    def build(
        ckpt_dir: str,
        gen_args: GenArgs,
        tokenizer_path: Optional[str] = None,
        xclbin_dir: str = "build",
    ) -> "NPUGenerator":
        """
        Load model checkpoint and build generator.
        
        Args:
            ckpt_dir: Directory containing model checkpoints
            gen_args: Generation arguments
            tokenizer_path: Path to tokenizer model
            xclbin_dir: Directory containing compiled NPU kernels
        """
        start_time = time.time()

        # Model args for prefill (no kernel) and decode (with kernel)
        model_args_prefill = fast.ModelArgs(
            use_kernel=False,
        )
        model_args_decode = fast.ModelArgs(
            use_kernel=True,
            xclbin_dir=xclbin_dir,
        )
        
        # Load tokenizer
        if tokenizer_path is None:
            tokenizer_path = str(Path(__file__).parent / "tokenizer.model")
        tokenizer = Tokenizer(tokenizer_path)

        # Set dtype
        torch.set_default_dtype(torch.bfloat16)

        # Build models
        prefill_model = fast.Transformer(model_args_prefill)
        decode_model = fast.Transformer(model_args_decode)

        # Load checkpoints
        ckpt_path = Path(ckpt_dir)
        
        fp16_ckpt_path = ckpt_path / "model_state_fp16.pt"
        int2_ckpt_path = ckpt_path / "model_state_int2.pt"
        
        if fp16_ckpt_path.exists():
            fp16_checkpoint = torch.load(str(fp16_ckpt_path), map_location="cpu")
            prefill_model.load_state_dict(fp16_checkpoint, strict=False)
            print(f"Loaded prefill model from {fp16_ckpt_path}")
        else:
            print(f"Warning: fp16 checkpoint not found at {fp16_ckpt_path}")
        
        if int2_ckpt_path.exists():
            int2_checkpoint = torch.load(str(int2_ckpt_path), map_location="cpu")
            decode_model.load_state_dict(int2_checkpoint, strict=False)
            print(f"Loaded decode model from {int2_ckpt_path}")
        else:
            print(f"Warning: int2 checkpoint not found at {int2_ckpt_path}")

        prefill_model.eval()
        decode_model.eval()
        
        print(f"Loaded model in {time.time() - start_time:.2f} seconds")

        return NPUGenerator(
            gen_args,
            model_args_decode,
            prefill_model,
            decode_model,
            tokenizer,
        )

    def __init__(
        self,
        args: GenArgs,
        model_args: fast.ModelArgs,
        prefill_model: fast.Transformer,
        decode_model: fast.Transformer,
        tokenizer: Tokenizer,
    ):
        self.gen_args = args
        self.max_seq_length = args.prompt_length + args.gen_length
        self.model_args = model_args
        self.prefill_model = prefill_model
        self.decode_model = decode_model
        self.tokenizer = tokenizer
        self._cache = None

    def _init_cache(self):
        """Initialize KV cache."""
        if self._cache is None:
            self._cache = fast.make_cache(
                args=self.model_args,
                length=self.gen_args.gen_bsz * self.max_seq_length,
                device="cpu",
            )
        return self._cache

    @torch.inference_mode()
    def generate_all(
        self, 
        prompts: List[List[int]], 
        use_sampling: bool,
    ) -> Tuple[Stats, List[List[int]]]:
        """
        Generate text for a batch of prompts.
        
        Args:
            prompts: List of token ID lists
            use_sampling: Use sampling instead of greedy decoding
        
        Returns:
            Tuple of (stats, generated_token_lists)
        """
        bs = len(prompts)
        prompt_lens = [len(p) for p in prompts]
        max_prompt_length = max(prompt_lens)
        gen_length = self.gen_args.gen_length
        max_seq_length = max_prompt_length + gen_length
        
        print(f"Prompt length: {max_prompt_length}, Gen length: {gen_length}")

        # Initialize cache
        cache = self._init_cache()

        # Pad prompts to same length
        padded_prompts = [
            prompt + [1] * (self.gen_args.prompt_length - len(prompt)) 
            for prompt in prompts
        ]
        tokens = torch.tensor(sum(padded_prompts, []), dtype=torch.int).reshape(bs, -1)
        out_tokens = torch.zeros((gen_length, bs), dtype=torch.int)

        stats = Stats()
        stats.phase("prefill")

        # Prefill
        output = self.prefill_model(tokens, cache, start_pos=0)
        
        # Get logits for last token of each sequence
        logits = torch.stack([output[i, prompt_lens[i] - 1, :] for i in range(bs)])
        
        # Sample first token
        if use_sampling:
            probs = torch.softmax(logits / self.gen_args.temperature, dim=-1)
            next_token = sample_utils.top_p(probs, self.gen_args.top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)

        next_token = next_token.reshape(bs)
        out_tokens[0, :] = next_token

        stats.phase("decode")

        # Decode loop
        eos_id = self.tokenizer.eot_id
        kv_seqlen = torch.tensor(prompt_lens, dtype=torch.int)
        
        for niter in range(1, gen_length):
            kv_seqlen = kv_seqlen + 1
            
            # Single token forward
            output = self.decode_model(
                next_token.unsqueeze(1), 
                cache, 
                start_pos=kv_seqlen[0].item() - 1,
            )

            logits = output[:, 0, :]

            if use_sampling:
                probs = torch.softmax(logits / self.gen_args.temperature, dim=-1)
                next_token = sample_utils.top_p(probs, self.gen_args.top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)

            next_token = next_token.reshape(bs)
            out_tokens[niter, :] = next_token

            if next_token.eq(eos_id).any():
                break

        stats.end_phase(tokens=niter * bs)

        def trim_answer(prompt_len, tokens):
            """Trim answer at EOS token."""
            tokens = tokens[:gen_length]
            eos_id = self.tokenizer.eot_id
            if eos_id in tokens:
                return tokens[:tokens.index(eos_id) + 1]
            return tokens

        answers = [
            trim_answer(prompt_len, answer)
            for prompt_len, answer in zip(prompt_lens, out_tokens.t().tolist())
        ]
        return stats, answers


def get_prompts(interactive: bool) -> Iterable[List[str]]:
    """Get prompts from user or use defaults."""
    if interactive:
        while True:
            try:
                prompts = input("Enter prompt: ").split("\n")
            except EOFError:
                print("Exiting")
                sys.exit(0)
            yield prompts
    else:
        yield ["Hello, my name is"]


def main(
    ckpt_dir: str, 
    interactive: bool = False, 
    chat_format: bool = False, 
    sampling: bool = False,
    xclbin_dir: str = "build",
    gen_length: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.95,
):
    """
    BitNet text generation on AMD NPU.
    
    Args:
        ckpt_dir: Directory containing model checkpoints
        interactive: Enable interactive prompt mode
        chat_format: Use chat template formatting
        sampling: Use sampling instead of greedy decoding
        xclbin_dir: Directory containing compiled NPU kernels
        gen_length: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling probability
    """
    gen_args = GenArgs(
        gen_length=gen_length,
        use_sampling=sampling,
        temperature=temperature,
        top_p=top_p,
    )
    
    g = NPUGenerator.build(
        ckpt_dir,
        gen_args,
        xclbin_dir=xclbin_dir,
    )

    if chat_format:
        g.tokenizer = ChatFormat(g.tokenizer)

    for prompts in get_prompts(interactive):
        if chat_format:
            tokens = [
                g.tokenizer.encode_dialog_prompt(
                    dialog=[{"role": "user", "content": prompt}],
                    completion=True
                )
                for prompt in prompts
            ]
        else:
            tokens = [g.tokenizer.encode(x, bos=False, eos=False) for x in prompts]

        print(f"Token IDs: {tokens}")
        
        stats, out_tokens = g.generate_all(tokens, use_sampling=sampling)

        for i, prompt in enumerate(prompts):
            print(f"\n> {prompt}")
            answer = g.tokenizer.decode(out_tokens[i])
            print(f"< {answer}")
            print("-" * 40)

        for phase_stats in stats.phases:
            print(phase_stats.show())


if __name__ == "__main__":
    fire.Fire(main)
