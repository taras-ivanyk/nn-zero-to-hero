"""GPT-2 (124M) model.

Scaffold only: every method raises NotImplementedError until implemented.
Work through part A of the private guide (`instructions.md`) before touching
the training script.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # 50,000 BPE merges + 256 byte tokens + <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # TODO: one Linear producing q, k, v for all heads (c_attn), an output projection (c_proj).
        # Use the Hugging Face parameter names so from_pretrained can copy weights by name.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: (B, T, C) -> split heads -> causal attention -> merge heads -> c_proj
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # TODO: c_fc (n_embd -> 4 * n_embd), GELU with the tanh approximation, c_proj back to n_embd
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # TODO: ln_1, attn, ln_2, mlp (pre-LayerNorm residual block)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # TODO: transformer = ModuleDict(wte, wpe, h, ln_f); lm_head without bias, weight tied to wte
        raise NotImplementedError

    def _init_weights(self, module: nn.Module) -> None:
        # TODO: normal(0, 0.02); residual projections scaled by (2 * n_layer) ** -0.5; zero biases
        raise NotImplementedError

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Return (logits of shape (B, T, vocab_size), loss or None)."""
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, model_type: str = "gpt2") -> "GPT":
        """Load OpenAI weights from Hugging Face `transformers` (checkpoint 1 in the guide)."""
        # TODO: remember that HF stores Conv1D weights transposed relative to nn.Linear
        raise NotImplementedError

    def configure_optimizers(self, weight_decay: float, learning_rate: float, device_type: str):
        """AdamW with weight decay on 2-D parameters only."""
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, top_k: int = 50) -> torch.Tensor:
        raise NotImplementedError
