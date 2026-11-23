"""Train GPT-2 (124M) on FineWeb-Edu shards.

Scaffold only. Single GPU:   python train_gpt2.py
               Multi-GPU:    torchrun --standalone --nproc_per_node=8 train_gpt2.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def load_tokens(path: Path) -> torch.Tensor:
    """Load a uint16 .npy shard as a long tensor."""
    raise NotImplementedError


class DataLoaderLite:
    """Serves (B, T) batches from consecutive shards; each DDP rank reads its own slice."""

    def __init__(self, B: int, T: int, process_rank: int, num_processes: int, split: str, data_root: Path):
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


def setup_ddp() -> tuple[bool, int, int, int, str]:
    """Return (ddp, rank, local_rank, world_size, device); fall back to one process when not under torchrun."""
    raise NotImplementedError


def get_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Linear warmup, then cosine decay to min_lr."""
    raise NotImplementedError


@torch.no_grad()
def evaluate_val_loss(model, val_loader: DataLoaderLite, steps: int, device: str) -> float:
    raise NotImplementedError


def save_checkpoint(path: Path, model, optimizer, step: int, val_loss: float) -> None:
    raise NotImplementedError


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data/edu_fineweb10B"))
    p.add_argument("--log-dir", type=Path, default=Path("log"))
    p.add_argument("--total-batch-size", type=int, default=2**19, help="tokens per optimizer step")
    p.add_argument("--micro-batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-lr", type=float, default=6e-4)
    p.add_argument("--warmup-steps", type=int, default=715)
    p.add_argument("--max-steps", type=int, default=19073)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: DDP setup, seeds, TF32, data loaders, model, optimizer,
    # loop with gradient accumulation + bf16 autocast + grad clipping,
    # periodic val loss / HellaSwag / samples / checkpoints, tokens/sec logging.
    raise NotImplementedError


if __name__ == "__main__":
    main()
