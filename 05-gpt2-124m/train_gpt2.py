"""Train GPT-2 (124M).

Current stage: single process, tiny Shakespeare, constant LR.
    python train_gpt2.py --micro-batch-size 4 --seq-len 32 --max-steps 50

Planned: FineWeb-Edu shards, multi-GPU via
    torchrun --standalone --nproc_per_node=8 train_gpt2.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tiktoken
import torch

from model import GPT, GPTConfig


def load_tokens(path: Path) -> torch.Tensor:
    """Load a uint16 .npy shard as a long tensor."""
    raise NotImplementedError


class DataLoaderLite:
    """Serves consecutive (B, T) batches from one text file, wrapping around at the end.

    Later this will read FineWeb-Edu shards and give each DDP rank its own slice.
    """

    def __init__(self, B: int, T: int, input_file: Path):
        self.B, self.T = B, T
        enc = tiktoken.get_encoding("gpt2")
        self.tokens = torch.tensor(enc.encode(input_file.read_text()), dtype=torch.long)
        print(f"loaded {len(self.tokens):,} tokens, 1 epoch = {len(self.tokens) // (B * T)} batches")
        self.reset()

    def reset(self) -> None:
        self.current_position = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)  # targets: the next token for every position
        self.current_position += B * T
        # if the next batch would run past the end, start over
        if self.current_position + B * T + 1 > len(self.tokens):
            self.reset()
        return x, y


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
    p.add_argument("--input-file", type=Path, default=Path("data/input.txt"), help="text file to train on (tiny Shakespeare)")
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
    # TODO (next steps): DDP setup, TF32, LR schedule, gradient accumulation,
    # bf16 autocast, grad clipping, val loss / HellaSwag / samples / checkpoints.

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")
    torch.manual_seed(args.seed)

    train_loader = DataLoaderLite(B=args.micro_batch_size, T=args.seq_len, input_file=args.input_file)

    model = GPT(GPTConfig())  # random init, 124M parameters
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr)
    for step in range(args.max_steps):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()  # wait for the GPU so the timing is real
        dt = time.time() - t0
        tokens_per_sec = x.numel() / dt
        print(f"step {step:4d} | loss {loss.item():.4f} | dt {dt * 1000:.0f}ms | tok/s {tokens_per_sec:.0f}")


if __name__ == "__main__":
    main()
