"""HellaSwag evaluation, completion style.

Scaffold only. For each example the model scores the four endings by the
average token loss over the ending only; the lowest loss is the prediction.
Validate on OpenAI GPT-2 (124M) first, then run the same code on your model.

    python hellaswag.py --model gpt2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import torch

DATA_DIR = Path("data/hellaswag")


def download(split: str = "val") -> Path:
    raise NotImplementedError


def iterate_examples(split: str = "val") -> Iterator[dict]:
    raise NotImplementedError


def render_example(example: dict) -> tuple[dict, torch.Tensor, torch.Tensor, int]:
    """Return (data, tokens (4, T), mask (4, T) marking ending tokens, label)."""
    raise NotImplementedError


def get_most_likely_row(tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor) -> int:
    """Index of the ending with the lowest average loss over masked positions."""
    raise NotImplementedError


@torch.no_grad()
def evaluate(model, device: str, split: str = "val", limit: int | None = None) -> dict:
    """Return {"num_total", "acc", "acc_norm"}."""
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt2", help="Hugging Face model id or path to your checkpoint")
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N examples")
    args = p.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
