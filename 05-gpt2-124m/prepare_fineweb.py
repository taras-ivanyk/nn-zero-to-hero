"""Download FineWeb-Edu (sample-10BT), tokenize with the GPT-2 encoding and write uint16 shards.

Scaffold only. Output: <out-dir>/edufineweb_{val,train}_000000.npy, ...
The first shard is the validation shard.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def tokenize(doc: dict) -> np.ndarray:
    """<|endoftext|> followed by the document's GPT-2 tokens, as uint16."""
    raise NotImplementedError


def write_shard(path: Path, tokens: np.ndarray) -> None:
    raise NotImplementedError


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("data/edu_fineweb10B"))
    p.add_argument("--remote-name", default="sample-10BT")
    p.add_argument("--shard-size", type=int, default=100_000_000, help="tokens per shard")
    p.add_argument("--num-proc", type=int, default=None, help="tokenizer worker processes")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: load_dataset("HuggingFaceFW/fineweb-edu", name=args.remote_name, split="train"),
    # tokenize in a multiprocessing pool, fill fixed-size buffers, write each full shard.
    raise NotImplementedError


if __name__ == "__main__":
    main()
