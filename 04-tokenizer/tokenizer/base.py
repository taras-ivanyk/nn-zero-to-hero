"""Shared byte-level BPE machinery: pair statistics, merging, decode, save/load."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

Pair = Tuple[int, int]


class Tokenizer:
    """Base class. Subclasses implement `train` and `encode`."""

    pattern: str = ""

    def __init__(self):
        self.merges: Dict[Pair, int] = {}
        self.vocab: Dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}

    def merge(self, ids: List[int], pair: Pair, idx: int) -> List[int]:
        """Replace every occurrence of `pair` in `ids` with the new token `idx`."""
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def get_stats(self, ids: List[int]) -> Dict[Pair, int]:
        """Count consecutive pairs."""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")

    def _build_vocab(self) -> Dict[int, bytes]:
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    def save(self, path: Path | str) -> None:
        """Write the merges (and split pattern) as JSON; the vocab is rebuilt on load."""
        data = {
            "type": type(self).__name__,
            "pattern": self.pattern,
            "merges": [[p0, p1, idx] for (p0, p1), idx in sorted(self.merges.items(), key=lambda kv: kv[1])],
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    def load(self, path: Path | str) -> "Tokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["type"] != type(self).__name__:
            raise ValueError(f"{path} holds a {data['type']}, not a {type(self).__name__}")
        self.pattern = data["pattern"]
        self.merges = {(p0, p1): idx for p0, p1, idx in data["merges"]}
        self.vocab = self._build_vocab()
        return self
