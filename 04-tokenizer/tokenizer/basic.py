"""Byte-level BPE trained directly on the UTF-8 bytes of the text."""

from __future__ import annotations

from typing import List

from .base import Tokenizer


class BasicTokenizer(Tokenizer):

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i

            ids = self.merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def encode(self, text: str) -> List[int]:
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        while len(ids) >= 2:
            stats = self.get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = self.merge(ids, pair, idx)
        return ids
