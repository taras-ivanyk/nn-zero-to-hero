"""Byte-level BPE with the GPT-4 regex split pattern.

The text is first split into chunks (words, numbers, punctuation, whitespace)
so merges never cross chunk boundaries.
"""

from __future__ import annotations

from typing import List

import regex

from .base import Tokenizer

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern: str = GPT4_SPLIT_PATTERN):
        super().__init__()
        self.pattern = pattern

    @property
    def pattern(self) -> str:
        return self._pattern

    @pattern.setter
    def pattern(self, value: str) -> None:
        self._pattern = value
        self.compiled_pattern = regex.compile(value)

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        num_merges = vocab_size - 256
        chunks = self.compiled_pattern.findall(text)
        ids = [list(chunk.encode("utf-8")) for chunk in chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = {}

            for chunk_ids in ids:
                for pair, count in self.get_stats(chunk_ids).items():
                    stats[pair] = stats.get(pair, 0) + count

            if not stats:
                break

            pair = max(stats, key=stats.get)
            idx = 256 + i

            ids = [self.merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def encode(self, text: str) -> List[int]:
        chunks = self.compiled_pattern.findall(text)
        result = []

        for chunk in chunks:
            ids = list(chunk.encode("utf-8"))

            while len(ids) >= 2:
                stats = self.get_stats(ids)
                pair = min(
                    stats,
                    key=lambda p: self.merges.get(p, float("inf"))
                )

                if pair not in self.merges:
                    break

                idx = self.merges[pair]
                ids = self.merge(ids, pair, idx)

            result.extend(ids)

        return result
