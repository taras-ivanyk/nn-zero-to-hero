"""Dataset helpers shared by the makemore models."""

from __future__ import annotations

import random
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import torch

NAMES_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DEFAULT_PATH = Path("data/names.txt")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_words(path: Path | str = DEFAULT_PATH, download: bool = True) -> List[str]:
    """Read one name per line, downloading `names.txt` first if it is missing."""
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(NAMES_URL, path)
    return path.read_text(encoding="utf-8").splitlines()


def build_vocab(words: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Map characters to integers 1..26 and the boundary token '.' to 0."""
    chars = sorted(list(set("".join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def build_dataset(words: List[str], stoi: Dict[str, int], block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sliding-window (context -> next character) examples for every word."""
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


def split_words(words: List[str], seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """Shuffle with `random.seed(seed)` and split 80/10/10, as in the notebooks."""
    words = list(words)
    random.seed(seed)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    return words[:n1], words[n1:n2], words[n2:]
