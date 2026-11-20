"""Bigram character-level language models (makemore part 1).

Two equivalent models: a smoothed count table, and a single linear layer on
one-hot inputs trained by gradient descent. Run from `02-makemore/`:

    python -m makemore.bigram --model counts
    python -m makemore.bigram --model neural --steps 50
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .data import build_vocab, load_words

SEED = 2147483647


def bigram_pairs(words: List[str], stoi: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    return torch.tensor(xs), torch.tensor(ys)


class BigramCounts:
    """P(next | current) from bigram counts with add-`smoothing` smoothing."""

    def __init__(self, vocab_size: int = 27, smoothing: int = 1):
        self.N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
        self.smoothing = smoothing
        self.P = None

    def fit(self, xs: torch.Tensor, ys: torch.Tensor) -> "BigramCounts":
        for x1, x2 in zip(xs.tolist(), ys.tolist()):
            self.N[x1, x2] += 1
        P = (self.N + self.smoothing).float()
        self.P = P / P.sum(1, keepdim=True)
        return self

    def nll(self, xs: torch.Tensor, ys: torch.Tensor) -> float:
        """Average negative log-likelihood of the pairs."""
        return -self.P[xs, ys].log().mean().item()

    def next_probs(self, ix: int) -> torch.Tensor:
        return self.P[ix]


class NeuralBigram:
    """One linear layer (27x27 weights) on one-hot inputs, softmax output."""

    def __init__(self, vocab_size: int = 27, generator: torch.Generator | None = None):
        self.vocab_size = vocab_size
        self.W = torch.randn((vocab_size, vocab_size), generator=generator, requires_grad=True)

    def probs(self, xs: torch.Tensor) -> torch.Tensor:
        xenc = F.one_hot(xs, num_classes=self.vocab_size).float()
        logits = xenc @ self.W
        counts = logits.exp()
        return counts / counts.sum(1, keepdims=True)

    def loss(self, xs: torch.Tensor, ys: torch.Tensor, reg: float = 0.001) -> torch.Tensor:
        probs = self.probs(xs)
        return -probs[torch.arange(xs.nelement()), ys].log().mean() + reg * (self.W**2).mean()

    def train(self, xs: torch.Tensor, ys: torch.Tensor, steps: int = 50, lr: float = 50.0,
              reg: float = 0.001, verbose: bool = True) -> float:
        loss = None
        for _ in range(steps):
            loss = self.loss(xs, ys, reg)
            if verbose:
                print(loss.item())
            self.W.grad = None
            loss.backward()
            self.W.data += -lr * self.W.grad
        return loss.item()

    @torch.no_grad()
    def next_probs(self, ix: int) -> torch.Tensor:
        return self.probs(torch.tensor([ix]))[0]


def sample(model, itos: Dict[int, str], num_samples: int, generator: torch.Generator) -> List[str]:
    names = []
    for _ in range(num_samples):
        out = []
        ix = 0
        while True:
            p = model.next_probs(ix)
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=generator).item()
            out.append(itos[ix])
            if ix == 0:
                break
        names.append("".join(out))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["counts", "neural"], default="counts")
    parser.add_argument("--data", default="data/names.txt")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=50.0)
    parser.add_argument("--reg", type=float, default=0.001)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    words = load_words(args.data)
    stoi, itos = build_vocab(words)
    xs, ys = bigram_pairs(words, stoi)
    print(f"number of examples = {xs.nelement()}")

    if args.model == "counts":
        model = BigramCounts().fit(xs, ys)
        print(f"average nll: {model.nll(xs, ys):.4f}")
    else:
        g = torch.Generator().manual_seed(args.seed)
        model = NeuralBigram(generator=g)
        final = model.train(xs, ys, steps=args.steps, lr=args.lr, reg=args.reg, verbose=False)
        print(f"final loss (incl. regularisation): {final:.4f}")

    g = torch.Generator().manual_seed(args.seed)
    for name in sample(model, itos, args.num_samples, g):
        print(name)


if __name__ == "__main__":
    main()
