"""WaveNet-style hierarchical character model (makemore part 5).

Layers follow the PyTorch `nn` API but are written from scratch. Run from
`02-makemore/`:

    python -m makemore.wavenet --steps 200000
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import torch
import torch.nn.functional as F

from .data import build_dataset, build_vocab, get_device, load_words, split_words

SEED = 2147483647


class Linear:
    def __init__(self, fan_in: int, fan_out: int, bias: bool = True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    """BatchNorm over dim 0 for 2-D inputs and dims (0, 1) for 3-D inputs."""

    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    def __init__(self, num_embeddings: int, embeddings_dim: int):
        self.weight = torch.randn((num_embeddings, embeddings_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]


class FlattenConsecutive:
    """(B, T, C) -> (B, T // n, C * n); squeezes the time dim when it reaches 1."""

    def __init__(self, n: int):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def train(self, mode: bool = True) -> None:
        for layer in self.layers:
            layer.training = mode


def build_model(vocab_size: int = 27, n_embd: int = 24, n_hidden: int = 128) -> Sequential:
    """Three FlattenConsecutive(2) stages, so it expects a context of 8 characters."""
    model = Sequential([
        Embedding(vocab_size, n_embd),
        FlattenConsecutive(2), Linear(n_embd * 2, n_hidden), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden), BatchNorm1d(n_hidden), Tanh(),
        FlattenConsecutive(2), Linear(n_hidden * 2, n_hidden), BatchNorm1d(n_hidden), Tanh(),
        Linear(n_hidden, vocab_size),
    ])
    with torch.no_grad():
        model.layers[-1].weight *= 0.1
    for p in model.parameters():
        p.requires_grad = True
    return model


def train(model: Sequential, Xtr: torch.Tensor, Ytr: torch.Tensor, steps: int = 200000, batch_size: int = 32,
          lr: float = 0.1, lr_decay_step: int = 150000, log_every: int = 10000) -> List[float]:
    parameters = model.parameters()
    lossi = []
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
        Xb, Yb = Xtr[ix], Ytr[ix]

        loss = F.cross_entropy(model(Xb), Yb)

        for p in parameters:
            p.grad = None
        loss.backward()

        step_lr = lr if i < lr_decay_step else lr * 0.1
        for p in parameters:
            p.data += -step_lr * p.grad

        if log_every and i % log_every == 0:
            print(f"{i:7d}/{steps:7d}: {loss.item():.4f}")
        lossi.append(loss.log10().item())
    return lossi


@torch.no_grad()
def evaluate(model: Sequential, X: torch.Tensor, Y: torch.Tensor) -> float:
    return F.cross_entropy(model(X), Y).item()


@torch.no_grad()
def sample(model: Sequential, itos: Dict[int, str], block_size: int, num_samples: int,
           seed: int = SEED + 10) -> List[str]:
    g = torch.Generator(device=torch.get_default_device()).manual_seed(seed)
    names = []
    for _ in range(num_samples):
        out = []
        context = [0] * block_size
        while True:
            probs = F.softmax(model(torch.tensor([context])), dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        names.append("".join(itos[i] for i in out))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/names.txt")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--n-embd", type=int, default=24)
    parser.add_argument("--n-hidden", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: auto-detect)")
    args = parser.parse_args()

    block_size = 8
    device = args.device or get_device()
    torch.set_default_device(device)

    words = load_words(args.data)
    stoi, itos = build_vocab(words)
    train_w, dev_w, _ = split_words(words)
    Xtr, Ytr = (t.to(device) for t in build_dataset(train_w, stoi, block_size))
    Xdev, Ydev = (t.to(device) for t in build_dataset(dev_w, stoi, block_size))

    torch.manual_seed(args.seed)
    model = build_model(len(itos), args.n_embd, args.n_hidden)
    print(f"device {device}, parameters {sum(p.nelement() for p in model.parameters())}")

    train(model, Xtr, Ytr, steps=args.steps, batch_size=args.batch_size, lr=args.lr,
          lr_decay_step=int(args.steps * 0.75))
    model.train(False)
    print(f"train {evaluate(model, Xtr, Ytr):.4f}")
    print(f"val   {evaluate(model, Xdev, Ydev):.4f}")
    for name in sample(model, itos, block_size, args.num_samples):
        print(name)


if __name__ == "__main__":
    main()
