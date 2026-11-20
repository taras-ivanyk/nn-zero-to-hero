"""MLP character-level language model (makemore part 2, after Bengio et al. 2003).

Run from `02-makemore/`:

    python -m makemore.mlp --steps 200000
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import torch
import torch.nn.functional as F

from .data import build_dataset, build_vocab, get_device, load_words, split_words

SEED = 2147483647


class MLP:
    """Embedding table -> concatenated context -> tanh hidden layer -> logits."""

    def __init__(self, vocab_size: int = 27, block_size: int = 3, n_embd: int = 10, n_hidden: int = 200,
                 generator: torch.Generator | None = None, device: str = "cpu"):
        g = generator
        self.block_size = block_size
        self.n_embd = n_embd
        self.C = torch.randn((vocab_size, n_embd), generator=g)
        self.W1 = torch.randn((block_size * n_embd, n_hidden), generator=g)
        self.b1 = torch.randn(n_hidden, generator=g)
        self.W2 = torch.randn((n_hidden, vocab_size), generator=g)
        self.b2 = torch.randn(vocab_size, generator=g)
        self.parameters = [p.to(device) for p in (self.C, self.W1, self.b1, self.W2, self.b2)]
        self.C, self.W1, self.b1, self.W2, self.b2 = self.parameters
        for p in self.parameters:
            p.requires_grad = True

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        emb = self.C[X]
        h = torch.tanh(emb.view(-1, self.block_size * self.n_embd) @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def num_params(self) -> int:
        return sum(p.nelement() for p in self.parameters)


def train(model: MLP, Xtr: torch.Tensor, Ytr: torch.Tensor, steps: int = 200000, batch_size: int = 32,
          lr: float = 0.1, lr_decay_step: int = 100000, log_every: int = 10000) -> List[float]:
    """Minibatch SGD; the learning rate drops 10x after `lr_decay_step` steps."""
    lossi = []
    for i in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), device=Xtr.device)
        loss = F.cross_entropy(model(Xtr[ix]), Ytr[ix])

        for p in model.parameters:
            p.grad = None
        loss.backward()

        step_lr = lr if i < lr_decay_step else lr * 0.1
        for p in model.parameters:
            p.data += -step_lr * p.grad

        if log_every and i % log_every == 0:
            print(f"{i:7d}/{steps:7d}: {loss.item():.4f}")
        lossi.append(loss.log10().item())
    return lossi


@torch.no_grad()
def evaluate(model: MLP, X: torch.Tensor, Y: torch.Tensor) -> float:
    return F.cross_entropy(model(X), Y).item()


@torch.no_grad()
def sample(model: MLP, itos: Dict[int, str], num_samples: int, seed: int = SEED + 10) -> List[str]:
    device = model.C.device
    g = torch.Generator(device=device).manual_seed(seed)
    names = []
    for _ in range(num_samples):
        out = []
        context = [0] * model.block_size
        while True:
            probs = F.softmax(model(torch.tensor([context], device=device)), dim=1)
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
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--n-embd", type=int, default=10)
    parser.add_argument("--n-hidden", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: auto-detect)")
    args = parser.parse_args()

    device = args.device or get_device()
    torch.manual_seed(args.seed)

    words = load_words(args.data)
    stoi, itos = build_vocab(words)
    train_w, dev_w, _ = split_words(words)
    Xtr, Ytr = (t.to(device) for t in build_dataset(train_w, stoi, args.block_size))
    Xdev, Ydev = (t.to(device) for t in build_dataset(dev_w, stoi, args.block_size))

    g = torch.Generator().manual_seed(args.seed)
    model = MLP(len(itos), args.block_size, args.n_embd, args.n_hidden, generator=g, device=device)
    print(f"device {device}, parameters {model.num_params()}")

    train(model, Xtr, Ytr, steps=args.steps, batch_size=args.batch_size, lr=args.lr, lr_decay_step=args.steps // 2)
    print(f"train {evaluate(model, Xtr, Ytr):.4f}")
    print(f"dev   {evaluate(model, Xdev, Ydev):.4f}")
    for name in sample(model, itos, args.num_samples):
        print(name)


if __name__ == "__main__":
    main()
