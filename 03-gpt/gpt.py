"""Character-level GPT trained on Tiny Shakespeare.

Decoder-only Transformer: token + learned position embeddings, pre-LayerNorm
blocks of multi-head causal self-attention and a ReLU feed-forward layer, a
final LayerNorm and a linear head. Run from `03-gpt/`:

    python gpt.py                                   # train with the default 10.8M-parameter config
    python gpt.py --max-iters 500 --n-layer 2       # quick run
    python gpt.py --sample-only --checkpoint checkpoints/gpt.pt
"""

from __future__ import annotations

import argparse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List

import torch
import torch.nn as nn
from torch.nn import functional as F

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, config: GPTConfig, head_size: int):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = (q @ k.transpose(-2, -1)) * k.shape[-1] ** -0.5  # (B, T, T), scaled by 1/sqrt(head_size)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication (attention) followed by computation (MLP)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(config, config.n_head, head_size)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(idx.shape[1], device=idx.device))  # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class CharTokenizer:
    def __init__(self, chars: List[str]):
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(list(set(text))))

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_text(path: Path) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    return path.read_text(encoding="utf-8")


def make_batcher(train_data: torch.Tensor, val_data: torch.Tensor, block_size: int, batch_size: int,
                 device: str) -> Callable[[str], tuple]:
    def get_batch(split: str):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    return get_batch


@torch.no_grad()
def estimate_loss(model: GPT, get_batch, eval_iters: int) -> Dict[str, float]:
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(path: Path, model: GPT, optimizer, tokenizer: CharTokenizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "chars": tokenizer.chars,
        "config": asdict(model.config),
    }, path)
    print(f"saved {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/input.txt")
    p.add_argument("--checkpoint", default="checkpoints/gpt.pt")
    p.add_argument("--sample-only", action="store_true", help="load --checkpoint and generate text")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-iters", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default=None, help="cuda, mps or cpu (default: auto-detect)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or get_device()
    torch.manual_seed(args.seed)

    if args.sample_only:
        ckpt = torch.load(args.checkpoint, map_location=device)
        tokenizer = CharTokenizer(ckpt["chars"])
        model = GPT(GPTConfig(**ckpt["config"])).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        text = load_text(Path(args.data))
        tokenizer = CharTokenizer.from_text(text)
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        get_batch = make_batcher(data[:n], data[n:], args.block_size, args.batch_size, device)

        config = GPTConfig(len(tokenizer.chars), args.block_size, args.n_embd, args.n_head, args.n_layer, args.dropout)
        model = GPT(config).to(device)
        print(f"device {device}, {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

        try:
            for it in range(args.max_iters):
                if it % args.eval_interval == 0 or it == args.max_iters - 1:
                    losses = estimate_loss(model, get_batch, args.eval_iters)
                    print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                xb, yb = get_batch("train")
                _, loss = model(xb, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        finally:
            save_checkpoint(Path(args.checkpoint), model, optimizer, tokenizer)

    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tokenizer.decode(model.generate(context, max_new_tokens=args.max_new_tokens)[0].tolist()))


if __name__ == "__main__":
    main()
