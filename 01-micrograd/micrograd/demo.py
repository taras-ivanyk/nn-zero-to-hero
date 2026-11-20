"""Train a micrograd MLP on the two-moons dataset.

Run from `01-micrograd/`:

    python -m micrograd.demo --steps 100
"""

from __future__ import annotations

import argparse
import random

import numpy as np

from .engine import Value
from .nn import MLP


def make_data(n_samples: int, noise: float, seed: int):
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    y = y * 2 - 1  # labels in {-1, 1}
    return X, y


def loss(model: MLP, X, y, batch_size: int | None = None, alpha: float = 1e-4):
    """SVM max-margin loss with L2 regularisation; returns (loss, accuracy)."""
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    inputs = [list(map(Value, xrow)) for xrow in Xb]
    scores = list(map(model, inputs))

    losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    reg_loss = alpha * sum((p * p for p in model.parameters()))
    total_loss = data_loss + reg_loss

    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)


def train(model: MLP, X, y, steps: int, log_every: int = 1) -> tuple[float, float]:
    total_loss, acc = None, 0.0
    for k in range(steps):
        total_loss, acc = loss(model, X, y)

        model.zero_grad()
        total_loss.backward()

        learning_rate = 1.0 - 0.9 * k / steps
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if log_every and k % log_every == 0:
            print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")
    return total_loss.data, acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, nargs="+", default=[16, 16])
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    X, y = make_data(args.n_samples, args.noise, args.seed)
    model = MLP(2, args.hidden + [1])
    print(model)
    print("number of parameters", len(model.parameters()))
    train(model, X, y, args.steps)


if __name__ == "__main__":
    main()
