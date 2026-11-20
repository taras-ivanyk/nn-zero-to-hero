"""Scalar-valued autograd engine.

Each `Value` stores a number, its gradient, and the operation that produced it.
Calling `backward()` on the output runs reverse-mode differentiation over the
graph in topological order.
"""

from __future__ import annotations

import math
from typing import Callable, Union

Number = Union[int, float]


class Value:
    """A scalar node in a computation graph."""

    def __init__(self, data: Number, _children: tuple = (), _op: str = "", label: str = ""):
        self.data = data
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other: Union["Value", Number]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Number) -> "Value":
        return self + other

    def __mul__(self, other: Union["Value", Number]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Number) -> "Value":
        return self * other

    def tanh(self) -> "Value":
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other: Number) -> "Value":
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other: Union["Value", Number]) -> "Value":
        return self * other**-1

    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other: Union["Value", Number]) -> "Value":
        return self + (-other)

    def backward(self) -> None:
        """Backpropagate from this node, accumulating `.grad` on every ancestor."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
