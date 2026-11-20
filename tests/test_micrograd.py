import random

import pytest
import torch

from micrograd import MLP, Value


def test_neuron_matches_pytorch():
    x1, x2 = Value(2.0), Value(0.0)
    w1, w2 = Value(-3.0), Value(1.0)
    b = Value(6.8813735870195432)
    o = (x1 * w1 + x2 * w2 + b).tanh()
    o.backward()

    tx1, tx2, tw1, tw2, tb = (torch.tensor([v], dtype=torch.double, requires_grad=True)
                              for v in (2.0, 0.0, -3.0, 1.0, 6.8813735870195432))
    to = torch.tanh(tx1 * tw1 + tx2 * tw2 + tb)
    to.backward()

    assert o.data == pytest.approx(to.item())
    for v, t in [(x1, tx1), (x2, tx2), (w1, tw1), (w2, tw2), (b, tb)]:
        assert v.grad == pytest.approx(t.grad.item())


def test_composite_expression_matches_pytorch():
    a, b = Value(-4.0), Value(2.0)
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + f**-1 * 10.0
    h = (g * 0.01).exp() + (g * 0.001).tanh()
    h.backward()

    ta = torch.tensor(-4.0, dtype=torch.double, requires_grad=True)
    tb = torch.tensor(2.0, dtype=torch.double, requires_grad=True)
    tc = ta + tb
    td = ta * tb + tb**3
    tc = tc + tc + 1
    tc = tc + 1 + tc + (-ta)
    td = td + td * 2 + (tb + ta).relu()
    td = td + 3 * td + (tb - ta).relu()
    te = tc - td
    tf = te**2
    tg = tf / 2.0
    tg = tg + 10.0 / tf
    th = (tg * 0.01).exp() + (tg * 0.001).tanh()
    th.backward()

    assert h.data == pytest.approx(th.item(), rel=1e-9)
    assert a.grad == pytest.approx(ta.grad.item(), rel=1e-9)
    assert b.grad == pytest.approx(tb.grad.item(), rel=1e-9)


def test_gradient_accumulates_when_node_is_reused():
    a = Value(3.0)
    b = a + a
    b.backward()
    assert a.grad == 2.0


def test_mlp_gradients_match_pytorch():
    random.seed(0)
    model = MLP(3, [4, 1])
    x = [0.5, -1.0, 2.0]
    out = model(x)
    out.backward()

    l1, l2 = model.layers
    W1 = torch.tensor([[w.data for w in n.w] for n in l1.neurons], dtype=torch.double, requires_grad=True)
    b1 = torch.tensor([n.b.data for n in l1.neurons], dtype=torch.double, requires_grad=True)
    W2 = torch.tensor([[w.data for w in n.w] for n in l2.neurons], dtype=torch.double, requires_grad=True)
    b2 = torch.tensor([n.b.data for n in l2.neurons], dtype=torch.double, requires_grad=True)
    tx = torch.tensor(x, dtype=torch.double)
    tout = (torch.relu(W1 @ tx + b1) @ W2.T + b2).sum()
    tout.backward()

    assert out.data == pytest.approx(tout.item())
    for i, n in enumerate(l1.neurons):
        assert [w.grad for w in n.w] == pytest.approx(W1.grad[i].tolist())
        assert n.b.grad == pytest.approx(b1.grad[i].item())
    assert [w.grad for w in l2.neurons[0].w] == pytest.approx(W2.grad[0].tolist())


def test_zero_grad():
    model = MLP(2, [3, 1])
    model([1.0, 2.0]).backward()
    model.zero_grad()
    assert all(p.grad == 0 for p in model.parameters())
