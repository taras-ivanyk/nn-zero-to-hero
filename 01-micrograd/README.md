# 01 · micrograd

A scalar-valued autograd engine and a small neural network library on top of it, following Karpathy's [micrograd lecture](https://www.youtube.com/watch?v=VMj-3S1tku0).

- `micrograd.ipynb`: building `Value` step by step, drawing computation graphs with graphviz, checking gradients against PyTorch, training an MLP on two moons.
- `micrograd/engine.py`: `Value` with `+ * ** / -`, `exp`, `tanh`, `relu` and `backward()` (reverse-mode autodiff over a topological sort).
- `micrograd/nn.py`: `Neuron`, `Layer`, `MLP`.
- `micrograd/demo.py`: two-moons training from the command line.

Written following the lecture; the `nn.py` classes and the two-moons training loop follow Karpathy's micrograd demo.

## Run

```bash
cd 01-micrograd
python -m micrograd.demo --steps 100
pytest ../tests/test_micrograd.py
```

## Results

| Task | Result |
|---|---|
| MLP(2, [16, 16, 1]) on 100 two-moons points, SVM max-margin loss | 100% training accuracy at step 47 (run stopped manually; see notebook) |

## What I learned

TBD
