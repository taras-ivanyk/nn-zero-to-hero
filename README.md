# Neural Networks: Zero to Hero

From-scratch implementations of neural networks and language models in Python and PyTorch, from a scalar autograd engine to a GPT-2 (124M) reproduction, following Andrej Karpathy's [*Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html).

## Contents

| Folder | What is implemented | Key concepts | Status |
|---|---|---|---|
| [`01-micrograd`](01-micrograd/) | scalar autograd engine and MLP library | computation graphs, chain rule, reverse-mode autodiff | Done |
| [`02-makemore`](02-makemore/) | character-level models: bigram, MLP, BatchNorm MLP, manual backprop, WaveNet | embeddings, cross-entropy, initialisation, BatchNorm, hand-derived gradients, hierarchical context | Done |
| [`03-gpt`](03-gpt/) | character-level GPT on Tiny Shakespeare | causal self-attention, multi-head attention, residual blocks, LayerNorm, dropout | Done |
| [`04-tokenizer`](04-tokenizer/) | byte-level BPE, with and without the GPT-4 regex split | UTF-8 bytes, merge training, encode/decode, pre-tokenisation | Done |
| [`05-gpt2-124m`](05-gpt2-124m/) | GPT-2 small, FineWeb-Edu training, HellaSwag eval | weight tying, mixed precision, FlashAttention, gradient accumulation, DDP | In progress |

## Highlights

- Backpropagation from scratch on scalars (`Value.backward()` over a topological sort), checked against PyTorch.
- Manual backward pass through cross-entropy, BatchNorm, tanh, linear and embedding layers, checked tensor by tensor against autograd.
- WaveNet-style hierarchical model that fuses pairs of characters at each layer.
- Self-attention and Transformer blocks written out from the masked-averaging trick up to a 10.8M-parameter GPT.
- Byte-level BPE with the GPT-4 regex split pattern, with save/load and a round-trip test suite.
- GPT-2 (124M) training on FineWeb-Edu (in progress).

## Results

Numbers come from notebook outputs or from runs of the scripts in this repo. Anything not measured yet is marked TBD.

| Model | Dataset | Metric | Value |
|---|---|---|---|
| micrograd MLP(2, [16, 16, 1]) | two moons, 100 points | training accuracy | 100% (step 47) |
| Bigram (counts) | names, 32,033 words | train NLL | 2.4544 |
| MLP (part 2) | names, 80/10/10 split | train / val loss | 2.0690 / 2.1685 |
| MLP + BatchNorm (part 3) | names | train / val loss | 2.0666 / 2.1051 |
| MLP, manual backprop (part 4) | names | train / val loss | 2.0714 / 2.1095 |
| WaveNet-style (part 5), 76,963 params | names | train / val loss | 1.7669 / 1.9845 |
| Char-level GPT (notebook, 4 layers, 64-d, 0.21M params) | Tiny Shakespeare | train / val loss | 1.6635 / 1.8226 |
| Char-level GPT (`gpt.py`, 10.8M params) | Tiny Shakespeare | train / val loss | TBD |
| BPE tokenizers, vocab 1000 | `taylorswift.txt` | compression (basic / regex) | 3.19x / 2.64x |
| GPT-2 124M (mine) | FineWeb-Edu | train / val loss | TBD |
| GPT-2 124M (mine) | HellaSwag (10,042 val) | acc_norm vs OpenAI GPT-2 124M | TBD / TBD |
| GPT-2 124M (mine) | | tokens trained, hardware, wall-clock time | TBD |

## Quickstart

```bash
git clone https://github.com/taras-ivanyk/nn-zero-to-hero.git
cd nn-zero-to-hero
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                     # 32 tests, a few seconds on CPU, no downloads

cd 03-gpt
python gpt.py --max-iters 500 --n-layer 2 --n-embd 64 --n-head 4 --block-size 32 --batch-size 16
```

Each folder's README has its own run commands. Datasets are downloaded into a git-ignored `data/` folder on first use.

## Repository structure

```
nn-zero-to-hero/
├── 01-micrograd/
│   ├── micrograd.ipynb
│   └── micrograd/            engine.py, nn.py, demo.py
├── 02-makemore/
│   ├── notebooks/            part1_bigram … part5_wavenet
│   └── makemore/             data.py, bigram.py, mlp.py, wavenet.py
├── 03-gpt/
│   ├── gpt_from_scratch.ipynb
│   ├── gpt.py
│   └── experiments/          v1_single_head.py, v2_transformer.py
├── 04-tokenizer/
│   ├── tokenizer/            base.py, basic.py, regex.py
│   ├── train_tokenizer.py
│   ├── tiktoken_demo.py
│   └── examples/             trained vocabularies
├── 05-gpt2-124m/             model.py, train_gpt2.py, prepare_fineweb.py, hellaswag.py (scaffolds)
├── tests/                    test_micrograd.py, test_tokenizer.py, test_gpt_shapes.py
├── requirements.txt
└── LICENSE
```

## Acknowledgements

This repository follows Andrej Karpathy's [Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) course. The code was written while working through the lectures, so much of it closely follows the lecture code; each folder's README says what is adapted and what I added. Related repositories: [micrograd](https://github.com/karpathy/micrograd), [makemore](https://github.com/karpathy/makemore), [nanoGPT](https://github.com/karpathy/nanoGPT), [ng-video-lecture](https://github.com/karpathy/ng-video-lecture), [minbpe](https://github.com/karpathy/minbpe), [build-nanogpt](https://github.com/karpathy/build-nanogpt).

## Author

Taras Ivanyk · [GitHub](https://github.com/taras-ivanyk) · LinkedIn: TBD
