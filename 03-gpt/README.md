# 03 · GPT from scratch

A character-level decoder-only Transformer trained on Tiny Shakespeare, following Karpathy's [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) lecture.

- `gpt_from_scratch.ipynb`: based on Karpathy's companion notebook for the lecture: bigram baseline, the masked-averaging trick behind self-attention, single head, scaled dot-product attention, LayerNorm, full model.
- `gpt.py`: clean script with CLI arguments: multi-head causal self-attention, ReLU feed-forward, pre-LayerNorm residual blocks, dropout, AdamW, train/val loss estimation, checkpointing and sampling.
- `experiments/v1_single_head.py`, `experiments/v2_transformer.py`: my earlier script versions, kept unchanged to show the progression. v1 has one attention head (and computes `k @ q.T` instead of `q @ k.T`); v2 is the full Transformer but does not yet apply the final LayerNorm or the dropout after the attention projection. Both are fixed in `gpt.py`.

The scripts follow the structure of Karpathy's `gpt.py` from [ng-video-lecture](https://github.com/karpathy/ng-video-lecture). Differences in `gpt.py`: no global hyperparameters, a `GPTConfig` dataclass, device auto-detection, checkpoint save/load with a `--sample-only` mode, and no custom weight initialisation.

## Run

```bash
cd 03-gpt
python gpt.py                       # default config: 6 layers, 6 heads, 384-d, block 256, 10.79M parameters
python gpt.py --max-iters 500 --n-layer 2 --n-embd 64 --n-head 4 --block-size 32 --batch-size 16   # quick run
python gpt.py --sample-only --checkpoint checkpoints/gpt.pt
```

Tiny Shakespeare is downloaded to `03-gpt/data/input.txt` on first use.

## Results

| Model | Train loss | Val loss | Hardware | Time |
|---|---|---|---|---|
| `gpt.py` default config, 5,000 iterations | TBD | TBD | TBD | TBD |

## What I learned

TBD
