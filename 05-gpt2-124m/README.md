# 05 · GPT-2 (124M) reproduction

**Status: in progress.** The files here are scaffolds (signatures and TODOs); the implementation and training run are not done yet. No results below have been measured.

Goal: implement GPT-2 small from scratch in PyTorch, verify it against the OpenAI weights, train it on the FineWeb-Edu 10B-token sample and evaluate it on HellaSwag, following Karpathy's [Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU) and [build-nanogpt](https://github.com/karpathy/build-nanogpt).

| File | Purpose |
|---|---|
| `model.py` | GPT-2 model, weight loading from Hugging Face, optimizer setup |
| `prepare_fineweb.py` | download and tokenize FineWeb-Edu into `uint16` shards of 100M tokens |
| `train_gpt2.py` | training loop: gradient accumulation, bf16, `torch.compile`, DDP via `torchrun` |
| `hellaswag.py` | completion-style HellaSwag evaluation (10,042 validation examples) |

## Results

| Metric | My model | Reference |
|---|---|---|
| Validation loss (FineWeb-Edu) | TBD | TBD |
| HellaSwag acc_norm | TBD | OpenAI GPT-2 124M: TBD (measure with `hellaswag.py`) |
| Tokens trained | TBD | |
| Hardware | TBD | |
| Wall-clock time | TBD | |
| Throughput (tokens/s), MFU | TBD | |
