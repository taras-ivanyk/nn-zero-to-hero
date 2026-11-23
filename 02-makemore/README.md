# 02 · makemore

Character-level language models trained on 32,033 names, from a bigram count table to a WaveNet-style hierarchical network, following parts 1–5 of Karpathy's makemore lectures.

| Notebook | Topic | Lecture |
|---|---|---|
| `notebooks/part1_bigram.ipynb` | bigram counts, sampling, NLL, the same model as a one-layer net | [video](https://www.youtube.com/watch?v=PaCmpygFfXo) |
| `notebooks/part2_mlp.ipynb` | MLP with learned embeddings (Bengio et al. 2003), LR search, splits | [video](https://youtu.be/TCH_1BHY58I) |
| `notebooks/part3_batchnorm.ipynb` | activation/gradient statistics, Kaiming init, BatchNorm from scratch | [video](https://youtu.be/P6sfmUTpUmc) |
| `notebooks/part4_backprop_ninja.ipynb` | manual backward pass through cross-entropy, BatchNorm, tanh, linear and embedding layers | [video](https://youtu.be/q8SA3rM6ckI) |
| `notebooks/part5_wavenet.ipynb` | 8-character context, hierarchical `FlattenConsecutive` model | [video](https://youtu.be/t3YJ5hKiMQ0) |

Parts 1–3 and 5 were written following the lectures. Part 4 is Karpathy's exercise notebook with the gradient derivations filled in by me.

The `makemore/` package holds clean versions of the final models: `bigram.py`, `mlp.py`, `wavenet.py`, plus `data.py` (download, vocabulary, datasets, 80/10/10 split with `random.seed(42)`).

## Run

```bash
cd 02-makemore
python -m makemore.bigram --model counts
python -m makemore.mlp --steps 200000
python -m makemore.wavenet --steps 200000
```

`names.txt` is downloaded to `02-makemore/data/` on first use. Device is picked automatically (`cuda`, then `mps`, then `cpu`); override with `--device`.

## Results

All numbers are from the notebook outputs (loss = mean cross-entropy, nats per character).

| Model | Train | Val |
|---|---|---|
| Bigram counts, add-one smoothing (part 1) | 2.4544 | not split |
| MLP, 3-char context, 10-d embeddings, 200 hidden, 11,897 parameters (part 2) | 2.0690 | 2.1685 |
| MLP with BatchNorm (part 3) | 2.0666 | 2.1051 |
| MLP trained with my manual backward pass (part 4) | 2.0714 | 2.1095 |
| WaveNet-style, 8-char context, n_embd 24, n_hidden 128, 76,963 parameters (part 5) | 1.7669 | 1.9845 |

Part 2 also contains my batch-size experiment (train/dev loss and training time for batch sizes 32 to 1024), and parts 3 and 5 contain my loss logs after each change.

## What I learned

- BatchNorm acts as a regulariser: in part 3 the training loss went up slightly after adding it while the validation loss stayed about the same.
- In part 2, larger batches gave a lower loss for the same number of steps but cost more time per run (8.4 s at batch 32 vs 65.5 s at batch 1024).
- Initialisation matters a lot. In part 3, fixing the saturated tanh layer alone took val loss from 3.30 to 2.11, and Kaiming-style scaling brought it to 2.10. After the fixes, all five tanh layers had only about 2–4% saturated units and a steady std of about 0.65 through the depth.
- The hardest gradients in part 4 were BatchNorm and cross-entropy. Both look simple in the forward pass but have many paths in the backward pass. The analytic form at the end is much shorter than the step-by-step version.
- From the batch-size experiment I would pick 256: it reached 2.303 dev loss in 21 s, while 1024 was only 0.008 better and took 3x longer.
- In part 5, more context helped more than more structure: going from 3 to 8 characters dropped val loss from 2.105 to 2.034, while the hierarchical layout alone only got to 2.029.
- A BatchNorm bug with 3-D inputs trained without any error and only showed up in the loss (2.029 → 2.022 after the fix). Some bugs only show up in the numbers.
