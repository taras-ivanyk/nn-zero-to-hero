"""Train a BPE tokenizer on a text file, save it, reload it and check round-trips.

Run from `04-tokenizer/`:

    python train_tokenizer.py --type regex --vocab-size 512
    python train_tokenizer.py --type basic --vocab-size 1000 --input path/to/text.txt

Writes `<out-dir>/<type>.json` (merges) and `<out-dir>/vocab_<type>.txt`, a
human-readable list of every token and the pair it was merged from.
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

from tokenizer import BasicTokenizer, RegexTokenizer, Tokenizer

TEXT_URL = "https://raw.githubusercontent.com/karpathy/minbpe/master/tests/taylorswift.txt"
TOKENIZERS = {"basic": BasicTokenizer, "regex": RegexTokenizer}


def display(token_bytes: bytes) -> str:
    return token_bytes.decode("utf-8", errors="replace")


def write_vocab(tokenizer: Tokenizer, path: Path) -> None:
    """One line per token: `[left][right] -> [token] id` for merges, `[token] id` for bytes."""
    parents = {idx: pair for pair, idx in tokenizer.merges.items()}

    with open(path, "w", encoding="utf-8") as f:
        for idx, token_bytes in tokenizer.vocab.items():
            token = display(token_bytes)

            if idx in parents:
                p0, p1 = parents[idx]
                left = display(tokenizer.vocab[p0])
                right = display(tokenizer.vocab[p1])
                line = f"[{left}][{right}] -> [{token}] {idx}"
            else:
                line = f"[{token}] {idx}"

            line = line.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
            f.write(line + "\n")


def load_text(path: Path) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TEXT_URL, path)
    return path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/taylorswift.txt",
                        help="training text (downloaded from the minbpe repo if missing)")
    parser.add_argument("--type", choices=sorted(TOKENIZERS), default="regex")
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    text = load_text(Path(args.input))
    tokenizer = TOKENIZERS[args.type]()

    t0 = time.time()
    tokenizer.train(text, args.vocab_size, verbose=args.verbose)
    print(f"trained {args.type} tokenizer, vocab {len(tokenizer.vocab)}, in {time.time() - t0:.1f} s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{args.type}.json"
    tokenizer.save(model_path)
    write_vocab(tokenizer, out_dir / f"vocab_{args.type}.txt")

    reloaded = TOKENIZERS[args.type]().load(model_path)
    ids = reloaded.encode(text)
    assert reloaded.decode(ids) == text, "round-trip failed"
    print(f"{len(text.encode('utf-8'))} bytes -> {len(ids)} tokens "
          f"(compression {len(text.encode('utf-8')) / len(ids):.2f}x); round-trip ok")

    sample = "Hello world! It's 2024 and I'm tokenizing   text."
    sample_ids = reloaded.encode(sample)
    print(sample_ids)
    print([display(reloaded.vocab[i]) for i in sample_ids])


if __name__ == "__main__":
    main()
