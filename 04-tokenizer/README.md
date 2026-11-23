# 04 · BPE tokenizer

Byte-level Byte Pair Encoding tokenizers, following Karpathy's [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) lecture and the [minbpe](https://github.com/karpathy/minbpe) exercise.

- `tokenizer/base.py`: shared `Tokenizer` with pair counting, merging, `decode`, `save` and `load` (merges stored as JSON).
- `tokenizer/basic.py`: `BasicTokenizer`, BPE on the raw UTF-8 bytes of the whole text.
- `tokenizer/regex.py`: `RegexTokenizer`, which first splits text with the GPT-4 regex pattern so merges never cross word, number or punctuation boundaries.
- `train_tokenizer.py`: train on a text file, save, reload, check the encode/decode round-trip and write a readable vocabulary file.
- `tiktoken_demo.py`: encodes a sentence with every `tiktoken` encoding for comparison.
- `examples/vocab_basic.txt`, `examples/vocab_regex.txt`: vocabularies of size 1000 trained on `taylorswift.txt` (from minbpe). Each line shows a token and the pair it was merged from.

## Run

```bash
cd 04-tokenizer
python train_tokenizer.py --type regex --vocab-size 1000
python train_tokenizer.py --type basic --vocab-size 1000 --input path/to/text.txt
pytest ../tests/test_tokenizer.py
```

## Results

Measured with `train_tokenizer.py` on `taylorswift.txt` (185,768 bytes), vocabulary size 1000:

| Tokenizer | Tokens | Compression | Training time (Apple Silicon Mac, CPU) |
|---|---|---|---|
| `BasicTokenizer` | 58,300 | 3.19x | 7.7 s |
| `RegexTokenizer` (GPT-4 split) | 70,362 | 2.64x | 15.2 s |

The regex tokenizer compresses less on this text because merges cannot span chunk boundaries (for example `'s ` and ` and ` are single tokens for the basic tokenizer but not for the regex one).

## What I learned

- Byte-level BPE never hits an unknown token: any text is just UTF-8 bytes, and merges only add shortcuts on top of the 256 base bytes.
- The vocabulary reflects its training text. On `taylorswift.txt` the very first merge was `20` (token 257), followed early by `201` and `202`, because the text is full of years. The GPT-4 regex splits numbers into chunks of at most 3 digits, so a year like `2014` can never be a single token.
- The regex split costs compression (2.64x vs 3.19x) and doubles training time. In return, tokens stay inside words: ` October` becomes one token, but `'s ` across a boundary does not.
- Tokenization explains some odd model behaviour: a word can be split in unexpected places (for example `S` + `eptember`), so the model never actually sees the letters of a word.
- English-trained tokenizers handle Ukrainian poorly: Cyrillic takes 2 bytes per character in UTF-8, and the merges are learned from English text. When I trained a WordPiece tokenizer on Ukrainian text, I got much better results.
