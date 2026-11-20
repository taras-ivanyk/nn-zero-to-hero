import regex

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer:

    def __init__(self, pattern=GPT4_SPLIT_PATTERN):
        super().__init__()
        self.pattern = regex.compile(pattern)
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def merge(self, ids, pair, idx):

        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids


    def get_stats(self, ids):
        counts = {}

        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts


    def train(self, text, vocab_size, verbose=False):
        num_merges = vocab_size - 256
        chunks = self.pattern.findall(text)
        ids = [list(chunk.encode("utf-8")) for chunk in chunks] # Differ from BasicTokenizer, because of addition of regex patter used in GPT-4

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        for i in range(num_merges):
            stats = {}
            
            for chunk_ids in ids:
                for pair, count in self.get_stats(chunk_ids).items():
                    stats[pair] = stats.get(pair, 0) + count

            if not stats:
                break

            pair = max(stats, key=stats.get)
            idx = 256 + i

            ids = [self.merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        
        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        chunks = self.pattern.findall(text)
        result = []

        for chunk in chunks:
            ids = list(chunk.encode("utf-8"))

            while len(ids) >= 2:
                stats = self.get_stats(ids)
                pair = min(
                    stats,
                    key=lambda p: self.merges.get(p, float("inf"))
                )

                if pair not in self.merges:
                    break

                idx = self.merges[pair]
                ids = self.merge(ids, pair, idx)

            result.extend(ids)

        return result

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")

        return text
    
with open("/Users/tivanyk001/Desktop/Andrej Karpathy/Tokenization/taylorswift.txt", "r") as file:
    text = file.read()
    
tokenizer = RegexTokenizer()
tokenizer.train(text, vocab_size=1000) # ± 22 minutes, ± 0.026 seconds per merge

def display(token_bytes):
    return token_bytes.decode("utf-8", errors="replace")


# Map each merged token ID back to its component pair.
parents = {idx: pair for pair, idx in tokenizer.merges.items()}

with open("vocab_Regex.txt", "w", encoding="utf-8") as f:
    for idx, token_bytes in tokenizer.vocab.items():
        token = display(token_bytes)

        if idx in parents:
            p0, p1 = parents[idx]
            left = display(tokenizer.vocab[p0])
            right = display(tokenizer.vocab[p1])
            line = f"[{left}][{right}] -> [{token}] {idx}"
        else:
            line = f"[{token}] {idx}"

        # Keep newline and tab characters visible on one line.
        line = line.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

        f.write(line + "\n")