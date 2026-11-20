import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in ("01-micrograd", "02-makemore", "03-gpt", "04-tokenizer"):
    sys.path.insert(0, str(ROOT / folder))
