"""Train Viron's own BPE tokenizer from raw text files.
  python scripts/train_tokenizer.py --input data/raw --out artifacts/tokenizer.json --vocab 32000
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viron.tokenizer import BPETokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, help="directory of .txt files")
ap.add_argument("--out", default="artifacts/tokenizer.json")
ap.add_argument("--vocab", type=int, default=32000)
a = ap.parse_args()
files = [str(p) for p in Path(a.input).rglob("*.txt")]
if not files:
    sys.exit(f"no .txt files under {a.input}")
tk = BPETokenizer.train(files, a.out, a.vocab)
print(f"tokenizer saved -> {a.out} (vocab {tk.vocab_size}, trained on {len(files)} files)")
