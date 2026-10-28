"""Tokenize raw .txt / .jsonl ({"text": ...}) into train.bin / val.bin shards.
python scripts/prepare_data.py --input data/raw --tokenizer artifacts/tokenizer.json --out data/tokens
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viron.data import dtype_for  # noqa: E402
from viron.tokenizer import load_tokenizer  # noqa: E402


def iter_docs(root: Path):
    for p in sorted(root.rglob("*")):
        if p.suffix == ".txt":
            yield p.read_text(encoding="utf-8", errors="ignore")
        elif p.suffix == ".jsonl":
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line).get("text", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tokenizer", default=None, help="omit for byte-level tokenizer")
    ap.add_argument("--out", default="data/tokens")
    ap.add_argument("--val-every", type=int, default=100, help="every Nth doc goes to validation")
    a = ap.parse_args()

    tok = load_tokenizer(a.tokenizer)
    dt = dtype_for(tok.vocab_size)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    with open(out / "train.bin", "wb") as ftr, open(out / "val.bin", "wb") as fva:
        for i, doc in enumerate(iter_docs(Path(a.input))):
            if not doc.strip():
                continue
            ids = np.array(tok.encode(doc) + [tok.eos_id], dtype=dt)
            split = "val" if i % a.val_every == a.val_every - 1 else "train"
            (fva if split == "val" else ftr).write(ids.tobytes())
            counts[split] += len(ids)
    (out / "meta.json").write_text(json.dumps({"vocab_size": tok.vocab_size, "tokenizer": a.tokenizer, **counts}))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
