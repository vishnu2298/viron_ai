"""Tokenize data/clean/<source>/*.jsonl into data/tokens/<source>/{train,val}-NNNNN.bin + meta.json.
python scripts/prepare_shards.py --input data/clean --tokenizer artifacts/tokenizer.json --out data/tokens
Shards end on document boundaries; every Nth document goes to validation so each source has its own val set.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viron.data import dtype_for  # noqa: E402
from viron.tokenizer import load_tokenizer  # noqa: E402


class ShardWriter:
    def __init__(self, out: Path, split: str, dtype, shard_tokens: int):
        self.out, self.split, self.dtype, self.cap = out, split, dtype, shard_tokens
        self.index, self.f, self.n, self.total = 0, None, 0, 0

    def write(self, ids: list[int]) -> None:
        if self.f is None or self.n >= self.cap:
            self.close()
            self.f = open(self.out / f"{self.split}-{self.index:05d}.bin", "wb")
            self.index += 1
            self.n = 0
        self.f.write(np.asarray(ids, dtype=self.dtype).tobytes())
        self.n += len(ids)
        self.total += len(ids)

    def close(self) -> None:
        if self.f:
            self.f.close()
            self.f = None


def prepare(inp: Path, out: Path, tokenizer: str | None, val_every: int, shard_tokens: int) -> dict:
    tok = load_tokenizer(tokenizer)
    dt = dtype_for(tok.vocab_size)
    summary = {"vocab_size": tok.vocab_size, "tokenizer": tokenizer, "sources": {}}
    for src_dir in sorted(p for p in inp.iterdir() if p.is_dir()):
        dest = out / src_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        writers = {s: ShardWriter(dest, s, dt, shard_tokens) for s in ("train", "val")}
        docs = 0
        for part in sorted(src_dir.glob("part-*.jsonl")):
            for line in part.open(encoding="utf-8"):
                text = json.loads(line).get("text", "")
                if not text.strip():
                    continue
                split = "val" if docs % val_every == val_every - 1 else "train"
                writers[split].write(tok.encode(text) + [tok.eos_id])
                docs += 1
        for w in writers.values():
            w.close()
        meta = {"docs": docs, "train_tokens": writers["train"].total, "val_tokens": writers["val"].total}
        (dest / "meta.json").write_text(json.dumps(meta))
        summary["sources"][src_dir.name] = meta
    (out / "meta.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/clean")
    ap.add_argument("--out", default="data/tokens")
    ap.add_argument("--tokenizer", default=None, help="omit for byte-level tokenizer")
    ap.add_argument("--val-every", type=int, default=200, help="every Nth doc goes to validation")
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    print(json.dumps(prepare(Path(a.input), Path(a.out), a.tokenizer, a.val_every, a.shard_tokens)["sources"]))
