"""Train Viron's own BPE on a mixture-weighted sample of data/clean and report tokens-per-word per source.
python scripts/train_tokenizer.py --input data/clean --mix configs/data/mix_v1.yaml --out artifacts/tokenizer.json
Without --mix every source directory under --input gets an equal share (also accepts legacy *.txt files).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viron.mixture import load_mixture  # noqa: E402
from viron.tokenizer import BPETokenizer  # noqa: E402
from viron.tokenizer_data import char_quotas, iter_source_texts, split_sample, tokens_per_word  # noqa: E402


def run(inp: Path, out: Path, vocab: int, sample_chars: int, holdout_docs: int, weights: dict[str, float]) -> dict:
    samples, heldout = {}, {}
    for name, quota in char_quotas(weights, sample_chars).items():
        src = inp / name
        if not src.is_dir():
            raise SystemExit(f"source {name!r} has no directory under {inp} (run download + clean first)")
        samples[name], heldout[name] = split_sample(iter_source_texts(src), quota, holdout_docs)
        print(
            json.dumps(
                {
                    "event": "tokenizer.sample",
                    "source": name,
                    "docs": len(samples[name]),
                    "chars": sum(map(len, samples[name])),
                    "quota_chars": quota,
                }
            ),
            flush=True,
        )
    tok = BPETokenizer.train_from_iterator((t for name in samples for t in samples[name]), out, vocab)
    report = {
        "vocab_size": tok.vocab_size,
        "sample_chars": {n: sum(map(len, s)) for n, s in samples.items()},
        # measured on documents the tokenizer did not train on
        "tokens_per_word": {n: tokens_per_word(tok.encode, h) for n, h in heldout.items()},
        "heldout_docs": {n: len(h) for n, h in heldout.items()},
    }
    out.with_name(out.stem + "_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/clean")
    ap.add_argument("--out", default="artifacts/tokenizer.json")
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--mix", default=None, help="data mixture YAML; weights set each source's share of the sample")
    ap.add_argument("--sample-chars", type=lambda v: int(v.replace("_", "")), default=500_000_000)
    ap.add_argument("--holdout-docs", type=int, default=2000)
    a = ap.parse_args()
    inp = Path(a.input)
    if a.mix:
        weights = load_mixture(a.mix).weights
    else:
        names = sorted(p.name for p in inp.iterdir() if p.is_dir()) or ["."]
        weights = {n: 1 / len(names) for n in names}
    print(json.dumps(run(inp, Path(a.out), a.vocab, a.sample_chars, a.holdout_docs, weights)))
