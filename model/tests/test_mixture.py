import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_shards import prepare  # noqa: E402

from viron.data import MixtureDataset, ShardedDataset  # noqa: E402
from viron.mixture import load_mixture, parse_mixture  # noqa: E402


def test_shipped_mix_v1_is_valid_and_85_15():
    m = load_mixture(ROOT / "configs/data/mix_v1.yaml")
    assert m.weights["commonpile_comma"] == pytest.approx(0.85)
    assert m.weights["sangraha_verified_hin"] + m.weights["wikipedia_hi"] == pytest.approx(0.15)


def test_parse_mixture_normalizes_and_validates():
    m = parse_mixture({"sources": {"commonpile_comma": 3, "wikipedia_hi": 1, "sangraha_verified_hin": 0}})
    assert m.weights == {"commonpile_comma": 0.75, "wikipedia_hi": 0.25}
    with pytest.raises(ValueError, match="not approved"):
        parse_mixture({"sources": {"fineweb": 1.0}})
    with pytest.raises(ValueError, match="non-negative"):
        parse_mixture({"sources": {"wikipedia_hi": -1}})
    with pytest.raises(ValueError, match="positive"):
        parse_mixture({"sources": {"wikipedia_hi": 0}})
    with pytest.raises(ValueError, match="non-empty"):
        parse_mixture({})


def shard(path: Path, values: list[int]) -> Path:
    np.asarray(values, dtype=np.uint16).tofile(path)
    return path


def test_sharded_windows_never_cross_shards(tmp_path):
    a = shard(tmp_path / "train-00000.bin", [1] * 40)
    b = shard(tmp_path / "train-00001.bin", [2] * 40)
    ds = ShardedDataset([a, b], vocab_size=300)
    g = torch.Generator().manual_seed(0)
    x, y = ds.sample(200, 16, g)
    assert x.shape == y.shape == (200, 16)
    rows = torch.cat([x, y], dim=1)
    assert all(len(set(r.tolist())) == 1 for r in rows)  # each window is entirely one shard
    with pytest.raises(ValueError, match="shorter than seq_len"):
        ds.sample(1, 64, g)


def test_mixture_sampling_follows_weights(tmp_path):
    en = ShardedDataset([shard(tmp_path / "en.bin", [1] * 500)], 300)
    hi = ShardedDataset([shard(tmp_path / "hi.bin", [2] * 500)], 300)
    mix = MixtureDataset({"en": en, "hi": hi}, {"en": 0.85, "hi": 0.15})
    x, _ = mix.batch(4000, 8, "cpu", torch.Generator().manual_seed(1))
    share_hi = (x[:, 0] == 2).float().mean().item()
    assert x.shape == (4000, 8) and 0.12 < share_hi < 0.18


def test_prepare_shards_splits_per_source(tmp_path):
    clean = tmp_path / "clean"
    for name, word in (("commonpile_comma", "river"), ("wikipedia_hi", "नदी")):
        d = clean / name
        d.mkdir(parents=True)
        rows = [json.dumps({"text": f"{word} {i} " * 20}, ensure_ascii=False) for i in range(20)]
        (d / "part-00000.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    out = tmp_path / "tokens"
    summary = prepare(clean, out, tokenizer=None, val_every=5, shard_tokens=2000)
    for name in ("commonpile_comma", "wikipedia_hi"):
        meta = summary["sources"][name]
        assert meta["docs"] == 20
        trains = sorted((out / name).glob("train-*.bin"))
        assert len(trains) > 1  # rotated by shard_tokens
        assert sum(p.stat().st_size for p in trains) // 2 == meta["train_tokens"]
        assert len(list((out / name).glob("val-*.bin"))) == 1 and meta["val_tokens"] > 0
    assert json.loads((out / "meta.json").read_text())["vocab_size"] == 263


def test_train_with_mixture_reports_per_source_val(tmp_path, capsys):
    clean = tmp_path / "clean"
    for name, word in (("commonpile_comma", "stone path"), ("wikipedia_hi", "पत्थर रास्ता")):
        d = clean / name
        d.mkdir(parents=True)
        rows = [json.dumps({"text": f"{word} {i % 7} " * 30}, ensure_ascii=False) for i in range(40)]
        (d / "part-00000.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    prepare(clean, tmp_path / "tokens", None, val_every=5, shard_tokens=10_000)
    mix = tmp_path / "mix.yaml"
    mix.write_text(f"tokens_dir: {tmp_path / 'tokens'}\nsources:\n  commonpile_comma: 0.85\n  wikipedia_hi: 0.15\n")
    import train

    train.main(
        [
            "--preset",
            "nano",
            "--mix",
            str(mix),
            "--out",
            str(tmp_path / "run"),
            "--steps",
            "6",
            "--batch",
            "4",
            "--accum",
            "1",
            "--seq",
            "32",
            "--warmup",
            "2",
            "--eval-every",
            "6",
            "--eval-iters",
            "2",
            "--save-every",
            "6",
            "--device",
            "cpu",
        ]
    )
    evals = [json.loads(line) for line in capsys.readouterr().out.splitlines() if '"event": "eval"' in line]
    assert {"val", "val_commonpile_comma", "val_wikipedia_hi"} <= set(evals[-1])
    assert (tmp_path / "run" / "best.pt").exists()
    with pytest.raises(SystemExit):
        train.parse(["--preset", "nano"])  # neither --data nor --mix
