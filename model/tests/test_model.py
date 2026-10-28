import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viron.config import preset  # noqa: E402
from viron.model import Viron  # noqa: E402
from viron.tokenizer import ByteTokenizer  # noqa: E402


@pytest.fixture
def cfg():
    return preset("nano", vocab_size=263)


def test_forward_shapes_and_loss(cfg):
    m = Viron(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = m(x, y)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert abs(loss.item() - np.log(cfg.vocab_size)) < 0.5  # ~uniform at init


def test_causality(cfg):
    """Changing a future token must not change earlier logits."""
    torch.manual_seed(0)
    m = Viron(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 12))
    y = x.clone()
    y[0, 8] = (y[0, 8] + 1) % cfg.vocab_size
    a, _ = m(x, x)
    b, _ = m(y, y)
    assert torch.allclose(a[0, :8], b[0, :8], atol=1e-5)
    assert not torch.allclose(a[0, 8:], b[0, 8:], atol=1e-5)


def test_kv_cache_matches_full_recompute(cfg):
    torch.manual_seed(0)
    m = Viron(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 5))
    fast = m.generate(x, 20, temperature=0, use_cache=True)
    slow = m.generate(x, 20, temperature=0, use_cache=False)
    assert torch.equal(fast, slow)


def test_overfits_single_batch(cfg):
    """If it can't memorize one batch, something is broken."""
    torch.manual_seed(0)
    m = Viron(cfg)
    opt = m.configure_optimizer(3e-3, 0.0)
    x = torch.randint(0, cfg.vocab_size, (4, 32))
    first = None
    for _ in range(150):
        _, loss = m(x[:, :-1], x[:, 1:])
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.2


def test_gqa_and_tied_embeddings(cfg):
    m = Viron(cfg)
    assert m.lm_head.weight is m.tok_emb.weight
    assert m.layers[0].attn.wk.out_features == cfg.n_kv_heads * cfg.head_dim


def test_byte_tokenizer_roundtrip():
    t = ByteTokenizer()
    s = "Viron says नमस्ते 👋"
    assert t.decode(t.encode(s)) == s


def test_end_to_end_train_resume_sample(tmp_path):
    """prepare_data -> train -> resume -> load checkpoint -> generate, on CPU in seconds."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.txt").write_text("the quick brown fox jumps over the lazy dog. " * 400)
    (raw / "b.jsonl").write_text("\n".join(json.dumps({"text": f"doc {i} " * 50}) for i in range(150)))

    import subprocess

    root = Path(__file__).resolve().parents[1]
    tokens = tmp_path / "tokens"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/prepare_data.py"),
            "--input",
            str(raw),
            "--out",
            str(tokens),
            "--val-every",
            "10",
        ],
        check=True,
    )

    sys.path.insert(0, str(root))
    import train

    args = [
        "--preset",
        "nano",
        "--data",
        str(tokens),
        "--out",
        str(tmp_path / "run"),
        "--steps",
        "60",
        "--batch",
        "8",
        "--accum",
        "1",
        "--seq",
        "64",
        "--warmup",
        "5",
        "--lr",
        "3e-3",
        "--eval-every",
        "30",
        "--eval-iters",
        "5",
        "--save-every",
        "30",
        "--device",
        "cpu",
    ]
    best1 = train.main(args)
    best2 = train.main(args[: args.index("--steps") + 1] + ["90"] + args[args.index("--steps") + 2 :] + ["--resume"])
    assert best2 <= best1
    assert best1 < np.log(263)  # learned something vs uniform

    from sample import load

    model, tok = load(tmp_path / "run" / "best.pt", "cpu")
    out = model.generate(torch.tensor([tok.encode("the quick")]), 30, temperature=0)
    assert isinstance(tok.decode(out[0].tolist()), str)
