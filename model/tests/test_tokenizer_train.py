import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train_tokenizer import run  # noqa: E402

from viron.tokenizer import SPECIALS, BPETokenizer  # noqa: E402
from viron.tokenizer_data import char_quotas, split_sample, tokens_per_word  # noqa: E402

EN = [
    f"The ferry number {i} crossed the harbour while passengers watched the cliffs and the lighthouse."
    for i in range(400)
]
HI = [f"नाव संख्या {i} बंदरगाह पार कर रही थी और यात्री किनारे की चट्टानों को देख रहे थे।" for i in range(400)]


def write_clean(root: Path, name: str, texts: list[str]) -> None:
    d = root / name
    d.mkdir(parents=True)
    rows = "".join(json.dumps({"text": t, "source": name}, ensure_ascii=False) + "\n" for t in texts)
    (d / "part-00000.jsonl").write_text(rows, encoding="utf-8")


def test_quotas_and_split():
    assert char_quotas({"en": 0.85, "hi": 0.15}, 1000) == {"en": 850, "hi": 150}
    train, held = split_sample(["aaaa", "bbbb", "cccc", "dddd", "eeee"], quota_chars=6, holdout_docs=2)
    assert train == ["aaaa", "bbbb"] and held == ["cccc", "dddd"]


def test_tokens_per_word():
    assert tokens_per_word(lambda t: list(t.replace(" ", "")), ["ab cd", "e"]) == round(5 / 3, 3)
    assert tokens_per_word(len, []) is None


def test_train_reports_fertility_per_source_and_keeps_specials(tmp_path):
    clean = tmp_path / "clean"
    write_clean(clean, "commonpile_comma", EN)
    write_clean(clean, "wikipedia_hi", HI)
    out = tmp_path / "artifacts" / "tokenizer.json"
    report = run(
        clean,
        out,
        vocab=600,
        sample_chars=40_000,
        holdout_docs=20,
        weights={"commonpile_comma": 0.85, "wikipedia_hi": 0.15},
    )
    tok = BPETokenizer(out)
    assert [tok.token_id(s) for s in SPECIALS] == list(range(len(SPECIALS)))  # chat tokens reserved at fixed ids
    assert report["heldout_docs"] == {"commonpile_comma": 20, "wikipedia_hi": 20}
    tpw = report["tokens_per_word"]
    assert 1 <= tpw["commonpile_comma"] < tpw["wikipedia_hi"]  # small Hindi share -> worse Hindi fertility
    assert json.loads(out.with_name("tokenizer_report.json").read_text()) == report
    text = HI[0]
    assert tok.decode(tok.encode(text)) == text
