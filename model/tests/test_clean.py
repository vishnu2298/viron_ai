import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from clean_data import clean_tree  # noqa: E402

from viron.clean import (  # noqa: E402
    Thresholds,
    clean_document,
    content_hash,
    dup_line_char_frac,
    dup_ngram_char_frac,
    normalize,
    reject_reason,
    script_ratio,
)
from viron.redact import redact  # noqa: E402

T = Thresholds()
PII_CASES = json.loads((ROOT.parent / "orchestrator/test/fixtures/pii-cases.json").read_text(encoding="utf-8"))

# Synthetic prose written for these tests: varied wording so the repetition filters pass
EN = (
    "Early in the morning the ferry left the harbour while gulls circled above the grey water. "
    "Passengers wrapped in thick coats watched the coastline shrink, and a child counted the fishing boats aloud. "
    "By noon the wind had dropped, the sea turned calm, and the captain announced that lunch was being served "
    "below deck. Most travellers stayed outside anyway, unwilling to miss the cliffs that rose sharply from the "
    "waves near the northern island, where a small lighthouse still guided ships through narrow channels at night."
)
HI = (
    "सुबह होते ही गाँव के लोग अपने खेतों की ओर निकल पड़ते हैं। बच्चे पास के स्कूल में पढ़ने जाते हैं, "
    "जबकि बुज़ुर्ग चौपाल पर बैठकर मौसम और फसल की बातें करते हैं। दोपहर में सूरज तेज़ हो जाता है, "
    "इसलिए किसान पेड़ों की छाया में आराम करते हैं और घर से लाया हुआ खाना खाते हैं। शाम को नदी के किनारे "
    "हल्की ठंडी हवा चलती है, और लौटते समय लोग बाज़ार से सब्ज़ियाँ तथा ज़रूरी सामान खरीद लेते हैं।"
)


@pytest.mark.parametrize("case", PII_CASES, ids=lambda c: c["input"][:30])
def test_redact_matches_js_fixture(case):
    assert redact(case["input"]) == case["expected"]


def test_normalize_nfc_controls_whitespace():
    raw = "á  b\t\tc\r\n\x00\x07line2\n\n\n\n\nend  "
    assert normalize(raw) == "á b c\nline2\n\nend"


def test_content_hash_ignores_case_and_spacing():
    assert content_hash("Hello   World\n") == content_hash("hello world")
    assert content_hash("hello world") != content_hash("hello worlds")


def test_script_ratio_en_hi_and_code_mixed():
    assert script_ratio(EN, "en") == 1.0
    assert script_ratio(HI, "hi") > 0.95
    assert script_ratio(HI, "en") < 0.1
    mixed = HI + " Wi-Fi password app download"
    assert 0.5 < script_ratio(mixed, "hi") < 1.0
    assert script_ratio("12345 !!!", "en") == 0.0
    assert script_ratio(EN, "ta") == 1.0  # no rule yet -> never filtered


def test_repetition_metrics():
    assert dup_line_char_frac("a line\nb line\nc line") == 0.0
    assert dup_line_char_frac("same\nsame\nsame\nother") == pytest.approx(8 / 17)
    assert dup_ngram_char_frac(EN, 5) == 0.0
    assert dup_ngram_char_frac("buy now cheap deal today " * 20, 5) > 0.9


@pytest.mark.parametrize(
    "text,lang,reason",
    [
        ("short text", "en", "too_short"),
        ("x" * 250, "en", "too_few_words"),
        (HI, "en", "wrong_language"),
        (EN, "hi", "wrong_language"),
        ("\n".join(["Subscribe to our newsletter for the latest updates today"] * 12), "en", "repeated_lines"),
        ("click here to buy cheap deals now " * 30, "en", "repeated_ngrams"),
    ],
)
def test_reject_reasons(text, lang, reason):
    assert reject_reason(normalize(text), lang, T) == reason


def test_good_documents_pass():
    assert reject_reason(EN, "en", T) is None
    assert reject_reason(HI, "hi", T) is None


def test_clean_document_redacts_before_filtering():
    doc, reason = clean_document(EN + " Write to someone@example.com or call +91 98765 43210.", "en", T)
    assert reason is None
    assert "[EMAIL]" in doc and "[PHONE]" in doc and "example.com" not in doc


def write_raw(root: Path, name: str, texts: list[str]) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "part-00000.jsonl").write_text("".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8")


def test_clean_tree_end_to_end_with_cross_source_dedup(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "clean"
    write_raw(raw, "commonpile_comma", [EN, EN.upper().lower(), "tiny", HI])
    write_raw(raw, "wikipedia_hi", [HI, HI + " ", EN])
    stats = clean_tree(raw, out, T)
    assert stats["commonpile_comma"] == {
        "docs_in": 4,
        "docs_out": 1,
        "chars_out": len(EN),
        "drop_duplicate": 1,
        "drop_too_short": 1,
        "drop_wrong_language": 1,
    }
    assert stats["wikipedia_hi"]["docs_out"] == 1
    assert stats["wikipedia_hi"]["drop_duplicate"] == 1
    assert stats["wikipedia_hi"]["drop_wrong_language"] == 1
    rows = [json.loads(x) for x in (out / "wikipedia_hi" / "part-00000.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows == [{"text": HI, "source": "wikipedia_hi"}]
    assert json.loads((out / "stats.json").read_text()) == stats
