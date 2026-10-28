"""
Document cleaning for pretraining data: pure functions, applied per document in this order:
normalize -> redact PII -> length -> language (script ratio) -> repetition -> exact dedup.
Thresholds follow the Gopher/FineWeb heuristics, scaled down; tune them with the stats this produces.
"""

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from .redact import redact

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BLANKS = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class Thresholds:
    min_chars: int = 200
    max_chars: int = 1_000_000
    min_words: int = 30
    min_script_ratio: dict = None  # lang -> min share of letters in the expected script
    max_dup_line_char_frac: float = 0.20
    max_dup_ngram_char_frac: float = 0.15
    ngram: int = 5

    def script_ratio(self, lang: str) -> float:
        # Hindi text legitimately mixes in English words and Latin brand names, so its bar is lower
        return (self.min_script_ratio or {"en": 0.80, "hi": 0.50}).get(lang, 0.0)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = "\n".join(_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip()


def content_hash(text: str) -> str:
    """Exact-dup key: case- and whitespace-insensitive, so trivially reformatted copies collapse."""
    return hashlib.sha1(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def _is_devanagari(ch: str) -> bool:
    return "ऀ" <= ch <= "ॿ" or "꣠" <= ch <= "ꣿ"


def script_ratio(text: str, lang: str) -> float:
    """Share of letters (incl. combining marks, which Indic scripts need) written in `lang`'s script."""
    letters = [c for c in text if unicodedata.category(c)[0] in "LM"]
    if not letters:
        return 0.0
    if lang == "hi":
        hit = sum(_is_devanagari(c) for c in letters)
    elif lang == "en":
        hit = sum(c.isascii() for c in letters)
    else:
        return 1.0  # no rule for this language yet: do not filter
    return hit / len(letters)


def dup_line_char_frac(text: str) -> float:
    lines = [ln for ln in text.split("\n") if ln.strip()]
    total = sum(len(ln) for ln in lines)
    if not total:
        return 0.0
    counts = Counter(lines)
    return sum(len(ln) * (n - 1) for ln, n in counts.items() if n > 1) / total


def dup_ngram_char_frac(text: str, n: int) -> float:
    """Share of word characters inside word n-grams that occur more than once (boilerplate / loops)."""
    words = text.split()
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    covered = [False] * len(words)
    for i, g in enumerate(grams):
        if counts[g] > 1:
            for j in range(i, i + n):
                covered[j] = True
    total = sum(len(w) for w in words)
    return sum(len(w) for w, c in zip(words, covered, strict=True) if c) / total


def reject_reason(text: str, lang: str, t: Thresholds) -> str | None:
    """None if the document passes every quality filter, else a short reason code for the stats."""
    if len(text) < t.min_chars:
        return "too_short"
    if len(text) > t.max_chars:
        return "too_long"
    if len(text.split()) < t.min_words:
        return "too_few_words"
    if script_ratio(text, lang) < t.script_ratio(lang):
        return "wrong_language"
    if dup_line_char_frac(text) > t.max_dup_line_char_frac:
        return "repeated_lines"
    if dup_ngram_char_frac(text, t.ngram) > t.max_dup_ngram_char_frac:
        return "repeated_ngrams"
    return None


def clean_document(text: str, lang: str, t: Thresholds) -> tuple[str | None, str | None]:
    """(cleaned text, None) on success, (None, reason) when dropped. Redaction runs before any filter."""
    doc = redact(normalize(text))
    reason = reject_reason(doc, lang, t)
    return (None, reason) if reason else (doc, None)
