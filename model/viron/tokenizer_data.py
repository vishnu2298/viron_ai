"""Mixture-weighted text sampling for tokenizer training, and the tokens-per-word report (roadmap M1.5)."""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path


def iter_source_texts(src_dir: Path) -> Iterator[str]:
    """Texts of one source in data/clean: part-*.jsonl (cleaned) or legacy *.txt files."""
    for p in sorted(src_dir.glob("part-*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                text = json.loads(line).get("text", "")
                if text.strip():
                    yield text
    for p in sorted(src_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            yield text


def char_quotas(weights: dict[str, float], total_chars: int) -> dict[str, int]:
    return {name: int(total_chars * w) for name, w in weights.items()}


def split_sample(texts: Iterable[str], quota_chars: int, holdout_docs: int) -> tuple[list[str], list[str]]:
    """First `quota_chars` of text for training, then the next `holdout_docs` documents for the report.
    Sources are read in file order; downloads and cleaning are already order-randomized enough for a vocab."""
    train, held, used = [], [], 0
    for text in texts:
        if used < quota_chars:
            train.append(text)
            used += len(text)
        elif len(held) < holdout_docs:
            held.append(text)
        else:
            break
    return train, held


def tokens_per_word(encode, texts: Iterable[str]) -> float | None:
    """Fertility: tokens per whitespace-separated word. ~1.3 is typical for English with a 32k BPE;
    much higher values for Hindi mean the vocabulary under-serves Devanagari."""
    tokens = words = 0
    for t in texts:
        tokens += len(encode(t))
        words += len(t.split())
    return round(tokens / words, 3) if words else None
