"""
PII redaction, a line-for-line port of orchestrator/src/utils/redact.js (US + India formats).
Both implementations are checked against orchestrator/test/fixtures/pii-cases.json, so change them together.
"""

import re

# JS semantics: \d and \b are ASCII-only, but JS \s is Unicode whitespace. re.ASCII matches the first two;
# JS's \s set is spelled out so NBSP-separated numbers are treated the same in both runtimes.
_WS = "\t\n\v\f\r    -     　﻿"
_A = re.ASCII


def _digits(s: str) -> int:
    return len(re.sub(r"\D", "", s, flags=_A))


_RULES: list[tuple[re.Pattern, object]] = [
    (re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", _A | re.IGNORECASE), "[EMAIL]"),
    (re.compile(r"\b(?:sk|pk|api|key)[-_][A-Za-z0-9_-]{16,}\b", _A), "[SECRET]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b", _A), "[SSN]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", _A), "[PAN]"),
    (
        re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,7}\b", _A),
        lambda m: "[CARD]" if _digits(m.group()) >= 13 else m.group(),
    ),
    (re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b", _A), "[AADHAAR]"),
    # Phone: optional +CC, then digit groups; only redact when 10-13 digits total
    (
        re.compile(rf"(?:\+\d{{1,3}}[{_WS}-]?)?\(?\d{{2,5}}\)?(?:[{_WS}-]?\d{{2,5}}){{1,3}}", _A),
        lambda m: "[PHONE]" if 10 <= _digits(m.group()) <= 13 else m.group(),
    ),
]


def redact(text: str) -> str:
    for pattern, rep in _RULES:
        text = pattern.sub(rep, text)
    return text
