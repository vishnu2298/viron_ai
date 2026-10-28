import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_traces import build_datasets, strip_context


def t(msg, ans, adapter=None):
    return {"message": msg, "answer": ans, "decision": {"adapters": [{"name": adapter}] if adapter else []}}


def test_buckets_dedupes_and_strips_context():
    traces = [
        t("Context:\n[1] doc\n\nHow do refunds work?", "Within 7 days.", "support"),
        t("how do refunds work?", "dup", "support"),
        t("Fix this regex", "Use a lazy quantifier.", "code"),
        t("hi", "", None),
    ]
    ds = build_datasets(traces)
    assert set(ds) == {"support", "code"}
    assert len(ds["support"]) == 1
    assert ds["support"][0]["messages"][1]["content"] == "How do refunds work?"


def test_min_examples_filters_small_buckets():
    assert build_datasets([t("question one", "answer one", "code")], min_examples=2) == {}


def test_strip_context_passthrough():
    assert strip_context("plain question") == "plain question"
