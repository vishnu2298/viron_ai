import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_traces import build_datasets, export


def t(msg, ans, adapter=None):
    return {"message": msg, "answer": ans, "decision": {"adapters": [{"name": adapter}] if adapter else []}}


def test_buckets_and_dedupes():
    traces = [
        t("How do refunds work?", "Within 7 days.", "support"),
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


def test_user_message_is_kept_verbatim():
    # A real question that happens to look like the old inline-context format must not be truncated
    msg = "Context:\nwe migrated last week\n\nWhy are refunds slow?"
    ds = build_datasets([t(msg, "Queue backlog.", "support")])
    assert ds["support"][0]["messages"][1]["content"] == msg


class FakeCollection:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def find(self, flt, projection, batch_size):
        self.calls.append((flt, projection, batch_size))
        return iter(self.docs)


def test_export_writes_jsonl_per_adapter(tmp_path):
    now = datetime(2026, 1, 31, tzinfo=UTC)
    col = FakeCollection([t("How do refunds work?", "Within 7 days.", "support"), t("Fix this regex", "Lazy.", "code")])
    counts = export(col, tmp_path / "data", days=30, min_examples=1, batch=50, now=now)

    assert counts == {"support": 1, "code": 1}
    flt, projection, batch = col.calls[0]
    assert flt == {"escalated": True, "createdAt": {"$gte": now - timedelta(days=30)}}
    assert projection["answer"] == 1 and batch == 50
    row = json.loads((tmp_path / "data" / "support.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["messages"][2]["content"] == "Within 7 days."


def test_export_nothing_ready_writes_no_files(tmp_path):
    counts = export(
        FakeCollection([t("question one", "answer one", "code")]), tmp_path, days=1, min_examples=5, batch=10
    )
    assert counts == {}
    assert list(tmp_path.iterdir()) == []
