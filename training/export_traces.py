"""
Distillation data export: MongoDB traces -> per-adapter JSONL chat datasets.

Escalated traces are teacher-labelled (the large model answered what the small one couldn't),
so they are exactly the gap a new LoRA adapter should close.

Usage:
  python export_traces.py --out data/ --days 30 --min-examples 200
"""

import argparse
import hashlib
import json
import logging
import os
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='{"level":"%(levelname)s","event":"%(message)s"}')
log = logging.getLogger("export")

SYSTEM = "You are a concise, accurate assistant."


def to_example(trace: dict) -> dict | None:
    # Traces hold the user's message verbatim: retrieved context travels in the system prompt
    # (never the user turn), so adapters learn skills, not stale facts, with no stripping needed.
    q = (trace.get("message") or "").strip()
    a = (trace.get("answer") or "").strip()
    if len(q) < 3 or len(a) < 3:
        return None
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
    }


def bucket(trace: dict) -> str:
    adapters = (trace.get("decision") or {}).get("adapters") or []
    return adapters[0]["name"] if adapters else "general"


def build_datasets(traces, min_examples: int = 1):
    """Pure function (testable): dedupe, bucket by adapter, drop small buckets."""
    seen, buckets = set(), defaultdict(list)
    for t in traces:
        ex = to_example(t)
        if not ex:
            continue
        key = hashlib.sha1(ex["messages"][1]["content"].lower().encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        buckets[bucket(t)].append(ex)
    return {k: v for k, v in buckets.items() if len(v) >= min_examples}


def export(col, out: Path, days: int, min_examples: int, batch: int, now: datetime | None = None) -> dict[str, int]:
    """Read escalated traces from `col` and write one JSONL per adapter bucket. Returns example counts."""
    since = (now or datetime.now(UTC)) - timedelta(days=days)
    cursor = col.find(
        {"escalated": True, "createdAt": {"$gte": since}},
        {"message": 1, "answer": 1, "decision.adapters": 1},
        batch_size=batch,
    )
    datasets = build_datasets(cursor, min_examples)

    out.mkdir(parents=True, exist_ok=True)
    for name, rows in datasets.items():
        path = out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info(json.dumps({"adapter": name, "examples": len(rows), "path": str(path)}))
    if not datasets:
        log.info("no adapter bucket reached --min-examples yet; keep collecting traces")
    return {name: len(rows) for name, rows in datasets.items()}


def main():  # pragma: no cover - CLI wiring around export()
    from pymongo import MongoClient

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--min-examples", type=int, default=200)
    ap.add_argument("--batch", type=int, default=500)
    args = ap.parse_args()

    client = MongoClient(os.environ["MONGODB_URI"])
    col = client[os.environ.get("MONGODB_DB", "viron_ai")]["traces"]
    export(col, Path(args.out), args.days, args.min_examples, args.batch)


if __name__ == "__main__":  # pragma: no cover
    main()
