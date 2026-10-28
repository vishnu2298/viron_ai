"""Clean data/raw/<source>/*.jsonl into data/clean/<source>/part-*.jsonl and write stats.json.
python scripts/clean_data.py --input data/raw --out data/clean
Exact dedup is global across all sources in one run (first occurrence wins, sources processed in sorted order).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viron.clean import Thresholds, clean_document, content_hash  # noqa: E402
from viron.sources import SOURCES  # noqa: E402


def clean_tree(inp: Path, out: Path, t: Thresholds, part_docs: int = 100_000) -> dict:
    seen: set[str] = set()  # ~40 bytes/doc; fine for the few-million-doc scale of M1-M3
    stats: dict[str, Counter] = {}
    for src_dir in sorted(p for p in inp.iterdir() if p.is_dir()):
        name = src_dir.name
        lang = SOURCES[name].lang if name in SOURCES else "unknown"
        st = stats.setdefault(name, Counter())
        dest = out / name
        dest.mkdir(parents=True, exist_ok=True)
        f, kept = None, 0
        for part in sorted(src_dir.glob("part-*.jsonl")):
            for line in part.open(encoding="utf-8"):
                st["docs_in"] += 1
                doc, reason = clean_document(json.loads(line)["text"], lang, t)
                if reason is None:
                    h = content_hash(doc)
                    reason = "duplicate" if h in seen else None
                    seen.add(h)
                if reason:
                    st[f"drop_{reason}"] += 1
                    continue
                if kept % part_docs == 0:
                    if f:
                        f.close()
                    f = open(dest / f"part-{kept // part_docs:05d}.jsonl", "w", encoding="utf-8")
                f.write(json.dumps({"text": doc, "source": name}, ensure_ascii=False) + "\n")
                kept += 1
                st["docs_out"] += 1
                st["chars_out"] += len(doc)
        if f:
            f.close()
    result = {name: dict(c) for name, c in stats.items()}
    (out / "stats.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw")
    ap.add_argument("--out", default="data/clean")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(clean_tree(Path(a.input), out, Thresholds())))
