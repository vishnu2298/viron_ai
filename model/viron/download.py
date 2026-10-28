"""
Stream approved sources from a Hugging Face-compatible hub into `data/raw/<source>/part-*.jsonl`.

Lists files through the public tree API, streams `.jsonl` / `.jsonl.gz` line by line, reads `.parquet` in row-group
batches (pyarrow; Parquet needs random access, so each file is spooled to a temp file first), and stops at a token
budget.
Progress lives in `manifest.json` so an interrupted or budget-limited download resumes at the next file.
"""

import gzip
import io
import json
import os
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path

from .sources import Source

HUB = "https://huggingface.co"
READABLE = (".jsonl", ".jsonl.gz", ".json.gz", ".parquet")
PART_BYTES = 256 * 1024 * 1024  # rotate output files so no single file gets unwieldy


def log(event: str, **kw) -> None:
    print(json.dumps({"event": event, **kw}, ensure_ascii=False), flush=True)


def estimate_tokens(text: str) -> int:
    """Rough budget unit before Viron's tokenizer exists: ~4 UTF-8 bytes per BPE token. Exact counts come later."""
    return max(1, len(text.encode("utf-8")) // 4)


def _request(url: str, token: str | None, retries: int = 4) -> urllib.request.addinfourl:
    headers = {"user-agent": "viron-download/0.1"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    for attempt in range(retries + 1):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
        except urllib.error.HTTPError as err:
            if err.code < 500 or attempt == retries:
                raise
        except urllib.error.URLError:
            if attempt == retries:
                raise
        time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def _next_link(link_header: str | None) -> str | None:
    m = re.search(r'<([^>]+)>;\s*rel="next"', link_header or "")
    return m.group(1) if m else None


def list_files(src: Source, hub: str = HUB, token: str | None = None) -> list[str]:
    """All file paths under `src.prefix`, sorted so runs are reproducible; follows tree-API pagination."""
    prefix = src.prefix.rstrip("/")
    url = f"{hub}/api/datasets/{src.repo}/tree/main/{urllib.parse.quote(prefix)}?recursive=true"
    paths: list[str] = []
    while url:
        with _request(url, token) as res:
            entries = json.load(res)
            url = _next_link(res.headers.get("link"))
        paths += [e["path"] for e in entries if e.get("type") == "file"]
    return sorted(paths)


def iter_texts(stream: io.BufferedIOBase, path: str, field: str) -> Iterator[str]:
    raw = gzip.GzipFile(fileobj=stream) if path.endswith(".gz") else stream
    for line in io.TextIOWrapper(raw, encoding="utf-8", errors="replace"):
        if not line.strip():
            continue
        try:
            text = json.loads(line).get(field)
        except (json.JSONDecodeError, AttributeError):
            continue  # one bad record must not abort a multi-GB shard
        if isinstance(text, str) and text.strip():
            yield text


def iter_parquet_texts(stream: io.BufferedIOBase, field: str, spool_dir: Path, batch_rows: int = 1024) -> Iterator[str]:
    import pyarrow.parquet as pq

    with tempfile.NamedTemporaryFile(dir=spool_dir, suffix=".parquet") as tmp:
        shutil.copyfileobj(stream, tmp, length=1 << 20)
        tmp.flush()
        pf = pq.ParquetFile(tmp.name)
        if field not in pf.schema_arrow.names:
            raise ValueError(f"column {field!r} not in parquet schema {pf.schema_arrow.names}")
        for batch in pf.iter_batches(columns=[field], batch_size=batch_rows):
            for text in batch.column(0).to_pylist():
                if isinstance(text, str) and text.strip():
                    yield text


class _PartWriter:
    def __init__(self, out: Path, start: int):
        self.out, self.index, self.f, self.size = out, start, None, 0

    def write(self, record: dict) -> None:
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        if self.f is None or self.size + len(line) > PART_BYTES:
            self.close()
            self.f = open(self.out / f"part-{self.index:05d}.jsonl", "ab")
            self.index += 1
            self.size = 0
        self.f.write(line)
        self.size += len(line)

    def close(self) -> None:
        if self.f:
            self.f.close()
            self.f = None


def download(
    src: Source,
    out_root: Path,
    max_tokens: int,
    hub: str = HUB,
    token: str | None = None,
    opener: Callable[[str, str | None], io.BufferedIOBase] | None = None,
) -> dict:
    """Download `src` until `max_tokens` (estimated, cumulative across resumes) is reached. Returns the manifest."""
    out = out_root / src.name
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    manifest = (
        json.loads(mpath.read_text())
        if mpath.exists()
        else {"source": asdict(src), "files_done": [], "skipped": [], "docs": 0, "bytes": 0, "est_tokens": 0}
    )
    open_file = opener or (lambda url, tok: _request(url, tok))
    writer = _PartWriter(out, start=len(list(out.glob("part-*.jsonl"))))

    def save(complete: bool) -> None:
        manifest["complete"] = complete
        mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    try:
        for path in list_files(src, hub, token):
            if manifest["est_tokens"] >= max_tokens:
                save(complete=False)
                log("download.budget_reached", source=src.name, est_tokens=manifest["est_tokens"])
                return manifest
            if path in manifest["files_done"] or path in manifest["skipped"]:
                continue
            if not path.endswith(READABLE):
                manifest["skipped"].append(path)  # README, scripts, images, ...
                continue
            url = f"{hub}/datasets/{src.repo}/resolve/main/{urllib.parse.quote(path)}"
            with open_file(url, token) as stream:
                texts = (
                    iter_parquet_texts(stream, src.text_field, out)
                    if path.endswith(".parquet")
                    else iter_texts(stream, path, src.text_field)
                )
                for text in texts:
                    writer.write({"text": text, "source": src.name})
                    manifest["docs"] += 1
                    manifest["bytes"] += len(text.encode("utf-8"))
                    manifest["est_tokens"] += estimate_tokens(text)
                    if manifest["est_tokens"] >= max_tokens:
                        break
            # A file cut by the budget still counts as done: resuming continues with the next file, never re-reads
            manifest["files_done"].append(path)
            save(complete=False)
            log("download.file_done", source=src.name, path=path, est_tokens=manifest["est_tokens"])
    finally:
        writer.close()
    save(complete=manifest["est_tokens"] < max_tokens)
    return manifest


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or None
