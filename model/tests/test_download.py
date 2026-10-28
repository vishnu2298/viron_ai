import gzip
import io
import json
import sys
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viron.download import download, estimate_tokens, list_files  # noqa: E402
from viron.sources import SOURCES, Source  # noqa: E402

DOC = "synthetic document text " * 8  # 200 bytes -> 50 estimated tokens


def jsonl(n: int, start: int = 0) -> bytes:
    return "".join(json.dumps({"text": f"{DOC}{i}"}) + "\n" for i in range(start, start + n)).encode()


def parquet(n: int, start: int) -> bytes:
    buf = io.BytesIO()
    texts = [f"{DOC}{i}" for i in range(start, start + n)] + ["", None]  # empty/null rows are skipped
    pq.write_table(pa.table({"id": list(range(n + 2)), "text": texts}), buf, row_group_size=2)
    return buf.getvalue()


FILES = {
    "data/a.jsonl": jsonl(3),
    "data/b.jsonl.gz": gzip.compress(jsonl(3, 100) + b"not json\n" + b'{"text": ""}\n' + b'{"other": 1}\n'),
    "data/c.parquet": parquet(3, 300),
    "data/notes.md": b"# readme",
    "data/d.jsonl": jsonl(3, 200),
}


class FakeHub(BaseHTTPRequestHandler):
    requests: list[str] = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        FakeHub.requests.append(self.path)
        if self.path.startswith("/api/datasets/org/ds/tree/main/data"):
            # Two pages, to exercise Link-header pagination
            paths = sorted(FILES)
            page2 = "cursor=2" in self.path
            body = json.dumps(
                [{"type": "file", "path": p} for p in (paths[2:] if page2 else paths[:2])]
                + ([] if page2 else [{"type": "directory", "path": "data/sub"}])
            )
            self.send_response(200)
            if not page2:
                nxt = f"http://{self.headers['host']}/api/datasets/org/ds/tree/main/data?recursive=true&cursor=2"
                self.send_header("link", f'<{nxt}>; rel="next"')
            self.end_headers()
            self.wfile.write(body.encode())
            return
        prefix = "/datasets/org/ds/resolve/main/"
        if self.path.startswith(prefix) and self.path[len(prefix) :] in FILES:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(FILES[self.path[len(prefix) :]])
            return
        self.send_response(404)
        self.end_headers()


@pytest.fixture()
def hub():
    FakeHub.requests = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeHub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


SRC = Source("fake", "org/ds", "data/", "en", "CC0")
run = partial(download, SRC)


def read_out(out: Path) -> list[dict]:
    return [json.loads(line) for p in sorted((out / "fake").glob("part-*.jsonl")) for line in p.open()]


def test_estimate_tokens_counts_utf8_bytes():
    assert estimate_tokens("abcd" * 10) == 10
    assert estimate_tokens("हिन्दी") == 4  # 6 chars x 3 bytes = 18 bytes
    assert estimate_tokens("") == 1


def test_list_files_follows_pagination_and_skips_dirs(hub):
    assert list_files(SRC, hub) == sorted(FILES)


def test_full_download_reads_jsonl_gzip_and_parquet_skips_bad_rows(hub, tmp_path):
    m = run(tmp_path, max_tokens=10**9, hub=hub)
    rows = read_out(tmp_path)
    assert len(rows) == 12 and {r["source"] for r in rows} == {"fake"}
    assert sum(r["text"].endswith(("300", "301", "302")) for r in rows) == 3
    assert m["docs"] == 12 and m["complete"] is True
    assert m["skipped"] == ["data/notes.md"]
    assert not list((tmp_path / "fake").glob("*.parquet*"))  # spooled temp files are removed
    saved = json.loads((tmp_path / "fake" / "manifest.json").read_text())
    assert saved["source"]["license"] == "CC0"


def test_budget_stops_then_resume_continues_with_next_file(hub, tmp_path):
    m = run(tmp_path, max_tokens=90, hub=hub)  # 2 docs (48 est. tokens each) of a.jsonl reach it
    assert m["docs"] == 2 and m["complete"] is False and m["files_done"] == ["data/a.jsonl"]
    m = run(tmp_path, max_tokens=10**9, hub=hub)
    texts = [r["text"] for r in read_out(tmp_path)]
    assert m["complete"] is True and m["docs"] == 11  # the 3rd doc of the cut file is not re-read
    assert len(texts) == len(set(texts)) == 11
    assert sum("/resolve/main/data/a.jsonl" in r for r in FakeHub.requests) == 1


def test_rerun_after_completion_downloads_nothing(hub, tmp_path):
    run(tmp_path, max_tokens=10**9, hub=hub)
    before = len(FakeHub.requests)
    m = run(tmp_path, max_tokens=10**9, hub=hub)
    assert m["docs"] == 12
    assert not any("/resolve/" in r for r in FakeHub.requests[before:])


def test_parquet_missing_text_column_is_an_error(tmp_path):
    from viron.download import iter_parquet_texts

    buf = io.BytesIO()
    pq.write_table(pa.table({"body": ["x"]}), buf)
    with pytest.raises(ValueError, match="column 'text'"):
        list(iter_parquet_texts(io.BytesIO(buf.getvalue()), "text", tmp_path))


def test_parquet_temp_file_is_closed_before_reopen_and_removed_on_early_stop(tmp_path, monkeypatch):
    """Regression for Windows WinError 32: the spooled file must not be held open when pyarrow opens it."""
    from viron import download as dl

    real_open = pq.ParquetFile
    seen = {}

    def strict_open(path, *args, **kwargs):
        # On Windows an open writer handle makes this fail; emulate by checking the file is fully written and closed
        seen["path"] = path
        with open(path, "rb") as f:  # would raise PermissionError on Windows if a handle were still open
            assert f.read(4) == b"PAR1"
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", strict_open)
    texts = dl.iter_parquet_texts(io.BytesIO(parquet(5, 0)), "text", tmp_path, batch_rows=2)
    assert next(texts).endswith("0")
    texts.close()  # caller stops early, like a token budget cut
    assert Path(seen["path"]).parent == tmp_path
    assert list(tmp_path.iterdir()) == []


def test_registry_only_has_approved_licenses():
    assert set(SOURCES) == {"commonpile_comma", "wikipedia_hi", "sangraha_verified_hin"}
    assert all("synthetic" not in s.prefix and "unverified" not in s.prefix for s in SOURCES.values())
