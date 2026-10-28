import gzip
import json
import sys
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viron.download import download, estimate_tokens, list_files  # noqa: E402
from viron.sources import SOURCES, Source  # noqa: E402

DOC = "synthetic document text " * 8  # 200 bytes -> 50 estimated tokens


def jsonl(n: int, start: int = 0) -> bytes:
    return "".join(json.dumps({"text": f"{DOC}{i}"}) + "\n" for i in range(start, start + n)).encode()


FILES = {
    "data/a.jsonl": jsonl(3),
    "data/b.jsonl.gz": gzip.compress(jsonl(3, 100) + b"not json\n" + b'{"text": ""}\n' + b'{"other": 1}\n'),
    "data/c.parquet": b"PAR1",
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


def test_full_download_reads_jsonl_and_gzip_skips_bad_rows_and_parquet(hub, tmp_path):
    m = run(tmp_path, max_tokens=10**9, hub=hub)
    rows = read_out(tmp_path)
    assert len(rows) == 9 and {r["source"] for r in rows} == {"fake"}
    assert m["docs"] == 9 and m["complete"] is True
    assert m["skipped"] == ["data/c.parquet"]
    saved = json.loads((tmp_path / "fake" / "manifest.json").read_text())
    assert saved["source"]["license"] == "CC0"


def test_budget_stops_then_resume_continues_with_next_file(hub, tmp_path):
    m = run(tmp_path, max_tokens=90, hub=hub)  # 2 docs (48 est. tokens each) of a.jsonl reach it
    assert m["docs"] == 2 and m["complete"] is False and m["files_done"] == ["data/a.jsonl"]
    m = run(tmp_path, max_tokens=10**9, hub=hub)
    texts = [r["text"] for r in read_out(tmp_path)]
    assert m["complete"] is True and m["docs"] == 8  # the 3rd doc of the cut file is not re-read
    assert len(texts) == len(set(texts)) == 8
    assert sum("/resolve/main/data/a.jsonl" in r for r in FakeHub.requests) == 1


def test_rerun_after_completion_downloads_nothing(hub, tmp_path):
    run(tmp_path, max_tokens=10**9, hub=hub)
    before = len(FakeHub.requests)
    m = run(tmp_path, max_tokens=10**9, hub=hub)
    assert m["docs"] == 9
    assert not any("/resolve/" in r for r in FakeHub.requests[before:])


def test_registry_only_has_approved_licenses():
    assert set(SOURCES) == {"commonpile_comma", "wikipedia_hi", "sangraha_verified_hin"}
    assert all("synthetic" not in s.prefix and "unverified" not in s.prefix for s in SOURCES.values())
