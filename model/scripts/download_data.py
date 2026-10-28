"""Download approved sources (docs/DATA.md) into data/raw/<source>/ with a token budget.
python scripts/download_data.py --source commonpile_comma --max-tokens 850_000_000
python scripts/download_data.py --source sangraha_verified_hin --max-tokens 150_000_000
Set HF_TOKEN for gated datasets. `--list` checks a source's repo/path without downloading.
Re-running resumes from the manifest; raise --max-tokens to fetch more.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viron.download import HUB, download, hf_token, list_files  # noqa: E402
from viron.sources import SOURCES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, choices=sorted(SOURCES))
ap.add_argument("--max-tokens", type=lambda v: int(v.replace("_", "")), help="estimated-token budget")
ap.add_argument("--list", action="store_true", help="only list the source's files (verifies repo id and path)")
ap.add_argument("--out", default="data/raw")
ap.add_argument("--hub", default=HUB)
a = ap.parse_args()
if a.list:
    files = list_files(SOURCES[a.source], a.hub, hf_token())
    print(json.dumps({"source": a.source, "files": len(files), "first": files[:5]}, indent=2))
    sys.exit(0 if files else f"no files found for {a.source}: check repo/prefix in viron/sources.py")
if a.max_tokens is None:
    ap.error("--max-tokens is required unless --list is given")
manifest = download(SOURCES[a.source], Path(a.out), a.max_tokens, hub=a.hub, token=hf_token())
print(json.dumps({k: manifest[k] for k in ("docs", "bytes", "est_tokens", "complete")}))
