"""Download approved sources (docs/DATA.md) into data/raw/<source>/ with a token budget.
python scripts/download_data.py --source commonpile_comma --max-tokens 850_000_000
python scripts/download_data.py --source sangraha_verified_hin --max-tokens 150_000_000
Set HF_TOKEN for gated datasets. Re-running resumes from the manifest; raise --max-tokens to fetch more.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viron.download import HUB, download, hf_token  # noqa: E402
from viron.sources import SOURCES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, choices=sorted(SOURCES))
ap.add_argument("--max-tokens", type=lambda v: int(v.replace("_", "")), required=True, help="estimated-token budget")
ap.add_argument("--out", default="data/raw")
ap.add_argument("--hub", default=HUB)
a = ap.parse_args()
manifest = download(SOURCES[a.source], Path(a.out), a.max_tokens, hub=a.hub, token=hf_token())
print(json.dumps({k: manifest[k] for k in ("docs", "bytes", "est_tokens", "complete")}))
