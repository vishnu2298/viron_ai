"""Serve a Viron checkpoint as an OpenAI-compatible chat endpoint (with logprobs) for the orchestrator.
python serve.py --ckpt checkpoints/tiny/best.pt --port 8081 --name viron-tiny
Then run the orchestrator with TINY_MODEL_URL=http://localhost:8081 and open http://localhost:8080/ui/
"""

import argparse
import json

import torch

from sample import load
from viron.serve import make_server

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8081)
ap.add_argument("--name", default="viron")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
model, tok = load(a.ckpt, a.device)
srv = make_server(model, tok, a.host, a.port, a.device, a.name)
print(json.dumps({"event": "serve.started", "port": a.port, "model": a.name, "params": model.num_params()}), flush=True)
srv.serve_forever()
