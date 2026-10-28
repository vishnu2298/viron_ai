# Viron AI

> A standalone language model trained from scratch (`model/`), plus an elastic runtime that serves it with the least compute each request needs. No external AI APIs, no pretrained weights.

Viron **shrinks** to the least compute a request needs and **expands** only when it must, entirely on its own
weights. Target footprint: ~2–4 GB of weights on one machine.

```
Client → Cloudflare Worker (auth, KV cache of confident answers; origin down → 503, cache hits still served)
       → Orchestrator (Node.js)
           route → retrieve (MongoDB text/vector search)
           → Viron tiny (+LoRA) ──unsure──▶ Viron core (+LoRA) ──unsure──▶ best local answer (flagged in meta.attempts)
           → trace (PII-redacted) → MongoDB
model/:    own BPE tokenizer → pretraining from random init → SFT → checkpoints → GGUF (M5) → llama-server tiers
       ↻ training/: redacted Viron traces → LoRA skill adapters on Viron checkpoints (M5)
```
No external AI APIs, no pretrained weights or tokenizers, no training on other AI products' outputs (`CLAUDE.md`).

| Tier | What | Footprint |
|---|---|---|
| edge-cache | repeated questions with a confident answer | 0 compute |
| tiny | Viron small preset, quantized | ~0.5–1 GB |
| core | larger Viron preset, same tokenizer | ~2 GB |
| adapters | LoRA skills on a specific Viron checkpoint | 10–100 MB each |
| knowledge | MongoDB, outside the weights | unbounded |

When neither tier is confident, the core answer is still returned (degrade, don't fail); every tier down → 503.

## How it improves
Each training run is logged in `docs/EXPERIMENTS.md` and gated on validation loss/eval per dollar (`docs/ROADMAP.md`).
At runtime, redacted traces show where Viron is unsure; they feed data selection, SFT, and per-skill LoRA adapters.
**North-star:** best eval score per dollar of compute, then the share of requests answered confidently by `tiny`.

## Run
```bash
cp .env.example .env               # set keys
# 1. train Viron checkpoints from scratch (model/README.md); no pretrained weights are used
# 2. export to GGUF into ./models (roadmap M5)
docker compose up -d
curl -s localhost:8080/api/v1/chat -H 'authorization: Bearer change-me' \
  -H 'content-type: application/json' -d '{"message":"How do refunds work?"}'
```
Config is validated at startup (`orchestrator/src/config/load.js`); an invalid env value aborts boot with every problem listed by variable name.
Until adapters exist, remove the `--lora` lines from `llm-core`.
Retrieval: leave `EMBED_URL` unset to use MongoDB text search; vector search needs embeddings from a Viron model,
not a pretrained embedder.
The per-request `lora` field needs a recent `llama-server`; the image is pinned in `docker-compose.yml` (`x-llama-image`).

### Upgrading llama.cpp
1. Pick a `server-b<build>` tag from `ghcr.io/ggml-org/llama.cpp` and get its index digest:
   `docker buildx imagetools inspect ghcr.io/ggml-org/llama.cpp:server-b<build>` (the top-level `Digest:`).
2. Update the tag and digest in `x-llama-image`, then re-verify per-request `lora`, `logprobs`, and `/v1/embeddings`.

## Learning loop
```bash
cd training && pip install -r requirements.txt
MONGODB_URI=... python export_traces.py --out data --days 30 --min-examples 200
python train_lora.py --base <local-viron-core-checkpoint> --data data/support.jsonl --out adapters/support-core
LLAMA_CPP=~/llama.cpp BASE=<local-viron-core-checkpoint> NAME=core ./build_artifacts.sh
```
Adapters are tied to their base weights: train per Viron tier. `--base`/`BASE` must be a local Viron checkpoint
exported in HF layout (roadmap M5); Hub model IDs are rejected. Until M5 lands, this loop has no valid input.

## Behind a proxy
- `TRUST_PROXY` is passed to Express `trust proxy` (`false` default, `true`, a hop count like `1`, or subnets/names like `loopback`), so `req.ip` reflects `X-Forwarded-For` from trusted hops only.
- `TRUST_CF_IP=true` keys the rate limiter on `CF-Connecting-IP`. Enable it only when the origin accepts traffic solely from Cloudflare (Tunnel or IP allow-list); otherwise clients can spoof the header.

## Edge
```bash
cd edge && wrangler kv namespace create CACHE   # put id in wrangler.toml
wrangler secret put EDGE_API_KEY && wrangler secret put ORIGIN_API_KEY && wrangler deploy
```

## Tests
```bash
cd model && pip install -r requirements.txt && pytest -q   # shapes, causality, KV cache, overfit, train/resume/sample
cd orchestrator && npm i && npm test     # routing, confidence gate, tiny→core ladder, API, PII redaction
cd training && pip install -r requirements-dev.txt && pytest -q   # dataset export, LoRA base guard (+coverage)
cd edge && node test.mjs                 # auth, cache policy, origin-down 503
```
Coverage: `cd orchestrator && npm run test:coverage` (fails below 80% lines/branches/functions on `src/services/`);
`cd training && pytest` reports coverage for `export_traces.py` (fails below 80%).

## Lint
```bash
cd orchestrator && npm run lint          # ESLint + Prettier check (npm run format to fix)
cd edge && npm ci && npm run lint
cd training && pip install -r requirements-dev.txt && ruff check . && ruff format --check .
```

## Known limits
- No trained Viron checkpoint yet; the runtime is tested against fake model servers (roadmap M1–M5).
- The router is heuristic. Replace it with a classifier trained on traces after ~5k requests.
- Logprob confidence is a proxy. Add a verifier step for high-stakes domains.
- Tiers are discrete model sizes. Nested-width / early-exit elasticity in one checkpoint is roadmap M7.
- Exact-match edge cache only. Semantic caching is a next step.

## Contributing with Claude Code
See `CLAUDE.md` for project rules and `docs/ROADMAP.md` for the task list.
