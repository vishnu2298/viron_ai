# Viron AI

> Elastic AI runtime: a small local model that shrinks to the least compute a request needs and expands only when it must.

A small AI system that **shrinks** to the least compute a request needs and **expands** only when it must.
Target footprint: ~2–4 GB of weights on one machine, instead of a giant always-on model.

```
Client → Cloudflare Worker (auth, KV answer cache, Workers AI fallback)
       → Orchestrator (Node.js)
           route → retrieve (MongoDB vector search)
           → tiny model (+LoRA)  ──unsure──▶ core model (+LoRA) ──unsure──▶ escalation (large hosted model)
           → trace (PII-redacted) → MongoDB
       ↻ training/: escalated traces → distilled LoRA adapters → fewer escalations next cycle
```

| Tier | What | Footprint |
|---|---|---|
| edge-cache | repeated questions | 0 compute |
| tiny | small quantized model | ~0.5–1 GB |
| core | larger quantized model of the same family | ~2 GB |
| adapters | LoRA skills, hot-attached per request | 10–100 MB each |
| knowledge | MongoDB, outside the weights | unbounded |
| escalation | hosted large model | pay per hard request |

## Why it gets cheaper over time
Every escalation is stored as a teacher-labelled example. `training/export_traces.py` buckets them per skill,
`train_lora.py` distills a new adapter, and the next deployment answers those questions locally.
**North-star metric:** `localResolutionRate` from `GET /api/v1/stats`.

## Run
```bash
cp .env.example .env               # set keys
# 1. pick a small open-weight model family with two sizes; check its license
# 2. build GGUF artifacts into ./models  (training/build_artifacts.sh)
docker compose up -d
curl -s localhost:8080/api/v1/chat -H 'authorization: Bearer change-me' \
  -H 'content-type: application/json' -d '{"message":"How do refunds work?"}'
```
Config is validated at startup (`orchestrator/src/config/load.js`); an invalid env value aborts boot with every problem listed by variable name.
Until adapters exist, remove the `--lora` lines from `llm-core`.
The per-request `lora` field needs a recent `llama-server`; the image is pinned in `docker-compose.yml` (`x-llama-image`).

### Upgrading llama.cpp
1. Pick a `server-b<build>` tag from `ghcr.io/ggml-org/llama.cpp` and get its index digest:
   `docker buildx imagetools inspect ghcr.io/ggml-org/llama.cpp:server-b<build>` (the top-level `Digest:`).
2. Update the tag and digest in `x-llama-image`, then re-verify per-request `lora`, `logprobs`, and `/v1/embeddings`.

## Learning loop
```bash
cd training && pip install -r requirements.txt
MONGODB_URI=... python export_traces.py --out data --days 30 --min-examples 200
python train_lora.py --base <core-base-model> --data data/support.jsonl --out adapters/support-core
LLAMA_CPP=~/llama.cpp BASE=<core-base-dir> NAME=core ./build_artifacts.sh
```
Adapters are tied to their base weights: train per tier.

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
cd orchestrator && npm i && npm test     # routing, confidence gate, expand/escalate ladder, API, PII redaction
cd training && pip install -r requirements-dev.txt && pytest -q   # dataset export (+coverage)
cd edge && node test.mjs                 # cache + fallback
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
- The router is heuristic. Replace it with a classifier trained on traces after ~5k requests.
- Logprob confidence is a proxy. Add a verifier step for high-stakes domains.
- Tiers are discrete model sizes. True nested-weight elasticity (one checkpoint, variable depth) is phase 3.
- Exact-match edge cache only. Semantic caching is a next step.

## Contributing with Claude Code
See `CLAUDE.md` for project rules and `docs/ROADMAP.md` for the task list.
