# Viron AI roadmap

Status: v0.1 scaffold — all suites pass against mock model servers. No real model wired yet.
North-star metric: `localResolutionRate` (`GET /api/v1/stats`) at equal or better answer quality.

## Phase 0 — Hardening the scaffold
- [x] Add ESLint + Prettier (orchestrator, edge) and ruff (training); add lint jobs to CI.
- [x] Add coverage (`node --test --experimental-test-coverage`, `pytest --cov`); target ≥80% on `services/`.
- [x] `app.set('trust proxy', …)` and key the rate limiter on `CF-Connecting-IP` when behind Cloudflare.
- [x] Edge: use a constant-time compare for `EDGE_API_KEY`.
- [ ] Edge: cache only answers that were confident locally or came from escalation (check `meta.attempts`); never cache `edge-fallback`.
- [ ] Orchestrator: send retrieved context as a separate system/context message, not prefixed into the user turn, so history stays clean. Update `export_traces.py` accordingly.
- [ ] Config validation at startup (Joi schema over `config/index.js`); fail fast on bad values.
- [ ] Graceful shutdown: drain in-flight requests before closing Mongo.
- [ ] Pin `llama.cpp` server image to a tag/digest instead of `:server`.

## Phase 1 — Real models (needs user decision: base model family + license)
- [ ] Write `docs/MODELS.md` comparing 2–3 small open-weight families that ship ≥2 sizes; stop for user choice.
- [ ] `scripts/fetch_models.sh`: download chosen tiny/core/embedding models, convert + quantize via `build_artifacts.sh`.
- [ ] Verify on the pinned `llama-server`: per-request `lora`, `logprobs` in chat completions, `/v1/embeddings`. Record findings in `docs/MODELS.md`.
- [ ] `scripts/smoke.sh`: bring up compose, hit `/api/v1/chat` with 5 canned prompts, assert each tier is reachable.
- [ ] Calibrate confidence thresholds on a small labelled set; document chosen values.

## Phase 2 — Knowledge layer
- [ ] `orchestrator/scripts/ingest.js`: files/URLs → chunk (by tokens, with overlap) → embed → `knowledge.upsert`. Idempotent by content hash.
- [ ] `scripts/create_vector_index.js` for the `knowledge_vector` Atlas index (dims from the embedder).
- [ ] Hybrid retrieval: vector + text with reciprocal-rank fusion; tests with fake repos.
- [ ] Return citations (`contextIds`) in the API `meta`.

## Phase 3 — Evaluation & observability
- [ ] `eval/` harness: golden JSONL (synthetic), run per tier and full ladder, report accuracy (LLM-judge via escalation model, opt-in), latency p50/p95, escalation rate, cost estimate. Markdown report output.
- [ ] Prometheus `/metrics`: requests by `servedBy`, latency histograms, escalation counter, adapter usage.
- [ ] Extend `/stats` with per-adapter resolution rate.

## Phase 4 — Learned router
- [ ] Export router features + outcome labels from traces (which tier first succeeded).
- [ ] Train a small classifier (logistic regression on embeddings) in `training/router/`; export weights as JSON.
- [ ] `router.learned.js` loads weights; feature-flag + shadow mode comparing against the heuristic; promote when better on eval.

## Phase 5 — Product surface
- [ ] SSE streaming on `/api/v1/chat/stream` (local tiers stream; escalation streams via the hosted API).
- [ ] Semantic cache at the edge (Cloudflare Vectorize) with similarity threshold + TTL.
- [ ] Per-tenant API keys and quotas (Mongo-backed), replacing the single shared key.
- [ ] OpenAI-compatible facade (`/v1/chat/completions`) so existing clients can point at Viron AI.

## Phase 6 — Research track (true elasticity)
- [ ] Spike: early-exit / nested-width (Matryoshka-style) inference on one checkpoint; write findings before any implementation.
- [ ] Spike: speculative decoding with tiny as draft for core.
