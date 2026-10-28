# Viron AI — guide for Claude Code

Elastic AI runtime: serve each request with the least compute that answers it well.
Ladder: edge cache → tiny model → core model (+LoRA adapters, +retrieved knowledge) → hosted large model (escalation).
Escalated answers are traced and distilled into new LoRA adapters, so the local resolution rate rises over time.
Roadmap and task list: `docs/ROADMAP.md`. Work it top to bottom unless the user says otherwise.

## Layout
- `orchestrator/` — Node 22, Express, CommonJS. Controller → Service → Repository.
  - `services/router.service.js` picks tier + adapters (heuristic for now)
  - `services/confidence.service.js` gates on token logprobs
  - `services/elastic.service.js` is the pipeline (route → retrieve → local ladder → escalate → trace)
  - `services/inference.service.js` talks to llama.cpp `llama-server` (OpenAI-compatible, per-request `lora`)
  - `repositories/` MongoDB: `knowledge` (vector/text search), `traces` (learning signal, 90-day TTL)
  - `utils/redact.js` PII redaction, applied before anything is persisted
- `training/` — Python 3.12. `export_traces.py` (traces → per-adapter JSONL), `train_lora.py` (QLoRA), `build_artifacts.sh` (GGUF + Q4_K_M)
- `edge/` — Cloudflare Worker: auth, KV exact-match cache, Workers AI fallback
- `docker-compose.yml` — llm-tiny, llm-core, embedder, mongo (atlas-local), orchestrator

## Commands
- `cd orchestrator && npm ci && npm test` (`npm run test:coverage` enforces ≥80% on `services/`)
- `cd training && pip install -r requirements-dev.txt && pytest -q` (includes coverage)
- `cd edge && node test.mjs`
- Lint: `npm run lint` (orchestrator, edge); `ruff check . && ruff format --check .` (training)
- `docker compose up -d` (needs GGUF files in `./models`, not in git)
CI (`.github/workflows/ci.yml`) runs all three suites; keep it green.

## Hard rules
1. **Public repo.** Never commit secrets, `.env`, model weights, training data, traces, real user data, PII, or any employer/company-internal code, names, schemas, or data. Examples and fixtures must be synthetic and generic.
2. Everything persisted or exported for training passes through `redact()` first. New PII formats need a test in `test/unit.test.js`.
3. Dependencies are injected via factory functions (`createXService({ deps })`). Keep services testable without network; tests use local fake servers, never real APIs.
4. API response shape: `{ success, message, data, meta }` / errors `{ success:false, message, code }`. Version routes under `/api/v1`; do not break v1.
5. A LoRA adapter only works on the base model it was trained on. Never wire an adapter to a tier with a different base.
6. Degrade, don't fail: retrieval or trace failures must not fail a request; a down tier falls through to the next.
7. Ask the user before: adding a dependency, changing the public API or env var names, choosing a base model or license, deleting files, or anything that costs money (cloud resources, paid API calls).

## Workflow
- One roadmap task per branch: `feat/<short-name>` or `fix/<short-name>`. Conventional commits.
- Before each commit: run the affected test suites; add tests for new behavior; update README/ROADMAP if behavior or commands change.
- Open a PR per task with: what changed, why, how it was tested, follow-ups. Do not push to `main` directly.
- Tick the task in `docs/ROADMAP.md` in the same PR.
- If a task is blocked on a decision, write the options with trade-offs in the PR/issue and stop; don't guess.

## Style
- JS: async/await, no classes unless state demands it, structured winston logs (`{ event, ... }`), Joi validation on every input.
- Python: type hints, pure functions for logic (testable), I/O at the edges, batch DB reads.
- Keep files small and single-purpose. Comments explain *why*, not *what*.
