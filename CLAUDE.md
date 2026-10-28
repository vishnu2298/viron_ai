# Viron AI — guide for Claude Code

**Mission:** build Viron, a standalone language model trained from scratch, plus an elastic runtime that serves it
with the least compute per request. It improves in small, measured steps: every session leaves a better model,
better data, or better tooling, and a log entry proving it.

Task list: `docs/ROADMAP.md` (work top to bottom). Experiment log: `docs/EXPERIMENTS.md`.

## Non-negotiable: standalone
1. No external AI APIs anywhere in the product path (no OpenAI/Anthropic/Gemini/Workers AI/etc. calls).
2. No pretrained weights or pretrained tokenizers. Viron starts from random init and its own BPE.
3. No training on outputs of other AI products (distillation from them), and no data without a license that permits training. Record every dataset's source and license in `docs/DATA.md`.
Libraries (PyTorch, numpy, `tokenizers` as a BPE algorithm, llama.cpp as an inference engine) are fine.
Existing code that violates this (`orchestrator/src/services/escalation.service.js`, Workers AI fallback in `edge/`, `training/train_lora.py` on HF bases) is removed or rewired in roadmap M0.

## Layout
- `model/` — **the model.** PyTorch, Python 3.12.
  - `viron/config.py` presets nano→1b · `viron/model.py` transformer (RMSNorm, RoPE, GQA, SwiGLU, KV cache)
  - `viron/tokenizer.py` own BPE + byte fallback; chat tokens `<|system|> <|user|> <|assistant|> <|end|>` reserved
  - `viron/data.py` memmap token shards · `scripts/` tokenizer + data prep · `train.py` (DDP, resume) · `sample.py`
- `orchestrator/` — Node 22 Express runtime: route → retrieve (MongoDB) → tiny → core. Controller → Service → Repository.
- `training/` — trace export (keep) and LoRA tooling (to be retargeted at Viron checkpoints).
- `edge/` — Cloudflare Worker: auth + KV cache.

## Commands
- `cd model && pip install -r requirements.txt && pytest -q` (CPU torch is enough for tests)
- `cd orchestrator && npm ci && npm test` (`npm run test:coverage` enforces ≥80% on `services/`)
- `cd training && pip install -r requirements-dev.txt && pytest -q` (includes coverage) · `cd edge && npm ci && node test.mjs`
- Lint: `npm run lint` (orchestrator, edge); `ruff check . && ruff format --check .` (training)
CI runs all four suites; keep it green.

## Hard rules
1. **Public repo.** Never commit secrets, `.env`, checkpoints (`*.pt`), token shards (`*.bin`), raw datasets, real user data, PII, or any employer/company-internal code, names, or data. Fixtures are synthetic.
2. Model changes need tests: shapes, causality, KV-cache == full recompute, overfit-one-batch must stay green.
3. Every training run gets an entry in `docs/EXPERIMENTS.md`: date, git SHA, preset, data + token count, hyperparams, steps, final train/val loss, tokens/sec, hardware, cost, conclusion, next step. No entry = run didn't happen.
4. Change one variable per experiment; compare against the previous best on the same validation set.
5. Ask the user before: choosing datasets or licenses, spending money (GPU rental, cloud), runs longer than ~1 hour, adding dependencies, changing public API/env names, deleting files.
6. Degrade, don't fail (runtime): a failing tier falls through; retrieval/trace failures never fail a request.
7. PII: anything persisted or used as training data passes through redaction first.

## Workflow
- One roadmap task per branch (`feat/…`, `fix/…`, `exp/…` for experiments), conventional commits, PR per task with what/why/how-tested. Never push to `main`.
- Tick the task in `docs/ROADMAP.md` in the same PR. Blocked on a decision → write options with trade-offs and stop.
- Start of each session: read `docs/EXPERIMENTS.md` tail and the next unchecked roadmap task; state the plan before editing.

## Style
- Python: type hints, pure functions for logic, I/O at the edges, `json` one-line logs (`{"event": ...}`).
- JS: async/await, factory-injected deps, Joi validation, winston structured logs.
- Small single-purpose files. Comments explain *why*.
