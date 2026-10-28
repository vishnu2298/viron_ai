# Viron roadmap

Status: model code (from scratch) + runtime scaffold, all suites green on CPU. No trained checkpoint yet.
North-star: best validation loss / eval score per dollar of compute, then local resolution rate in the runtime.

## M0 — Go standalone (cleanup)
- [x] Remove `escalation.service.js`, its config/env/tests; the ladder ends at the `core` tier. Return 503 `NO_TIER_AVAILABLE` if all tiers fail.
- [x] Edge: remove Workers AI fallback and the `[ai]` binding; on origin failure return 503 (cache hits still served).
- [x] Edge: constant-time key compare; cache only confident local answers (check `meta.attempts`).
- [x] Mark `training/train_lora.py` as targeting Viron checkpoints (M5); remove HF-pretrained-base assumptions from docs.
- [x] Update root README diagram and `.env.example` to the standalone architecture.
- [x] Lint/format: ruff (Python), ESLint + Prettier (JS); add to CI.

## M1 — Data (needs user decision)
- [x] `docs/DATA.md`: compare 3–5 openly licensed pretraining corpora (license, size, languages incl. Hindi/Indic, quality); stop for user choice.
- [x] `model/scripts/download_data.py`: stream chosen sources to `data/raw/` with a token budget flag.
- [x] Cleaning: exact-dup removal (content hash), length/repetition/language filters, PII redaction (port `orchestrator/src/utils/redact.js` rules to Python with tests).
- [ ] Multi-shard token files + data mixture config (YAML: source → weight).
- [ ] Train the 32k BPE on a representative sample; report tokens-per-word for English and Hindi.

## M2 — Training infrastructure
- [ ] YAML run configs (`model/configs/*.yaml`) instead of long CLIs; `train.py --config`.
- [ ] Log tokens/sec and MFU; optional gradient checkpointing; FSDP path for `base`/`1b`.
- [ ] `model/eval/`: held-out perplexity per data source + a small open-licensed benchmark set (record licenses); markdown report per checkpoint.
- [ ] `docs/EXPERIMENTS.md` template and a script that appends a run summary from logs.

## M3 — First real model
- [ ] Run `tiny` on ~0.5–1B tokens on one GPU (ask before renting). Log it. Write `model/MODEL_CARD.md`.
- [ ] 3–5 single-variable experiments (lr, batch, seq len, data mix); keep the best.

## M4 — Chat
- [ ] `model/sft.py`: loss only on assistant tokens, chat template with reserved tokens, packing.
- [ ] SFT data plan in `docs/DATA.md` (human-written / openly licensed; no other-AI outputs).
- [ ] `model/chat.py` interactive CLI with KV cache and streaming.

## M5 — Serve Viron in the runtime
- [ ] GGUF export: map Viron weights to llama.cpp's llama architecture; verify llama.cpp logits ≈ PyTorch on fixed prompts.
- [ ] Viron `tiny` and `small` become the orchestrator's tiny/core tiers in `docker-compose.yml`.
- [ ] LoRA on Viron checkpoints from runtime traces (redacted), per skill.

## M6 — Scale & align
- [ ] `small` → `base` → `1b`, each gated on eval improvement per dollar.
- [ ] Preference tuning (DPO) with own preference data; long-context extension (RoPE scaling).

## M7 — Elastic Viron (the original idea)
- [ ] Early-exit heads trained jointly so easy tokens leave after a few layers; measure speed vs quality.
- [ ] Nested-width (Matryoshka-style) training so one checkpoint serves several sizes.
- [ ] Speculative decoding: Viron tiny drafts for Viron core.
