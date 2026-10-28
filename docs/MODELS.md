# Base model family — decision record

Status: **awaiting decision** (Phase 1, task 1). Nothing downstream (`scripts/fetch_models.sh`, adapters, threshold
calibration) starts until a family is chosen, because LoRA adapters are tied to their base weights (CLAUDE.md rule 5).

Researched 2026-10-28. Hugging Face model cards could not be read from the build environment, so figures below come from
vendor docs and secondary sources; anything marked **verify** must be confirmed on the model card before we commit.

## What the runtime needs from a family
| Requirement | Why | Hard? |
|---|---|---|
| ≥2 sizes sharing one tokenizer and chat template | tiny/core tiers; same prompts, same trace format; enables speculative decoding (Phase 6) | yes |
| Permissive license for commercial use and for distilling teacher outputs into adapters | public repo, product use | yes |
| GGUF + llama.cpp `llama-server` support incl. **per-request `lora`** and `logprobs` | `inference.service.js` relies on both | yes |
| Standard projection names LoRA can target (`q/k/v/o_proj`, `gate/up/down_proj`) | `train_lora.py` targets these today | strongly preferred |
| Footprint at Q4_K_M: tiny ≈ 0.5–1 GB, core ≈ 2–3 GB | README target: 2–4 GB total on one box | soft |
| Thinking can be switched off | logprob confidence and latency assume a direct answer, not a reasoning trace | soft |
| A matching small embedding model (same or permissive license) | retrieval (`EMBED_URL`) | soft |

`logprobs` and per-request `lora` are implemented model-agnostically in `llama-server`; the real risk is
**architecture support for LoRA on non-standard layers** (linear attention, state-space, per-layer embeddings).

## Candidates

### A. Qwen3.5 small — 0.8B / 2B / 4B / 9B (Alibaba, Mar 2026)
- **License:** Apache 2.0.
- **Architecture:** hybrid — Gated DeltaNet (linear attention) and full softmax attention in roughly 3:1, dense. Native
  vision (separate projector; can be skipped for text-only). 262K context.
- **Tiers:** tiny = 0.8B (~0.6 GB Q4), core = 4B (~2.6 GB Q4). 2B is a middle option if 4B is too slow on CPU.
- **Strengths:** strongest quality per parameter of the three on public leaderboards; one tokenizer across sizes; Apache.
- **Risks:**
  - LoRA on DeltaNet layers in llama.cpp is unproven for per-request hot-swap (**verify**). Our `target_modules` would
    only hit the full-attention and MLP blocks unless extended to the DeltaNet projection names.
  - Recurrent state limits llama.cpp prompt-cache reuse across requests (latency, not correctness).
  - Reasoning-style models: must confirm a non-thinking mode via the chat template (**verify**).
- **Embeddings:** Qwen3-Embedding-0.6B (Apache 2.0, 1024 dims).

### B. Gemma 4 — E2B / E4B (+ 26B-A4B MoE, 31B) (Google, Apr 2026)
- **License:** Apache 2.0 (a change from the custom Gemma terms of earlier generations) — **verify** on the card.
- **Architecture:** standard attention with Per-Layer Embeddings (PLE) and KV-cache sharing across later layers.
  "E" = effective parameters (~2.3B / ~4.5B); PLE tables are a large share of total weights. Text+image (+audio on E-models).
  128K context on E2B/E4B.
- **Tiers:** tiny = E2B, core = E4B. There is no sub-1B size.
- **Strengths:** Apache, strong multilingual and on-device focus; standard projection names, so `train_lora.py` works as is.
  If E2B is nested inside E4B (as with Gemma 3n's MatFormer — **verify** for Gemma 4), it is directly relevant to the
  Phase 6 nested-width research track.
- **Risks:**
  - GGUF files are larger than effective size suggests because PLE weights ship with the model (tiny likely >1 GB at Q4;
    **verify**), which breaks the footprint target unless PLE is offloaded.
  - llama.cpp handling of PLE + shared-KV with LoRA hot-swap needs the Phase 1 verification step.
  - Big jump from E4B to 26B; no mid-size dense model.
- **Embeddings:** EmbeddingGemma-300m is under the older Gemma terms (**verify** whether it was relicensed);
  otherwise pair with a permissive embedder (Qwen3-Embedding-0.6B or Granite Embedding).

### C. Granite 4.0 — 350M / 1B / Micro 3B (dense) and H-variants (hybrid) (IBM)
- **License:** Apache 2.0.
- **Architecture:** IBM ships both hybrid Mamba-2/transformer ("H") and **conventional dense transformer** variants of the
  small sizes, and explicitly recommends the dense ones where Mamba-2 support is immature (llama.cpp, PEFT).
- **Tiers:** tiny = 1B dense (or 350M), core = Micro 3B dense (~2 GB Q4).
- **Strengths:** the lowest-risk path for our exact stack — dense transformers with standard projections, so per-request
  LoRA and `train_lora.py` should work unchanged; enterprise-oriented provenance documentation; Apache throughout,
  including Granite Embedding models (English R2, 768 dims).
- **Risks:** lower general quality than A or B at equal size, so more traffic escalates early on (higher teacher cost
  until adapters close the gap); confirm all chosen sizes share one tokenizer (**verify**).

### Not shortlisted
- **Qwen3 (2025) 0.6B/1.7B/4B/8B**, Apache 2.0, standard transformer with mature llama.cpp + LoRA support. Not newest, but
  the **fallback if A fails verification** — same vendor, same tokenizer family as Qwen3-Embedding.
- Llama 3.2 1B/3B: custom community license with use restrictions and attribution requirements.
- Phi-4-mini (MIT): single small size, no tier pair.

## Comparison
| | A. Qwen3.5 small | B. Gemma 4 E-series | C. Granite 4.0 dense |
|---|---|---|---|
| License | Apache 2.0 | Apache 2.0 (verify) | Apache 2.0 |
| Tiny / core | 0.8B / 4B | E2B / E4B | 1B / 3B |
| Footprint fit (Q4) | good | at risk (PLE) | good |
| Expected quality | highest | high | moderate |
| LoRA hot-swap risk in llama.cpp | **high** (DeltaNet) | medium (PLE, shared KV) | **low** |
| `train_lora.py` unchanged | partial | yes | yes |
| Non-thinking answers | verify | yes | yes |
| Permissive embedder | Qwen3-Embedding-0.6B | needs pairing | Granite Embedding |

## Recommendation
**A (Qwen3.5 small: 0.8B tiny, 4B core) with Qwen3 as the pre-agreed fallback**, gated on the Phase 1 verification task:
per-request `lora` on a DeltaNet-hybrid checkpoint, `logprobs`, non-thinking mode, and `/v1/embeddings` on the pinned
`llama-server` (`server-b11946`). Quality per GB matters most for `localResolutionRate`, and Qwen3.5 leads there; if
hot-swapped LoRA is broken or degrades the hybrid layers, switch to Qwen3 (same vendor and license, mature support)
before any adapter is trained, so no adapter work is lost.

Choose **C** instead if predictability matters more than headline quality (e.g. a fixed demo date): it is the only
option where every hard requirement is expected to hold today.

## Decision
- Family: _pending_
- Tiny / core / embedder: _pending_
- License reviewed by: _pending_
- llama.cpp verification results (Phase 1, task 3): _pending_
