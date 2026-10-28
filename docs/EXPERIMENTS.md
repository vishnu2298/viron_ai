# Experiments log

Newest last. One entry per run. Copy the template.

## Template
- **Date / SHA:**
- **Goal (one variable changed):**
- **Preset / params:**
- **Data (sources, tokens):**
- **Hyperparams:** lr, min_lr, warmup, batch×accum×seq, steps, wd
- **Hardware / time / cost:**
- **Result:** final train loss, val loss, val ppl, tokens/sec
- **Conclusion & next step:**

## 2026-10-28 — scaffold sanity (CPU)
- Goal: verify pipeline end to end. Preset nano (2.1M), byte tokenizer, synthetic text.
- Result: tests green — causality, KV-cache equivalence, single-batch overfit, train → resume → sample.
- Next: M0 cleanup, then M1 data selection.
