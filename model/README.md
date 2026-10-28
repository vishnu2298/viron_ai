# Viron model — trained from scratch

A decoder-only transformer written from first principles in PyTorch. No pretrained weights, no external AI APIs.
Llama-style blocks (pre-RMSNorm, RoPE, grouped-query attention, SwiGLU, tied embeddings) so checkpoints can later be
converted to GGUF and served by the `llama-server` tiers in `docker-compose.yml`.

## Pipeline
```bash
pip install -r requirements.txt
# approved sources only (docs/DATA.md); budgets are estimated tokens, re-run to resume; HF_TOKEN for gated sets
python scripts/download_data.py --source commonpile_comma --max-tokens 850_000_000
python scripts/download_data.py --source sangraha_verified_hin --max-tokens 150_000_000
python scripts/train_tokenizer.py --input data/raw --out artifacts/tokenizer.json --vocab 32000
python scripts/prepare_data.py --input data/raw --tokenizer artifacts/tokenizer.json --out data/tokens
python train.py --preset tiny --data data/tokens --out checkpoints/tiny        # add --resume to continue
python sample.py --ckpt checkpoints/tiny/best.pt --prompt "Once upon a time"
torchrun --nproc_per_node=8 train.py --preset small --data data/tokens ...    # multi-GPU
```

## Growth path ("improve day by day")
| Preset | Params | Hardware | What to expect |
|---|---|---|---|
| nano | 2M | CPU | tests only |
| tiny | 22M | 1 consumer GPU, hours | fluent short text in a narrow domain |
| small | 100M | 1 GPU, 1–2 days, ~2–5B tokens | coherent general text, weak reasoning |
| base | 300M | multi-GPU, days | usable after instruction tuning for simple tasks |
| 1b | 1.04B | GPU cluster, ~20B+ tokens | small assistant tier |

Quality comes from three stages, in order: **pretraining** (data scale and quality), **SFT** (instruction/chat data
with the reserved `<|system|> <|user|> <|assistant|> <|end|>` tokens), then **preference tuning**. Every stage
must use data you have the rights to; do not train on outputs of other AI products.

Frontier-scale models (the hundreds-of-billions class) need compute budgets far beyond a single team; this repo
is built to climb the table above efficiently and to make each checkpoint servable in the elastic runtime.

## Tests
`pytest -q` — shapes, causality, KV-cache equivalence, single-batch overfit, end-to-end train/resume/sample on CPU.
