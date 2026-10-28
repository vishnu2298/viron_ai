#!/usr/bin/env bash
# Convert a Viron checkpoint (exported in HF layout, roadmap M5) + its adapters to GGUF for llama.cpp, 4-bit.
# Standalone: BASE must be a local Viron export, never a pretrained model.
# Requires a llama.cpp checkout: git clone https://github.com/ggml-org/llama.cpp
set -euo pipefail
: "${LLAMA_CPP:?set LLAMA_CPP to your llama.cpp checkout}"
: "${BASE:?set BASE to the local Viron checkpoint dir (HF layout)}"
[ -f "$BASE/config.json" ] || { echo "BASE=$BASE is not a local checkpoint dir with config.json" >&2; exit 1; }
NAME="${NAME:-core}"
OUT="${OUT:-../models}"
mkdir -p "$OUT"

python "$LLAMA_CPP/convert_hf_to_gguf.py" "$BASE" --outtype f16 --outfile "$OUT/$NAME-f16.gguf"
"$LLAMA_CPP/build/bin/llama-quantize" "$OUT/$NAME-f16.gguf" "$OUT/$NAME-q4_k_m.gguf" Q4_K_M
rm "$OUT/$NAME-f16.gguf"

for adapter in adapters/*-"$NAME"; do
  [ -d "$adapter" ] || continue
  python "$LLAMA_CPP/convert_lora_to_gguf.py" "$adapter" --base "$BASE" \
    --outfile "$OUT/lora-$(basename "$adapter").gguf"
done
ls -lh "$OUT"
