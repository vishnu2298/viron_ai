"""Generate text from a Viron checkpoint.
python sample.py --ckpt checkpoints/run/best.pt --prompt "Once upon a time"
"""

import argparse

import torch

from viron.config import ModelConfig
from viron.model import Viron
from viron.tokenizer import load_tokenizer


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Viron(ModelConfig(**ck["config"])).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), load_tokenizer(ck.get("tokenizer"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    model, tok = load(a.ckpt, a.device)
    ids = tok.encode(a.prompt) or [tok.token_id("<|bos|>")]
    x = torch.tensor([ids], device=a.device)
    out = model.generate(x, a.max_new, a.temperature, a.top_p, eos_id=tok.eos_id)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
