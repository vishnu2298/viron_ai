"""
Pretrain Viron from random init.
Single GPU:   python train.py --preset tiny --data data/tokens --tokenizer artifacts/tokenizer.json
Multi-GPU:    torchrun --nproc_per_node=8 train.py --preset small ...
Resume:       add --resume (picks up out_dir/ckpt.pt)
"""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from viron.config import ModelConfig, preset
from viron.data import TokenDataset


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    if step >= a.steps:
        return a.min_lr
    r = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (a.lr - a.min_lr)


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", default="checkpoints/run")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16, help="micro-batch per device")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seq", type=int, default=None, help="defaults to preset max_seq_len")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank, local, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
        a.device = f"cuda:{local}"
        torch.cuda.set_device(a.device)
    else:
        rank, world = 0, 1
    master = rank == 0
    torch.manual_seed(a.seed + rank)
    dev_type = "cuda" if a.device.startswith("cuda") else "cpu"
    if dev_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    amp = torch.autocast(dev_type, dtype=torch.bfloat16) if dev_type == "cuda" else nullcontext()

    meta = json.loads((Path(a.data) / "meta.json").read_text())
    out = Path(a.out)
    ckpt_path = out / "ckpt.pt"
    step, best = 0, float("inf")

    if a.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**ck["config"])
    else:
        cfg = preset(a.preset, vocab_size=meta["vocab_size"])
    seq = min(a.seq or cfg.max_seq_len, cfg.max_seq_len)

    from viron.model import Viron

    model = Viron(cfg).to(a.device)
    opt = model.configure_optimizer(a.lr, a.wd, device_type=dev_type)
    if a.resume and ckpt_path.exists():
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, best = ck["step"], ck["best_val"]
    raw = model
    if a.compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    train = TokenDataset(str(Path(a.data) / "train.bin"), cfg.vocab_size)
    val = TokenDataset(str(Path(a.data) / "val.bin"), cfg.vocab_size)

    if master:
        out.mkdir(parents=True, exist_ok=True)
        tokens_per_step = a.batch * a.accum * seq * world
        print(
            json.dumps(
                {
                    "event": "start",
                    "params": raw.num_params(),
                    "preset": a.preset,
                    "tokens_per_step": tokens_per_step,
                    "train_tokens": len(train),
                    "device": a.device,
                }
            )
        )

    @torch.no_grad()
    def evaluate():
        model.eval()
        res = {}
        for name, ds in (("train", train), ("val", val)):
            losses = torch.zeros(a.eval_iters)
            for i in range(a.eval_iters):
                x, y = ds.batch(a.batch, seq, a.device)
                with amp:
                    _, loss = model(x, y)
                losses[i] = loss.item()
            res[name] = losses.mean().item()
        model.train()
        return res

    def save(tag="ckpt"):
        torch.save(
            {
                "model": raw.state_dict(),
                "optimizer": opt.state_dict(),
                "config": cfg.to_dict(),
                "step": step,
                "best_val": best,
                "tokenizer": meta.get("tokenizer"),
            },
            out / f"{tag}.pt",
        )

    model.train()
    t0 = time.time()
    while step < a.steps:
        lr = lr_at(step, a)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_acc = 0.0
        for micro in range(a.accum):
            x, y = train.batch(a.batch, seq, a.device)
            sync = nullcontext() if (not ddp or micro == a.accum - 1) else model.no_sync()
            with sync, amp:
                _, loss = model(x, y)
                loss = loss / a.accum
            loss.backward()
            loss_acc += loss.item()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        if master and step % 10 == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(
                json.dumps(
                    {
                        "event": "train",
                        "step": step,
                        "loss": round(loss_acc, 4),
                        "lr": lr,
                        "gnorm": round(gnorm, 3),
                        "sec_per_10": round(dt, 2),
                    }
                ),
                flush=True,
            )
        if master and (step % a.eval_every == 0 or step == a.steps):
            ev = evaluate()
            print(
                json.dumps(
                    {
                        "event": "eval",
                        "step": step,
                        **{k: round(v, 4) for k, v in ev.items()},
                        "val_ppl": round(math.exp(min(ev["val"], 20)), 2),
                    }
                ),
                flush=True,
            )
            if ev["val"] < best:
                best = ev["val"]
                save("best")
        if master and (step % a.save_every == 0 or step == a.steps):
            save()

    if ddp:
        dist.destroy_process_group()
    return best


if __name__ == "__main__":
    main()
