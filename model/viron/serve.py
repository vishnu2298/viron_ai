"""
Serve a Viron checkpoint over an OpenAI-compatible `/v1/chat/completions` endpoint with token logprobs.

This is the dev path for live verification before GGUF export (roadmap M5): point the orchestrator's
TINY_MODEL_URL / CORE_MODEL_URL here. Stdlib HTTP + one lock, so it serves one generation at a time.
Per-request `lora` is accepted and ignored (no adapters exist for Viron yet), so the orchestrator needs no changes.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn.functional as F

from .model import sample_next

ROLE_TOKENS = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}


def encode_chat(tok, messages: list[dict]) -> list[int]:
    """<|role|> text <|end|> per turn, then <|assistant|>; specials are added by id so any tokenizer works."""
    ids = [tok.token_id("<|bos|>")]
    for m in messages:
        role = ROLE_TOKENS.get(m.get("role"))
        if role is None:
            raise ValueError(f"unsupported role {m.get('role')!r}")
        ids += [tok.token_id(role), *tok.encode(str(m.get("content", ""))), tok.token_id("<|end|>")]
    return ids + [tok.token_id("<|assistant|>")]


@torch.no_grad()
def generate(model, ids: list[int], max_new: int, temperature: float, top_p: float, stop_ids: set[int], device: str):
    """Returns (generated ids, logprob of each under the untempered distribution, finish_reason)."""
    model.eval()
    limit = model.cfg.max_seq_len
    ids = ids[-(limit - 1) :]  # keep the most recent context if the prompt is too long
    caches = [{} for _ in model.layers]
    logits, _ = model(torch.tensor([ids], device=device), kv_caches=caches, start_pos=0)
    pos, out, lps = len(ids), [], []
    for _ in range(max_new):
        last = logits[:, -1, :].float()
        nxt = sample_next(last, temperature, top_p)
        tid = int(nxt.item())
        if tid in stop_ids:
            return out, lps, "stop"
        out.append(tid)
        lps.append(float(F.log_softmax(last, dim=-1)[0, tid]))
        if pos + 1 >= limit:
            return out, lps, "length"
        logits, _ = model(nxt, kv_caches=caches, start_pos=pos)
        pos += 1
    return out, lps, "length"


def completion(model, tok, body: dict, device: str) -> dict:
    messages = body.get("messages") or []
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    prompt = encode_chat(tok, messages)
    max_new = max(1, min(int(body.get("max_tokens") or 256), 2048))
    temperature = float(body.get("temperature", 0.7))
    stop = {tok.token_id("<|end|>"), tok.eos_id}
    ids, lps, finish = generate(model, prompt, max_new, temperature, float(body.get("top_p", 0.95)), stop, device)
    choice = {"index": 0, "message": {"role": "assistant", "content": tok.decode(ids)}, "finish_reason": finish}
    if body.get("logprobs"):
        choice["logprobs"] = {
            "content": [{"token": tok.decode([t]), "logprob": round(lp, 4)} for t, lp in zip(ids, lps, strict=True)]
        }
    return {
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "viron",
        "choices": [choice],
        "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(ids), "total_tokens": len(prompt) + len(ids)},
    }


def make_server(model, tok, host: str, port: int, device: str, name: str = "viron") -> ThreadingHTTPServer:
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, status: int, payload: dict):
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"status": "ok", "model": name, "params": model.num_params()})
            self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self._json(404, {"error": {"message": "not found"}})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
                t0 = time.time()
                with lock:
                    res = completion(model, tok, body, device)
                res["model"] = name
                print(
                    json.dumps({"event": "serve.completion", "ms": int((time.time() - t0) * 1000), **res["usage"]}),
                    flush=True,
                )
                self._json(200, res)
            except (ValueError, KeyError, json.JSONDecodeError) as err:
                self._json(400, {"error": {"message": str(err)}})

    return ThreadingHTTPServer((host, port), Handler)
