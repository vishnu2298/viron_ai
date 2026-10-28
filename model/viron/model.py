"""
Decoder-only transformer, written from scratch in PyTorch.
Llama-style blocks (pre-RMSNorm, RoPE, GQA, SwiGLU) so checkpoints can be converted to GGUF later.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_f = x.float()
        return (x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_cache(head_dim: int, max_len: int, theta: float, device=None):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(max_len, device=device).float(), inv)
    return freqs.cos(), freqs.sin()  # (T, head_dim/2)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D); rotate pairs (even, odd)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None], sin[None, None]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.hd, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.hd, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.hd, cfg.dim, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if kv_cache is not None:
            if kv_cache.get("k") is not None:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"], kv_cache["v"] = k, v

        rep = self.n_heads // self.n_kv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # causal only when queries span multiple positions; single-token decode attends to all cached keys
        is_causal = T > 1
        if is_causal and k.size(2) != T:  # prefill on top of an existing cache
            mask = torch.ones(T, k.size(2), dtype=torch.bool, device=x.device).tril(k.size(2) - T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal, dropout_p=self.dropout if self.training else 0.0)
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)  # gate
        self.w3 = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)  # up
        self.w2 = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)  # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = FeedForward(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, kv_cache=None):
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin, kv_cache))
        return x + self.drop(self.ffn(self.ffn_norm(x)))


class Viron(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        # scaled init on residual projections (GPT-2 style) for stable deep training
        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "w2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, kv_caches=None, start_pos: int = 0):
        B, T = idx.shape
        assert start_pos + T <= self.cfg.max_seq_len, "sequence exceeds max_seq_len"
        x = self.tok_emb(idx)
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, kv_caches[i] if kv_caches is not None else None)
        x = self.norm(x)
        if targets is None:
            return self.lm_head(x[:, -1:, :]), None  # only last position needed for generation
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    def configure_optimizer(self, lr: float, weight_decay: float, betas=(0.9, 0.95), device_type="cpu"):
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=(device_type == "cuda"))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_p: float = 0.95,
                 eos_id: int | None = None, use_cache: bool = True):
        self.eval()
        caches = [{} for _ in self.layers] if use_cache else None
        logits, _ = self(idx, kv_caches=caches, start_pos=0) if use_cache else self(idx)
        pos = idx.size(1)
        for _ in range(max_new_tokens):
            nxt = sample_next(logits[:, -1, :], temperature, top_p)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
            if idx.size(1) >= self.cfg.max_seq_len:
                break
            if use_cache:
                logits, _ = self(nxt, kv_caches=caches, start_pos=pos)
                pos += 1
            else:
                logits, _ = self(idx)
        return idx


def sample_next(logits, temperature: float, top_p: float):
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    probs = F.softmax(logits / temperature, dim=-1)
    sorted_p, sorted_i = probs.sort(descending=True)
    cum = sorted_p.cumsum(-1)
    sorted_p[(cum - sorted_p) > top_p] = 0.0
    sorted_p /= sorted_p.sum(-1, keepdim=True)
    return sorted_i.gather(-1, torch.multinomial(sorted_p, 1))
