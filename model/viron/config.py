from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4  # grouped-query attention: fewer KV heads = smaller KV cache
    hidden_mult: float = 8 / 3  # SwiGLU hidden size = dim * hidden_mult (rounded)
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def hidden_dim(self) -> int:
        h = int(self.dim * self.hidden_mult)
        return 64 * ((h + 63) // 64)

    def to_dict(self) -> dict:
        return asdict(self)


# Growth path. Approximate params assume vocab 32k with tied embeddings.
PRESETS = {
    "nano": dict(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=128),  # tests / CPU
    "tiny": dict(dim=384, n_layers=6, n_heads=6, n_kv_heads=2, max_seq_len=512),  # ~22M,  1 GPU, hours
    "small": dict(dim=768, n_layers=12, n_heads=12, n_kv_heads=4, max_seq_len=1024),  # ~100M, 1 GPU, ~1-2 days
    "base": dict(dim=1024, n_layers=24, n_heads=16, n_kv_heads=4, max_seq_len=2048),  # ~300M, multi-GPU
    "1b": dict(dim=2048, n_layers=22, n_heads=16, n_kv_heads=4, max_seq_len=2048),  # ~1.04B, GPU cluster
}


def preset(name: str, **overrides) -> ModelConfig:
    return ModelConfig(**{**PRESETS[name], **overrides})
