"""Token shards on disk (numpy memmap) and random-window batching — scales past RAM."""

from pathlib import Path

import numpy as np
import torch


def dtype_for(vocab_size: int):
    return np.uint16 if vocab_size < 65536 else np.uint32


class TokenDataset:
    def __init__(self, path: str, vocab_size: int):
        self.data = np.memmap(path, dtype=dtype_for(vocab_size), mode="r")

    def __len__(self):
        return len(self.data)

    def batch(self, batch_size: int, seq_len: int, device: str, generator: torch.Generator | None = None):
        hi = len(self.data) - seq_len - 1
        if hi <= 0:
            raise ValueError(f"dataset ({len(self.data)} tokens) shorter than seq_len {seq_len}")
        ix = torch.randint(hi, (batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy(self.data[i : i + seq_len].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1 : i + 1 + seq_len].astype(np.int64)) for i in ix])
        if device.startswith("cuda"):
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)


class ShardedDataset:
    """Several memmapped shards of one source. Windows never cross a shard boundary (shards end at doc ends)."""

    def __init__(self, paths: list[str | Path], vocab_size: int):
        self.shards = [np.memmap(p, dtype=dtype_for(vocab_size), mode="r") for p in sorted(map(str, paths))]
        if not self.shards:
            raise ValueError("no shards given")

    def __len__(self):
        return sum(len(s) for s in self.shards)

    def sample(
        self, n: int, seq_len: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pick shards proportional to how many windows they hold, so every token is equally likely
        room = torch.tensor([max(len(s) - seq_len - 1, 0) for s in self.shards], dtype=torch.float64)
        if room.sum() == 0:
            raise ValueError(f"every shard is shorter than seq_len {seq_len}")
        which = torch.multinomial(room, n, replacement=True, generator=generator)
        xs, ys = [], []
        for k in which.tolist():
            i = int(torch.randint(int(room[k]), (1,), generator=generator))
            w = torch.from_numpy(self.shards[k][i : i + seq_len + 1].astype(np.int64))
            xs.append(w[:-1])
            ys.append(w[1:])
        return torch.stack(xs), torch.stack(ys)

    def batch(self, batch_size: int, seq_len: int, device: str, generator: torch.Generator | None = None):
        return _to_device(*self.sample(batch_size, seq_len, generator), device)


class MixtureDataset:
    """Each window comes from a source drawn by mixture weight; per-source datasets stay addressable for eval."""

    def __init__(self, parts: dict[str, ShardedDataset], weights: dict[str, float]):
        self.parts, self.names = parts, sorted(parts)
        self.probs = torch.tensor([weights[n] for n in self.names], dtype=torch.float64)

    def __len__(self):
        return sum(len(p) for p in self.parts.values())

    def batch(self, batch_size: int, seq_len: int, device: str, generator: torch.Generator | None = None):
        pick = torch.multinomial(self.probs, batch_size, replacement=True, generator=generator)
        xs, ys = [], []
        for k, name in enumerate(self.names):
            n = int((pick == k).sum())
            if n:
                x, y = self.parts[name].sample(n, seq_len, generator)
                xs.append(x)
                ys.append(y)
        return _to_device(torch.cat(xs), torch.cat(ys), device)


def _to_device(x: torch.Tensor, y: torch.Tensor, device: str):
    if device.startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def load_source_split(tokens_dir: Path, source: str, split: str, vocab_size: int) -> ShardedDataset:
    paths = sorted((tokens_dir / source).glob(f"{split}-*.bin"))
    if not paths:
        raise FileNotFoundError(f"no {split} shards for {source!r} under {tokens_dir} (run scripts/prepare_shards.py)")
    return ShardedDataset(paths, vocab_size)
