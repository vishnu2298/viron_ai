"""Token shards on disk (numpy memmap) and random-window batching — scales past RAM."""

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
