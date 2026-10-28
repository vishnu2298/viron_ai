"""
Tokenizers trained on YOUR data — no pretrained vocabularies.
- BPETokenizer: byte-level BPE trained from scratch with the `tokenizers` library (an algorithm library, not a model).
- ByteTokenizer: zero-dependency 256+specials fallback for tests and tiny experiments.
Chat special tokens are reserved from day one so SFT later needs no vocab change.
"""

from collections.abc import Iterable
from pathlib import Path

SPECIALS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]


class ByteTokenizer:
    def __init__(self):
        self.offset = len(SPECIALS)
        self.vocab_size = 256 + self.offset
        self.special = {t: i for i, t in enumerate(SPECIALS)}

    def token_id(self, t: str) -> int:
        return self.special[t]

    @property
    def eos_id(self) -> int:
        return self.special["<|eos|>"]

    def encode(self, text: str) -> list[int]:
        return [b + self.offset for b in text.encode("utf-8")]

    def decode(self, ids: Iterable[int]) -> str:
        return bytes(i - self.offset for i in ids if i >= self.offset).decode("utf-8", errors="replace")


class BPETokenizer:
    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer

        self.tk = Tokenizer.from_file(str(path))
        self.vocab_size = self.tk.get_vocab_size()

    @staticmethod
    def _new(vocab_size: int, min_frequency: int):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tk = Tokenizer(models.BPE(byte_fallback=False))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIALS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        return tk, trainer

    @staticmethod
    def _save(tk, out_path: str | Path) -> "BPETokenizer":
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        tk.save(str(out_path))
        return BPETokenizer(out_path)

    @staticmethod
    def train(files: list[str], out_path: str | Path, vocab_size: int = 32000, min_frequency: int = 2):
        tk, trainer = BPETokenizer._new(vocab_size, min_frequency)
        tk.train(files, trainer)
        return BPETokenizer._save(tk, out_path)

    @staticmethod
    def train_from_iterator(
        texts: Iterable[str], out_path: str | Path, vocab_size: int = 32000, min_frequency: int = 2
    ):
        """Train on an in-memory stream (e.g. a mixture-weighted sample of data/clean) without temp files."""
        tk, trainer = BPETokenizer._new(vocab_size, min_frequency)
        tk.train_from_iterator(texts, trainer)
        return BPETokenizer._save(tk, out_path)

    def token_id(self, t: str) -> int:
        return self.tk.token_to_id(t)

    @property
    def eos_id(self) -> int:
        return self.token_id("<|eos|>")

    def encode(self, text: str) -> list[int]:
        return self.tk.encode(text).ids

    def decode(self, ids: Iterable[int]) -> str:
        return self.tk.decode(list(ids), skip_special_tokens=True)


def load_tokenizer(path: str | None):
    return BPETokenizer(path) if path else ByteTokenizer()
