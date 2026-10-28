"""Viron: a decoder-only transformer language model, trained from scratch."""

from .config import PRESETS, ModelConfig
from .model import Viron

__all__ = ["ModelConfig", "PRESETS", "Viron"]
