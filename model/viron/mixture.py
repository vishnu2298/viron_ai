"""Data-mixture config (YAML): which approved sources to train on, and with what sampling weight."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from .sources import SOURCES


@dataclass(frozen=True)
class Mixture:
    tokens_dir: Path
    weights: dict[str, float]  # normalized to sum to 1


def parse_mixture(raw: dict, base: Path = Path(".")) -> Mixture:
    sources = raw.get("sources") or {}
    if not isinstance(sources, dict) or not sources:
        raise ValueError("mixture needs a non-empty 'sources' mapping of source -> weight")
    unknown = sorted(set(sources) - set(SOURCES))
    if unknown:
        # The registry is the license gate: a source must be approved in docs/DATA.md before it can be mixed in
        raise ValueError(f"sources not approved in viron/sources.py: {unknown}")
    if any(not isinstance(w, int | float) or w < 0 for w in sources.values()):
        raise ValueError("weights must be non-negative numbers")
    total = float(sum(sources.values()))
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    weights = {k: v / total for k, v in sorted(sources.items()) if v > 0}
    return Mixture(tokens_dir=base / raw.get("tokens_dir", "data/tokens"), weights=weights)


def load_mixture(path: str | Path) -> Mixture:
    return parse_mixture(yaml.safe_load(Path(path).read_text()) or {})
