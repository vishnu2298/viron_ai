"""Approved pretraining sources (docs/DATA.md, Mix B). Adding a source here requires a recorded user decision."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    repo: str  # Hugging Face dataset repo
    prefix: str  # path inside the repo; only files under it are read
    lang: str  # expected language, used by the cleaning filters
    license: str
    text_field: str = "text"


SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        # Repo ids and paths are marked "verify" in docs/DATA.md: confirm on the dataset card before a full download
        Source("commonpile_comma", "common-pile/comma_v0.1_training_dataset", "", "en", "per-document open licenses"),
        Source("wikipedia_hi", "wikimedia/wikipedia", "20231101.hi/", "hi", "CC BY-SA 4.0"),
        Source("sangraha_verified_hin", "ai4bharat/sangraha", "verified/hin/", "hi", "CC BY 4.0"),
    ]
}
