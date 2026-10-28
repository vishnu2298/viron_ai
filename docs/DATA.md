# Viron training data — sources, licenses, decisions

Every dataset used to train Viron is recorded here with its source and license (CLAUDE.md, standalone rule 3).
Nothing is downloaded until the user approves a source in the **Decision** section.

Status: **awaiting decision** (M1, task 1). Researched 2026-10-28 from dataset cards, papers and announcements via search;
Hugging Face and several project sites could not be opened directly from the build environment, so figures marked
**verify** must be checked on the dataset card before download.

## What we need
- **Scale (near term):** M3 trains `tiny` on ~0.5–1B tokens; `small` needs ~2–5B. Any candidate below is large enough
  for the next several milestones, so license and quality matter more than raw size.
- **Languages:** English first, Hindi second, other Indic languages a stretch. One shared 32k BPE (M1 task 5).
- **License:** must permit training. No outputs of other AI products (that includes machine-translated text).
- **Hygiene:** exact dedup, length/repetition/language filters, PII redaction before tokenization (M1 task 3).

## Two kinds of "open"
1. **Openly licensed content.** The rights holders released the text itself under public domain / CC0 / CC-BY /
   CC-BY-SA / permissive code licenses, curated per document. The lowest legal risk.
2. **Openly licensed compilation of web crawl.** The *dataset* is released under ODC-By, CC0 or CC-BY, but the
   underlying pages remain under their authors' copyright (and Common Crawl's terms of use). Training use relies on
   text-and-data-mining exceptions / fair use, which vary by jurisdiction. Much higher quality per token, and the
   only practical source of large-scale Hindi today.

## Candidates

| Corpus | License | Size | English | Hindi / Indic | Kind | Quality notes |
|---|---|---|---|---|---|---|
| **Common Pile v0.1** (EleutherAI, 2025) | Per-document open licenses (public domain, CC0/BY/BY-SA, permissive code), checked against an Open Knowledge Foundation-style list | ~8 TB, 30 sources; a filtered mix trained the Comma 7B models on 1–2T tokens | Yes (language-ID filtered to English) | Essentially none | 1 | Books (public domain), papers (arXiv, PubMed Central), legal, patents, StackExchange, Wikimedia, code. Comma 7B ≈ Llama 1/2 7B. Skewed to technical/legal/old text; little conversational or modern web prose |
| **Common Corpus** (Pleias, 2024–25) | Uncopyrighted or openly licensed; per-document provenance | ~2.0T tokens | Yes (largest share with French) | ≥1B tokens for 30+ languages; Hindi amount **verify** (likely small) | 1 | Heavy on public-domain books, newspapers, government and scientific text; OCR noise in older sources |
| **Wikipedia dumps** (en, hi, + other Indic) | CC BY-SA 4.0 (attribution + share-alike) | en: billions of tokens; hi: order of 10⁸ tokens **verify** | Yes | Yes, small but clean; also bn, ta, te, mr, … | 1 | High quality, encyclopedic register only. Already included in Common Pile for English |
| **FineWeb-Edu** (Hugging Face, 2024) | ODC-By 1.0; also subject to Common Crawl terms | 1.3T tokens (from 15T FineWeb) | Yes | No | 2 | Web filtered by an educational-quality classifier; among the strongest small-model pretraining results per token |
| **FineWeb-2** (Hugging Face, 2025) | ODC-By 1.0; Common Crawl terms | ~3T words across 1000+ languages | Separate (FineWeb) | Yes; Hindi was one of the tuned "canary" languages; Hindi size **verify** | 2 | Same pipeline, language-specific filtering and dedup; no edu-classifier for Hindi |
| **Sangraha** (AI4Bharat, 2024) | CC BY 4.0 (dataset card) | ~251B tokens / 22 languages: verified 64B, unverified 24B, synthetic 162B | Some Indian English | Hindi ≈ 12.6B verified + 12.3B unverified + 9.6B synthetic | 2 (verified/unverified) | *Verified*: human-checked sites, OCR'd PDFs, transcripts — the best Hindi quality available. *Unverified*: filtered from existing web corpora. **Synthetic = machine-translated English Wikimedia → excluded (other-AI output, rule 3)** |
| **IndicCorp v2** (AI4Bharat, 2022) | CC0 (dataset); underlying pages copyrighted | 20.9B tokens: 14.4B across 23 Indic languages + 6.5B Indian English | Indian English | Yes; v2 Hindi count **verify** | 2 | Mostly news crawl; older; partly overlaps Sangraha's sources |

Not shortlisted: mC4 / CulturaX / MADLAD-400 (web compilations with weaker filtering, already partly inside Sangraha
*unverified*); Dolma (English web plus sources Common Pile covers more strictly); any instruction/chat dataset (that
is M4, and many contain other-AI outputs).

## Mixes for the next milestones (M3: ~1B tokens)
| | A. Strict open | B. Strict English + curated Hindi (recommended) | C. Quality first |
|---|---|---|---|
| English | Common Pile (Comma filtered mix) | Common Pile (Comma filtered mix) | FineWeb-Edu |
| Hindi | Wikipedia hi (+ Common Corpus hi) | Wikipedia hi + **Sangraha verified (hin)** | FineWeb-2 hin + Sangraha verified (hin) |
| Hindi share achievable | ~1–5% (data-limited) | 10–20% (choice) | 10–30% (choice) |
| License risk | lowest (kind 1 only) | low for English; Hindi relies on a CC-BY compilation of curated web | highest (kind 2 for everything) |
| Expected quality at 1B tokens | good English, weak Hindi | good English, usable Hindi | best English and Hindi |
| Attribution duties | CC-BY/BY-SA notices | + AI4Bharat CC-BY attribution | + ODC-By attribution |

## Recommendation
**Mix B.** English from Common Pile keeps the bulk of the model on openly licensed content. Hindi from Sangraha
*verified* is the only source with real volume **and** human checking, and it carries a clear CC BY 4.0 dataset
license. Exclude Sangraha *synthetic* (machine translation) and *unverified*. Start with ~85% English / 15% Hindi
by tokens; the M1 tokenizer report (tokens-per-word for English and Hindi) will tell us whether 32k BPE handles both.

Choose **A** if the standard is "only content whose authors licensed it". Expect weak Hindi until more openly licensed
Hindi text exists. Choose **C** if quality per token matters most and web-crawl compilations are acceptable.

## Open questions for the user
1. Which license standard: kind 1 only (A), kind 1 for English + curated kind 2 for Hindi (B), or kind 2 (C)?
2. Is CC BY-SA share-alike acceptable in training data (Wikipedia, parts of Common Pile)? It is common practice;
   whether share-alike reaches model weights is legally unsettled.
3. Hindi share for the first `tiny` run (default proposal: 15%).

## Decision
- Mix: _pending_
- Sources approved (name · version/date · license · URL · tokens used): _pending_
- Attribution notices file: _pending (needed for CC-BY / ODC-By sources)_
