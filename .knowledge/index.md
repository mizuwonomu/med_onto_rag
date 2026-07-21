# index.md (current architecture — structural, durable)

## ICD-10 parser
The ICD-10 feature parses the bilingual two-column `data/raw/ICD_10.pdf` (English left, Vietnamese right, ~841 pages) into `data/processed/icd_concepts.jsonl`, one flat record per ICD code, serving as Entity-Linking candidates for the CHAN_DOAN branch. It is split into: `src/kb/icd.py` (page column detection + code-anchored parsing + EN/VI join + record building), `src/kb/icd_hierarchy.py` (static range→chapter/block table), `src/kb/icd_text.py` (head/tail name-vs-note splitter), `src/configs/config.py` (tuning constants), and `scripts/build_kb/icd/build_icd_kb.py` (entrypoint). The defining boundary: the **ICD code is the anchor**; the two language columns are parsed independently into `{code → description}` streams and joined **by code**, never by geometric row/position — so page breaks and EN/VI line drift are harmless.

- **Cross-cutting:** the flat JSONL record schema (`text`, `name_vi`, `name_en`, `note_vi`, `note_en`, `code`, `chapter`, `block`, `kb`, `role`, `vi_glyph_ok`) is meant to be shared with other KB branches (RxNorm etc.). `text` is the sole field sent to embedding + BM25. Changing this shape forces re-ingestion — warrants a pointer line at the top of `index.md`.

