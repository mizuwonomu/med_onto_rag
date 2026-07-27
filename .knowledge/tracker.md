# tracker.md (living status — flags + pointers only)

## ICD Parser
- Date: 2026-07-21
- `icd_parser: DONE -> features/icd_parser/log.md`

**Known limitations / debt left open:**

- **Glyph detector misses consonant-only drops** ("th c qu n" = thực quản): only nguyen-am-moc drops are caught; consonant-only losses go undetected (deliberate — broadening caused ~1170 false positives). Real corruption is under-counted.
- **17 codes have no Vietnamese at all** (leading letter dropped by font → code-anchor regex misses the row in the VI stream), so they embed English text; not recovered (crop cannot; word-level could).
- **5 codes are heavily corrupted** (≥4 glyph errors, e.g. Y98, T98.3, E15) yet still embed their damaged Vietnamese — no per-code severity gate.
- **1 phantom code `B12`** from a cross-reference fragment remains in output.
- **`find_column_split` fallback (`page_width/2`) is silent** — no log/counter when it fires; a future layout regression would fail quietly (0/841 hit it now, but it is unobserved).
- **Unused deps `pymupdf`/`fitz`** left in `pyproject.toml` from the dropped word-level spike.
- **`tests/` is gitignored** — the 51 tests will not be committed as currently configured.
- **Detector/head-tail markers are Vietnamese-corpus-specific regexes** — brittle if the PDF edition or wording changes.

## RxNORM-etl Parser

- Date: 2026-07-08
- `rxnorm-etl: IN PROGRESS -> features/rxnorm-etl/log.md`
- **Known limitations / debt left open:**
    - The brand→generic join in `export_brand_jsonl.sql` has **no `rela` filter**. It matches on `rxcui1` alone and therefore pulls in every relation type, including `has_dose_form`, `consists_of`, and others that are not ingredient relations. Recall is inflated and precision is unquantified. The user planned to run a survey query over actual `rela` values before deciding the filter; that survey has not been run, so the export currently in hand should be treated as provisional.
    - `resolve_min` is untested end-to-end. No multi-ingredient brand has been verified as emitting a populated `resolve_min` alongside its `resolve_in`.
    - Single-hop traversal only. Brands reachable from their ingredients only through an intermediate concept are dropped silently, with no logging or count of exclusions.
    - Alias docs may duplicate per brand concept (multiple `b_<rxaui>` rows for one brand). The duplication rate has not been measured, and no decision has been made on whether it degrades retrieval.
    - No provenance is recorded for the RxNorm release version used, despite `rela` vocabulary and `suppress` flags being version-dependent.

## Vector index

- Date: 2026-07-27
- `vector_index: DONE -> features/vector_index/log.md`
- Scope this session: load + ingest only. Retrieval / BM25 / hybrid / reranker are NOT built yet.

**Known limitations / debt left open:**

- **One-shot full rebuild only.** Every `build_vector` run `rmtree`s the persist dir and re-encodes the entire KB. No incremental update; when the KB grows large, re-encoding cost is the trigger to add one.
- **`SharedSystemClient.clear_system_cache()` is process-global, not path-scoped.** It is only safe because `ingest_collection` is called sequentially (icd then rxnorm). Parallel/concurrent ingestion in one process would clear each other's clients and break.
- **`sanity_check_asymmetry` is a shallow check.** It uses a single hardcoded Vietnamese sample and only proves "query != document + norm ~ 1.0". It catches a missing prompt, not deeper encoding regressions.
- **`ingest_collection` reads `store._collection.count()`** — private langchain-chroma internals. Fragile if the wrapper's attribute layout changes.
- **CPU fallback is silent.** On a CUDA-less machine `build_embeddings` quietly builds a CPU index (correct but very slow) with no warning — fine for tests, a footgun for a real run on the wrong host.
- **No provenance recorded** for the embedding model version or index build (mirrors the RxNorm gap).
- **`tests/index/` is gitignored** (same repo-wide `tests/` ignore as the ICD suite) — the 5 new smoke tests will not be committed as currently configured.
- **ICD debt flags flow into metadata unacted-on:** `vi_glyph_ok=False` records and English-fallback rows are embedded as-is; no per-record quality gate at ingest time.