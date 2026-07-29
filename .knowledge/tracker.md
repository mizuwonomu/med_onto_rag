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
## Retrieval layer

- Date: 2026-07-29
- `retrieval: DONE (2026-07-29) -> features/retrieval/log.md`
- RxNorm thresholds frozen; ICD thresholds tentative. Scope: retrieval + threshold calibration only. Not yet wired to NER.

**Known limitations / debt left open:**

- **Every number here is measured on synthetic data only.** All thresholds, recall figures and score distributions come from 399 LLM-generated rows; nothing has been validated against a real clinical note or against real NER output. Real mentions bring span-boundary errors, multi-concept spans and a different distribution — none of which the synthetic set simulates. The first genuine validation is the organizer's test set, and a drop there should be read as scope, not regression. RxNorm is the partial exception: its `margin = 0` follows from the KB structure (one alias expands to the full RXCUI set), which synthetic data confirmed rather than established.
- **ICD thresholds are not trustworthy for production.** sweep-dev = +0.104 (sweep 0.636 vs dev 0.532) — the ICD pair fits the tuning split better than unseen data. Bootstrap is stable, so this is not sampling noise in the threshold itself.
- **The ICD reranker cannot separate siblings.** distractor p75 (3.477) exceeds gold p25 (3.156): the distributions overlap mid-scale, where no floor can cut. Worst clusters by name: `ICD_Malnutrition_Severe` (median gap 0.203), `ICD_Paralysis_Tetraplegia` (0.387, 3 of 6 rows rank a distractor above gold), `Rx_Ear_Drops_Cortisporin` (0.023). **Adding synthetic data will not fix this** — untried directions are a larger reranker (Qwen3-Reranker-4B) or feeding ICD block information into the doc text given to the reranker.
- **`multi_sibling` recall is 0.860 and did not move** between the dirty and clean datasets — the weakest group, and it is a retrieval-stage loss, not a threshold one.
- **The hybrid tokenizer variant was never measured.** Indexing both whitespace tokens and segmented compounds (`[viêm, cầu, thận, viêm_cầu_thận]`) is additive and cheap, but was deprioritized behind the reranker bottleneck. The original docstring claim that Vietnamese segmenters are "brittle on medical terminology" was an **untested assumption** and has been softened.
- **Near-miss validation is ontology-only.** The gate accepts a distractor if it shares an ICD block or a `resolve_in` group. Purely lexical near-misses (`dexamethasone` vs `dexmedetomidine`) are unchecked, because a deterministic test would require the gate to load the BM25 index and stop being a pure-data script. Consequently the distractor set may under-represent the hardest real-world case.
- **The distractor label in the dump is not typed.** Ontology-neighbour and lexical-neighbour distractors are pooled, so "which kind of confusion does the reranker actually lose to" is currently unanswerable.
- **`scripts/retrieval/common.py` `dedupe_records` is a workaround for a data defect**, kept as a default-on safety net. It hides duplicates rather than preventing them; the generator is the correct place to enforce uniqueness.
- **Plot histograms use `density=True`**, which normalizes each label group independently and makes gold/distractor bar heights visually incomparable. This already caused one misreading of the score distribution. Switching to counts would be clearer.
- **`TOP_K_DENSE` / `TOP_K_BM25` = 30 were never swept.** The values are inherited from the plan; recall@union may be cheaply improvable by raising them, at the cost of reranker time.
- **The reranker instruction was frozen without ablation.** It is a single hand-written sentence; no alternative was measured, so its contribution to the score distribution is unknown.
- **`dense_search` is not a `Runnable`**, so the dense branch cannot be dropped into an LCEL chain as-is. Deliberate (see `decisions.md`), but it is a real constraint the NER integration will meet.
- **`tests/retrieval/` is gitignored** (repo-wide `tests/` ignore, same as the ICD and index suites) — the 69 new tests will not be committed as configured.
- **No provenance recorded** for the reranker model revision or the BM25 index build, mirroring the same gap already open for RxNorm and the vector index.
