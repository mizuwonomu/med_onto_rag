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

- Date: 2026-07-08 (granularity measured 2026-08-03)
- `rxnorm-etl: IN PROGRESS (SCD widening evaluated and reverted 2026-08-03) -> features/rxnorm-etl_parser/log.md`

**Term-type survey against the organizer's gold (2026-08-03, 13 RXCUIs).** Queried
`rxnconso` directly for every drug code appearing in the organizer's worked examples:

- **12 of 13 are present; only `360047` is absent from `rxnconso` entirely.** The
  release is not truncated - this is a filter problem, not a reload problem. `360047`
  was left unresolved: `rxnatomarchive` has never been loaded, and loading a table to
  chase 1 code in 13 is not worth it while 181 records score zero for another reason.
- **The gold term type is `SCD`, not `SCDC`.** Ten of the twelve resolve to exactly one
  `SCD` row (`amlodipine 10 MG Oral Tablet` = `308135`). The exception is `7597` =
  `nystatin`, a bare `IN` - so gold mixes ingredient-level and strength-level codes and
  the export must carry **both**. No `SCDC` appears anywhere in the sample.
- **Concept counts under `sab='RXNORM'`** (`COUNT(DISTINCT rxcui)` = `COUNT(*)`, one atom
  per RXCUI): `IN/N` 14,648 | `SCD/N` 17,552 | `SCD/O` 20,301 | `SCD/E` 2,174. The chosen
  filter `tty IN ('IN','SCD') AND suppress='N'` yields **32,200 concepts, 2.2x the
  current 14,648** - a re-embed cost of minutes, not the thing to worry about.
- **`suppress='E'` exists and was not previously considered.** The current `suppress='N'`
  filter already excludes it, but any future rewrite as `suppress <> 'O'` would silently
  pull in 2,174 obsolete concepts.

**Known limitations / debt left open:**

- **`suppress='O'`/`'E'` deliberately excluded from candidates, not overlooked.** Obsolete
  SCDs are mostly discontinued strength variants of drugs that are still active, so they
  add siblings to exactly the cluster the reranker already loses on while gold never
  selects them: precision falls, recall does not rise. This also preserves the asymmetry
  already decided for this layer - alias docs keep `suppress='O'` so discontinued brand
  names stay *findable*, while what they resolve *to* must be currently active. Revisit
  only if gold is shown to contain an obsolete code.
- **Widening to SCD invalidates two frozen RxNorm retrieval settings.** The `strip_dose`
  query variant removes the only field that separates `197527` (`clonazepam 0.5 MG`) from
  `197528` (`1 MG`), and `margin = 0` rested on the structural claim that one alias
  expands to the full RXCUI set - which no longer holds once thousands of strength-level
  siblings exist. Both must be re-swept against `candidates_score` before the next
  submission, along with `TOP_K_DENSE`/`TOP_K_BM25 = 30` (never swept, and the gold code
  must reach the reranker before ranking matters).
- **Gold strength does not match KB strength verbatim.** The organizer maps
  `clonazepam 1.5 mg po qhs` to `197528`, whose string is `clonazepam 1 MG Oral Tablet`.
  Exact-dose matching will miss. `metoprolol succinate xl 50 mg` -> `866436`
  (`24 HR ... Extended Release Oral Tablet`) needs `xl` -> `Extended Release`, a vocabulary
  gap of the same family as the colloquial-ICD one already recorded under NER.
- **`synonyms` is exported, stored, and never searched.** `load_rxnorm_documents` puts it
  in metadata as a JSON string while `page_content` receives `text` alone, and both Chroma
  and BM25 index `page_content` - so `ASA` (aspirin), `APAP` (acetaminophen) and
  `DOSS Sodium` are unreachable today. Widening the synonym term types changes nothing
  until a consumer exists. The clean fix is to let BM25 build its own index string from
  `text + synonyms` while `page_content` stays intact for dense, keeping `doc_id` as the
  union key; putting synonyms into `page_content` would blur the embedding.
- **`PSN` was considered as a synonym source and rejected.** It duplicates the SCD string
  with tall-man letters (`amLODIPine besylate 10 MG Oral Tablet`) rather than adding
  vocabulary, and BM25 is case-insensitive already. Case folding belongs in the tokenizer.
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

## NER extraction layer

- **SCOREBOARD LOG (2026-08-03).** Five submissions, NER layer identical to four decimal
  places throughout (`WER 77.8966` / `J_assertion 26.7302` in every run), so each delta is
  attributable to retrieval alone. Best configuration is the original one:

  | KB | margin | variant | J_candidates | total |
  |---|---|---|---|---|
  | IN only | 0.0 | strip_dose | **9.1937** | **18.3275** |
  | IN only | 0.0 | raw | **9.1937** | **18.3275** |
  | IN only | 0.2 | raw | 9.1292 | 18.3017 |
  | IN + SCD | 0.2 | raw | 8.3495 | 17.9899 |
  | IN + SCD | 0.0 | raw | 8.1404 | 17.9063 |

  Reading: KB granularity moved the score `-1.05`; `margin` moved it `±0.2` inconsistently;
  query variant moved it `0.0000` (byte-identical). Threshold work on the RxNorm branch has
  a measured ceiling of about `±0.08` on the final score.
- **BEST TO DATE: `18.4827`** (2026-08-04, `c8d3e7f`) - `align.postprocess` on the ingredient KB.
  `WER 77.6252` | `J_assertion 26.9255` | `J_candidates 9.2314`. The first submission where all
  three metrics moved together, and the first time `WER` moved at all. `J_candidates` rose with
  **no change to retrieval**, so the gain is denominator-side: ~55 junk spans of 1946 removed.
- **FIRST REAL SCORE (organizer's scoreboard, 2026-08-02): `18.3275`** on 100/100 documents.
  `WER 77.8966` -> text 22.10 | `J_assertion 26.7302` | `J_candidates 9.1937`.
  Verified: `0.3*22.1034 + 0.3*26.7302 + 0.4*9.1937 = 18.327`, so the formula is read correctly
  and `num_scored = 100` confirms no submission file was missing or malformed.
- Date: 2026-08-02
- `ner_extraction: IN PROGRESS -> features/ner_extraction/log.md`
- **Benchmark stage only.** Every number so far comes from a 10-document hand-authored synthetic set. The organizer's `data/input/` (100 documents) has not been run even once, and no submission output has been produced.
- Committed so far: `src/extract/` (schema, prompt, llm, align, linker), `src/pipeline.py`, `scripts/ner/serve_llm.sh`, config + pyproject. `scripts/ner/predict.py` and `run_ner_synthetic.py` are written and working but **not yet committed**; `build_synthetic.py`, `validate_ner_synthetic.py`, `scripts/README.md` and the synthetic data are intentionally kept out of version control.

**Known limitations / debt left open:**

- **Ten documents is not a sample, it is an anecdote.** 80 gold concepts, and whole categories appear once or twice — one `CHẨN_ĐOÁN`+`isHistorical`, one `THUỐC`+`isFamily`. No claim about the layer's accuracy survives contact with a larger set, and the current scores should be read as "the pipeline runs and is not obviously broken", nothing more.
- **The prompt was tuned on the same ten documents it is scored against.** Three revisions each used the failing cases as new examples. Rounds one and two targeted errors that recurred across documents (framing, modifiers) and are defensible; anything further would be memorization. There is no held-out split, so the reported gain from 0.6917 to 0.8747 is a training-set number.
- **Gold quality is unresolved and was found lacking in both directions.** Five labels were corrected during this work; separately, at least five genuine symptoms present in the text (`mệt mỏi`, `đau bụng`, `nôn ói`, `nóng rát sau xương ức`) are missing from gold and are currently scored as model hallucinations. The counts for "spurious" are therefore inflated by an unknown amount.
- **The evaluation harness pairs greedily and can mis-assign.** The overlap pass walks predictions in order and takes the longest raw common substring, so an early short prediction can claim a gold that a later, better prediction needed — observed in `syn_008`, where a correct span carrying the right ICD code was pushed onto the wrong gold and scored zero. A Dice-normalized score with global (Hungarian) assignment was designed but not implemented.
- **`wrong_type_count` under-reports.** It only counts a type error when the same text survives in both the missed and spurious lists, so any type error whose prediction gets absorbed by the overlap pass is silently invisible. It reported zero on a run that demonstrably contained one.
- **ICD linking is weak and it is not this layer's fault.** `candidates_score` splits 0.719 RxNorm / 0.500 ICD, consistent with the reranker limitation already recorded under the retrieval feature. A distinct sub-case appears here: parent-vs-child confusion (`A09` returned where gold is `A09.0`) scores zero under Jaccard despite being one level away — arguably the most painful failure mode, and unaddressed.
- **`--skip-link` output is not submittable** and nothing enforces that. It writes records without `candidates`; a run made for inspection could be submitted by accident. `predict.py` warns in the log and no further.
- **No verification that all 100 outputs exist.** Documents that error are deliberately not written (an empty list would disguise an infrastructure failure as a zero score), but nothing counts the results afterwards, so a partial run can be mistaken for a complete one. Re-running fills gaps via skip-existing; remembering to re-run is manual.
- **Context budget has no headroom check.** Prompt is ~4.2k tokens against `-c` 10132; the longest organizer document adds ~1.5k and `max_tokens` another 2048. Any prompt growth silently eats this margin, and the failure mode is a length error mid-run rather than a warning at startup.
- **Token estimates are approximate.** All context arithmetic uses a measured ~3.2 characters-per-token ratio for Vietnamese rather than actual tokenization, so the margins above are rough.
- **Retrieval is invoked per mention with no caching.** The same drug repeated across documents re-runs dense + BM25 + reranking every time. Irrelevant at this scale, a real cost at 100 documents.
- **RxNorm gold codes are not in the KB - drugs cannot score at all.** The organizer's example gives `Chlorpheniramine 0.4 MG/ML = 360047` and `Capsaicin 0.38 MG/ML = 1660761`; neither RXCUI exists in `data/processed/rxnorm_concepts.jsonl`. The export is ingredient-level, so the pipeline returns `2400` / `1992`. All 181 drug records in the submission are structurally incapable of matching. Fix is upstream in the SQL export, not in retrieval. **Measured 2026-08-03, the term type is `SCD`, not the `SCDC` originally hypothesized** - see the RxNORM-etl section for the query results and the chosen filter.
- **ICD `margin = 0.399` is too tight for a set-valued metric.** Gold for the same example is `K21.0` + `K21.9`; both are in the KB but only `K21.0` survives thresholding, capping that mention at J = 0.5. 226 of 471 diagnoses in the submission returned exactly one code. Jaccard punishes under-returning harder than over-returning, so the margin should be reswept against `candidates_score` rather than recall.
- **BM25 recall collapses on colloquial vocabulary.** `viêm dạ dày` puts K29* at ranks 1-6; `viêm bao tử` (the everyday word for the same organ) returns cervicitis and bursitis and no digestive code in the top 30. The 399-row synthetic set never exposed this because its mentions were written from KB vocabulary. A colloquial-to-formal query map is the cheap fix; it does not exist yet.
- **84 ICD records carry English-only text**, including `E11` (type 2 diabetes, `vi_glyph_ok=False`), `F43.1`, `F05.9`, `H54`, `E07`. A Vietnamese query cannot reach them at all. The Vietnamese is present in `ICD_10.pdf`; the code-anchor regex missed those rows. This is the ICD-parser debt already listed above, now measured and shown to hit high-frequency codes.
- **KB text carries undetected typos**: `K25` reads `Loét dạy dày` (should be `dạ dày`) yet passes `vi_glyph_ok`.
- **8.3% of drug records are junk spans.** The organizer redacts drug names with asterisks; the model labels the asterisks as a drug. 8 records are pure `*`, 4 more mix asterisks into a span (2 of those still received an RXCUI), 3 carry framing verbs (`Tăng liều bactrim`, `Uống thuốc`). All are removable deterministically in `align.py`; none needs the model.
- **15 pairs of overlapping spans** in the submission - the same concept emitted twice at different boundaries (`Cơn rối loạn ý thức thoáng qua` [132,162] and `rối loạn ý thức thoáng qua` [136,162]). Also removable deterministically: drop a span fully contained in another of the same type.
- **The compliance flags explain only ~10% of records (195/1966) and cannot account for `WER 77.9`.** Something more systematic is wrong - most likely a concept-granularity mismatch with gold (the submission averages 19.7 concepts per document, one per ~103 characters, while the organizer's own worked example runs about four times denser). This is unmeasurable from our side; the only instrument is submitting a deliberate variant and comparing scores.
- **`tests/extract/` is gitignored** (same repo-wide `tests/` rule as earlier features), so the two smoke tests — one of which caught a silent Unicode offset bug — will not be committed.

**Measured 2026-08-03, after the SCD experiment:**

- **The `24/157` dose split is measured on our own predictions, not on gold.** A drug mention the NER layer missed entirely is absent from that count, so the ratio is biased by whatever the extractor systematically drops.
- **No gold exists for any bare drug name.** That bare names map to ingredient codes is inferred from score movement across five submissions, never observed. If it is wrong, the SCD revert is wrong with it.
- **`floor = 2.157` has never been re-measured.** Every sweep this session held it fixed and varied `margin` only. It was calibrated on synthetic data at ingredient granularity and has no independent confirmation.
- **`export_candidate_jsonl.sql` and the built index now disagree.** The SQL carries `tty IN ('IN','SCD')` while the KB in use is ingredient-only. Either revert the SQL or record why it differs; a future rebuild from that file silently reintroduces the regression.
- **The dose-routing idea is untested.** Allowing SCD only for dose-bearing mentions would cap the loss at zero while keeping the 24-mention upside. Designed, never implemented, because the ceiling on the whole branch is about `±0.08` on the final score.
- **Five scoreboard submissions were spent on a branch with a `±0.08` ceiling**, while `WER 77.8966` - carrying weight `0.3` against candidates' `0.4` - never moved and was never probed. Ten points of `text_score` are worth `+3` on the final score, roughly forty times the entire remaining headroom in candidates.
- **The fixture has no ICD counterpart.** The same circularity applies to ICD thresholds, which were also calibrated on synthetic gold drawn from the KB under test.

**Measured 2026-08-04, after `align.postprocess`:**

- **`postprocess` touches ~55 spans of 1946 (2.8%) and bought `+0.155`.** Extrapolating, perfect span hygiene is worth a few points at most. `WER 77.63` needs to fall by roughly twenty for the layer to change character, and nothing tried so far has touched its main cause.
- **Three flag groups remain and none is reachable by string code.** `KẾT_QUẢ_XÉT_NGHIỆM` with no digits (48) is a type error - `chụp ct sọ`, `chọc dò dịch não tủy` are test names, and a type error costs double under the organizer's rule. Test phrases opening with a verb (43) need a human decision on whether `chụp ct sọ não` keeps its verb. `CHẨN_ĐOÁN`/`THUỐC` with no code (71) is mostly correct to leave empty (`kháng sinh`, `intravenous fluids`, redacted names), so forcing a code would trade zero for zero.
- **`TRIỆU_CHỨNG` sits at 47.8% of records (935) and has not moved through any experiment.** No criterion exists to separate a real symptom from an invented one without gold, so the largest single category is also the one with no available instrument.
- **The two test types are emitted in near-exact 1:1 pairs** (187/187, and 254/252 after a counter-example was added). Real notes contain ordered-but-unreported tests and free-standing values, so the pairing is few-shot imitation. It survived a direct attempt to break it.
- **Span-length assumptions are unverified.** Every trim rule rests on gold being shorter than what the model emits, inferred from the organizer's examples, never confirmed.
- **The trim rule deliberately under-cuts.** A correct trim leaving one word (`sốt cao` -> `sốt`) is now refused, because a wrong cut fabricates a concept while a missed cut costs only part of one WER. The size of what this gives up is unmeasured.
- **`scripts/ner/check_submission.py` is untracked**, like the rest of `scripts/ner/` tooling. It is the only instrument that measures the 100 unlabelled documents, and it is the one most likely to be wanted again.
