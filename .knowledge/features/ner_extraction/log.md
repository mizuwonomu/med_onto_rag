# log.md — NER extraction layer

## How it was actually built

The layer was assembled bottom-up so that each piece could be accepted before the next one depended on it: schema, then prompt, then the client, then alignment, then linking, then the orchestrator, then the runnable scripts.

`schema.py` came first because it is not merely a validation model — `with_structured_output(..., method="json_schema")` converts it to JSON Schema, llama-server converts that to GBNF, and the grammar constrains decoding. Every field is therefore a decoding constraint: `Literal` enums make an unknown entity type or assertion *unreachable* rather than merely invalid. The wrapper key `entities` exists only because the OpenAI protocol requires a JSON-Schema root to be an object; the submission format is a bare list, and the key disappears before anything is written to disk.

`align.py` was written as pure string code with no model import, which is what makes it the only component in the layer that can be verified deterministically. It carries the two rules that shape everything above it: a forward cursor so repeated mentions receive distinct offsets, and unconditional overwriting of `text` with the raw slice. A smoke test was built from shortened excerpts of real `data/input/*.txt` files rather than invented strings — the real corpus contains phenomena (a diagnosis phrase repeated verbatim across two sections, glued words like `nhịp xoangnhịp xoang`) that would not have occurred to anyone composing test data by hand. That test immediately earned its place by catching the per-character NFC bug, which produced no exception and would have surfaced only as an unexplained score drop.

`linker.py` deliberately reuses the retrieval pipeline as-is: `retrieve_candidates → rerank → apply_thresholds → expand_codes`, with `build_backends` loading embeddings, both Chroma stores, both BM25 indexes and the reranker once per run. It reads `RETRIEVAL_THRESHOLDS` per KB together with the query variant that pair was calibrated under, since floor and variant are a single measurement and separating them compares scores across incompatible scales.

The prompt was authored in English for the rules and Vietnamese for the labels and examples, on the reasoning that instruction-following is stronger in English while `CHẨN_ĐOÁN` is a generation target rather than a concept to translate. It went through three measured revisions. The first draft covered types, assertions and the verbatim rule. Measurement showed the dominant failure was over-long spans — the model swallowing narrative framing, severity modifiers and reporting verbs — so a MINIMAL SPAN section plus a fourth worked example were added, built from the actual failing cases; `text_score` moved from 0.6917 to 0.8747 on the same three documents. The third revision addressed type confusion between diagnosis and symptom, written as an ordered five-step test rather than a keyword list.

Bringing the layer up against a live server surfaced four failures in sequence, each masking the next: the chat template rejecting a second system message (HTTP 400), thinking mode consuming the entire token budget, prompt plus completion exceeding the context window, and finally a degenerate decoding loop under greedy sampling. The last one produced correct output for three concepts and then cycled indefinitely; it was fixed at three levels — `maxItems` in the grammar so the array closes, a repeat penalty to break the tie between `]` and `,`, and a reduced `max_tokens` to bound the damage. `preflight()` was extracted into `extract/llm.py` afterwards so that a single short probe call runs *before* the multi-minute GPU load, turning each of these into an immediate error with a targeted hint rather than a failure discovered after a long wait.

Evaluation is stage-separated by design, because the organizer's single blended score cannot distinguish a missed mention from a mislabelled type from a retrieval miss, and those three demand different fixes. Pairing runs in two passes — exact `(text, type)` first, then string-overlap within the same type — after the exact-only version was found to inflate `text_score` to a meaningless 1.0.

## Result

The pipeline runs end to end on the hand-authored synthetic set: LLM extraction → position alignment → verbatim repair → type routing → retrieval → assembled records in the organizer's format. `predict.py` writes one `.json` per input document, skipping documents that already have output so an interrupted run resumes.

Latest full measurement, 10 documents / 80 gold concepts: `text_score` 0.8634, `assertions_score` 0.9190, `candidates_score` 0.6670 (RxNorm 0.719 across 30 mentions, ICD 0.500 across 10), 74 of 80 concepts paired, zero type errors on that run. These figures come from an internal harness that is more lenient than the organizer's, and every number to date is from the synthetic benchmark — the layer has not yet been run against the organizer's `data/input/`.

## Measurement session, 2026-08-03 (branch `evals`, `8bba52d`)

This session was measurement, not construction. It began from a defect recorded in the previous handoff: the organizer's gold drug RXCUIs did not exist in the exported KB, so all 181 drug records were believed structurally incapable of scoring.

The first move was to confirm the term type. The standing hypothesis was `SCDC`; querying `rxnconso` for the published gold refuted it - `1660761` is `SCD`, and a survey of all 13 gold RXCUIs found 12 present, 10 of them resolving to exactly one `SCD` row, with `nystatin` = `7597` a bare `IN`. Gold therefore mixes two granularities, which set the export filter to `tty IN ('IN','SCD')` rather than a replacement.

Before rebuilding, a fixture was written against those 11 gold mentions, because the 399-row synthetic set could not answer the question being asked: its gold was drawn from the KB under test, so it can only confirm that the index contains what the index contains. The fixture measured `raw` at 3/11 and `strip_dose` at 0/11, and its `--show-rank` output showed gold sitting at rank 2 in six cases - a `margin` problem, not a retrieval one. A Jaccard sweep over `margin` peaked at `0.2`.

The scoreboard disagreed with all of it. Five submissions, with an NER layer identical to four decimal places throughout, produced: ingredient KB + `margin 0` = 9.1937 (both `strip_dose` and `raw`, byte-identical); ingredient KB + `margin 0.2` = 9.1292; SCD KB + `margin 0.2` = 8.3495; SCD KB + `margin 0` = 8.1404. The KB change dominated every threshold effect by an order of magnitude, and it was negative.

The explanation came from one count that could have been run at the very start: of 181 drug mentions in the existing submission, 24 carry a dose and 157 do not. The organizer's published examples are prescription lines with full dosing, so both the SCD hypothesis and the `margin` optimum were extrapolated from an unrepresentative sample. Adding 17,552 strength-level concepts could help at most 24 mentions while displacing the correct ingredient for up to 157.

Outcome: the configuration is back where it started, and that is the result. What the session produced is exclusion - the RxNorm branch is not where `J_candidates` is lost, and threshold tuning on it has a measured ceiling of roughly ±0.2 on `J_candidates`, i.e. ±0.08 on the final score. The reusable artifact is `scripts/retrieval/fixture_rxnorm_gold.py`, which measures against gold our KB did not generate.

## Span cleanup session, 2026-08-04 (branch `evals`, `c8d3e7f`)

The RxNorm session had closed with a measured ceiling of about ±0.08 on the final score for anything in the retrieval branch, and with `WER 77.8966` untouched across five submissions while carrying weight 0.3. This session moved to the extraction layer.

The instrument came first. `check_compliance` already scored output against the layer's own rules rather than against gold, but it only ran inside the synthetic harness; `scripts/ner/check_submission.py` was written as a thin adapter so the same checks run over the 100 unlabelled organizer documents. It reports the type distribution, the rule flags, and span length per type. Nothing in it is a verdict - each flag marks a place a human should look.

The first run gave the shape of the problem: 1966 records, 19.7 per document, `TRIỆU_CHỨNG` at 47.8%, and `TÊN_XÉT_NGHIỆM` exactly equal to `KẾT_QUẢ_XÉT_NGHIỆM` at 187 each. That equality is not something independent categories produce; it is the model reproducing the 1:1 pairing shown in every few-shot example. Span lengths were long against the organizer's own examples: `CHẨN_ĐOÁN` averaged 4.82 words where published gold runs two or three.

Two rounds followed, and the contrast between them is the finding.

The first round put the rules in the prompt: a sharper severity-trimming rule with the failing phrases as examples, a redefinition of `KẾT_QUẢ_XÉT_NGHIỆM` as a measured value, and a fifth few-shot showing tests without results. It made things worse. Severity violations went `81 -> 80`. The `KẾT_QUẢ` flag **doubled** to 91, because admitting `âm tính` as a valid result taught the model that results need no digits without teaching it where the span ends. Total records rose to 2172 while the goal was to remove surplus. The prompt was reverted whole.

The second round put the same intent in `align.py` as `postprocess()`: drop asterisk-only spans, drop a span nested inside another of the same type, trim a trailing severity adverb. A first version over-trimmed - comparing the two output directories record by record showed `tiểu ít -> tiểu`, `sốt nhẹ -> sốt`, `uống nhiều -> uống`, each destroying a named symptom. The fix was to stop enumerating exceptions and instead require two words to survive the cut, a condition that holds for every observed correct trim and fails for every observed wrong one.

Outcome: the first submission in which all three metrics moved together - `WER 77.8966 -> 77.6252`, `J_assertion 26.7302 -> 26.9255`, `J_candidates 9.1937 -> 9.2314`, total `18.3275 -> 18.4827`. `J_candidates` rose with no change to retrieval at all, which is only possible if the denominator shrank: deleting a junk span is a gain in all three metrics, the mirror of the dropped-mention rule. It is also the first evidence on the density question - the submission carries surplus concepts rather than missing them. The effect is small because `postprocess` touches about 55 spans of 1946.
