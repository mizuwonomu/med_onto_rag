# decisions.md (why — the expensive part)

**Chosen approach + why:**

- Split the ingestion layer into `index/ingestion.py` (a generic, embedding-agnostic `ingest_collection()`) and `index/build_vector.py` (the Qwen-specific entrypoint) (commit 0d7a58b, branch build_index, 2026-07-27). The original plan put everything in one `ingestion.py` with `main()`. It was split on the user's request because `ingest_collection` writes to Chroma given *any* `Embeddings` object and knows nothing about Qwen — so the BM25/hybrid session and any future embedding swap can reuse it untouched. The model-specific wiring (device selection, fp16, asymmetric prompt, sanity check, orchestration) all lives in `build_vector.py`. The boundary is "how to write vectors" vs "which vectors, from which model."

- Embeddings are **dependency-injected** into `ingest_collection(docs, embeddings, ...)` rather than constructed inside it (commit 0d7a58b, branch build_index, 2026-07-27) so tests inject `DeterministicFakeEmbedding(size=32)` and never download the 0.6B model or need a GPU. This is what makes the whole ingestion path testable on a CPU-only, `data/`-less checkout.

- **One-shot rebuild semantics**: `ingest_collection` wipes `persist_dir` (`rmtree`) and rebuilds from scratch every run (commit 0d7a58b, branch build_index, 2026-07-27). Chosen over incremental upsert because the KB is a static snapshot and a full rebuild is trivially correct and idempotent — no stale-vector reconciliation, no partial-state bugs. The cost (re-encoding the whole KB each run) is accepted at current KB size.

- **Fail-early asymmetry check**: `sanity_check_asymmetry` runs before any ingestion and raises if `embed_query(x) == embed_documents([x])[0]`, or if a vector norm is not ~1.0 (commit 0d7a58b, branch build_index, 2026-07-27). The asymmetric setup (query gets `prompt_name="query"`, document side plain) is silent when misconfigured — the vectors just come out wrong and retrieval quietly degrades. Checking it up front converts a silent quality regression into a hard startup failure.

- **`device` falls back cuda -> cpu** in `build_embeddings` (commit 0d7a58b, branch build_index, 2026-07-27) so the entrypoint imports and the sanity check runs on machines without CUDA. fp16 is only requested on `cuda`.

**Assumptions it rests on:**

- `data/processed/{icd_concepts,rxnorm_concepts,rxnorm_aliases}.jsonl` already exist before `build_vector` runs. This layer only reads; the `scripts/build_kb/*` entrypoints must have run first.
- The KB is small enough that a full re-encode + rebuild on every run is acceptable. When the KB (especially full RxNorm) grows large enough that re-encoding hurts, incremental update becomes the signal to revisit.
- `doc_id` is unique within a KB. On collision the loader keeps the first occurrence and logs the count; if a later duplicate carried the "correct" record, it is silently dropped.
- The embedding model, cosine space, normalization, and asymmetric prompt form a **frozen contract**. Changing any of them invalidates every stored vector and forces a full re-ingestion (this is the CLAUDE.md guardrail).
- The real run has a CUDA GPU (3060 12GB) where `Qwen3-Embedding-0.6B` in fp16 fits. On a CUDA-less machine the fallback silently builds a CPU index — correct but very slow, with no warning.

**Failed approaches:**

- Tried: rerunning `ingest_collection` against the same `persist_dir` within one Python process (as the idempotency test does) → Failed because: chromadb caches its `PersistentClient` per `persist_directory` at the process level (`SharedSystemClient`); after `rmtree` deletes the sqlite file, a second `Chroma(persist_directory=...)` on the same path hands back the cached client still holding the now-orphaned sqlite handle, and the write fails with `attempt to write a readonly database` → Avoid when: any code deletes and recreates a Chroma persist dir inside a single process (rebuild loops, tests). Fix: call `SharedSystemClient.clear_system_cache()` right after the `rmtree`. Note this is **cross-cutting** — it governs any future in-process rebuild.

**Nuances agreed with the user:**

- `synonyms` / `resolve_in` / `resolve_min` are serialized with `json.dumps(..., ensure_ascii=False)`. The nuance: `ensure_ascii` is **not** a correctness knob — `ensure_ascii=True` would escape `ê` to `ê` but `json.loads` round-trips both to the identical Python string, so a diacritic brand name is never lost either way. `ensure_ascii=False` was chosen purely for (1) consistency with `page_content` and the other string metadata, which are stored as raw NFC UTF-8, so the DB does not mix two representations of Vietnamese, and (2) readability when inspecting the Chroma sqlite by eye. The **implicit contract** this creates is the thing to remember: because these fields are JSON strings stuffed into scalar-only metadata, the retrieval side MUST `json.loads` them before use and must never substring-match the raw stored string.

- Chroma metadata is scalar-only, so lists are the *only* thing that needs serializing; `None`/`""` keys are dropped entirely rather than stored as empty, keeping the metadata shape sparse and predictable.

- Text is NFC-normalized at load time (`normalize_text`) but **not** lowercased and diacritics are **not** stripped — the dense side keeps original case and accents. Lowercasing/normalization for lexical matching belongs to the later BM25 session, not here.

- `resolve_in` and `resolve_min` are kept as two separate metadata fields, never merged — mirroring the KB schema decision so the IN-vs-MIN choice stays open at retrieval time.
