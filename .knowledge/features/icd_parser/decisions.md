# decisions.md (why — the expensive part)

- Date: 2026-07-21
- Branch: build_kb
- Commit hash: 27f2855 (feat scripts entrypoint) — feature spans 3 commits: `4c78662` chore(deps) → `300c1b8` feat(kb) → `27f2855` feat(scripts)

**Chosen approach + why:**

- **Code-anchor parsing (not row/geometry join):** parse each column independently to `{code → desc}` then outer-join by code. Chosen because page breaks and EN/VI vertical drift would corrupt any position-based pairing; code is the one stable key present in both columns.
- **Dynamic column-split detection (histogram, nearest-center gap):** for each page, bucket word `x0` into a histogram and pick the empty band whose right edge is nearest page center. Chosen over a fixed `width/2` split because pages vary, and over "widest gap" because hanging-indented codes create a left-margin band wider than the true gutter.
- **Crop architecture over word-level clustering:** chose crop for maintainability — it auto-detects the split per page (no hard-coded band constants), so it survives layout changes. Word-level was faster (~37x) and had marginally fewer nulls (12 vs 17) but hard-codes 3 band boundaries.
- **Font glyph-drop treated as data flaw, flagged not fixed:** the source PDF's Arial,Bold font drops some Vietnamese diacritic glyphs (~434 codes). Flagged via `vi_glyph_ok`; `text` still keeps the Vietnamese.
- `text` **keeps Vietnamese even when glyph-broken:** only falls back to English when Vietnamese is entirely absent (`head_vi is None`, 17 codes).
- **Dagger/asterisk (†/*) stripped + dedup-merged:** normalized out of codes because they are dual-classification marks, not part of the mapping key.
- **Head/tail split, embed head only:** cut description at the first note marker (Loai tru / Bao gom / Dung ma / Excl / Incl / [See); head → name + `text`, tail → `note_`*.

**Assumptions it rests on:**

- The PDF stays a fixed 2-column bilingual layout (English left / Vietnamese right), ~595pt wide, with the inter-column gutter near page center.
- ICD codes always appear in *both* language columns (needed for the R51→"51" leading-letter recovery idea and for the code-anchor to work per column).
- The disease name always precedes note markers in the description (head-first structure).
- Data stays small/static enough that an 80s single-threaded crop pass is acceptable (speed was explicitly deemed unimportant).
- `text` is the only embedded field and the shared schema is stable (breaking it = re-ingest).

**Failed approaches:**

- Tried: `PyMuPDF (fitz) instead of pdfplumber to fix missing Vietnamese diacritics` — Failed because: `the glyph loss is in the source PDF's embedded Arial,Bold font (bad CID mapping), so both extractors read the same corrupt data` — Avoid when: `tempted to swap PDF libraries hoping to recover missing diacritics — it is a font-level defect, not an extractor difference`.
- Tried: `"widest gap" heuristic for the column split` — Failed because: `hanging-indented ICD codes create a left-margin empty band wider than the real inter-column gutter, so the split landed at ~70pt and bled English into Vietnamese on ~80/841 pages` — Avoid when: `writing any column detector on a page with hanging indents — anchor to page center, not gap width`. **(cross-cutting-ish: this is the core column-detection contract — worth a pointer.)**
- Tried: `extending the glyph-drop detector to flag "single isolated letter between two spaces" (to catch "th c qu n")` — Failed because: `"u" (khối u / tumor) and "y" (y học) are valid one-character Vietnamese words, causing ~1170 false positives` — Avoid when: `tempted to broaden the glyph regex to catch consonant-only drops — the false-positive cost far exceeds the ~4 real cases`.
- Tried: `word-level bbox clustering as the main parser` — Failed because: `not a correctness failure, but it hard-codes 3 band-boundary constants (VI_HALF_X, EN_CODE_MAX_X, VI_CODE_MAX_X) that break silently if the PDF layout shifts; maintainability lost to crop's self-adapting split` — Avoid when: `null-code recovery becomes a priority — then revisit word-level, since only it can rebuild codes with a dropped leading letter`.

**Nuances agreed with the user:**

- Glyph corruption usually loses only **one syllable**; 82% of the 417 flagged codes have exactly 1 error, so the head name is still query-usable → keep Vietnamese in `text` rather than "waste" it on English.
- `vi_glyph_ok` became pure metadata (a trace flag), it no longer controls `text`.
- One legitimate phantom accepted: `B12` (a "vitamin B12" reference fragment that matched the code regex) — 1/11,125, not worth hardening the regex against.
- Notes (Loai tru / Bao gom) are kept in `note_*` fields, not discarded — reserved for possible secondary BM25 use — rather than thrown away.
- `KXĐK` / `NOS` are part of disease names, explicitly NOT treated as cut markers.

