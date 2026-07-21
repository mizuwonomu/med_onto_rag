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