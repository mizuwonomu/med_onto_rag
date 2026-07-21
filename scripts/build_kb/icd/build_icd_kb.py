"""
    Module build KB từ logic
"""
import json
from pathlib import Path

import pdfplumber

from kb.icd import build_icd_concepts

PDF_PATH = Path("data/raw/ICD_10.pdf")
OUTPUT_PATH = Path("data/processed/icd_concepts.jsonl")


def main() -> None:
    with pdfplumber.open(PDF_PATH) as pdf:
        records = build_icd_concepts(pdf)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    broken = sum(1 for r in records if not r["vi_glyph_ok"])
    fallback_en = sum(1 for r in records if r["name_vi"] is None and r["name_en"] is not None)
    print(f"Wrote {len(records)} ICD concepts to {OUTPUT_PATH}")
    print(f"vi_glyph_ok=False (flagged, text vẫn giữ tiếng Việt): {broken} ({broken / len(records) * 100:.2f}%)")
    print(f"text fallback sang name_en (name_vi mất hẳn): {fallback_en}")


if __name__ == "__main__":
    main()
