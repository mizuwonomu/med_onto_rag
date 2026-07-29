"""
    Gán nhãn gold_kind cho tập test entity-linking (JSONL).

    gold_kind phân loại độ khó của mention theo số lượng mã gold và loại KB:
      - single         : đúng 1 mã gold duy nhất
      - multi_sibling  : kb="icd", >=2 mã gold (anh em cùng nhánh, mention mơ hồ
                         về mức độ chi tiết — vd I50.0 + I50.9)
      - multi_solving  : kb="rxnorm", >=2 mã gold (biệt dược đa hoạt chất phân
                         giải ra nhiều RXCUI — vd bactrim → 10180 + 10829)

    Script chỉ chịu trách nhiệm cho `gold_kind`. Mọi trường khác (`cluster_name`,
    `distractor`...) được giữ nguyên; `distractor` chỉ được thêm dưới dạng list
    rỗng KHI record chưa có trường này, để cố định schema mà không đè mất kết quả
    sinh nhiễu từ bước LLM.

    Script KHÔNG ghi đè file input.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_INPUT = Path("data/test/rxnorm_icd_synthetic_400.jsonl")

MULTI_KIND_BY_KB = {
    "icd": "multi_sibling",
    "rxnorm": "multi_solving",
}


def classify_gold_kind(record: dict) -> str:
    """Suy ra gold_kind từ 1 record. Ném ValueError nếu record không hợp lệ."""
    gold = record.get("gold")
    if not isinstance(gold, list) or not gold:
        raise ValueError(f"gold rỗng hoặc không phải list: {record!r}")

    # Trùng mã trong cùng 1 dòng không làm mention khó hơn -> đếm mã phân biệt.
    distinct = len(set(gold))
    if distinct == 1:
        return "single"

    kb = record.get("kb")
    if kb not in MULTI_KIND_BY_KB:
        raise ValueError(f"kb không hỗ trợ cho trường hợp nhiều gold: {kb!r} ({record!r})")
    return MULTI_KIND_BY_KB[kb]


def label_records(records: list[dict]) -> list[dict]:
    """Trả về bản sao các record đã thêm gold_kind + distractor placeholder."""
    labeled = []
    for line_no, record in enumerate(records, start=1):
        try:
            gold_kind = classify_gold_kind(record)
        except ValueError as exc:
            raise ValueError(f"dòng {line_no}: {exc}") from exc

        # Giữ nguyên thứ tự key gốc + mọi trường sẵn có (cluster_name, distractor
        # đã gen bằng LLM...). `distractor` chỉ được điền placeholder khi VẮNG MẶT
        # — không bao giờ ghi đè list đã có.
        labeled_record = {**record, "gold_kind": gold_kind}
        labeled_record.setdefault("distractor", [])
        labeled.append(labeled_record)
    return labeled


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gán gold_kind cho tập test JSONL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="mặc định: <input>-labeled.jsonl (không ghi đè input)",
    )
    args = parser.parse_args()

    output_path = args.output or args.input.with_name(f"{args.input.stem}-labeled.jsonl")
    if output_path.resolve() == args.input.resolve():
        raise SystemExit("output trùng input — script này không ghi đè dữ liệu gốc.")

    records = read_jsonl(args.input)
    labeled = label_records(records)
    write_jsonl(output_path, labeled)

    counts = Counter(r["gold_kind"] for r in labeled)
    print(f"Wrote {len(labeled)} records to {output_path}")
    for kind in ("single", "multi_sibling", "multi_solving"):
        print(f"  {kind:14s}: {counts.get(kind, 0)}")


if __name__ == "__main__":
    main()
