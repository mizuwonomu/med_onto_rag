"""
    Kiểm tra tính nhất quán của tập test entity-linking (JSONL) trước khi dùng
    để eval / train reranker.

    Script này KHÔNG sửa và KHÔNG sinh dữ liệu — nó chỉ đọc và báo lỗi. Mục đích
    là chặn các lỗi nhãn "im lặng": không gây crash ở downstream, không lộ ra ở
    đâu cả, chỉ làm điểm eval xấu đi một cách khó truy nguyên.

    Luật kiểm tra (tất cả đều ở phạm vi TỪNG DÒNG):

      1. gold ∩ distractor == ∅
         Lỗi nặng nhất. Cùng một mention mà một mã vừa là đáp án vừa là nhiễu
         thì tín hiệu train ngược chiều nhau, không có cách nào đúng.
         LƯU Ý: cố tình KHÔNG kiểm tra giao chéo giữa các dòng — một mã là gold
         ở dòng này và distractor ở dòng khác là hợp lệ và cần thiết, vì nhãn
         đúng phụ thuộc vào mention chứ không phụ thuộc vào mã.

      2. gold không rỗng và không trùng lặp nội bộ.

      3. kb ∈ {icd, rxnorm}.

      4. distractor không trùng lặp nội bộ (mã lặp làm lệch trọng số negative).

      5. gold_kind (nếu có) khớp với luật phân loại: >=2 gold phân biệt thì
         icd -> multi_sibling, rxnorm -> multi_solving; còn lại là single.

      6. cluster_name (nếu có) và distractor phải nhất quán: có nhiễu thì phải
         có tên cụm, và ngược lại — cảnh báo chứ không phải lỗi cứng.

    Exit code 1 nếu có bất kỳ lỗi nào (dùng được trong CI / trước khi commit data).
"""
import argparse
import json
import sys
from pathlib import Path

VALID_KB = {"icd", "rxnorm"}
MULTI_KIND_BY_KB = {"icd": "multi_sibling", "rxnorm": "multi_solving"}


def expected_gold_kind(kb: str, gold: list) -> str | None:
    """gold_kind đáng lẽ phải có. None nếu không suy ra được (kb lạ)."""
    if len(set(gold)) == 1:
        return "single"
    return MULTI_KIND_BY_KB.get(kb)


def duplicates(values: list) -> list:
    seen, dups = set(), []
    for v in values:
        if v in seen and v not in dups:
            dups.append(v)
        seen.add(v)
    return dups


def check_record(record: dict) -> tuple[list[str], list[str]]:
    """Trả về (errors, warnings) của 1 record."""
    errors: list[str] = []
    warnings: list[str] = []

    gold = record.get("gold")
    if not isinstance(gold, list) or not gold:
        errors.append(f"gold rỗng hoặc không phải list: {gold!r}")
        return errors, warnings

    if dups := duplicates(gold):
        errors.append(f"gold có mã trùng lặp: {dups}")

    kb = record.get("kb")
    if kb not in VALID_KB:
        errors.append(f"kb không hợp lệ: {kb!r}")

    distractor = record.get("distractor", [])
    if not isinstance(distractor, list):
        errors.append(f"distractor không phải list: {distractor!r}")
        distractor = []

    # Luật 1 — lỗi nặng nhất.
    if overlap := sorted(set(distractor) & set(gold)):
        errors.append(f"distractor giao với gold trong cùng dòng: {overlap}")

    if dups := duplicates(distractor):
        errors.append(f"distractor có mã trùng lặp: {dups}")

    if "gold_kind" in record:
        expected = expected_gold_kind(kb, gold)
        if expected is not None and record["gold_kind"] != expected:
            errors.append(
                f"gold_kind sai: có {record['gold_kind']!r}, đáng lẽ {expected!r}"
            )

    if "cluster_name" in record:
        has_cluster = record["cluster_name"] is not None
        has_distractor = bool(distractor)
        if has_distractor and not has_cluster:
            warnings.append("có distractor nhưng cluster_name là null")
        elif has_cluster and not has_distractor:
            warnings.append(f"có cluster_name {record['cluster_name']!r} nhưng distractor rỗng")

    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra nhất quán tập test JSONL.")
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="coi warning là lỗi (exit 1)",
    )
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as f:
        records = [(i, json.loads(line)) for i, line in enumerate(f, 1) if line.strip()]

    n_errors = n_warnings = 0
    for line_no, record in records:
        errors, warnings = check_record(record)
        for msg in errors:
            n_errors += 1
            print(f"[ERROR] dòng {line_no}: {msg}")
        for msg in warnings:
            n_warnings += 1
            print(f"[WARN ] dòng {line_no}: {msg}")

    print(f"\nĐã kiểm tra {len(records)} dòng: {n_errors} lỗi, {n_warnings} cảnh báo.")

    if n_errors or (args.strict and n_warnings):
        sys.exit(1)


if __name__ == "__main__":
    main()
