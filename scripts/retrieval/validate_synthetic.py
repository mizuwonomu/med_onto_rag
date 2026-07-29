"""
Cổng dữ liệu (data gate) cho tập synthetic dùng để hiệu chỉnh ngưỡng retrieval.

Khác với `scripts/validate_test_set.py` (chỉ kiểm tra TỪNG DÒNG, không cần KB),
script này neo tập test vào KB thật và vào cấu trúc cụm (cluster):

  1. Schema đầy đủ và không có field lạ — bắt được lỗi gõ `distractors` vs
     `distractor`, vốn sẽ khiến mọi hard-negative bị bỏ qua trong im lặng.
  2. Mọi `gold` / `distractor` phải TỒN TẠI trong KB tương ứng. ICD: tập `code`
     của icd_concepts. RxNorm: tập `code` của candidate HỢP với mọi phần tử
     `resolve_in` của alias (vì một brand giải về nhiều generic RXCUI).
  3. Không trùng (mention, kb).
  4. `gold_kind` khớp với luật: 1 gold -> single; nhiều gold ICD -> multi_sibling
     và phải cùng block; nhiều gold RxNorm -> multi_solving và phải cùng nằm
     trong một `resolve_in` nào đó (tức là chúng thật sự là các generic của
     cùng một brand).
  5. Nhất quán cụm: có `cluster_name` <=> có `distractor`; cụm phải có >= 2 dòng;
     mọi distractor phải là near-miss THẬT của cụm - cùng block ICD, hoặc chung
     nhóm `resolve_in` với gold nào đó của cụm. Distractor lấy từ một vùng khác
     hẳn thì không cạnh tranh nổi với gold, và cụm mất tác dụng hard-negative.
     Xem `check_clusters` để biết vì sao luật này KHÔNG đòi distractor phải là
     gold của dòng khác. Với cụm ICD, toàn bộ gold+distractor phải cùng block.

Exit code 1 nếu có bất kỳ lỗi nào.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from configs.config import (  # noqa: E402
    ICD_JSONL_PATH,
    RXNORM_ALIASES_PATH,
    RXNORM_CONCEPTS_PATH,
    SYNTHETIC_EVAL_PATH,
)
from kb.icd_hierarchy import lookup_hierarchy  # noqa: E402

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"mention", "kb", "gold", "noise", "gold_kind"}
OPTIONAL_FIELDS = {"cluster_name", "distractor"}
VALID_KB = {"icd", "rxnorm"}
MULTI_KIND_BY_KB = {"icd": "multi_sibling", "rxnorm": "multi_solving"}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_valid_codes(
    icd_path: Path, rxnorm_concepts: Path, rxnorm_aliases: Path
) -> tuple[dict[str, set[str]], list[set[str]]]:
    """Trả về ({kb -> tập mã hợp lệ}, danh sách các nhóm resolve_in của alias)."""
    icd_codes = {rec["code"] for rec in read_jsonl(icd_path) if rec.get("code")}

    rxnorm_codes = {
        str(rec["code"]) for rec in read_jsonl(rxnorm_concepts) if rec.get("code")
    }
    resolve_groups: list[set[str]] = []
    for rec in read_jsonl(rxnorm_aliases):
        resolve_in = rec.get("resolve_in") or []
        group = {str(c) for c in resolve_in}
        if group:
            resolve_groups.append(group)
            rxnorm_codes |= group

    return {"icd": icd_codes, "rxnorm": rxnorm_codes}, resolve_groups


def block_of(code: str) -> str | None:
    return lookup_hierarchy(code)[1]


def check_schema(rec: dict[str, Any]) -> list[str]:
    errors = []
    keys = set(rec)
    if missing := REQUIRED_FIELDS - keys:
        errors.append(f"thiếu field bắt buộc: {sorted(missing)}")
    if unknown := keys - REQUIRED_FIELDS - OPTIONAL_FIELDS:
        errors.append(f"field lạ (gõ sai tên?): {sorted(unknown)}")
    if rec.get("kb") not in VALID_KB:
        errors.append(f"kb không hợp lệ: {rec.get('kb')!r}")
    for field in ("gold", "distractor"):
        value = rec.get(field, [])
        if value is not None and not isinstance(value, list):
            errors.append(f"{field} phải là list, nhận {type(value).__name__}")
    if not rec.get("gold"):
        errors.append("gold rỗng")
    return errors


def check_codes_exist(
    rec: dict[str, Any], valid: dict[str, set[str]]
) -> list[str]:
    kb = rec.get("kb")
    if kb not in valid:
        return []
    errors = []
    for field in ("gold", "distractor"):
        codes = [str(c) for c in (rec.get(field) or [])]
        if unknown := [c for c in codes if c not in valid[kb]]:
            errors.append(f"{field} có mã không tồn tại trong KB {kb}: {unknown}")
    return errors


def check_gold_kind(
    rec: dict[str, Any], resolve_groups: list[set[str]]
) -> list[str]:
    kb, gold = rec.get("kb"), [str(c) for c in (rec.get("gold") or [])]
    kind = rec.get("gold_kind")
    distinct = set(gold)
    if not distinct or kb not in VALID_KB:
        return []

    if len(distinct) == 1:
        return [] if kind == "single" else [f"gold_kind={kind!r}, đáng lẽ 'single'"]

    expected = MULTI_KIND_BY_KB[kb]
    errors = []
    if kind != expected:
        errors.append(f"gold_kind={kind!r}, đáng lẽ {expected!r}")

    if kb == "icd":
        blocks = {block_of(c) for c in distinct}
        if len(blocks) > 1 or None in blocks:
            errors.append(f"multi_sibling nhưng gold không cùng block: {sorted(blocks)}")
    else:
        if not any(distinct <= group for group in resolve_groups):
            errors.append(
                "multi_solving nhưng không resolve_in nào chứa trọn bộ gold "
                f"{sorted(distinct)} - các mã này không phải generic của cùng một brand"
            )
    return errors


def check_cluster_shape(rec: dict[str, Any]) -> list[str]:
    cluster = rec.get("cluster_name")
    distractor = rec.get("distractor") or []
    if cluster and not distractor:
        return [f"cluster_name={cluster!r} nhưng distractor rỗng"]
    if distractor and not cluster:
        return ["có distractor nhưng cluster_name rỗng/null"]
    return []


def check_clusters(
    rows: list[tuple[int, dict[str, Any]]], resolve_groups: list[set[str]]
) -> list[tuple[int, str]]:
    """Kiểm tra chéo trong từng (kb, cluster_name): distractor phải là near-miss THẬT của cụm.

    Luật cũ bắt distractor phải là gold của một dòng khác cùng cụm. Luật đó SAI và
    đã bị chính dữ liệu bác bỏ: một cụm hợp lệ hoàn toàn có thể chứa mã anh em chỉ
    đóng vai nhiễu, không bao giờ là đáp án của ai. Ví dụ thật trong
    `ICD_AminoAcid_UreaCycle`: gold trải trên E72.2/E72.3, còn E72.4/E72.5 chỉ để
    làm nhiễu. Chúng là near-miss chuẩn mực - reranker phải học cách KHÔNG chọn
    chúng - nhưng luật cũ báo oan cả 48 dòng thuộc 8 cụm dạng này. Cổng báo oan
    hàng loạt còn nguy hiểm hơn cổng lỏng, vì người dùng sẽ quen tay bỏ qua rồi
    lỗi thật trượt theo.

    Luật hiện tại: distractor chỉ cần chứng minh mình cùng "vùng" với gold của cụm.
      * ICD    -> cùng block với gold nào đó của cụm.
      * RxNorm -> nằm chung một nhóm `resolve_in` với gold nào đó của cụm.
    Vẫn bắt được đúng lỗi thật ở v2 (distractor lấy từ một nhóm khái niệm khác hẳn,
    không dính dáng gì tới gold của cụm) mà không đụng tới cụm hợp lệ.

    Cố ý CHƯA kiểm near-miss theo mặt chữ (dexamethasone vs dexmedetomidine: khác
    vùng ontology nhưng cả BM25 lẫn dense đều lẫn). Lớp đó có thật và đáng đo,
    nhưng muốn kiểm tất định thì phải neo vào "distractor có lọt top-k BM25 của
    mention không" chứ không phải một ngưỡng khoảng cách chuỗi bịa ra - mà làm vậy
    thì cổng phải nạp BM25 index, hết còn là script thuần dữ liệu. Để dump điểm
    quyết định xem có đáng đánh đổi không.
    """
    groups: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    for line_no, rec in rows:
        if rec.get("cluster_name"):
            groups[(rec.get("kb"), rec["cluster_name"])].append((line_no, rec))

    errors: list[tuple[int, str]] = []
    for (kb, cluster), members in groups.items():
        if len(members) < 2:
            errors.append(
                (members[0][0], f"cụm {cluster!r} (kb={kb}) chỉ có 1 dòng - không thành hard-negative group")
            )

        cluster_gold = {str(c) for _, rec in members for c in (rec.get("gold") or [])}
        if kb == "icd":
            allowed_blocks = {block_of(c) for c in cluster_gold} - {None}

            def is_near_miss(code: str) -> bool:
                return block_of(code) in allowed_blocks
        else:
            # nhóm resolve_in nào có dính gold của cụm thì mọi thành viên của nó
            # đều là near-miss hợp lệ
            neighbours = {
                c for group in resolve_groups if group & cluster_gold for c in group
            }

            def is_near_miss(code: str) -> bool:
                return code in neighbours

        for line_no, rec in members:
            strangers = [
                str(c) for c in (rec.get("distractor") or []) if not is_near_miss(str(c))
            ]
            if strangers:
                errors.append(
                    (
                        line_no,
                        f"distractor không phải near-miss của cụm {cluster!r} "
                        f"(khác block/khác nhóm resolve so với mọi gold của cụm): {strangers}",
                    )
                )
        if kb == "icd":
            codes = {
                str(c)
                for _, rec in members
                for c in (rec.get("gold") or []) + (rec.get("distractor") or [])
            }
            blocks = {block_of(c) for c in codes}
            if len(blocks) > 1:
                errors.append(
                    (members[0][0], f"cụm ICD {cluster!r} trải trên nhiều block: {sorted(map(str, blocks))}")
                )
    return errors


def validate(
    path: Path, valid: dict[str, set[str]], resolve_groups: list[set[str]]
) -> list[tuple[int, str]]:
    rows = [(i, rec) for i, rec in enumerate(read_jsonl(path), 1)]
    errors: list[tuple[int, str]] = []

    seen: dict[tuple[str, str], int] = {}
    for line_no, rec in rows:
        for msg in (
            check_schema(rec)
            + check_codes_exist(rec, valid)
            + check_gold_kind(rec, resolve_groups)
            + check_cluster_shape(rec)
        ):
            errors.append((line_no, msg))

        key = (rec.get("mention"), rec.get("kb"))
        if key in seen:
            errors.append((line_no, f"trùng (mention, kb) với dòng {seen[key]}"))
        else:
            seen[key] = line_no

    errors.extend(check_clusters(rows, resolve_groups))
    errors.sort()
    logger.info("đã kiểm tra %d dòng", len(rows))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=SYNTHETIC_EVAL_PATH)
    parser.add_argument("--icd", type=Path, default=ICD_JSONL_PATH)
    parser.add_argument("--rxnorm-concepts", type=Path, default=RXNORM_CONCEPTS_PATH)
    parser.add_argument("--rxnorm-aliases", type=Path, default=RXNORM_ALIASES_PATH)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    valid, resolve_groups = build_valid_codes(args.icd, args.rxnorm_concepts, args.rxnorm_aliases)
    logger.info(
        "KB hợp lệ: icd=%d mã, rxnorm=%d mã (%d nhóm resolve_in)",
        len(valid["icd"]), len(valid["rxnorm"]), len(resolve_groups),
    )

    errors = validate(args.input, valid, resolve_groups)
    for line_no, msg in errors:
        print(f"[ERROR] dòng {line_no}: {msg}")

    print(f"\nTổng: {len(errors)} lỗi.")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
