"""
Sinh file .json nộp bài: mỗi `<tên>.txt` đầu vào -> một `<tên>.json` đầu ra.

ĐỊNH DẠNG ĐẦU RA LÀ LIST TRẦN, KHÔNG CÓ KHOÁ BAO NGOÀI
------------------------------------------------------
Mỗi file .json là một list dictionary theo đúng mẫu BTC:

    [
      {"text": "...", "type": "THUỐC", "candidates": ["308135"],
       "assertions": ["isHistorical"], "position": [58, 83]},
      ...
    ]

Khoá `entities` KHÔNG xuất hiện ở đây. Nó chỉ tồn tại bên trong lời gọi LLM, vì
`response_format: json_schema` bắt buộc root phải là object (ràng buộc của giao
thức OpenAI, không phải lựa chọn thiết kế) - nên `NerOutput` bọc list vào một
khoá. `pipeline.extract_document` trả về list trần, và script này ghi thẳng list
đó ra đĩa. Xem `src/extract/schema.py`.

CHẠY TRÊN DỮ LIỆU NÀO
---------------------
Mặc định đọc `--input-dir`. Trong lúc phát triển thì trỏ vào tập synthetic đã
ghép (`--from-jsonl`), chỉ đụng `data/input/` của BTC ở lần chạy cuối cùng - dữ
liệu đó là tập chấm điểm, không phải sân tập.

TÍNH TIẾP TỤC ĐƯỢC
------------------
Mặc định BỎ QUA file đã có .json (`--skip-existing`). Chạy 100 document qua một
model 9B mất hàng chục phút; đứt giữa chừng mà phải làm lại từ đầu là mất trắng
phần đã xong. Muốn ghi đè thì `--overwrite`.

Một document lỗi (LLM trả JSON không hợp lệ, timeout) KHÔNG làm hỏng cả mẻ: ghi
log, đếm vào bảng cuối, chạy tiếp. Nhưng file .json của nó sẽ KHÔNG được tạo -
im lặng ghi ra một list rỗng là biến lỗi hạ tầng thành điểm 0 trông như thật.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, get_args

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retrieval"))

from common import read_jsonl  # noqa: E402

from configs.config import (  # noqa: E402
    NER_INPUT_DIR,
    NER_OUTPUT_DIR,
    NER_SYNTHETIC_PATH,
)
from extract.schema import ENTITY_TYPES  # noqa: E402
from pipeline import (  # noqa: E402
    TYPES_WITH_ASSERTIONS,
    TYPES_WITH_CANDIDATES,
    extract_document,
)

logger = logging.getLogger(__name__)

VALID_TYPES = frozenset(get_args(ENTITY_TYPES))


def load_documents(
    input_dir: Path | None, from_jsonl: Path | None
) -> list[tuple[str, str]]:
    """Trả [(stem, raw_text)]. `stem` quyết định tên file .json đầu ra."""
    if from_jsonl is not None:
        rows = list(read_jsonl(from_jsonl))
        return [(str(r["doc_id"]), r["input"]) for r in rows]

    files = sorted(
        input_dir.glob("*.txt"),
        # 1.txt, 2.txt, ..., 100.txt - sắp theo SỐ chứ không theo chuỗi, để log
        # đọc được theo thứ tự tự nhiên.
        key=lambda p: (int(p.stem) if p.stem.isdigit() else 0, p.stem),
    )
    return [(p.stem, p.read_text(encoding="utf-8")) for p in files]


def verify_outputs(
    docs: list[tuple[str, str]], out_dir: Path
) -> tuple[list[str], list[str]]:
    """Soát thư mục nộp bài. Trả (lỗi CHẶN nộp, cảnh báo).

    VÌ SAO PHẢI CÓ BƯỚC NÀY
    -----------------------
    Document lỗi thì KHÔNG được ghi file - cố ý, vì một list rỗng sẽ nguỵ trang
    lỗi hạ tầng thành điểm 0 trông như thật. Nhưng hệ quả là một mẻ chạy dở
    trông y hệt một mẻ chạy xong: chỉ khác ở dòng log đã cuộn mất. Với 100 file
    thì thiếu một file là mất trắng một document mà không ai hay.

    BẤT BIẾN QUAN TRỌNG NHẤT là `raw[start:end] == text`. Đó là hợp đồng mà
    `align.py` dựng lên, và nếu nó vỡ thì `text` và `position` mâu thuẫn nhau -
    BTC chấm theo cái nào cũng sai. Kiểm ở đây là kiểm trên ĐÚNG chuỗi đã đọc từ
    đĩa, không phải trên biến còn nằm trong bộ nhớ.
    """
    errors: list[str] = []
    warnings: list[str] = []

    for stem, raw in docs:
        path = out_dir / f"{stem}.json"
        if not path.exists():
            errors.append(f"{stem}.json THIẾU (document không chạy được hoặc chưa chạy)")
            continue
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            errors.append(f"{stem}.json không parse được: {e}")
            continue
        if not isinstance(records, list):
            errors.append(f"{stem}.json root phải là LIST, đang là {type(records).__name__}")
            continue

        for k, r in enumerate(records):
            at = f"{stem}.json[{k}]"
            if not isinstance(r, dict):
                errors.append(f"{at} không phải object")
                continue

            typ, text = r.get("type"), r.get("text")
            if typ not in VALID_TYPES:
                errors.append(f"{at} type không hợp lệ: {typ!r}")
            if not isinstance(text, str) or not text:
                errors.append(f"{at} text rỗng hoặc không phải chuỗi")
                continue

            pos = r.get("position")
            if (
                not isinstance(pos, list)
                or len(pos) != 2
                or not all(isinstance(x, int) for x in pos)
            ):
                errors.append(f"{at} position phải là [int, int], đang là {pos!r}")
            elif not (0 <= pos[0] < pos[1] <= len(raw)):
                errors.append(f"{at} position {pos} nằm ngoài văn bản (dài {len(raw)})")
            elif raw[pos[0] : pos[1]] != text:
                # Hợp đồng nguyên văn vỡ: text và position chỉ vào hai thứ khác nhau.
                errors.append(
                    f"{at} position {pos} trỏ vào {raw[pos[0]:pos[1]]!r} "
                    f"nhưng text là {text!r}"
                )

            if "assertions" in r and typ not in TYPES_WITH_ASSERTIONS:
                errors.append(f"{at} loại {typ} KHÔNG được mang khoá `assertions`")
            if "candidates" in r and typ not in TYPES_WITH_CANDIDATES:
                errors.append(f"{at} loại {typ} KHÔNG được mang khoá `candidates`")
            if typ in TYPES_WITH_CANDIDATES and not r.get("candidates"):
                # Cảnh báo, không phải lỗi: gold RỖNG là hợp lệ về mặt định dạng.
                # Nhưng gold của hai loại này hầu như luôn có ít nhất một mã, nên
                # mỗi dòng như vậy gần như chắc chắn ăn 0 ở tầng candidates.
                warnings.append(f"{at} {typ} {text!r} không có mã nào")

    extra = {p.stem for p in out_dir.glob("*.json")} - {s for s, _ in docs}
    for s in sorted(extra):
        warnings.append(f"{s}.json thừa - không có document đầu vào tương ứng")

    return errors, warnings


def report_verification(errors: list[str], warnings: list[str], n: int, show: int = 15) -> bool:
    print(f"\n== SOÁT BÀI NỘP ({n} document)")
    for label, items in (("LỖI CHẶN NỘP", errors), ("cảnh báo", warnings)):
        print(f"   {label}: {len(items)}")
        for line in items[:show]:
            print(f"     {line}")
        if len(items) > show:
            print(f"     ... còn {len(items) - show}")
    if not errors:
        print("   -> định dạng hợp lệ: mỗi <tên>.txt có một <tên>.json, "
              "root là list, position khớp nguyên văn.")
    return not errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input-dir",
        type=Path,
        nargs="?",
        const=NER_INPUT_DIR,
        help=f"thư mục chứa các file .txt đầu vào (mặc định {NER_INPUT_DIR})",
    )
    source.add_argument(
        "--from-jsonl",
        type=Path,
        nargs="?",
        const=NER_SYNTHETIC_PATH,
        help=f"đọc trường `input` từ file JSONL (mặc định {NER_SYNTHETIC_PATH})",
    )
    parser.add_argument("--out-dir", type=Path, default=NER_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="chỉ chạy N document đầu")
    parser.add_argument(
        "--skip-link",
        action="store_true",
        help="không gắn candidates (không nạp GPU) - CHỈ để soi tầng NER, không dùng để nộp",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="chỉ soát thư mục output đã có, không chạy model",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="ghi đè file .json đã có (mặc định: bỏ qua để chạy tiếp được)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Không có nguồn nào -> tập synthetic. Chạy lên tập chấm điểm của BTC vẫn
    # phải là hành động CỐ Ý: `--input-dir` không tham số thì mới lấy NER_INPUT_DIR.
    if args.input_dir is None and args.from_jsonl is None:
        args.from_jsonl = NER_SYNTHETIC_PATH
        logger.info("không chỉ định nguồn -> dùng tập synthetic %s", args.from_jsonl)
    if args.input_dir is not None:
        logger.info("nguồn: %s -> ghi ra %s", args.input_dir, args.out_dir)

    if args.input_dir is not None and not args.input_dir.is_dir():
        raise SystemExit(f"không thấy thư mục input: {args.input_dir}")
    if args.from_jsonl is not None and not args.from_jsonl.exists():
        raise SystemExit(f"không thấy file JSONL: {args.from_jsonl}")

    docs = load_documents(args.input_dir, args.from_jsonl)
    if args.limit:
        docs = docs[: args.limit]
    if not docs:
        raise SystemExit("không có document nào để chạy")
    logger.info("nạp %d document", len(docs))

    if args.verify_only:
        errors, warnings = verify_outputs(docs, args.out_dir)
        raise SystemExit(0 if report_verification(errors, warnings, len(docs)) else 1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    from extract.llm import PreflightError, build_ner_chain, preflight

    ner_chain = build_ner_chain()
    try:  # bắt lỗi server TRƯỚC khi tốn hàng phút nạp GPU
        preflight(ner_chain)
    except PreflightError as e:
        raise SystemExit(f"preflight THẤT BẠI: {e}") from e

    backends = None
    if args.skip_link:
        logger.warning("--skip-link: output KHÔNG có candidates, không dùng để nộp bài")
    else:
        from extract.linker import build_backends

        logger.info("nạp embedding + reranker lên GPU (mất một lúc)...")
        backends = build_backends()

    done = skipped = failed = 0
    total_entities = 0
    started = time.time()

    for i, (stem, raw_text) in enumerate(docs, 1):
        out_path = args.out_dir / f"{stem}.json"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            records = extract_document(raw_text, ner_chain, backends)
        except Exception as e:
            # Không ghi file rỗng: một lỗi hạ tầng không được hoá trang thành
            # "document này không có khái niệm nào".
            failed += 1
            logger.error("[%s] LỖI, bỏ qua: %s: %s", stem, type(e).__name__, e)
            continue

        out_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        done += 1
        total_entities += len(records)
        logger.info("[%d/%d] %s -> %d khái niệm", i, len(docs), out_path.name, len(records))

    elapsed = time.time() - started
    print(
        f"\n== xong: {done} ghi mới, {skipped} bỏ qua (đã có), {failed} lỗi"
        f"\n   {total_entities} khái niệm, {elapsed:.1f}s"
        f"\n   -> {args.out_dir}"
    )
    if skipped and not args.overwrite:
        print("   (dùng --overwrite để ghi lại các file đã có)")

    errors, warnings = verify_outputs(docs, args.out_dir)
    ok = report_verification(errors, warnings, len(docs))
    if args.skip_link:
        print("\n   !! --skip-link: output KHÔNG có `candidates`, KHÔNG nộp được.")
        ok = False
    if failed or not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
