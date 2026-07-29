"""
Chạy mô hình MỘT LẦN, ghi toàn bộ điểm thô ra JSONL.

Nguyên tắc: GPU chỉ đụng vào ở script này. Mọi bước phân tích và sweep ngưỡng
sau đó chỉ đọc file dump - nên có thể quét lưới ngưỡng, bootstrap, vẽ lại hình
bao nhiêu lần cũng được mà không tốn thêm một giây GPU nào. Dump là hợp đồng
giữa "đo" và "phân tích".

Mỗi bản ghi mang theo TOÀN BỘ field nguồn (gold, noise, gold_kind, cluster_name,
distractor) để script downstream không bao giờ phải mở lại file synthetic - tránh
hoàn toàn nguy cơ lệch dòng giữa hai file.

Chạy cho cả 2 biến thể query (raw, strip_dose) để quyết định có cần tầng chuẩn
hóa query hay không - xem `retrieval/query_variants.py`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from common import (
    PLOTS_DIR,
    dump_path,
    plot_score_distributions,
    print_table,
    read_jsonl,
    write_jsonl,
)

from configs.config import (  # noqa: E402
    BM25_ICD_DIR,
    BM25_RXNORM_DIR,
    EVAL_OUTPUT_DIR,
    ICD_DB_DIR,
    RXNORM_DB_DIR,
    SYNTHETIC_EVAL_PATH,
    TOP_K_BM25,
    TOP_K_DENSE,
)
from index.build_vector import build_embeddings  # noqa: E402
from retrieval.bm25_retriever import BM25SRetriever  # noqa: E402
from retrieval.dense_retriever import load_dense_store  # noqa: E402
from retrieval.hybrid import expand_codes, rerank, retrieve_candidates  # noqa: E402
from retrieval.query_variants import QUERY_VARIANTS, apply_variant  # noqa: E402
from retrieval.reranker import Qwen3Reranker  # noqa: E402

logger = logging.getLogger(__name__)

KB_SPECS = {
    "icd": {"chroma_dir": ICD_DB_DIR, "collection": "icd", "bm25_dir": BM25_ICD_DIR},
    "rxnorm": {"chroma_dir": RXNORM_DB_DIR, "collection": "rxnorm", "bm25_dir": BM25_RXNORM_DIR},
}


def build_backends(kbs: list[str]) -> dict[str, dict]:
    embeddings = build_embeddings()
    backends = {}
    for kb in kbs:
        spec = KB_SPECS[kb]
        backends[kb] = {
            "dense": load_dense_store(spec["chroma_dir"], spec["collection"], embeddings),
            "bm25": BM25SRetriever.load(spec["bm25_dir"]),
        }
    return backends


def dump_variant(
    rows: list[dict[str, Any]],
    variant: str,
    backends: dict[str, dict],
    reranker: Qwen3Reranker,
    k_dense: int,
    k_bm25: int,
) -> list[dict[str, Any]]:
    records = []
    for i, row in enumerate(rows, 1):
        kb = row["kb"]
        query = apply_variant(row["mention"], variant)
        docs = retrieve_candidates(
            query, backends[kb]["dense"], backends[kb]["bm25"], k_dense, k_bm25
        )
        scored = rerank(query, docs, reranker)

        candidates = []
        for doc, score in scored:
            candidates.append(
                {
                    "doc_id": doc.metadata["doc_id"],
                    "role": doc.metadata.get("role", "candidate"),
                    "codes": sorted(expand_codes([doc])),
                    "dense_score": doc.metadata.get("dense_score"),
                    "bm25_score": doc.metadata.get("bm25_score"),
                    "rerank_score": score,
                }
            )

        gold = {str(c) for c in row["gold"]}
        union_codes = {c for cand in candidates for c in cand["codes"]}
        records.append(
            {
                "mention": row["mention"],
                "query": query,
                "query_variant": variant,
                "kb": kb,
                "gold": row["gold"],
                "noise": row.get("noise"),
                "gold_kind": row.get("gold_kind"),
                "cluster_name": row.get("cluster_name"),
                "distractor": row.get("distractor") or [],
                "candidates": candidates,
                "recall_union": gold <= union_codes,
            }
        )
        if i % 50 == 0:
            logger.info("[%s] %d/%d", variant, i, len(rows))
    return records


def _branch_hit(records: list[dict], branch: str) -> float:
    """Tỉ lệ dòng mà RIÊNG nhánh này đã phủ hết gold - đo đóng góp thật của từng nhánh."""
    key = "dense_score" if branch == "dense" else "bm25_score"
    hits = 0
    for rec in records:
        codes = {
            c
            for cand in rec["candidates"]
            if cand.get(key) is not None
            for c in cand["codes"]
        }
        if {str(g) for g in rec["gold"]} <= codes:
            hits += 1
    return hits / len(records) if records else 0.0


def report_recall(records: list[dict], variant: str) -> None:
    def rate(subset: list[dict]) -> str:
        if not subset:
            return "-"
        return f"{sum(r['recall_union'] for r in subset) / len(subset):.3f} (n={len(subset)})"

    print_table(
        f"recall@union - biến thể {variant}",
        [("TỔNG", rate(records))]
        + [
            (f"kb={kb}", rate([r for r in records if r["kb"] == kb]))
            for kb in sorted({r["kb"] for r in records})
        ]
        + [
            (f"noise={n}", rate([r for r in records if r["noise"] == n]))
            for n in sorted({str(r["noise"]) for r in records})
        ]
        + [
            (f"gold_kind={g}", rate([r for r in records if r["gold_kind"] == g]))
            for g in sorted({str(r["gold_kind"]) for r in records})
        ],
        ("nhóm", "recall@union"),
    )

    print_table(
        f"recall riêng từng nhánh - biến thể {variant}",
        [
            (kb, f"{_branch_hit(sub, 'dense'):.3f}", f"{_branch_hit(sub, 'bm25'):.3f}", f"{rate(sub)}")
            for kb in sorted({r["kb"] for r in records})
            for sub in [[r for r in records if r["kb"] == kb]]
        ],
        ("kb", "dense-only", "bm25-only", "union"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=SYNTHETIC_EVAL_PATH)
    parser.add_argument("--out-dir", type=Path, default=EVAL_OUTPUT_DIR)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--variants", nargs="+", default=list(QUERY_VARIANTS))
    parser.add_argument("--k-dense", type=int, default=TOP_K_DENSE)
    parser.add_argument("--k-bm25", type=int, default=TOP_K_BM25)
    parser.add_argument("--limit", type=int, default=None, help="chỉ chạy N dòng đầu (dùng để smoke test)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch của reranker")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    rows = list(read_jsonl(args.input))
    if args.limit:
        rows = rows[: args.limit]
    logger.info("nạp %d dòng từ %s", len(rows), args.input)

    backends = build_backends(sorted({r["kb"] for r in rows}))
    reranker = Qwen3Reranker(batch_size=args.batch_size)

    for variant in args.variants:
        records = dump_variant(rows, variant, backends, reranker, args.k_dense, args.k_bm25)
        out = dump_path(variant, args.out_dir)
        write_jsonl(out, records)
        logger.info("đã ghi %d bản ghi vào %s", len(records), out)
        report_recall(records, variant)
        plot_score_distributions(records, variant, args.plots_dir)


if __name__ == "__main__":
    main()
