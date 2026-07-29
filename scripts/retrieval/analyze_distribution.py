"""
Nhìn vào phân bố điểm TRƯỚC khi quét ngưỡng.

Sweep là bài toán tối ưu mù: nó luôn trả về một cặp (floor, margin) tốt nhất, kể
cả khi phân bố gold và non-gold chồng khít lên nhau và cặp đó thực chất vô nghĩa.
Ba biểu đồ ở đây trả lời câu hỏi "có chỗ cho ngưỡng không" trước khi tin vào con
số sweep đưa ra:

  1. gold vs non-gold vs distractor  -> khoảng trống = đất sống của FLOOR.
  2. Dòng multi (multi_sibling / multi_solving): khoảng cách giữa gold hạng 1 và
     gold hạng cuối -> margin PHẢI rộng ít nhất bằng đây mới bắt đủ gold. Đây là
     SÀN của margin.
  3. Dòng có distractor: khoảng cách giữa gold cao nhất và distractor cao nhất
     -> margin vượt quá đây là bắt đầu nuốt hard-negative. Đây là TRẦN của margin.

Nếu sàn (2) > trần (3) thì không tồn tại margin nào đúng cho mọi dòng, và con số
sweep chỉ là một thỏa hiệp - biết trước điều đó khác hẳn với việc phát hiện muộn.
"""

from __future__ import annotations

import argparse
import logging
import statistics
from collections import defaultdict
from pathlib import Path

from common import (
    PLOTS_DIR,
    dedupe_records,
    dump_path,
    label_candidates,
    plot_score_distributions,
    print_table,
    read_jsonl,
    save_fig,
)

from configs.config import EVAL_OUTPUT_DIR  # noqa: E402

logger = logging.getLogger(__name__)

MULTI_KINDS = {"multi_sibling", "multi_solving"}


def _fmt(values: list[float]) -> tuple:
    """(n, min, p25, median, p75, max) - đủ để thấy đuôi mà không cần vẽ."""
    if not values:
        return (0, "-", "-", "-", "-", "-")
    s = sorted(values)

    def q(p: float) -> str:
        return f"{s[min(len(s) - 1, int(p * len(s)))]:.3f}"

    return (len(s), f"{s[0]:.3f}", q(0.25), f"{statistics.median(s):.3f}", q(0.75), f"{s[-1]:.3f}")


def required_margin(rec: dict) -> float | None:
    """Margin nhỏ nhất để tập MÃ mở ra từ các doc sống sót phủ hết gold. None nếu không phủ nổi.

    Đo trên MÃ chứ không phải trên doc, và đây là điểm mấu chốt: một alias RxNorm
    mở thẳng ra nhiều RXCUI, nên một dòng multi_solving có thể chỉ cần đúng doc
    hạng 1 là đủ - margin cần bằng 0. Nếu đo bằng "khoảng cách giữa các doc gold"
    thì rxnorm bị báo là cần margin ~5 điểm trong khi thực tế cần 0, và mọi kết
    luận sàn-vs-trần sau đó đều sai.
    """
    candidates = sorted(rec["candidates"], key=lambda c: c["rerank_score"], reverse=True)
    if not candidates:
        return None
    gold = {str(c) for c in rec["gold"]}
    top = candidates[0]["rerank_score"]
    covered: set[str] = set()
    for cand in candidates:
        covered.update(cand["codes"])
        if gold <= covered:
            return top - cand["rerank_score"]
    return None


def gold_spread(records: list[dict]) -> dict[str, list[float]]:
    """SÀN của margin theo từng KB (chỉ xét dòng multi-gold, nơi margin thực sự phải làm việc)."""
    out: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        if rec.get("gold_kind") not in MULTI_KINDS:
            continue
        need = required_margin(rec)
        if need is not None:
            out[rec["kb"]].append(need)
    return out


def gold_distractor_gap(records: list[dict]) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Gold cao nhất - distractor cao nhất = TRẦN của margin. Trả về (theo kb, theo cluster)."""
    by_kb: dict[str, list[float]] = defaultdict(list)
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        labeled = list(label_candidates(rec))
        gold = [s for label, s in labeled if label == "gold"]
        distractor = [s for label, s in labeled if label == "distractor"]
        if not gold or not distractor:
            continue
        gap = max(gold) - max(distractor)
        by_kb[rec["kb"]].append(gap)
        by_cluster[f"{rec['kb']}/{rec['cluster_name']}"].append(gap)
    return by_kb, by_cluster


def plot_gap_hist(gaps: dict[str, list[float]], title: str, name: str, plots_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = {k: v for k, v in gaps.items() if v}
    if not values:
        logger.warning("%s: không có dữ liệu để vẽ", title)
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for kb, v in sorted(values.items()):
        ax.hist(v, bins=30, alpha=0.6, label=f"{kb} (n={len(v)})")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("chênh lệch điểm reranker")
    ax.set_ylabel("số dòng")
    ax.set_title(title)
    ax.legend()
    logger.info("đã lưu %s", save_fig(fig, name, plots_dir))
    plt.close(fig)


def report_recall_by_group(records: list[dict]) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        groups["TỔNG"].append(rec)
        groups[f"kb={rec['kb']}"].append(rec)
        groups[f"noise={rec['noise']}"].append(rec)
        groups[f"gold_kind={rec['gold_kind']}"].append(rec)
    print_table(
        "recall@union theo nhóm",
        [
            (name, len(sub), f"{sum(r['recall_union'] for r in sub) / len(sub):.3f}")
            for name, sub in sorted(groups.items())
        ],
        ("nhóm", "n", "recall@union"),
    )


def analyze(records: list[dict], variant: str, plots_dir: Path) -> None:
    print(f"\n######## biến thể query: {variant} ({len(records)} dòng) ########")
    report_recall_by_group(records)
    plot_score_distributions(records, variant, plots_dir)

    # (1) điểm gold vs non-gold ở dạng bảng, để đọc ngay không cần mở hình
    by_kb: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        for label, score in label_candidates(rec):
            by_kb[rec["kb"]][label].append(score)
    print_table(
        "phân bố điểm reranker (đất sống của floor nằm giữa gold và other)",
        [
            (kb, label, *_fmt(by_kb[kb][label]))
            for kb in sorted(by_kb)
            for label in ("gold", "distractor", "other")
        ],
        ("kb", "nhãn", "n", "min", "p25", "median", "p75", "max"),
    )

    # (2) sàn của margin
    spread = gold_spread(records)
    print_table(
        "SÀN margin - margin nhỏ nhất để phủ hết gold (dòng multi-gold)",
        [(kb, *_fmt(v)) for kb, v in sorted(spread.items())],
        ("kb", "n", "min", "p25", "median", "p75", "max"),
    )
    plot_gap_hist(spread, f"Sàn margin (multi-gold) - {variant}", f"margin_floor_{variant}.png", plots_dir)

    # (3) trần của margin
    gap_kb, gap_cluster = gold_distractor_gap(records)
    print_table(
        "TRẦN margin - (gold cao nhất - distractor cao nhất); giá trị <= 0 nghĩa là reranker xếp distractor trên gold",
        [(kb, *_fmt(v)) for kb, v in sorted(gap_kb.items())],
        ("kb", "n", "min", "p25", "median", "p75", "max"),
    )
    print_table(
        "TRẦN margin theo cụm - cụm có median thấp/âm là cụm reranker đang lẫn",
        sorted(
            [(cluster, len(v), f"{statistics.median(v):.3f}", f"{min(v):.3f}", sum(1 for x in v if x <= 0))
             for cluster, v in gap_cluster.items()],
            key=lambda r: float(r[2]),
        ),
        ("kb/cụm", "n", "median gap", "min gap", "số dòng gap<=0"),
    )
    plot_gap_hist(gap_kb, f"Trần margin (gold vs distractor) - {variant}", f"margin_ceiling_{variant}.png", plots_dir)

    for kb in sorted(set(spread) | set(gap_kb)):
        if spread.get(kb) and gap_kb.get(kb):
            need = statistics.median(spread[kb])
            allow = statistics.median(gap_kb[kb])
            verdict = "CÓ khoảng an toàn" if need < allow else "XUNG ĐỘT - không margin nào đúng cho mọi dòng"
            print(f"\n[{kb}] sàn(median)={need:.3f}  trần(median)={allow:.3f}  -> {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=["raw", "strip_dose"])
    parser.add_argument("--dump-dir", type=Path, default=EVAL_OUTPUT_DIR)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="giữ nguyên dòng trùng (mention, kb); mặc định bỏ - xem common.dedupe_records",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    for variant in args.variants:
        path = dump_path(variant, args.dump_dir)
        if not path.exists():
            logger.warning("bỏ qua %s - chưa có %s (chạy dump_scores.py trước)", variant, path)
            continue
        records = list(read_jsonl(path))
        if not args.keep_duplicates:
            records = dedupe_records(records)
        analyze(records, variant, args.plots_dir)


if __name__ == "__main__":
    main()
