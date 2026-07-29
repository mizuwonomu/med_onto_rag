"""
Quét lưới (floor, margin) cho TỪNG KB rồi chốt ngưỡng.

Vì sao mỗi KB một cặp ngưỡng riêng: ICD và RxNorm khác nhau cả về hình dạng bài
toán (mô tả chẩn đoán dài vs tên hoạt chất ngắn) lẫn về cách mã nở ra (alias
RxNorm mở thành nhiều RXCUI). Điểm reranker của hai KB không nằm trên cùng một
thang, nên một cặp ngưỡng chung sẽ luôn hy sinh một bên.

Ba thứ chống tự lừa mình, theo thứ tự quan trọng:

  1. **Chia sweep/dev theo cụm.** Chốt ngưỡng trên sweep, chấm trên dev. Chênh
     lệch sweep-dev lớn = đã overfit vào 300 dòng.
  2. **Bootstrap.** Lấy mẫu lại tập sweep ~300 lần, mỗi lần chọn lại cặp tối ưu.
     Nếu cặp thắng nhảy lung tung giữa các lần thì con số "tối ưu" chỉ là nhiễu
     của cỡ mẫu, không phải tín hiệu - lúc đó nên chọn điểm giữa vùng ổn định
     thay vì đỉnh nhọn.
  3. **Báo cáo rò distractor theo tên cụm.** Tổng thể "3% rò" là con số vô dụng;
     "cụm Loạn sản âm hộ rò 40%" thì hành động được ngay.

Script này KHÔNG đụng GPU - chỉ đọc dump của `dump_scores.py`.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from common import PLOTS_DIR, dedupe_records, dump_path, print_table, read_jsonl, save_fig

from configs.config import EVAL_OUTPUT_DIR  # noqa: E402
from retrieval.metrics import candidates_score  # noqa: E402
from retrieval.reranker import MED_LINKING_INSTRUCTION  # noqa: E402
from retrieval.splits import cluster_aware_split  # noqa: E402

logger = logging.getLogger(__name__)


def predict(rec: dict[str, Any], floor: float, margin: float) -> set[str]:
    """Áp ngưỡng lên điểm DOC rồi mới mở ra mã (xem hybrid.apply_thresholds/expand_codes).

    Lặp lại logic ở dạng thuần dữ liệu vì dump chỉ có dict chứ không có Document -
    nhưng phải giữ đúng thứ tự "lọc doc trước, mở mã sau", nếu không ngưỡng chốt
    ở đây sẽ khác ngưỡng chạy thật.
    """
    candidates = rec["candidates"]
    if not candidates:
        return set()
    top = max(c["rerank_score"] for c in candidates)
    cutoff = max(floor, top - margin)
    return {
        code
        for c in candidates
        if c["rerank_score"] >= cutoff
        for code in c["codes"]
    }


def score_grid(
    records: Sequence[dict], floors: Sequence[float], margins: Sequence[float]
) -> dict[tuple[float, float], float]:
    grid = {}
    for floor in floors:
        for margin in margins:
            grid[(floor, margin)] = candidates_score(
                [(rec["gold"], predict(rec, floor, margin)) for rec in records]
            )
    return grid


def best_pair(grid: dict[tuple[float, float], float]) -> tuple[float, float]:
    """Điểm cao nhất; hòa thì ưu tiên margin nhỏ rồi floor lớn (dự đoán chặt hơn)."""
    return max(grid, key=lambda key: (grid[key], -key[1], key[0]))


def make_axis(records: Sequence[dict], steps: int, quantile_clip: float = 0.99) -> tuple[list[float], list[float]]:
    """Lưới bám vào phân bố điểm thật thay vì hằng số cứng.

    Điểm reranker là logit chưa chuẩn hóa, thang của nó phụ thuộc model và
    instruction. Hardcode floor kiểu 0.5 là giả định về một thang không tồn tại.
    """
    scores = sorted(c["rerank_score"] for rec in records for c in rec["candidates"])
    if not scores:
        raise ValueError("dump rỗng")
    lo, hi = scores[0], scores[min(len(scores) - 1, int(quantile_clip * len(scores)))]
    span = hi - lo or 1.0
    floors = [lo + span * i / (steps - 1) for i in range(steps)]
    margins = [span * i / (steps - 1) for i in range(steps)]
    return floors, margins


def bootstrap_pairs(
    records: Sequence[dict],
    floors: Sequence[float],
    margins: Sequence[float],
    resamples: int,
    seed: int,
) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    n = len(records)
    winners = []
    for _ in range(resamples):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        winners.append(best_pair(score_grid(sample, floors, margins)))
    return winners


def distractor_leak(records: Sequence[dict], floor: float, margin: float) -> list[tuple]:
    """% dòng hard-negative mà distractor lọt vào tập dự đoán, theo tên cụm."""
    by_cluster: dict[str, list[bool]] = defaultdict(list)
    for rec in records:
        distractor = {str(c) for c in rec.get("distractor") or []}
        if not distractor:
            continue
        leaked = bool(predict(rec, floor, margin) & distractor)
        by_cluster[str(rec.get("cluster_name"))].append(leaked)
    return sorted(
        [
            (cluster, len(flags), sum(flags), f"{sum(flags) / len(flags):.2f}")
            for cluster, flags in by_cluster.items()
        ],
        key=lambda row: -float(row[3]),
    )


def plot_heatmap(grid, floors, margins, chosen, kb, variant, plots_dir) -> None:
    """Đỉnh nằm trên cao nguyên hay trên lưỡi dao - quyết định ngưỡng có sống nổi trên dữ liệu thật không."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = [[grid[(f, m)] for m in margins] for f in floors]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(
        matrix, origin="lower", aspect="auto", cmap="viridis",
        extent=[margins[0], margins[-1], floors[0], floors[-1]],
    )
    ax.plot(chosen[1], chosen[0], marker="*", color="red", markersize=16, label=f"chọn {chosen[0]:.2f}/{chosen[1]:.2f}")
    ax.set_xlabel("margin")
    ax.set_ylabel("floor")
    ax.set_title(f"candidates_score - kb={kb}, biến thể={variant}")
    ax.legend(loc="lower right")
    fig.colorbar(im, ax=ax)
    logger.info("đã lưu %s", save_fig(fig, f"sweep_heatmap_{variant}_{kb}.png", plots_dir))
    plt.close(fig)


def plot_bootstrap(winners, chosen, kb, variant, plots_dir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5))
    counts = Counter(winners)
    xs = [m for _, m in counts]
    ys = [f for f, _ in counts]
    sizes = [20 * c for c in counts.values()]
    ax.scatter(xs, ys, s=sizes, alpha=0.5)
    ax.plot(chosen[1], chosen[0], marker="*", color="red", markersize=16, label="cặp chốt")
    ax.set_xlabel("margin")
    ax.set_ylabel("floor")
    ax.set_title(f"Bootstrap: cặp thắng qua {len(winners)} lần lấy mẫu - kb={kb}")
    ax.legend()
    logger.info("đã lưu %s", save_fig(fig, f"sweep_bootstrap_{variant}_{kb}.png", plots_dir))
    plt.close(fig)


def plot_slices(grid, floors, margins, chosen, kb, variant, plots_dir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(floors, [grid[(f, chosen[1])] for f in floors])
    axes[0].axvline(chosen[0], color="red", linestyle="--")
    axes[0].set_xlabel("floor")
    axes[0].set_ylabel("candidates_score")
    axes[0].set_title(f"lát cắt theo floor (margin={chosen[1]:.2f})")

    axes[1].plot(margins, [grid[(chosen[0], m)] for m in margins])
    axes[1].axvline(chosen[1], color="red", linestyle="--")
    axes[1].set_xlabel("margin")
    axes[1].set_title(f"lát cắt theo margin (floor={chosen[0]:.2f})")
    fig.suptitle(f"Độ nhạy 1 chiều - kb={kb}, biến thể={variant}")
    logger.info("đã lưu %s", save_fig(fig, f"sweep_slices_{variant}_{kb}.png", plots_dir))
    plt.close(fig)


def sweep_kb(
    kb: str, records: list[dict], split: dict[str, list[int]], args
) -> dict[str, Any]:
    sweep_rows = [records[i] for i in split["sweep"]]
    dev_rows = [records[i] for i in split["dev"]]
    logger.info("[%s] sweep=%d dòng, dev=%d dòng", kb, len(sweep_rows), len(dev_rows))

    floors, margins = make_axis(records, args.grid_steps)
    grid = score_grid(sweep_rows, floors, margins)
    chosen = best_pair(grid)
    sweep_score = grid[chosen]
    dev_score = candidates_score([(r["gold"], predict(r, *chosen)) for r in dev_rows])

    winners = bootstrap_pairs(sweep_rows, floors, margins, args.bootstrap, args.seed)
    floor_vals = sorted(f for f, _ in winners)
    margin_vals = sorted(m for _, m in winners)

    def spread(values: list[float]) -> str:
        lo = values[int(0.05 * len(values))]
        hi = values[min(len(values) - 1, int(0.95 * len(values)))]
        return f"[{lo:.3f}, {hi:.3f}]"

    print_table(
        f"kết quả sweep - kb={kb}",
        [
            ("floor chốt", f"{chosen[0]:.3f}"),
            ("margin chốt", f"{chosen[1]:.3f}"),
            ("candidates_score (sweep)", f"{sweep_score:.4f}"),
            ("candidates_score (dev)", f"{dev_score:.4f}"),
            ("chênh lệch sweep-dev", f"{sweep_score - dev_score:+.4f}"),
            ("bootstrap floor 5-95%", spread(floor_vals)),
            ("bootstrap margin 5-95%", spread(margin_vals)),
            ("số cặp thắng phân biệt", len(set(winners))),
        ],
        ("chỉ số", "giá trị"),
    )
    if abs(sweep_score - dev_score) > args.gap_warn:
        logger.warning(
            "[%s] chênh lệch sweep-dev %.4f > %.4f - nghi overfit vào tập sweep",
            kb, sweep_score - dev_score, args.gap_warn,
        )

    leak = distractor_leak(records, *chosen)
    if leak:
        print_table(
            f"rò distractor tại ngưỡng đã chốt - kb={kb} (xếp theo mức rò)",
            leak,
            ("cụm", "số dòng", "số dòng rò", "tỉ lệ"),
        )

    plot_heatmap(grid, floors, margins, chosen, kb, args.variant, args.plots_dir)
    plot_bootstrap(winners, chosen, kb, args.variant, args.plots_dir)
    plot_slices(grid, floors, margins, chosen, kb, args.variant, args.plots_dir)

    return {
        "floor": chosen[0],
        "margin": chosen[1],
        "sweep_score": sweep_score,
        "dev_score": dev_score,
        "n_sweep": len(sweep_rows),
        "n_dev": len(dev_rows),
        "bootstrap_floor_p05_p95": [floor_vals[int(0.05 * len(floor_vals))], floor_vals[min(len(floor_vals) - 1, int(0.95 * len(floor_vals)))]],
        "bootstrap_margin_p05_p95": [margin_vals[int(0.05 * len(margin_vals))], margin_vals[min(len(margin_vals) - 1, int(0.95 * len(margin_vals)))]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="raw", help="biến thể query đã chốt TRƯỚC khi sweep")
    parser.add_argument("--dump-dir", type=Path, default=EVAL_OUTPUT_DIR)
    parser.add_argument("--out", type=Path, default=EVAL_OUTPUT_DIR / "thresholds.json")
    parser.add_argument("--split-out", type=Path, default=EVAL_OUTPUT_DIR / "split.json")
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--grid-steps", type=int, default=41)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--dev-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--gap-warn", type=float, default=0.05)
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="giữ nguyên dòng trùng (mention, kb); mặc định bỏ - xem common.dedupe_records",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    records = list(read_jsonl(dump_path(args.variant, args.dump_dir)))
    if not args.keep_duplicates:
        records = dedupe_records(records)
    split = cluster_aware_split(records, dev_ratio=args.dev_ratio, seed=args.seed)
    args.split_out.parent.mkdir(parents=True, exist_ok=True)
    args.split_out.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "dev_ratio": args.dev_ratio,
                "deduped": not args.keep_duplicates,
                "variant": args.variant,
                # ghi mention thay vì chỉ số để split còn tái lập được khi dump đổi thứ tự
                "sweep": [records[i]["mention"] for i in split["sweep"]],
                "dev": [records[i]["mention"] for i in split["dev"]],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("đã ghi split vào %s (sweep=%d, dev=%d)", args.split_out, len(split["sweep"]), len(split["dev"]))

    thresholds: dict[str, Any] = {}
    for kb in sorted({r["kb"] for r in records}):
        kb_records = [r for r in records if r["kb"] == kb]
        index_map = {id(r): i for i, r in enumerate(kb_records)}
        kb_split = {
            name: [index_map[id(records[i])] for i in idx if records[i]["kb"] == kb]
            for name, idx in split.items()
        }
        thresholds[kb] = sweep_kb(kb, kb_records, kb_split, args)

    thresholds["metadata"] = {
        "variant": args.variant,
        "seed": args.seed,
        "date": date.today().isoformat(),
        "grid_steps": args.grid_steps,
        "deduped": not args.keep_duplicates,
        "bootstrap_resamples": args.bootstrap,
        # instruction quyết định thang điểm -> ngưỡng chỉ có nghĩa khi đi kèm nó
        "reranker_instruction": MED_LINKING_INSTRUCTION,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("đã ghi ngưỡng vào %s", args.out)


if __name__ == "__main__":
    main()
