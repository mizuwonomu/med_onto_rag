"""Tiện ích dùng chung cho các script đo đạc retrieval: đường dẫn, đọc dump, lưu hình."""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from configs.config import EVAL_OUTPUT_DIR  # noqa: E402

PLOTS_DIR = EVAL_OUTPUT_DIR / "plots"


def dump_path(variant: str, out_dir: Path = EVAL_OUTPUT_DIR) -> Path:
    return Path(out_dir) / f"scores_{variant}.jsonl"


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_fig(fig, name: str, plots_dir: Path = PLOTS_DIR) -> Path:
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path


def dedupe_records(records: list[dict]) -> list[dict]:
    """Bỏ bản ghi trùng (mention, kb), giữ bản đầu tiên.

    Tập synthetic v3-labeled có 94 dòng lặp lại y hệt. Để nguyên thì hỏng hai
    chỗ: (a) mọi con số recall và candidates_score bị đánh trọng số gấp đôi ở
    đúng những dòng ngẫu nhiên bị lặp, (b) chia sweep/dev có thể ném bản gốc
    sang một phía và bản sao sang phía kia - dev khi đó chấm trên dòng mà ngưỡng
    đã được tune trực tiếp trên nó.

    Dedupe ở tầng đọc dump chứ không phải tầng dữ liệu nguồn: hai dòng giống hệt
    cho ra cùng một kết quả retrieval, nên bỏ bản sao sau khi chấm điểm cho kết
    quả y hệt bỏ trước, mà không tốn thêm một giây GPU nào.
    """
    seen: set[tuple] = set()
    kept = []
    for rec in records:
        key = (rec.get("mention"), rec.get("kb"))
        if key in seen:
            continue
        seen.add(key)
        kept.append(rec)
    dropped = len(records) - len(kept)
    if dropped:
        logging.getLogger(__name__).info(
            "dedupe: bỏ %d bản ghi trùng (mention, kb), còn %d", dropped, len(kept)
        )
    return kept


def label_candidates(rec: dict) -> Iterator[tuple[str, float]]:
    """Gán nhãn từng candidate của một bản ghi dump: gold / distractor / other.

    Một candidate được coi là gold nếu tập mã nó mở ra GIAO với gold - vì alias
    RxNorm mở ra nhiều RXCUI, đòi hỏi khớp trọn vẹn sẽ đánh rớt oan doc đúng.
    """
    gold = {str(c) for c in rec["gold"]}
    distractor = {str(c) for c in rec.get("distractor") or []}
    for cand in rec["candidates"]:
        codes = set(cand["codes"])
        if codes & gold:
            label = "gold"
        elif codes & distractor:
            label = "distractor"
        else:
            label = "other"
        yield label, cand["rerank_score"]


def plot_score_distributions(records: list[dict], variant: str, plots_dir: Path) -> list[Path]:
    """Histogram điểm reranker: gold vs non-gold vs distractor, mỗi KB một hình.

    Khoảng trống giữa hai phân bố chính là chỗ để đặt floor. Nếu chúng chồng lên
    nhau hoàn toàn thì không ngưỡng nào cứu được - vấn đề nằm ở reranker/
    instruction chứ không phải ở việc tune.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_kb: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"gold": [], "non-gold": [], "distractor": []}
    )
    for rec in records:
        buckets = by_kb[rec["kb"]]
        for label, score in label_candidates(rec):
            if label == "gold":
                buckets["gold"].append(score)
            else:
                buckets["non-gold"].append(score)
                if label == "distractor":
                    buckets["distractor"].append(score)

    paths = []
    for kb, buckets in sorted(by_kb.items()):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for label, values in buckets.items():
            if values:
                ax.hist(values, bins=50, alpha=0.55, density=True, label=f"{label} (n={len(values)})")
        ax.set_xlabel("điểm reranker")
        ax.set_ylabel("mật độ")
        ax.set_title(f"Phân bố điểm reranker - kb={kb}, biến thể={variant}")
        ax.legend()
        paths.append(save_fig(fig, f"scores_{variant}_{kb}.png", plots_dir))
        plt.close(fig)
    logging.getLogger(__name__).info("đã lưu %d hình phân bố điểm", len(paths))
    return paths


def print_table(title: str, rows: list[tuple], headers: tuple) -> None:
    """Bảng text - bản dự phòng luôn đọc được ngay trên terminal, không cần mở hình."""
    print(f"\n== {title}")
    widths = [
        max(len(str(headers[i])), *(len(str(r[i])) for r in rows)) if rows else len(str(headers[i]))
        for i in range(len(headers))
    ]
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
