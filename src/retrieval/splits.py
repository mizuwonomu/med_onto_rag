"""
Chia tập đo thành sweep/dev theo CỤM, có phân tầng.

Hai ràng buộc đánh nhau, và thứ tự ưu tiên giữa chúng quan trọng:

  * **Phân tầng theo (noise, gold_kind)** để dev không vô tình toàn dòng dễ.
  * **Toàn bộ dòng cùng `cluster_name` phải nằm cùng một phía.** Một cụm
    hard-negative được sinh ra như một khối: distractor của dòng này chính là
    gold của dòng kia. Xé cụm ra hai phía nghĩa là lúc tune ngưỡng ta đã nhìn
    thấy đúng cấu trúc gold/distractor mà lát nữa dùng để chấm - dev khi đó báo
    điểm cao giả.

Ràng buộc cụm thắng: cụm là đơn vị chia không thể tách, phân tầng chỉ được áp ở
mức đơn vị đó (tầng của một cụm = tầng chiếm đa số trong cụm). Hệ quả là tỉ lệ
300/100 chỉ đạt xấp xỉ chứ không chính xác tuyệt đối - đó là cái giá đúng để trả.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Sequence

SplitName = str


def _unit_key(row: dict[str, Any]) -> str:
    """Đơn vị chia: cụm nếu có, còn lại thì mỗi dòng là một đơn vị riêng."""
    cluster = row.get("cluster_name") #cụm distractor, tức chia ra tên các chương chứa các mã nhiễu
    if cluster:
        return f"cluster::{row.get('kb')}::{cluster}" #trong một cụm, distractor của dòng này là gold của dòng kia
    return f"row::{row.get('kb')}::{row.get('mention')}"


def _stratum(rows: Sequence[dict[str, Any]]) -> tuple:
    counts = Counter((r.get("noise"), r.get("gold_kind")) for r in rows)
    return counts.most_common(1)[0][0] #lấy tầng có cụm chiếm đa số trong 6 tầng (mild, single), (heavy, multi_sibling),..


def cluster_aware_split(
    rows: Sequence[dict[str, Any]], dev_ratio: float = 0.25, seed: int = 20260728
) -> dict[SplitName, list[int]]:
    """Trả về {'sweep': [chỉ số dòng], 'dev': [...]} - chỉ số theo `rows` truyền vào."""
    units: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        units[_unit_key(row)].append(i)

    by_stratum: dict[tuple, list[list[int]]] = defaultdict(list)
    for indices in units.values():
        by_stratum[_stratum([rows[i] for i in indices])].append(indices)

    rng = random.Random(seed) #trộn bằng rng có seed
    split: dict[SplitName, list[int]] = {"sweep": [], "dev": []} 
    for stratum in sorted(by_stratum, key=str):
        unit_list = sorted(by_stratum[stratum], key=lambda idx: idx[0]) #sắp đơn vị theo chỉ số dòng đầu
        rng.shuffle(unit_list)
        target_dev = round(dev_ratio * sum(len(u) for u in unit_list)) #dev = 25% số dòng của tầng
        dev_count = 0
        for unit in unit_list:
            # Nhận cụm vào dev chừng nào chưa vượt hạn mức; cụm to có thể làm lố
            # một chút - chấp nhận, vì xé cụm còn tệ hơn nhiều.
            if dev_count < target_dev:
                split["dev"].extend(unit)
                dev_count += len(unit)
            else:
                split["sweep"].extend(unit)

    split["sweep"].sort()
    split["dev"].sort()
    return split
