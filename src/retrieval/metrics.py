"""
Hàm mục tiêu của bước hiệu chỉnh ngưỡng: `candidates_score` theo đúng công thức BTC.

Điểm dễ hiểu nhầm nhất: `len(ground_truth) + 1` KHÔNG nằm ở mẫu số của Jaccard.
Jaccard từng candidate là Jaccard thuần `|gt ∩ pred| / |gt ∪ pred|`; `len(gt)+1`
là **trọng số** khi gộp trung bình các candidate lại. Hệ quả thực tế: dòng
multi-gold nặng ký hơn dòng single-gold (3 vs 2), nên sweep sẽ ưu ái margin đủ
rộng để bắt được multi_sibling / multi_solving. Nếu cài nhầm thành mẫu số thì
một dự đoán hoàn hảo cũng chỉ được 0.5 và toàn bộ bề mặt tối ưu bị bóp méo.

Hai trường hợp biên (gt rỗng) được giữ nguyên theo spec dù tập synthetic hiện
tại không có dòng nào gt rỗng - chúng sẽ xuất hiện khi nối với NER thật, lúc đó
một mention bịa ra phải bị phạt 0 chứ không phải bỏ qua.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def jaccard(gt: Iterable[str], pred: Iterable[str]) -> float:
    gt_set, pred_set = set(gt), set(pred)
    if not gt_set:
        return 1.0 if not pred_set else 0.0
    union = gt_set | pred_set
    return len(gt_set & pred_set) / len(union)


def candidate_weight(gt: Iterable[str]) -> int:
    return len(set(gt)) + 1


def candidates_score(pairs: Sequence[tuple[Iterable[str], Iterable[str]]]) -> float:
    """Trung bình Jaccard có trọng số (len(gt)+1) trên toàn bộ candidate."""
    total_weight = 0
    total = 0.0
    for gt, pred in pairs:
        weight = candidate_weight(gt)
        total += jaccard(gt, pred) * weight
        total_weight += weight
    return total / total_weight if total_weight else 0.0
