"""
Hàm mục tiêu của bước hiệu chỉnh ngưỡng: `candidates_score` theo đúng công thức BTC.

Điểm dễ hiểu nhầm nhất: `len(ground_truth) + 1` KHÔNG nằm ở mẫu số của Jaccard.
Jaccard từng candidate là Jaccard thuần `|gt ∩ pred| / |gt ∪ pred|`; `len(gt)+1`
là **trọng số** khi gộp trung bình các candidate lại. Hệ quả thực tế: dòng
multi-gold nặng ký hơn dòng single-gold (3 vs 2), nên sweep sẽ ưu ái margin đủ
rộng để bắt được multi_sibling / multi_solving. Nếu cài nhầm thành mẫu số thì
một dự đoán hoàn hảo cũng chỉ được 0.5 và toàn bộ bề mặt tối ưu bị bóp méo.

Hai trường hợp biên (gt rỗng) được giữ nguyên theo spec dù tập synthetic hiện
tại không có dòng nào gt rỗng - chúng xuất hiện khi nối với NER thật, vì mọi
khái niệm đều là một candidate `k` và loại không mang mã (TRIỆU_CHỨNG, hai loại
xét nghiệm) có `gt = []`.

GIỚI HẠN: `jaccard` KHÔNG tự phạt được mention bịa ra. Một TRIỆU_CHỨNG bịa có
`gt = []` và `pred = []`, rơi vào nhánh rỗng-khớp-rỗng và được 1.0 - tự thưởng
điểm cho một khái niệm không tồn tại trong gold. Muốn phạt thì phải biết cặp đó
có ghép được hay không, mà đó là khái niệm của tầng NER chứ không phải tầng
retrieval. Nên việc ép 0 nằm ở phía gọi (`scripts/ner/run_ner_synthetic.py`,
hàm `_weighted_candidates`), và module này cố ý không biết gì về nó.
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
