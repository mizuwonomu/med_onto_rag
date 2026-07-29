"""
Hai biến thể query đưa vào retrieval, để đo xem có cần tầng chuẩn hóa query không.

Mention lâm sàng tiếng Việt thường dính liều/đường dùng/tần suất
("uống viên augmentin 625mg", "paracetamol 500mg po bid"). Những token này KHÔNG
có trong text của KB (RxNorm ở mức ingredient), nên với BM25 chúng là token rác
làm loãng idf, còn với dense/reranker chúng có thể kéo lệch biểu diễn.

Nhưng cắt liều cũng có giá: "insulin 70/30" hay các dạng bào chế mà liều là một
phần định danh sẽ mất thông tin. Vì vậy không quyết định bằng cảm tính - dump cả
hai biến thể, so recall@union và phân bố điểm, rồi mới chọn. Ngưỡng floor/margin
phụ thuộc biến thể, nên phải chốt biến thể TRƯỚC khi sweep.
"""

from __future__ import annotations

import re

from index.load_documents import normalize_text

_DOSE_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mg/ml|mcg/ml|g/ml|mg|mcg|µg|ug|g|ml|l|iu|ui|%)\b",
    re.IGNORECASE,
)
# tỉ lệ phối hợp: "875/125", "70/30"
_RATIO_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*/\s*\d+(?:[.,]\d+)?\b")
_ROUTE_RE = re.compile(r"\b(?:po|iv|im|sc|sl|pr|td|inh)\b", re.IGNORECASE)
_FREQ_RE = re.compile(
    r"\b(?:bid|tid|qid|qd|od|daily|qam|qpm|qhs|q\d+h|prn|x\s*\d+\s*(?:lần|viên|ngày))\b",
    re.IGNORECASE,
)

QUERY_VARIANTS = ("raw", "strip_dose")


def raw(mention: str) -> str:
    return normalize_text(mention)


def strip_dose(mention: str) -> str:
    """Bỏ liều / tỉ lệ / đường dùng / tần suất, giữ nguyên phần còn lại."""
    text = normalize_text(mention)
    for pattern in (_DOSE_RE, _RATIO_RE, _ROUTE_RE, _FREQ_RE):
        text = pattern.sub(" ", text)
    text = normalize_text(text)
    # Cắt sạch thành chuỗi rỗng thì thà giữ nguyên bản gốc còn hơn query rỗng.
    return text or normalize_text(mention)


VARIANT_FUNCS = {"raw": raw, "strip_dose": strip_dose}


def apply_variant(mention: str, variant: str) -> str:
    try:
        return VARIANT_FUNCS[variant](mention)
    except KeyError:
        raise ValueError(f"biến thể query không hợp lệ: {variant!r}") from None
