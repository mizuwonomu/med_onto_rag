"""
Pipeline mỗi mention: union(dense, bm25) -> rerank -> lọc ngưỡng -> mở ra tập mã.

**Vì sao union chứ không RRF/ensemble.** Reranker chấm lại TOÀN BỘ union, rồi
floor/margin làm việc trên điểm reranker. Mọi thứ tự do fusion tạo ra đều bị vứt
đi ở bước sau. Thêm một tầng RRF chỉ thêm hai siêu tham số phải tune mà không
đổi được tập ứng viên - union đã là tập đó rồi. Hai nhánh chỉ có nhiệm vụ duy
nhất: đảm bảo recall@union.

**Vì sao ngưỡng đặt trên doc chứ không trên mã.** Một alias RxNorm giải về nhiều
RXCUI; nếu chấm điểm ở mức mã thì cùng một điểm doc bị nhân bản ra nhiều mã và
margin mất ý nghĩa. Nên: lọc doc trước, mở ra mã sau, hợp lại thành tập dự đoán.

LangSmith chỉ để debug định tính, bật bằng biến môi trường `LANGSMITH_TRACING`.
`@traceable` tự thành no-op khi biến đó không bật nên không cần rẽ nhánh tay.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.documents import Document

try:  # langsmith là dep phụ trợ - không được phép làm hỏng pipeline nếu thiếu
    from langsmith import traceable
except ImportError:  # pragma: no cover

    def traceable(*args: Any, **kwargs: Any):
        def decorator(func):
            return func

        return decorator(args[0]) if args and callable(args[0]) else decorator


logger = logging.getLogger(__name__)


@traceable(name="retrieve_candidates")
def retrieve_candidates(
    mention: str,
    dense_store,
    bm25_retriever,
    k_dense: int,
    k_bm25: int,
) -> list[Document]:
    """Hợp nhất theo doc_id. Doc trúng cả hai nhánh giữ đủ cả dense_score và bm25_score."""
    from retrieval.dense_retriever import dense_search

    merged: dict[str, Document] = {}
    for doc in dense_search(dense_store, mention, k_dense) + bm25_retriever.retrieve(
        mention, k=k_bm25
    ):
        doc_id = doc.metadata["doc_id"]
        if doc_id in merged:
            merged[doc_id].metadata.update(doc.metadata)
        else:
            merged[doc_id] = Document(
                page_content=doc.page_content, metadata=dict(doc.metadata)
            )
    return list(merged.values())


@traceable(name="rerank")
def rerank(mention: str, docs: list[Document], reranker) -> list[tuple[Document, float]]:
    """Trả (doc, điểm) sắp giảm dần."""
    scores = reranker.score(mention, docs)
    return sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)


def apply_thresholds(
    scored: list[tuple[Document, float]], floor: float, margin: float
) -> list[Document]:
    """Giữ doc có điểm >= floor VÀ >= top1 - margin.

    floor = ngưỡng tuyệt đối (mention không có đáp án nào trong KB -> trả rỗng).
    margin = ngưỡng tương đối (giữ các doc bám sát top-1 -> bắt được multi-gold).
    Thiếu floor thì mọi mention đều trả ít nhất 1 mã; thiếu margin thì mọi mention
    chỉ trả đúng 1 mã. Cần cả hai.
    """
    if not scored:
        return []
    top_score = max(score for _, score in scored)
    cutoff = max(floor, top_score - margin)
    return [doc for doc, score in scored if score >= cutoff]


def expand_codes(docs: list[Document]) -> set[str]:
    """doc -> tập mã: candidate lấy `code`; alias lấy toàn bộ `resolve_in`.

    `resolve_in` được Chroma lưu dưới dạng chuỗi JSON (metadata chỉ nhận scalar),
    nên LUÔN `json.loads` - so khớp chuỗi con trên chuỗi JSON là bug im lặng
    (RXCUI "723" khớp cả bên trong "17230").
    """
    codes: set[str] = set()
    for doc in docs:
        role = doc.metadata.get("role")
        if role == "alias":
            resolve_in = doc.metadata.get("resolve_in")
            if resolve_in:
                codes.update(str(c) for c in json.loads(resolve_in))
        else:
            code = doc.metadata.get("code")
            if code:
                codes.add(str(code))
    return codes
