"""
Định tuyến theo loại thực thể -> tầng retrieval -> tập mã ứng viên.

ĐỊNH TUYẾN BẰNG TYPE, KHÔNG PHẢI BẰNG HÌNH DẠNG MÃ
--------------------------------------------------
Chỉ CHẨN_ĐOÁN đi vào ICD và THUỐC đi vào RxNorm
`check_code_shape` vì thế KHÔNG phải bộ định tuyến dự phòng, mà là assertion
phòng thủ: mã ICD bắt đầu bằng chữ cái, RXCUI toàn chữ số. Sai hình dạng nghĩa
là dây bị nối chéo (nạp nhầm chroma dir, nhầm collection) - một lỗi cấu hình sẽ
trôi qua mọi bài test nhưng làm hỏng toàn bộ kết quả. Ném luôn còn hơn ghi ra
một file dump trông bình thường.

NGƯỠNG LẤY TỪ ĐÂU
-----------------
Đây là NƠI ĐẦU TIÊN trong repo thực sự đọc `RETRIEVAL_THRESHOLDS`. Ngưỡng được
lưu kèm biến thể query mà nó được đo dưới đó (icd: raw, rxnorm: strip_dose), nên
`link_entity` luôn lấy cả hai từ cùng một dict - dùng floor của KB này với biến
thể query của KB kia là so điểm trên hai thang khác nhau.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from configs.config import (
    BM25_ICD_DIR,
    BM25_RXNORM_DIR,
    ICD_DB_DIR,
    RETRIEVAL_THRESHOLDS,
    RXNORM_DB_DIR,
    TOP_K_BM25,
    TOP_K_DENSE,
)

logger = logging.getLogger(__name__)

TYPE_TO_KB = {"CHẨN_ĐOÁN": "icd", "THUỐC": "rxnorm"}

KB_SPECS = {
    "icd": {"chroma_dir": ICD_DB_DIR, "collection": "icd", "bm25_dir": BM25_ICD_DIR},
    "rxnorm": {
        "chroma_dir": RXNORM_DB_DIR,
        "collection": "rxnorm",
        "bm25_dir": BM25_RXNORM_DIR,
    },
}

CODE_SHAPE = {"icd": re.compile(r"^[A-Z]\d"), "rxnorm": re.compile(r"^\d+$")}


def build_backends(kbs: tuple[str, ...] = ("icd", "rxnorm")) -> dict[str, Any]:
    """Nạp store + reranker MỘT LẦN, dùng lại cho mọi document.

    Nạp lại theo từng document là nạp lại vài GB trọng số mỗi lần - chi phí đó
    lớn hơn toàn bộ phần còn lại của pipeline cộng lại.
    """
    from index.build_vector import build_embeddings
    from retrieval.bm25_retriever import BM25SRetriever
    from retrieval.dense_retriever import load_dense_store
    from retrieval.reranker import Qwen3Reranker

    # BẮT BUỘC qua build_embeddings: nó giữ tính bất đối xứng query/document
    # (query có prompt_name="query"). Tự dựng HuggingFaceEmbeddings bằng tay là
    # mất vế đó -> query được mã hoá như document, lệch khỏi toàn bộ index đã ghi.
    embeddings = build_embeddings()

    backends: dict[str, Any] = {"reranker": Qwen3Reranker()}
    for kb in kbs:
        spec = KB_SPECS[kb]
        backends[kb] = {
            "dense": load_dense_store(spec["chroma_dir"], spec["collection"], embeddings),
            "bm25": BM25SRetriever.load(spec["bm25_dir"]),
        }
    return backends


def check_code_shape(codes: set[str], kb: str) -> None:
    pattern = CODE_SHAPE[kb]
    if bad := sorted(c for c in codes if not pattern.match(c)):
        raise ValueError(
            f"mã trả về không đúng hình dạng của KB {kb!r}: {bad} - "
            "khả năng cao là nối nhầm store/collection giữa hai KB"
        )


def link_entity(
    text: str,
    ent_type: str,
    backends: dict[str, Any],
    k_dense: int = TOP_K_DENSE,
    k_bm25: int = TOP_K_BM25,
) -> set[str] | None:
    """Trả tập mã, hoặc None nếu loại thực thể này không thuộc KB nào.

    None (không chạm retrieval) khác hẳn set() (có hỏi, ngưỡng cắt sạch): tầng
    trên dùng đúng sự khác biệt đó để quyết định có gắn khoá `candidates` hay
    không - xem `pipeline.extract_document`.
    """
    kb = TYPE_TO_KB.get(ent_type)
    if kb is None:
        return None

    from retrieval.hybrid import apply_thresholds, expand_codes, rerank, retrieve_candidates
    from retrieval.query_variants import apply_variant

    th = RETRIEVAL_THRESHOLDS[kb]
    query = apply_variant(text, th["query_variant"])

    docs = retrieve_candidates(
        query, backends[kb]["dense"], backends[kb]["bm25"], k_dense, k_bm25
    )
    scored = rerank(query, docs, backends["reranker"])
    kept = apply_thresholds(scored, th["floor"], th["margin"])
    codes = expand_codes(kept)

    check_code_shape(codes, kb)
    return codes
