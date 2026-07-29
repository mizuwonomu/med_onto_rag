"""
Nhánh dày (dense) của hybrid retrieval: mở Chroma index CÓ SẴN để đọc.

Cố ý KHÔNG đi qua `index.ingestion.ingest_collection` - hàm đó `rmtree` persist
dir trước khi ghi. Một lần gọi nhầm từ phía retrieval là mất toàn bộ index đã
encode. Ở đây chỉ mở store ở chế độ đọc.

Embedding model lấy nguyên từ `index.build_vector.build_embeddings` để giữ
đúng hợp đồng bất đối xứng (query có `prompt_name="query"`, document thì không).
Tự dựng lại HuggingFaceEmbeddings ở đây là cách chắc chắn nhất để lệch prompt
mà không ai phát hiện.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


def load_dense_store(
    persist_dir: Path, collection_name: str, embeddings: Embeddings
) -> Chroma:
    """Mở Chroma collection đã build. Lỗi ngay nếu index trống/không tồn tại."""
    persist_dir = Path(persist_dir)
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"không có Chroma index tại {persist_dir} - chạy `python -m index.build_vector` trước"
        )
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    count = store._collection.count()
    if count == 0:
        raise RuntimeError(f"Chroma collection {collection_name!r} tại {persist_dir} rỗng")
    logger.info("dense[%s]: %d document", collection_name, count)
    return store


def dense_search(store: Chroma, query: str, k: int) -> list[Document]:
    """Trả Document kèm metadata['dense_score'] (relevance score, càng cao càng gần)."""
    hits = store.similarity_search_with_relevance_scores(query, k=k)
    return [
        Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "dense_score": float(score)},
        )
        for doc, score in hits
    ]
