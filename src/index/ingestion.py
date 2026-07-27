"""
Ghi Document vào một Chroma collection theo batch, không phụ thuộc embedding model cụ thể
"""

import logging
import shutil
from pathlib import Path

from chromadb.api.client import SharedSystemClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from configs.config import CHROMA_ADD_BATCH

logger = logging.getLogger(__name__)


def ingest_collection(
    docs: list[Document],
    embeddings: Embeddings,
    persist_dir: Path,
    collection_name: str,
) -> int:
    """Ghi đè persist_dir (rebuild one-shot) và nạp docs theo batch vào Chroma, trả về số doc đã nạp"""
    if persist_dir.exists():
        logger.info("ingest_collection[%s]: wiping existing %s", collection_name, persist_dir)
        shutil.rmtree(persist_dir)
        # chromadb cache client theo persist_directory trong process; xóa cache để
        # tránh dùng lại connection cũ trỏ vào file sqlite vừa bị rmtree
        SharedSystemClient.clear_system_cache()
    persist_dir.mkdir(parents=True, exist_ok=True)

    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(docs), CHROMA_ADD_BATCH):
        batch = docs[start : start + CHROMA_ADD_BATCH]
        ids = [doc.metadata["doc_id"] for doc in batch]
        store.add_documents(batch, ids=ids)
        logger.info(
            "ingest_collection[%s]: added %d/%d",
            collection_name,
            min(start + CHROMA_ADD_BATCH, len(docs)),
            len(docs),
        )

    count = store._collection.count()
    if count != len(docs):
        raise RuntimeError(
            f"ingest_collection[{collection_name}]: expected {len(docs)} docs, collection has {count}"
        )
    return count
