"""
Nhánh thưa (lexical) của hybrid retrieval: bọc thư viện `bm25s` vào interface
`BaseRetriever` của LangChain
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import bm25s
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr

from index.load_documents import normalize_text

logger = logging.getLogger(__name__)

SIDECAR_NAME = "documents.jsonl"


def tokenize(text: str) -> list[str]:
    """NFC + gộp khoảng trắng (dùng lại normalize_text) -> lowercase -> tách theo khoảng trắng."""
    return normalize_text(text).lower().split()


class BM25SRetriever(BaseRetriever):
    """Retriever BM25 trên corpus KB, trả Document kèm metadata['bm25_score']."""

    k: int = 30

    _bm25: bm25s.BM25 = PrivateAttr()
    _documents: list[Document] = PrivateAttr()

    def __init__(self, bm25: bm25s.BM25, documents: list[Document], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._bm25 = bm25
        self._documents = documents

    @classmethod
    def from_documents(cls, documents: list[Document], **kwargs: Any) -> "BM25SRetriever":
        corpus = [tokenize(doc.page_content) for doc in documents]
        bm25 = bm25s.BM25()
        bm25.index(corpus, show_progress=False)
        logger.info("BM25: đã index %d document", len(documents))
        return cls(bm25=bm25, documents=list(documents), **kwargs)

    @property
    def documents(self) -> list[Document]:
        return self._documents

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        return self.retrieve(query)

    def retrieve(self, query: str, k: int | None = None) -> list[Document]:
        """Như `invoke` nhưng cho phép ghi đè k theo từng lần gọi (hybrid cần điều này)."""
        tokens = tokenize(query)
        if not tokens:
            return []

        top_k = min(k or self.k, len(self._documents))
        indices, scores = self._bm25.retrieve([tokens], k=top_k, show_progress=False)

        results: list[Document] = []
        for idx, score in zip(indices[0], scores[0]):
            # bm25s đệm kết quả bằng score 0 khi không đủ tài liệu khớp; những
            # doc này không chia sẻ token nào với query nên chỉ là rác cho reranker.
            if score <= 0:
                continue
            source = self._documents[int(idx)]
            results.append(
                Document(
                    page_content=source.page_content,
                    metadata={**source.metadata, "bm25_score": float(score)},
                )
            )
        return results

    def save(self, directory: Path) -> None:
        """Lưu index bm25s + sidecar JSONL (doc_id/text/metadata) để load lại không cần đọc KB."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._bm25.save(str(directory))
        with (directory / SIDECAR_NAME).open("w", encoding="utf-8") as f:
            for doc in self._documents:
                f.write(
                    json.dumps(
                        {"page_content": doc.page_content, "metadata": doc.metadata},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        logger.info("BM25: đã lưu %d document vào %s", len(self._documents), directory)

    @classmethod
    def load(cls, directory: Path, **kwargs: Any) -> "BM25SRetriever":
        directory = Path(directory)
        sidecar = directory / SIDECAR_NAME
        if not sidecar.exists():
            raise FileNotFoundError(
                f"thiếu {sidecar} - chạy scripts/retrieval/build_bm25.py trước"
            )
        bm25 = bm25s.BM25.load(str(directory))
        documents = []
        with sidecar.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    documents.append(
                        Document(page_content=rec["page_content"], metadata=rec["metadata"])
                    )
        logger.info("BM25: đã nạp %d document từ %s", len(documents), directory)
        return cls(bm25=bm25, documents=documents, **kwargs)
