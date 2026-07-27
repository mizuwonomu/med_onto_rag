"""
Entrypoint: build embedding model bất đối xứng, load JSONL, nạp 2 Chroma vector index (icd_db, rxnorm_db)
"""

import logging

import numpy as np
import torch
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from configs.config import (
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    ICD_DB_DIR,
    ICD_JSONL_PATH,
    RXNORM_ALIASES_PATH,
    RXNORM_CONCEPTS_PATH,
    RXNORM_DB_DIR,
)
from index.ingestion import ingest_collection
from index.load_documents import load_icd_documents, load_rxnorm_documents

logger = logging.getLogger(__name__)


def build_embeddings() -> HuggingFaceEmbeddings:
    """Khởi tạo HuggingFaceEmbeddings bất đối xứng (query dùng prompt, document thì không)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs: dict = {"device": device}
    if device == "cuda":
        model_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True, "batch_size": EMBED_BATCH_SIZE},
        query_encode_kwargs={"prompt_name": "query", "normalize_embeddings": True},
    )


def sanity_check_asymmetry(emb: Embeddings) -> None:
    """Kiểm tra query embedding khác document embedding, và vector đã chuẩn hóa norm ~ 1.0"""
    sample_text = "viêm phổi"
    query_vec = np.array(emb.embed_query(sample_text))
    doc_vec = np.array(emb.embed_documents([sample_text])[0])

    if np.allclose(query_vec, doc_vec):
        raise RuntimeError(
            "query embedding trùng document embedding - prompt bất đối xứng không được áp dụng"
        )

    for name, vec in (("query", query_vec), ("document", doc_vec)):
        norm = np.linalg.norm(vec)
        if not np.isclose(norm, 1.0, atol=1e-2):
            raise RuntimeError(f"{name} embedding norm={norm:.4f} không xấp xỉ 1.0")

    logger.info("sanity_check_asymmetry: OK (query != document, norms ~ 1.0)")


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    embeddings = build_embeddings()
    sanity_check_asymmetry(embeddings)

    icd_docs = load_icd_documents(ICD_JSONL_PATH)
    icd_count = ingest_collection(icd_docs, embeddings, ICD_DB_DIR, "icd")
    logger.info("icd: ingested %d documents into %s", icd_count, ICD_DB_DIR)

    rxnorm_docs = load_rxnorm_documents(RXNORM_CONCEPTS_PATH, RXNORM_ALIASES_PATH)
    rxnorm_count = ingest_collection(rxnorm_docs, embeddings, RXNORM_DB_DIR, "rxnorm")
    logger.info("rxnorm: ingested %d documents into %s", rxnorm_count, RXNORM_DB_DIR)


if __name__ == "__main__":
    main()
