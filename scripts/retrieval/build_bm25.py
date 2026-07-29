"""
Entrypoint: dựng và lưu hai index BM25 (icd, rxnorm) từ các file JSONL đã xử lý.

Chạy lại bất cứ lúc nào KB đổi. Rẻ (không cần GPU, vài giây) nên không có chế
độ incremental - luôn dựng lại toàn bộ, giống `build_vector`.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from configs.config import (  # noqa: E402
    BM25_ICD_DIR,
    BM25_RXNORM_DIR,
    ICD_JSONL_PATH,
    RXNORM_ALIASES_PATH,
    RXNORM_CONCEPTS_PATH,
)
from index.load_documents import load_icd_documents, load_rxnorm_documents  # noqa: E402
from retrieval.bm25_retriever import BM25SRetriever  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--icd-out", type=Path, default=BM25_ICD_DIR)
    parser.add_argument("--rxnorm-out", type=Path, default=BM25_RXNORM_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    icd_docs = load_icd_documents(ICD_JSONL_PATH)
    BM25SRetriever.from_documents(icd_docs).save(args.icd_out)

    rxnorm_docs = load_rxnorm_documents(RXNORM_CONCEPTS_PATH, RXNORM_ALIASES_PATH)
    BM25SRetriever.from_documents(rxnorm_docs).save(args.rxnorm_out)

    logger.info("xong: icd=%d doc, rxnorm=%d doc", len(icd_docs), len(rxnorm_docs))


if __name__ == "__main__":
    main()
