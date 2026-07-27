"""
Đọc các file JSONL đã xử lý của KB và chuyển thành LangChain Document để nạp vào Chroma
"""

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterator

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Chuẩn hóa Unicode NFC, cắt khoảng trắng đầu/cuối, gộp khoảng trắng liên tiếp"""
    s = unicodedata.normalize("NFC", s)
    return _WHITESPACE_RE.sub(" ", s.strip())


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _clean_metadata(d: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in d.items():
        if value is None or value == "":
            continue
        assert isinstance(value, (str, int, float, bool)), (
            f"metadata field {key!r} must be scalar, got {type(value)}"
        )
        cleaned[key] = value
    return cleaned


def _dedupe_by_doc_id(docs: list[Document], kb_name: str) -> list[Document]:
    seen: set[str] = set()
    deduped: list[Document] = []
    duplicate_count = 0
    for doc in docs:
        doc_id = doc.metadata["doc_id"]
        if doc_id in seen:
            duplicate_count += 1
            continue
        seen.add(doc_id)
        deduped.append(doc)
    if duplicate_count:
        logger.warning(
            "%s: dropped %d duplicate doc_ids (kept first occurrence)",
            kb_name,
            duplicate_count,
        )
    return deduped


def load_icd_documents(path: Path) -> list[Document]:
    """Đọc icd_concepts.jsonl, trả về danh sách Document cho collection icd"""
    docs: list[Document] = []
    skipped = 0
    for rec in _read_jsonl(path):
        doc_id = rec.get("doc_id")
        text = rec.get("text")
        if not doc_id or not text:
            skipped += 1
            continue
        metadata = _clean_metadata(
            {
                "doc_id": doc_id,
                "code": rec.get("code"),
                "kb": rec.get("kb") or "icd",
                "role": rec.get("role") or "candidate",
                "name_en": rec.get("name_en"),
                "note_vi": rec.get("note_vi"),
                "note_en": rec.get("note_en"),
                "chapter": rec.get("chapter"),
                "block": rec.get("block"),
                "vi_glyph_ok": rec.get("vi_glyph_ok"),
            }
        )
        docs.append(Document(page_content=normalize_text(text), metadata=metadata))

    if skipped:
        logger.warning("icd: skipped %d records missing doc_id/text", skipped)

    return _dedupe_by_doc_id(docs, "icd")


def load_rxnorm_documents(concepts_path: Path, aliases_path: Path) -> list[Document]:
    """Đọc rxnorm_concepts.jsonl và rxnorm_aliases.jsonl, trả về danh sách Document cho collection rxnorm"""
    docs: list[Document] = []
    skipped = 0

    for rec in _read_jsonl(concepts_path):
        doc_id = rec.get("doc_id")
        text = rec.get("text")
        if not doc_id or not text:
            skipped += 1
            continue
        metadata: dict[str, Any] = {
            "doc_id": doc_id,
            "code": rec.get("code"),
            "kb": "rxnorm",
            "role": "candidate",
        }
        synonyms = rec.get("synonyms")
        if synonyms:
            metadata["synonyms"] = json.dumps(synonyms, ensure_ascii=False)
        docs.append(
            Document(page_content=normalize_text(text), metadata=_clean_metadata(metadata))
        )

    for rec in _read_jsonl(aliases_path):
        doc_id = rec.get("doc_id")
        text = rec.get("text")
        if not doc_id or not text:
            skipped += 1
            continue
        metadata = {
            "doc_id": doc_id,
            "kb": "rxnorm",
            "role": "alias",
        }
        resolve_in = rec.get("resolve_in")
        if resolve_in:
            metadata["resolve_in"] = json.dumps(resolve_in, ensure_ascii=False)
        resolve_min = rec.get("resolve_min")
        if resolve_min:
            metadata["resolve_min"] = json.dumps(resolve_min, ensure_ascii=False)
        docs.append(
            Document(page_content=normalize_text(text), metadata=_clean_metadata(metadata))
        )

    if skipped:
        logger.warning("rxnorm: skipped %d records missing doc_id/text", skipped)

    return _dedupe_by_doc_id(docs, "rxnorm")
