"""
Reranker cross-encoder (Qwen3-Reranker-0.6B) chấm lại toàn bộ union của hai nhánh.

Vì sao dùng `sentence_transformers.CrossEncoder` chứ không phải hai lựa chọn kia:

  * KHÔNG dùng `transformers` thuần: Qwen3-Reranker là causal LM chấm điểm bằng
    logit của token "yes"/"no" trên một chat template riêng. ST >= 5 đã cài sẵn
    template, cách lấy logit và batching - viết tay chỉ là chép lại và thêm chỗ
    để sai.
  * KHÔNG dùng `CrossEncoderReranker`/`ContextualCompressionRetriever` của
    LangChain: chúng là abstraction doc-vào/doc-ra, nuốt mất điểm thô. Mà điểm
    thô chính là thứ floor/margin và toàn bộ dump đo đạc cần.

**Instruction là một phần của phân bố điểm.** Đổi câu instruction là đổi thang
điểm, làm mọi ngưỡng đã hiệu chỉnh trở nên vô nghĩa. Vì vậy nó được đóng băng ở
hằng số dưới đây và được ghi vào metadata của `thresholds.json`.
"""

from __future__ import annotations

import logging

import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from configs.config import RERANKER_MODEL_NAME

logger = logging.getLogger(__name__)

PROMPT_NAME = "med_linking"

# ĐÃ ĐÓNG BĂNG - xem docstring. Thay đổi = phải hiệu chỉnh lại ngưỡng.
MED_LINKING_INSTRUCTION = (
    "Given a medical mention extracted from a Vietnamese clinical note, "
    "judge whether the document describes the same medical concept "
    "(the same drug ingredient or the same diagnosis) that the mention refers to."
)


class Qwen3Reranker:
    """Bọc mỏng CrossEncoder: nhận (mention, list[Document]) -> list điểm thô."""

    def __init__(
        self,
        model_name: str = RERANKER_MODEL_NAME,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            logger.warning(
                "Qwen3Reranker chạy trên CPU - chậm hơn GPU hàng chục lần. "
                "Kiểm tra torch.cuda.is_available() nếu máy có GPU."
            )

        model_kwargs = {"torch_dtype": torch.float16} if self.device == "cuda" else {}
        self.model = CrossEncoder(
            model_name,
            device=self.device,
            model_kwargs=model_kwargs,
            prompts={PROMPT_NAME: MED_LINKING_INSTRUCTION},
            default_prompt_name=PROMPT_NAME,
        )
        logger.info("Qwen3Reranker: %s trên %s", model_name, self.device)

    def score(self, query: str, docs: list[Document]) -> list[float]:
        """Điểm thô cho từng doc, THEO ĐÚNG THỨ TỰ docs truyền vào (không sắp xếp)."""
        if not docs:
            return []
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]
