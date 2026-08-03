"""
Schema đầu ra của LLM trích xuất - NGUỒN DUY NHẤT sinh ra grammar cho decoding.

Đây là hợp đồng hẹp nhất có thể: LLM CHỈ sinh `text/type/assertions`. Nó không
sinh `position` (vị trí là việc của string matching tất định - xem `align.py`,
để LLM đếm ký tự là mời gọi hallucination offset), và không sinh `candidates`
(mã ICD/RXCUI là việc của tầng retrieval đã hiệu chỉnh ngưỡng - xem `linker.py`;
LLM sinh mã thì đó là mã bịa từ trí nhớ, không neo vào KB nào cả).

`with_structured_output(..., method="json_schema")` dịch chính lớp Pydantic này
thành JSON Schema, llama-server chuyển tiếp thành GBNF để ràng buộc decoding. Do
đó mọi thay đổi ở đây đổi luôn grammar - xem cảnh báo fail-open trong `llm.py`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from configs.config import MAX_ENTITIES_PER_DOC
# Năm loại thực thể theo đề bài. Literal (không phải str) để enum đi thẳng vào
# grammar: model không thể sinh ra một loại thứ sáu.
ENTITY_TYPES = Literal[
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
    "CHẨN_ĐOÁN",
    "THUỐC",
]

ASSERTION_TYPES = Literal["isNegated", "isFamily", "isHistorical"]


class Entity(BaseModel):
    """Một mention trong văn bản. `text` phải là trích NGUYÊN VĂN từ input.

    Ràng buộc verbatim không thể ép bằng grammar (grammar không thấy được input),
    nên nó được ép ở hai chỗ khác: prompt yêu cầu, và `align.py` ghi đè
    `text = raw[start:end]` sau khi tìm được vị trí.
    """

    text: str = Field(description="Trích nguyên văn từ văn bản đầu vào")
    type: ENTITY_TYPES
    assertions: list[ASSERTION_TYPES] = Field(default_factory=list)


class NerOutput(BaseModel):
    """`max_length` chặn cứng số khái niệm ở tầng GRAMMAR, không phải ở prompt.

    Vì sao cần: greedy decoding (temperature=0) có thể kẹt trong vòng lặp thoái
    hoá - đo được trên câu "Bệnh nhân sốt cao, không ho.": model sinh đúng 3
    khái niệm đầu rồi lặp `sốt cao -> ho -> không ho` cho tới khi hết max_tokens.
    Sau khi đóng một object, xác suất của `]` và `,` gần bằng nhau; greedy chọn
    cứng cái nhỉnh hơn, mà trạng thái lặp lại y hệt nên nó chọn lại y hệt.

    GBNF ràng buộc HÌNH DẠNG chứ không ràng buộc ĐỘ DÀI, nên mảng dài bao nhiêu
    cũng "hợp lệ". `max_length` -> `maxItems` trong JSON Schema -> llama.cpp dịch
    sang GBNF (common/json-schema-to-grammar.cpp) là thứ DUY NHẤT khiến grammar
    tự đóng mảng. `max_tokens` chỉ giới hạn thiệt hại, không ngăn được vòng lặp.
    """

    entities: list[Entity] = Field(
        default_factory=list, max_length=MAX_ENTITIES_PER_DOC
    )
