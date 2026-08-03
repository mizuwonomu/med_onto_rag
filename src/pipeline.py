"""
Ghép ba tầng thành một document: trích xuất -> gán vị trí -> gắn mã ứng viên.


THỨ TỰ LÀ MỘT QUYẾT ĐỊNH, KHÔNG PHẢI NGẪU NHIÊN
-----------------------------------------------
`align_positions` chạy TRƯỚC `link_entity`. Vì align ép `text` về nguyên văn, nên
chuỗi đưa vào retrieval là chuỗi thật trong bệnh án, không phải bản LLM đã sửa.
Đảo thứ tự lại thì retrieval chấm trên một chuỗi mà bản ghi cuối cùng không hề
chứa - và align còn có thể vứt entity đó đi sau đó, phí trọn một lượt rerank.

HÌNH DẠNG BẢN GHI ĐẦU RA (theo mẫu của BTC)
-------------------------------------------
Hai khoá `assertions` và `candidates` VẮNG MẶT chứ không rỗng, và mỗi khoá theo
một luật riêng - chúng không đi cùng nhau:

  * `assertions` chỉ có ở CHẨN_ĐOÁN / THUỐC / TRIỆU_CHỨNG. Hai loại xét nghiệm
    không có: phủ định của một chỉ số nằm ở GIÁ TRỊ của nó ("CRP âm tính" thì
    thông tin nằm ở kết quả), không phải ở nhãn assertion.
  * `candidates` chỉ có ở CHẨN_ĐOÁN / THUỐC - đúng hai loại có KB tương ứng.

Vắng mặt khác rỗng: `"assertions": []` khẳng định "đã xét, không có assertion
nào"; không có khoá đó khẳng định "loại này không mang khái niệm assertion". BTC
đọc theo nghĩa thứ hai, nên `pop` khoá chứ đừng gán list rỗng.
"""

from __future__ import annotations

import logging
from typing import Any

from extract.align import align_positions
from extract.linker import link_entity

logger = logging.getLogger(__name__)

# Hai tập này ĐỘC LẬP nhau - TRIỆU_CHỨNG có assertions nhưng không có candidates.
TYPES_WITH_ASSERTIONS = {"CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"}
TYPES_WITH_CANDIDATES = {"CHẨN_ĐOÁN", "THUỐC"}


def extract_document(
    raw_text: str, ner_chain, backends: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Một document -> danh sách bản ghi ở đúng định dạng nộp.

    `backends=None` = chế độ chỉ-NER (`--skip-link`): bỏ hẳn bước gắn mã để soi
    chất lượng trích xuất mà không phải nạp store lên GPU. Đây là chế độ để đọc
    kết quả tầng NER một cách tách bạch, không phải đường tắt lúc chạy thật.
    """
    output = ner_chain.invoke({"input_text": raw_text})
    entities = output.entities if hasattr(output, "entities") else output["entities"]

    records: list[dict[str, Any]] = []
    for ent in align_positions(raw_text, entities):
        ent_type = ent["type"]
        record: dict[str, Any] = {"text": ent["text"], "type": ent_type}

        if backends is not None and ent_type in TYPES_WITH_CANDIDATES:
            codes = link_entity(ent["text"], ent_type, backends)
            if codes is not None:
                record["candidates"] = sorted(codes)

        # Loại không mang khái niệm assertion thì BỎ HẲN khoá, không gán rỗng.
        if ent_type in TYPES_WITH_ASSERTIONS:
            record["assertions"] = ent["assertions"]

        record["position"] = ent["position"]
        records.append(record)

    return records
