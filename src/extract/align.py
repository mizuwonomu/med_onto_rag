"""
Gán vị trí + sửa nguyên văn cho mention do LLM sinh ra. Chuỗi thuần, không model.

VÌ SAO GỘP LÀM MỘT BƯỚC
-----------------------
Tách "tìm vị trí" và "sửa text về nguyên văn" thành hai bước là tự tạo ra khả
năng lệch pha: bước sau có thể sửa text mà không cập nhật offset, và không ai
phát hiện. Ở đây chỉ có một luật duy nhất: tìm được `[start, end)` thì `text`
BỊ ÉP thành `raw[start:end]`. Text và position không thể mâu thuẫn, theo cấu
trúc chứ không nhờ kỷ luật của người viết code.

Hệ quả: LLM viết hoa lại, sửa chính tả, hay bỏ dấu thì kết quả vẫn khớp bản gốc
- miễn là còn tìm được span. Không cần dặn model "đừng sửa" rồi cầu may.

VÌ SAO CON TRỎ CHẠY TIẾN (CURSOR)
---------------------------------
"táo bón" xuất hiện hai lần trong bệnh án là chuyện thường, và LLM trả về hai
mention giống hệt nhau. `raw.find(text)` thuần sẽ gán cả hai vào cùng một offset
-> BTC chấm thành một thực thể trùng, mất một điểm. Con trỏ tiến theo thứ tự LLM
liệt kê (thường trùng thứ tự xuất hiện trong văn bản) nên lần thứ hai bắt được
đúng lần xuất hiện thứ hai.

Khi LLM liệt kê KHÔNG theo thứ tự văn bản, con trỏ trượt qua khỏi mention -> mới
có lần tìm lại từ 0. Đây là fallback đúng nhưng có giá: nó có thể gán trùng lại
một offset đã dùng, nên bước dedupe cuối cùng phải chốt bằng `(text, type, start)`.

VÌ SAO FUZZY LÀ NFC+CASEFOLD CHỨ KHÔNG PHẢI EDIT-DISTANCE
---------------------------------------------------------
Sai khác thực tế của LLM hầu hết là chuẩn hoá Unicode (NFD vs NFC: "ố" một
codepoint hay hai) và hoa/thường. Cả hai đều là ánh xạ TẤT ĐỊNH, không phải đoán.
Một ngưỡng edit-distance thì ngược lại: nó sẽ vui vẻ gán "đau bụng" vào "đau
lưng" ở đâu đó trong văn bản, tức là bịa ra một mention có vị trí hợp lệ - lỗi
tệ hơn hẳn việc bỏ sót.

RAW KHÔNG BAO GIỜ BỊ CHUẨN HOÁ
------------------------------
So khớp fuzzy chạy trên một BẢN SAO đã NFC+casefold, nhưng offset phải trỏ về
chuỗi gốc. NFC có thể ĐỔI ĐỘ DÀI chuỗi (2 codepoint -> 1), nên offset trên bản
sao KHÔNG dùng lại được trực tiếp. Vì vậy `_fold_with_map` giữ một bảng ánh xạ
index-bản-sao -> index-raw, và mọi offset trả ra đều đi qua bảng đó.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def _fold(s: str) -> str:
    return unicodedata.normalize("NFC", s).casefold()


def _fold_with_map(raw: str) -> tuple[str, list[int]]:
    """Trả (chuỗi đã NFC+casefold, ánh xạ index_folded -> index_raw).

    PHẢI fold CẢ CHUỖI, không được fold từng ký tự một. NFC là phép HỢP THÀNH
    nhiều ký tự: "u" + dấu mũ + dấu sắc -> "ấ". Chuẩn hoá lẻ từng ký tự thì các
    dấu tổ hợp không bao giờ gặp được ký tự cơ sở, chuỗi NFD ở lại nguyên dạng
    NFD, và mọi so khớp với needle đã NFC đều trượt - đúng thứ nhánh fuzzy sinh
    ra để cứu. (Lỗi này đã xảy ra thật; xem ca NFD trong scripts/ner/smoke_align.)

    Bảng ánh xạ vì thế dựng theo PREFIX: `len(_fold(raw[:i]))` là ranh giới của
    ký tự raw thứ `i` trên trục folded. Fold có tính đơn điệu theo prefix (thêm
    ký tự vào cuối không rút ngắn phần đầu đã fold), nên các ranh giới này không
    giảm và cho một ánh xạ hợp lệ. Ký tự raw nào bị nuốt vào phép hợp thành sẽ
    có ranh giới trùng với ký tự trước - tra ngược vẫn trả về đúng ký tự cơ sở.

    Phần tử cuối = len(raw) để tra `end` không phải xét trường hợp biên.
    """
    folded = _fold(raw)

    boundaries = [len(_fold(raw[: i + 1])) for i in range(len(raw))]
    index_map: list[int] = [0] * (len(folded) + 1)
    prev = 0
    for i, bound in enumerate(boundaries):
        bound = min(bound, len(folded))
        for pos in range(prev, bound):
            index_map[pos] = i
        prev = max(prev, bound)
    for pos in range(prev, len(folded)):
        index_map[pos] = len(raw) - 1 if raw else 0
    index_map[len(folded)] = len(raw)
    return folded, index_map


def _find_exact(raw: str, needle: str, cursor: int) -> tuple[int, int] | None:
    """Tìm tiến từ `cursor`, trượt thì tìm lại từ đầu."""
    if not needle:
        return None
    start = raw.find(needle, cursor)
    if start == -1:
        start = raw.find(needle)
    if start == -1:
        return None
    return start, start + len(needle)


def _find_fuzzy(
    raw: str, folded_raw: str, index_map: list[int], needle: str, cursor: int
) -> tuple[int, int] | None:
    """Khớp bỏ qua khác biệt NFC/hoa-thường, offset ánh xạ ngược về `raw`."""
    folded_needle = _fold(needle)
    if not folded_needle:
        return None

    # cursor nằm trên trục raw; đẩy sang trục folded bằng cách tìm vị trí folded
    # đầu tiên ánh xạ về >= cursor.
    folded_cursor = 0
    for pos, raw_idx in enumerate(index_map):
        if raw_idx >= cursor:
            folded_cursor = pos
            break

    start = folded_raw.find(folded_needle, folded_cursor)
    if start == -1:
        start = folded_raw.find(folded_needle)
    if start == -1:
        return None

    end = start + len(folded_needle)
    raw_start = index_map[start]
    # `end` là ranh giới SAU ký tự cuối. Tra index_map[end] trực tiếp là sai khi
    # ký tự raw cuối bị nuốt vào một phép hợp thành (ranh giới trùng nhau): khi đó
    # nó trả về chính ký tự cuối và lát cắt bị hụt mất một ký tự. Lấy ký tự raw
    # của vị trí folded cuối cùng THUỘC needle rồi +1 mới đúng exclusive end.
    raw_end = index_map[end - 1] + 1
    # Raw ở dạng NFD thì các dấu tổ hợp đứng SAU ký tự cơ sở và đã bị nuốt vào
    # một ký tự folded duy nhất - chúng thuộc về mention nhưng không có ranh giới
    # riêng để tra. Nuốt nốt chúng, nếu không lát cắt trả về sẽ mất dấu.
    while raw_end < len(raw) and unicodedata.combining(raw[raw_end]):
        raw_end += 1
    return raw_start, max(raw_end, raw_start + 1)


def align_positions(
    raw_input: str, entities: Iterable[Any]
) -> list[dict[str, Any]]:
    """Gán `position` + ép `text` về nguyên văn cho từng entity.

    Nhận `Entity` (Pydantic) hoặc dict - script kiểm tra dữ liệu synthetic nạp
    dict từ JSONL và dùng lại đúng hàm này làm cổng verbatim, nên không được ép
    kiểu Pydantic ở đây.

    Entity không tìm được span nào = LLM bịa ra một chuỗi không có trong văn bản.
    BỎ, không đoán vị trí: một mention có vị trí sai vừa mất điểm của chính nó
    vừa có thể đè lên mention đúng khi BTC ghép cặp.
    """
    folded_raw, index_map = _fold_with_map(raw_input)

    aligned: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    cursor = 0

    for ent in entities:
        if isinstance(ent, dict):
            text, ent_type = ent.get("text", ""), ent.get("type")
            assertions = list(ent.get("assertions") or [])
        else:
            text, ent_type = ent.text, ent.type
            assertions = list(ent.assertions or [])

        span = _find_exact(raw_input, text, cursor) or _find_fuzzy(
            raw_input, folded_raw, index_map, text, cursor
        )
        if span is None:
            logger.debug("bỏ mention không tìm thấy trong input: %r", text)
            continue

        start, end = span
        verbatim = raw_input[start:end]
        key = (verbatim, str(ent_type), start)
        if key in seen:
            continue
        seen.add(key)

        aligned.append(
            {
                "text": verbatim,
                "type": ent_type,
                "assertions": assertions,
                "position": [start, end],
            }
        )
        cursor = end

    return aligned
