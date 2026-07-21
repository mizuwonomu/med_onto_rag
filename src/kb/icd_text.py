"""Hậu xử lý mô tả ICD: tách "tên bệnh đầu" (head) khỏi "phần ghi chú đuôi" (tail).

Cấu trúc một mô tả ICD sau khi parse thường là:
    <tên bệnh chuẩn>  <ghi chú: Loại trừ / Bao gồm / Dùng mã / [See / synonym...>
Tên bệnh luôn nằm ngay đầu; các marker ghi chú mở đầu phần đuôi. Cắt tại marker đầu
tiên -> giữ head sạch để embed, tail để riêng (synonym/tham chiếu) cho BM25 phụ.

Dùng chung cho cả hai kiến trúc parse (crop nửa trang và word-level).
"""

import re

# Marker mở đầu phần đuôi, song ngữ. "Loại trừ:", "Bao gồm:" gần như luôn kèm dấu ':'.
# "Dùng mã..." không có ':'. Bên tiếng Anh có Excl./Incl./[See/Use additional/The following.
_TAIL_MARKER_RE = re.compile(
    r"(Loại trừ\s*:|Bao gồm\s*:|Dùng mã|Sử dụng thêm|Nếu cần"
    r"|Excl\.?\s*:?|Incl\.?\s*:?|\[See|Use additional|The following)"
)


def split_head_tail(text: str | None) -> tuple[str | None, str | None]:
    """Tách text thành (head, tail) tại marker ghi chú đầu tiên.

    - Không có marker -> (text, None): toàn bộ là tên bệnh.
    - Có marker -> (phần trước marker đã strip, phần từ marker trở đi đã strip).
    - head rỗng sau khi cắt (marker nằm ngay đầu) -> head=None để bước sau fallback.
    """
    if text is None:
        return None, None

    match = _TAIL_MARKER_RE.search(text)
    if not match:
        head = text.strip()
        return (head or None), None

    head = text[: match.start()].strip()
    tail = text[match.start() :].strip()
    return (head or None), (tail or None)
