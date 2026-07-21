"""
    Logic parser với regex, chuẩn hóa kí tự về mã thường, tìm điểm cắt gap, 
    và build record full dictionary của bộ mã ICD-10 với các field tương ứng
"""

import re

from configs.config import (
    COLUMN_BUCKET_WIDTH,
    COLUMN_GAP_THRESHOLD_RATIO,
    FOOTER_MARGIN,
)
from kb.icd_hierarchy import lookup_hierarchy
from kb.icd_text import split_head_tail

_CODE_LINE_RE = re.compile(r"^([A-Z]\d{2}(?:\.\d{1,4})?[†*]?)\s+(.*)$")

# Dấu hiệu rớt glyph: nguyên âm móc/mũ (ư,ơ,ô,ê,â,ă) đứng ngay trước 1-2 chữ mồ côi
# -- ví dụ "ngư c" (mất ợ của "ngược").
#
# Ghi chú: từng thử mở rộng bắt "một chữ cái đơn lẻ giữa 2 space" để vớt thêm ca kiểu
# "th c qu n" (thực quản), nhưng gây ~1170 false positive vì "u" (khối u), "y" (y học)...
# là từ 1-ký-tự HỢP LỆ rất phổ biến trong tên bệnh -> đã fallback về pattern gốc này.
_GLYPH_DROP_RE = re.compile(r"[ươôêâăƯƠÔÊÂĂ]\s[a-zà-ỹ]{1,2}\b")


def parse_code_stream(lines: list[str]) -> dict[str, str]:
    entries: dict[str, str] = {}
    current_code: str | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        match = _CODE_LINE_RE.match(line)
        if match:
            raw_code, description = match.groups()
            # Chuẩn hóa: bỏ ký hiệu dual-classification †/* khỏi mã. Nếu mã chuẩn hóa đã
            # tồn tại (vd E14 và E14†) -> gộp mô tả thay vì tạo entry trùng.
            current_code = raw_code.rstrip("†*")
            if current_code in entries:
                entries[current_code] = f"{entries[current_code]} {description}".strip()
            else:
                entries[current_code] = description
        elif current_code is not None:
            entries[current_code] = f"{entries[current_code]} {line}"

    return entries


def join_en_vi(en: dict[str, str], vi: dict[str, str]) -> list[dict[str, str | None]]:
    """
    Gộp 3 thành phần gồm code ICD, tên tiếng Việt và Anh thành dict
    """

    all_codes = sorted(set(en) | set(vi))
    return [
        {"code": code, "name_en": en.get(code), "name_vi": vi.get(code)}
        for code in all_codes
    ]


def find_column_split(x0_values: list[float], page_width: float) -> float:
    """
    Từ các tọa độ x theo point, tìm xem phần khoảng trống giữa 2 cột EN/ VI nằm ở x nào
    """
    if not x0_values:
        raise ValueError("x0_values must not be empty")

    buckets: dict[int, int] = {}
    for x in x0_values:
        bucket = int(x // COLUMN_BUCKET_WIDTH) #gom nhóm ô rộng 10 point -> xong đếm mật độ từng ô
        buckets[bucket] = buckets.get(bucket, 0) + 1 

    min_bucket, max_bucket = min(buckets), max(buckets)
    counts = [buckets.get(b, 0) for b in range(min_bucket, max_bucket + 1)] #đếm với các mật độ từ của từng group 10 point
                                                                            #nếu mật độ của một vùng ô bất kì từ thưa dần -> tăng lên, đó là vùng khê cột 

    #mật độ ô đông nhất * số % của mốc đó -> Nếu ô có số từ nhỏ hơn hoặc bằng thì coi là trống
    threshold = max(counts) * COLUMN_GAP_THRESHOLD_RATIO 

    # Gom TẤT CẢ các khoảng trống nằm giữa hai cụm dày (dense). Layout 2 cột có thể có
    # nhiều khoảng trống -- ví dụ dải lề trái rộng do mã ICD thụt lề treo (hanging-indent)
    # -- nên KHÔNG chọn khoảng rộng nhất. Khe giữa 2 cột thật là khoảng có mép phải gần
    # tâm trang nhất.
    gaps: list[tuple[int, int]] = []  # (chỉ_số_bắt_đầu, chỉ_số_kết_thúc_không_bao_gồm)
    dense = [i for i, c in enumerate(counts) if c > threshold] #danh sách các chỉ số vượt ngưỡng, những ô thật sự có chữ
    if len(dense) < 2: #trang có ít hơn 2 ô đông chữ -> gần như là trang trống hoặc trang bìa
        return page_width / 2 #fallback về chiều rộng page / 2

    first_dense, last_dense = dense[0], dense[-1]
    gap_start = None
    for i in range(first_dense, last_dense + 1):
        if counts[i] <= threshold:
            if gap_start is None:
                gap_start = i
        elif gap_start is not None:
            gaps.append((gap_start, i))
            gap_start = None

    if not gaps: #nếu không tồn tại gap -> không có ô trống kẹp giữa hai ô đông, tức chữ trải liênt ục từ trái sang phải
        return page_width / 2

    page_center = page_width / 2

    def gap_edge_x(gap: tuple[int, int]) -> float:
        return (min_bucket + gap[1]) * COLUMN_BUCKET_WIDTH

    best_gap = min(gaps, key=lambda g: abs(gap_edge_x(g) - page_center)) #lấy best gap giữa gap với hiệu của gap edge - chính giữa pdf
    return gap_edge_x(best_gap)


def extract_page_columns(page) -> tuple[list[str], list[str]]:
    """Trích xuất các text sau khi có gap"""

    words = page.extract_words()
    if not words:
        return [], []

    split = find_column_split([w["x0"] for w in words], page.width)
    content_bottom = page.height - FOOTER_MARGIN

    left = page.crop((0, 0, split, content_bottom))
    right = page.crop((split, 0, page.width, content_bottom))

    left_text = left.extract_text() or ""
    right_text = right.extract_text() or ""

    return left_text.split("\n"), right_text.split("\n")


def has_glyph_drop_signature(text: str | None) -> bool:
    """Metadata nếu glyph bị lỗi -> trả về true/ false"""

    if text is None:
        return False
    return bool(_GLYPH_DROP_RE.search(text))


def build_record(code: str, name_en: str | None, name_vi: str | None) -> dict:
    """Build full dict từng mã ICD

    Bao gồm các params:
    - doc_id: Mã ICD-10 theo định dạng Document cho langchain
    - code: Mã ICD-10
    - text: Tên căn bệnh tương ứng với mã Icd-10 đó (ưu tiên name_vi)
    - name_vi: Tên tiếng Việt tương ứng với mã ICD-10 (prioritize)
    - name_en: Tên tiếng Anh tương ứng với mã ICD-10 (fallback nếu không thể extract name_vi (= null))
    - head_vi, note_vi / head_en, note_en: Nếu một mã có description bắt nguồn từ loại trừ, bao gồm,... cắt phần thông tin này
    - Chapter, block: Mã này nằm ở chương nào, với dải block trong vùng chương
    - kb: đánh dấu knowledge base thuộc về ICD
    - role: candidate - ấn định làm một mã chuẩn
    - vi_glyph_ok: Biến bool để debug, nếu không bị lỗi âm tiết và mất nguyên âm -> True. Ngược lại -> False
    """

    chapter, block = lookup_hierarchy(code)

    # Cắt tên bệnh (head) khỏi phần ghi chú đuôi (Loại trừ/Bao gồm/...). Head đem đi
    # embed; note để riêng làm synonym/tham chiếu cho BM25 phụ.
    head_vi, note_vi = split_head_tail(name_vi)
    head_en, note_en = split_head_tail(name_en)

    # vi_glyph_ok là METADATA cảnh báo (head VI có và không rớt glyph), KHÔNG điều khiển
    # text nữa. Lý do: lỗi glyph thường chỉ mất 1 âm tiết lẻ ("hàm tr n"), phần tên chính
    # vẫn match tốt query tiếng Việt -> giữ name_vi đem embed vẫn lợi hơn dịch cả sang EN.
    vi_glyph_ok = head_vi is not None and not has_glyph_drop_signature(head_vi)

    # text (field đem đi embed/BM25): luôn ưu tiên head VI (kể cả khi glyph hỏng); chỉ
    # fallback head EN khi tiếng Việt mất HOÀN TOÀN (head_vi = None).
    text = head_vi if head_vi is not None else head_en

    return {
        "doc_id": f"icd_{code}",
        "code": code,
        "text": text,
        "name_vi": head_vi,
        "name_en": head_en,
        "note_vi": note_vi,
        "note_en": note_en,
        "chapter": chapter,
        "block": block,
        "kb": "icd",
        "role": "candidate",
        "vi_glyph_ok": vi_glyph_ok,
    }


def build_icd_concepts(pdf, pages: range | None = None) -> list[dict]:
    en_lines: list[str] = []
    vi_lines: list[str] = []

    page_iter = pdf.pages if pages is None else (pdf.pages[i] for i in pages)
    for page in page_iter:
        left_lines, right_lines = extract_page_columns(page)
        en_lines.extend(left_lines)
        vi_lines.extend(right_lines)

    en_entries = parse_code_stream(en_lines)
    vi_entries = parse_code_stream(vi_lines)
    joined = join_en_vi(en_entries, vi_entries)

    return [
        build_record(code=entry["code"], name_en=entry["name_en"], name_vi=entry["name_vi"])
        for entry in joined
    ]


