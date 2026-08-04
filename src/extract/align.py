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

VÌ SAO FUZZY LÀ NFC+CASEFOLD+GỘP-KHOẢNG-TRẮNG CHỨ KHÔNG PHẢI EDIT-DISTANCE
-------------------------------------------------------------------------
Sai khác thực tế của LLM có ba loại, cả ba đều là ánh xạ TẤT ĐỊNH chứ không phải
đoán: chuẩn hoá Unicode (NFD vs NFC: "ố" một codepoint hay hai), hoa/thường, và
KHOẢNG TRẮNG - model nuốt double-space ("nhịp  xoang" -> "nhịp xoang") hoặc dán
hai dòng vật lý làm một ("Lactat (Acid\nLactic): 0.8" viết liền). Gộp mọi cụm
khoảng trắng (kể cả \n) về đúng một space ở CẢ HAI vế rồi so chuỗi thường là đủ.

Vì sao gộp khoảng trắng KHÔNG phải edit-distance trá hình: nó chỉ nối được hai
cụm cách nhau THUẦN bằng khoảng trắng. "A B" khớp "A\n\nB" nhưng KHÔNG khớp
"A x B" - chen một ký tự không-trắng vào là trượt. Nên nó không thể gán "đau
bụng" vào "đau lưng" như một ngưỡng edit-distance sẽ làm; nó chỉ xoá sự phân
biệt về SỐ LƯỢNG và LOẠI khoảng trắng - đúng thứ vô nghĩa với WER (chấm theo TỪ)
và với position (không được BTC chấm). Span cứu được nhờ luật này vẫn giữ đúng
vị trí thật trong raw, chỉ là biên bao trùm cả cái newline nằm giữa.

RAW KHÔNG BAO GIỜ BỊ CHUẨN HOÁ
------------------------------
So khớp fuzzy chạy trên một BẢN SAO đã NFC+casefold, nhưng offset phải trỏ về
chuỗi gốc. NFC có thể ĐỔI ĐỘ DÀI chuỗi (2 codepoint -> 1), nên offset trên bản
sao KHÔNG dùng lại được trực tiếp. Vì vậy `_fold_with_map` giữ một bảng ánh xạ
index-bản-sao -> index-raw, và mọi offset trả ra đều đi qua bảng đó.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# Trạng từ mức độ/tần suất. KHÔNG đổi khái niệm, chỉ thổi WER - mà WER chuẩn hoá
# theo ĐỘ DÀI GOLD, nên một gold 2 từ chỉ cần thừa 1 từ là mất một nửa điểm.
#
# Danh sách này ĐÃ ĐƯỢC THỬ ĐẶT TRONG PROMPT VÀ THẤT BẠI: thêm luật + ví dụ vào
# khối MINIMAL SPAN gần như không đổi số ca vi phạm, kể cả với những từ viết
# thẳng ra trong luật. Model 9B không cưỡng chế được luật hình thức;
# `str.endswith` thì làm được đúng và miễn phí.
_SEVERITY_TAIL = (
    "dữ dội", "nghiêm trọng", "trầm trọng", "thoáng qua", "dai dẳng",
    "liên tục", "liên lục", "thường xuyên", "tột độ", "âm ỉ",
    "nặng", "nhẹ", "vừa", "nhiều", "ít",
)

# Cụm mà bỏ trạng từ đi thì phần còn lại KHÔNG còn là bệnh lý nữa. `nhịp tim` là
# trạng thái bình thường, `nhịp tim chậm` mới là bất thường - cắt `chậm` là xoá
# mất khái niệm. Danh sách này là chốt chặn, không phải danh sách đầy đủ.
_SEVERITY_KEEP = ("nhịp tim chậm", "nhịp tim nhanh", "huyết áp thấp", "huyết áp cao")

# Từ mà nếu đứng NGAY TRƯỚC trạng từ thì cụm đó là TÊN khái niệm chứ không phải
# một khái niệm bị chấm mức độ. `chuỗi nhẹ` trong `Bệnh amyloidosis chuỗi nhẹ`
# là thuật ngữ dịch của "light chain" - cắt `nhẹ` để lại `chuỗi` vô nghĩa.
_SEVERITY_BOUND = ("chuỗi",)

# Số TỪ tối thiểu còn lại sau khi cắt. Một từ trần hầu như luôn là động từ hoặc
# danh từ chưa thành khái niệm (`tiểu` từ `tiểu ít`, `đau` từ `đau vừa`), trong
# khi chính cụm hai từ mới là triệu chứng có mã (thiểu niệu). Danh sách cấm thì
# luôn thiếu ca; điều kiện về phần CÒN LẠI thì phủ được cả những ca chưa gặp.
_MIN_WORDS_AFTER_TRIM = 2


# Một cụm khoảng trắng bất kỳ (space, tab, \n, lẫn lộn) coi như một space.
_WS_RUN = re.compile(r"\s+")


def _fold(s: str) -> str:
    """NFC + casefold + gộp khoảng trắng. Cả ba TẤT ĐỊNH (xem docstring đầu file).

    KHÔNG strip đầu/cuối: `_fold_with_map` dựng bảng ánh xạ theo PREFIX và dựa
    vào tính đơn điệu "thêm ký tự vào cuối không rút ngắn phần đã fold". `re.sub`
    gộp-về-một-space giữ được tính đó (thêm một ký tự trắng vào prefix đang kết
    thúc bằng trắng thì độ dài folded không đổi); `strip()` thì phá nó, vì bỏ
    trắng ở ĐẦU làm folded của prefix ngắn đi khi ký tự sau xuất hiện.
    """
    return _WS_RUN.sub(" ", unicodedata.normalize("NFC", s).casefold())


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


def _trim_severity(rec: dict[str, Any], raw_input: str) -> bool:
    """Cắt trạng từ mức độ ở CUỐI span. Trả True nếu có cắt.

    Co `end` chứ không sửa `text` rời: bất biến `raw[start:end] == text` là hợp
    đồng của cả tầng này (xem docstring đầu file), nên mọi thay đổi text phải đi
    kèm thay đổi offset trong cùng một phép gán.

    Chỉ cắt ĐUÔI, không cắt đầu. Trạng từ đứng trước (`rất đau bụng`) hiếm hơn
    nhiều trong bệnh án và cắt đầu thì phải dịch `start`, dễ lệch với dấu cách.
    """
    text = rec["text"]
    low = text.casefold()
    if any(low.endswith(k) for k in _SEVERITY_KEEP):
        return False

    for tail in _SEVERITY_TAIL:
        if not low.endswith(" " + tail):
            continue
        cut = len(text) - len(tail) - 1
        trimmed = text[:cut].rstrip()
        # Phần CÒN LẠI phải còn là một khái niệm. Một từ trần gần như luôn không
        # phải: `tiểu ít` -> `tiểu`, `đau vừa` -> `đau`, `sốt nhẹ` -> `sốt` đều
        # xoá mất thứ đang được đặt tên. Đo trên bài nộp thật: mọi ca cắt SAI
        # đều rơi vào nhóm còn lại đúng một từ, mọi ca cắt ĐÚNG đều còn >= 2.
        if len(trimmed.split()) < _MIN_WORDS_AFTER_TRIM:
            return False
        # Trạng từ đứng sau một trong các từ này là một phần của TÊN, không phải
        # mức độ - xem `_SEVERITY_BOUND`.
        if trimmed.casefold().endswith(_SEVERITY_BOUND):
            return False
        start = rec["position"][0]
        end = start + len(trimmed)
        # Kiểm lại hợp đồng thay vì tin vào số học offset. `rstrip()` ở trên có
        # thể ăn nhiều hơn một khoảng trắng, và một lệch 1 ký tự ở đây tạo ra
        # đúng thứ cả tầng này được viết ra để ngăn: text và position trỏ vào
        # hai chuỗi khác nhau. Không khớp thì bỏ phép cắt, giữ span dài.
        if raw_input[start:end] != trimmed:
            return False
        rec["text"] = trimmed
        rec["position"] = [start, end]
        return True
    return False


def _drop_nested(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bỏ span nằm TRỌN trong một span khác CÙNG LOẠI.

    `sốt` [2281,2284] và `sốt siêu vi` [2281,2292] là một khái niệm bị phát hiện
    hai lần ở hai biên. BTC chấm chúng thành hai khái niệm, trong đó ít nhất một
    cái sai - và cái ngắn hơn gần như luôn là cái sai, vì nó là tiền tố của một
    cụm dài hơn có nghĩa đầy đủ.

    Đòi CÙNG LOẠI: `đại tiện ra máu đỏ tươi` (TRIỆU_CHỨNG) chứa `ra máu`
    (TRIỆU_CHỨNG) thì bỏ cái trong; nhưng một TÊN_XÉT_NGHIỆM nằm trong một
    KẾT_QUẢ dài hơn là lỗi loại, không phải trùng lặp, và cần người nhìn.
    """
    keep: list[dict[str, Any]] = []
    for i, r in enumerate(records):
        s1, e1 = r["position"]
        nested = any(
            j != i
            and o["type"] == r["type"]
            and o["position"][0] <= s1
            and e1 <= o["position"][1]
            and (o["position"][1] - o["position"][0]) > (e1 - s1)
            for j, o in enumerate(records)
        )
        if not nested:
            keep.append(r)
    return keep


def _is_junk(text: str) -> bool:
    """Span không mang khái niệm nào: BTC che tên thuốc bằng dấu sao.

    Model gắn nhãn THUỐC cho chính chuỗi dấu sao. Không có mã nào tra được, và
    nó vào mẫu số của cả ba metric - bỏ hẳn rẻ hơn giữ.
    """
    stripped = text.strip().strip("*").strip()
    return not stripped


def postprocess(records: list[dict[str, Any]], raw_input: str) -> list[dict[str, Any]]:
    """Ba luật TẤT ĐỊNH chạy sau khi đã có position.

    Vì sao ở đây chứ không ở prompt: cả ba đều kiểm tra được bằng phép so chuỗi,
    và lần thử đẩy chúng vào prompt đo được là gần như vô hiệu. Việc gì
    `str.endswith` làm đúng 100% thì không nhờ một model 9B làm.

    Thứ tự có ý nghĩa: cắt đuôi TRƯỚC khi lọc lồng nhau, vì cắt xong mới lộ ra
    hai span trùng khít (`buồn nôn thoáng qua` -> `buồn nôn`, trùng `buồn nôn`
    đã có sẵn).
    """
    out = [r for r in records if not _is_junk(r["text"])]
    n_junk = len(records) - len(out)

    n_trim = sum(_trim_severity(r, raw_input) for r in out)

    before = len(out)
    out = _drop_nested(out)
    n_nested = before - len(out)

    # Cắt đuôi có thể tạo ra bản trùng khít với một span khác; giữ bản đầu tiên.
    seen: set[tuple[str, int]] = set()
    deduped: list[dict[str, Any]] = []
    for r in out:
        key = (str(r["type"]), r["position"][0])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    n_dup = len(out) - len(deduped)

    # BƯỚC CUỐI: gộp khoảng trắng trong `text` xuất ra. Chạy SAU mọi phép tính
    # dựa trên vị trí (_trim_severity, _drop_nested) nên không phá số học offset
    # của chúng - `position` vẫn trỏ span raw thật (bao cả newline bên trong),
    # còn `text` thì sạch cho WER.
    #
    # ĐÂY LÀ CHỖ DUY NHẤT bất biến `text == raw[start:end]` được cố ý nới: nới
    # đúng ở khoảng trắng, GIỮ nguyên hoa/thường + chính tả của nguồn (vì text
    # vẫn lấy từ lát raw, không phải chuỗi thô của LLM - cái verbatim contract
    # làm tốt là snap về nguồn để sửa lỗi gõ của model, ta không vứt kèm).
    # An toàn vì: position không được chấm, và WER tách theo TỪ nên \n với
    # double-space vốn washout - gộp chỉ để chắc với tokenizer ngây thơ.
    for r in deduped:
        r["text"] = _WS_RUN.sub(" ", r["text"]).strip()

    if n_junk or n_trim or n_nested or n_dup:
        logger.info(
            "postprocess: bỏ %d span rác, cắt %d trạng từ mức độ, "
            "bỏ %d span lồng nhau, %d trùng sau khi cắt",
            n_junk, n_trim, n_nested, n_dup,
        )
    return deduped


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
    dropped: list[str] = []
    n_in = 0
    cursor = 0

    for ent in entities:
        n_in += 1
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
            dropped.append(str(text))
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

    # WARNING chứ không phải DEBUG. Mention bị bỏ ở đây mất TRỌN cả ba trường
    # được chấm - text, assertions và candidates - và nó biến mất mà không để
    # lại dấu vết nào trong file nộp bài: một document 20 khái niệm bị rơi 5
    # trông y hệt một document chỉ có 15 khái niệm. Ở mức DEBUG thì mất mát này
    # vô hình trong mọi lần chạy thật, vì các script chạy ở INFO.
    if dropped:
        logger.warning(
            "align: bỏ %d/%d mention không định vị được trong văn bản: %s",
            len(dropped),
            n_in,
            ", ".join(repr(t) for t in dropped[:5])
            + (f" ... (+{len(dropped) - 5})" if len(dropped) > 5 else ""),
        )

    return postprocess(aligned, raw_input)
