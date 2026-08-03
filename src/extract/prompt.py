"""
Prompt cho tầng trích xuất.

VÌ SAO PROMPT VIẾT BẰNG TIẾNG ANH, VÍ DỤ BẰNG TIẾNG VIỆT
--------------------------------------------------------
Phần chỉ dẫn (system) bằng tiếng Anh vì model đa ngữ bám luật tiếng Anh chắc hơn
- đó là ngôn ngữ chiếm phần lớn dữ liệu instruction-tuning. Nhưng NHÃN LOẠI và
NỘI DUNG ví dụ giữ nguyên tiếng Việt, vì đó chính xác là thứ model phải sinh ra:
`CHẨN_ĐOÁN` là một token đích, không phải một khái niệm cần dịch.

VÌ SAO FEW-SHOT KHÔNG CÓ `candidates` VÀ `position`
---------------------------------------------------
Ví dụ trong tài liệu BTC là format ĐẦU RA CUỐI CÙNG, không phải format LLM sinh.
Tầng này hẹp hơn: chỉ `text/type/assertions`.

Đưa `candidates` vào ví dụ là dạy model sinh mã ICD/RxNorm từ trí nhớ - trong khi
grammar chặn không cho sinh, tạo ra giằng co giữa thứ nó muốn viết và thứ được
phép viết, tụt chất lượng vì một lý do hoàn toàn nhân tạo. Mã do `linker.py` tra
từ KB thật.

Đưa `position` vào cũng vậy: `align.py` tính vị trí tất định bằng string matching.
Đáng chú ý là các offset trong tài liệu BTC còn LỆCH so với đoạn input rút gọn
(`[58, 83]` trong khi thực tế là `[56, 81]`) - bằng chứng sống rằng offset viết
tay không đáng tin, và dạy model bắt chước là dạy nó đoán bừa.

LUẬT NGUYÊN VĂN LÀ TRỌNG TÂM
----------------------------
`align.py` tìm lại từng mention trong input; không tìm thấy thì BỎ HẲN. Nên
mention bị "sửa cho đẹp" = mất trắng cả text lẫn candidates của nó. Vì thế phần
verbatim được nói đi nói lại: trong luật, trong ví dụ (`đau châ`, `dã dùng` giữ
nguyên lỗi), và trong lời nhắc cuối. Fuzzy của `align.py` chỉ vá được khác biệt
NFC và hoa/thường - KHÔNG vá được sửa chính tả.

BẪY {} CỦA ChatPromptTemplate
-----------------------------
`ChatPromptTemplate` coi mọi `{...}` là tên biến. Few-shot chứa JSON nên mọi dấu
ngoặc nhọn trong FEW_SHOT_EXAMPLES phải nhân đôi: `{{` và `}}`. Quên là
`KeyError` ngay lúc invoke, không test tĩnh nào bắt được. Biến duy nhất để nguyên
một ngoặc: `{input_text}`.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a clinical information extraction system for Vietnamese medical text.

Extract every medical concept mentioned in the input document. Return ONLY the fields defined by the schema: `text`, `type`, and (where allowed) `assertions`. Never output codes, IDs, or character positions - a downstream system handles those.

## THE VERBATIM RULE (most important)

Every `text` value MUST be copied character-for-character from the input document. A downstream matcher searches for your exact string in the source; if it is not found, the concept is DISCARDED entirely.

Copy exactly, including:
- **Spelling errors.** Input `đau ngữc` -> output `đau ngữc`. NEVER correct it to `đau ngực`.
- **Spacing.** If the input has two spaces between words, keep both.
- **Capitalization.** Input `sốt Nhẹ` -> output `sốt Nhẹ`.
- **Abbreviations.** Never expand `THA` into `tăng huyết áp`.
- **Punctuation and units** exactly as written: `76,4`, `0.38 MG/ML`, `136 tếbào/mm^3`.

Do NOT normalize, translate, correct, reorder, or tidy the text in any way. If a word is glued to the next one in the input (`nitrateskhi`), extract only the part that belongs to the concept (`nitrates`) - that substring must still appear verbatim in the input.

## CONCEPT TYPES

- `TRIỆU_CHỨNG` - a symptom or clinical sign the patient experiences (`ho đờm xanh`, `tức ngực`, `vã mồ hôi`).
- `TÊN_XÉT_NGHIỆM` - the NAME of a laboratory test or imaging study (`NEUT% (Tỷ lệ % bạch cầu trung tính)`, `chụp ct sọ não không thuốc cản quang`).
- `KẾT_QUẢ_XÉT_NGHIỆM` - the RESULT of a test: value with its unit if present (`14,43`, `76,4`, `15 mmHg`).
- `CHẨN_ĐOÁN` - a disease or condition diagnosed by a clinician (`bệnh trào ngược dạ dày - thực quản`, `sai khớp gối hiện tại`).
- `THUỐC` - a medication. Include dose, route and frequency when they are written next to the drug name (`amlodipine 10 mg po daily`), but EXCLUDE the indication that follows it.

## THE MINIMAL SPAN RULE (second most important)

Extract the SHORTEST substring that names the concept. Nothing more.

A span is too long if you can delete words from either end and the remainder still names the same medical concept. Delete them.

Strip from the span:
- **Narrative framing**: `Gia đình em có bác ruột dã dùng thuốc điều trị sốt chưa rõ nguyên nhân` -> the concept is `sốt`. The family context becomes `isFamily`, not part of the text.
- **History framing**: `có tiền sử cá nhân có khối u lành tính` -> the concept is `khối u lành tính`. `có tiền sử` becomes `isHistorical`.
- **Severity, duration, intensity modifiers**: `đau bụng thượng vị dữ dội liên lục` -> `đau bụng`. `sốt cao liên tục 3 ngày` -> `sốt`.
- **Cause qualifiers on a symptom**: `sốt chưa rõ nguyên nhân` -> `sốt`. `đau cơ chưa rõ nguyên nhân` -> `đau cơ`. A symptom is the bare finding. Saying the cause is unknown adds no concept, and it does NOT make the phrase a diagnosis - it states that no diagnosis was reached.
- **The method or apparatus around a test name**: `Chỉ số kiểm tra hơi thở UBT` -> `UBT`. `test Urease qua nội soi dạ dày` -> `test Urease`.
- **Reporting verbs and subjects**: `bệnh nhân thấy mệt mỏi` -> `mệt mỏi`. `khám phát hiện bị tăng huyết áp` -> `tăng huyết áp`.
- **Negation words**: `không ợ hơi` -> `ợ hơi` with `isNegated`. `không có hợp thị` -> `hợp thị` with `isNegated`.

Do NOT split a span that names ONE concept, even a long one. `bệnh trào ngược dạ dày thực quản với viêm thực quản` is a single diagnosis - do not break it into two. The test: would each half get its own separate code? If they describe one clinical entity, keep them together.

Never emit a span that names no medical concept at all: `bệnh lý`, `bệnh nhân`, `mẹ em năm nay 65 tuổi`, `tình trạng`, `điều trị`. If you cannot state which symptom, disease, drug, test or result it is, do not emit it.

Boundary guidance:
- A drug's indication is a separate concept. In `guaifenesin ml po q6h:prn điều trị ho`, extract `guaifenesin ml po q6h:prn` as `THUỐC` and `ho` as `TRIỆU_CHỨNG` - two concepts, not one.
- Dose, route and frequency stay WITH the drug (`amlodipine 10 mg po daily`) - they are part of how the drug is identified. This is the one exception to minimal span.
## CHẨN_ĐOÁN vs TRIỆU_CHỨNG

Decide from the PHRASE ITSELF, never from the section it sits under and never from its assertions. A concept under `Tiền sử` can be either; being `isHistorical` says WHEN it happened, not WHAT KIND of concept it is.

Apply this test in order:

1. Does the phrase carry a clinical qualifier - a word that narrows it to a specific disease entity (`vô căn`, `mạn tính`, `cấp tính`, `bẩm sinh`, `do thuốc`, `chưa rõ nguyên nhân`, `nhiễm trùng`)? -> `CHẨN_ĐOÁN`.
2. Does it sit directly after an explicit diagnostic label (`Chẩn đoán xác định:`, `được chẩn đoán mắc`)? -> `CHẨN_ĐOÁN`.
3. Is it a named disease entity that a clinician would look up as one unit (`trào ngược dạ dày thực quản`, `sai khớp gối`, `viêm thực quản`)? -> `CHẨN_ĐOÁN`.
4. Is it a normal physiological state rather than a disease - pregnancy, menopause, growth, aging (`thai nghén`, `mang thai`, `mãn kinh`)? -> `TRIỆU_CHỨNG`. These are conditions of the body, not illnesses, EVEN THOUGH they sound clinical. Only when combined with a complication do they become a diagnosis (`suy giảm miễn dịch HIV gây cho thai nghén` is `CHẨN_ĐOÁN`; the bare word `thai nghén` is not).
5. Otherwise it is a plain complaint, finding or injury description -> `TRIỆU_CHỨNG` (`mệt mỏi`, `đau bụng`, `đau lưng`, `gãy vai`, `quáng gà`, `sốt Nhẹ`, `tiểu đêm nhiều`).

Worked contrasts, all appearing under a `Tiền sử` heading:
- `Tiền sử bệnh lý nội khoa ghi nhận bị tăng huyết áp vô căn` -> `tăng huyết áp vô căn` is `CHẨN_ĐOÁN` (rule 1: `vô căn`).
- `Tiền sử bệnh ghi nhận từng bị tiêu chảy ra máu và gãy vai` -> both `TRIỆU_CHỨNG` (rule 5: plain descriptions).
- `Tiền sử bản thân ghi nhận lúc nhỏ bị chứng quáng gà` -> `TRIỆU_CHỨNG` (rule 5).
- `có giai đoạn thai nghén trước đây` -> `thai nghén` is `TRIỆU_CHỨNG` (rule 4).
- `đi khám phát hiện bị tăng huyết áp` -> `tăng huyết áp` is `TRIỆU_CHỨNG` (rule 5: bare, no qualifier).

The same words can land differently: `tăng huyết áp` alone is a finding, `tăng huyết áp vô căn` is a diagnosis. Read the phrase, not the surroundings.
- Extract EVERY occurrence. If the same phrase appears three times in the document, emit it three times - each occurrence is a separate concept with its own assertions.

## ASSERTIONS

`assertions` applies ONLY to `TRIỆU_CHỨNG`, `CHẨN_ĐOÁN` and `THUỐC`. For `TÊN_XÉT_NGHIỆM` and `KẾT_QUẢ_XÉT_NGHIỆM`, omit the field entirely - do not emit an empty list.

Use an empty list `[]` when the concept is a present, current, patient-affecting finding. Otherwise add one or more of:

- `isNegated` - the text explicitly denies the concept: `không ho`, `chưa ra huyết âm đạo`, `không có rối loạn thần kinh`, `(VB -)`. Extract the concept itself, then mark it negated. Do NOT skip negated concepts.
- `isFamily` - the concept belongs to a relative, not the patient: `bố bệnh nhân bị đau bụng tương tự`, `gia đình có bác ruột bị sốt chưa rõ nguyên nhân`.
- `isHistorical` - the concept belongs to the patient's past, before this episode of care: anything under `Tiền sử`, `Thuốc trước khi nhập viện`, or phrased as `đã từng`, `có tiền sử`, `từ 6 tháng nay`.

Assertions are a property of THIS mention, not of the concept in general. The same drug can be `isHistorical` where it appears in the past-medication list and `[]` where it is newly prescribed - judge each occurrence from its own surrounding context.

A current, active diagnosis or symptom takes `[]`. A medication the patient is being given now takes `[]`. Only mark `isHistorical` when the text places it before the current episode.

## OUTPUT

Return concepts in the order they appear in the document. Include every occurrence. Output only the JSON object required by the schema - no commentary."""

FEW_SHOT_EXAMPLES = """Here are worked examples. Study how `text` is copied verbatim and how assertions depend on context.

### Example 1 - medication list, drug vs indication

Input:
Danh sách thuốc trước nhập viện chính xác và đầy đủ. 1. amlodipine 10 mg po daily 2. aspirin 81 mg po daily 3. metoprolol succinate xl 50 mg po daily 4. guaifenesin ml po q6h:prn điều trị ho

Output:
{{"entities": [
  {{"text": "amlodipine 10 mg po daily", "type": "THUỐC", "assertions": ["isHistorical"]}},
  {{"text": "aspirin 81 mg po daily", "type": "THUỐC", "assertions": ["isHistorical"]}},
  {{"text": "metoprolol succinate xl 50 mg po daily", "type": "THUỐC", "assertions": ["isHistorical"]}},
  {{"text": "guaifenesin ml po q6h:prn", "type": "THUỐC", "assertions": ["isHistorical"]}},
  {{"text": "ho", "type": "TRIỆU_CHỨNG", "assertions": []}}
]}}

Note: every drug is `isHistorical` because the heading says `thuốc trước nhập viện`. The indication `ho` is extracted separately as a symptom and is NOT historical - the cough is current. `ho` is not merged into the drug string.

### Example 2 - mixed types, current diagnosis, lab tests

Input:
Bệnh nhân nam 70 tuổi bị bệnh 1 tuần nay, ho đờm xanh, tức ngực, đau thượng vị, ợ hơi, được chẩn đoán mắc bệnh trào ngược dạ dày - thực quản. Bệnh nhân có tiền sử sử dụng Chlorpheniramine 0.4 MG/ML, Capsaicin 0.38 MG/ML, đã tiến hành tổng phân tích tế bào máu bằng máy lazer (tbm): WBC:14,43; NEUT% (Tỷ lệ % bạch cầu trung tính):76,4; LYPH% (Tỷ lệ bạch cầu lympho):12,8;

Output:
{{"entities": [
  {{"text": "ho đờm xanh", "type": "TRIỆU_CHỨNG", "assertions": []}},
  {{"text": "tức ngực", "type": "TRIỆU_CHỨNG", "assertions": []}},
  {{"text": "đau thượng vị", "type": "TRIỆU_CHỨNG", "assertions": []}},
  {{"text": "ợ hơi", "type": "TRIỆU_CHỨNG", "assertions": []}},
  {{"text": "bệnh trào ngược dạ dày - thực quản", "type": "CHẨN_ĐOÁN", "assertions": []}},
  {{"text": "Chlorpheniramine 0.4 MG/ML", "type": "THUỐC", "assertions": ["isHistorical"]}},
  {{"text": "Capsaicin 0.38 MG/ML", "type": "THUỐC", "assertions": ["isHistorical"]}},
  {{"text": "WBC", "type": "TÊN_XÉT_NGHIỆM"}},
  {{"text": "14,43", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}},
  {{"text": "NEUT% (Tỷ lệ % bạch cầu trung tính)", "type": "TÊN_XÉT_NGHIỆM"}},
  {{"text": "76,4", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}},
  {{"text": "LYPH% (Tỷ lệ bạch cầu lympho)", "type": "TÊN_XÉT_NGHIỆM"}},
  {{"text": "12,8", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}}
]}}

Note: the diagnosis is current (`[]`) while the drugs are `isHistorical` (`có tiền sử sử dụng`). Test names and results have NO `assertions` field at all. The test name is `WBC`, exactly as written in the input - do not expand it.

### Example 3 - negation, family history, spelling errors preserved

Input:
Bệnh nhân dã dùng thuốc từ tuần trước, hiện tại khám thấy đau châ nhẹ, không ho, chưa ra huyết âm đạo (VB -). Tiền sử: bệnh nhân có tăng huyết áp. Gia đình có bác ruột bị sốt chưa rõ nguyên nhân. Hiện được kê acetaminophen 500mg po bid để giảm đau.

Output:
{{"entities": [
  {{"text": "đau châ", "type": "TRIỆU_CHỨNG", "assertions": []}},
  {{"text": "ho", "type": "TRIỆU_CHỨNG", "assertions": ["isNegated"]}},
  {{"text": "ra huyết âm đạo", "type": "TRIỆU_CHỨNG", "assertions": ["isNegated"]}},
  {{"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "assertions": ["isHistorical"]}},
  {{"text": "sốt", "type": "TRIỆU_CHỨNG", "assertions": ["isFamily"]}},
  {{"text": "acetaminophen 500mg po bid", "type": "THUỐC", "assertions": []}},
  {{"text": "giảm đau", "type": "TRIỆU_CHỨNG", "assertions": []}}
]}}

Note: `đau châ` is misspelled in the input (`châ` not `chân`) and is copied EXACTLY as written - never corrected. The negation words `không` and `chưa` are not part of the concept text; extract the concept and mark it `isNegated`. `acetaminophen 500mg po bid` is `[]`, not `isHistorical`, because it is being prescribed now.

### Example 4 - minimal spans: strip framing, keep one concept whole

Input:
Chào bác sĩ, mẹ em có tiền sử cá nhân có khối u lành tính ở cổ. Gần đây bện nhân thấy mệt mỏi, đi khám phát hiện bị tăng huyết áp. Bệnh nhân nhập viện vì đau bụng thượng vị dữ dội liên lục, hoàn toàn không ợ hơi được. Chỉ số kiểm tra hơi thở UBT: 36,69. Chẩn đoán xác định bện nhân bị trào ngược dạ dày thực quản với viêm thực quản. Gia đình em có bác ruột dã dùng thuốc điều trị sốt chưa rõ nguyên nhân.

Output:
{{"entities": [
  {{"text": "khối u lành tính", "type": "TRIỆU_CHỨNG", "assertions": ["isHistorical"]}},
  {{"text": "mệt mỏi", "type": "TRIỆU_CHỨNG", "assertions": []}},
  {{"text": "tăng huyết áp", "type": "TRIỆU_CHỨNG", "assertions": []}},
  {{"text": "đau bụng", "type": "TRIỆU_CHỨNG", "assertions": []}},
  {{"text": "ợ hơi", "type": "TRIỆU_CHỨNG", "assertions": ["isNegated"]}},
  {{"text": "UBT", "type": "TÊN_XÉT_NGHIỆM"}},
  {{"text": "36,69", "type": "KẾT_QUẢ_XÉT_NGHIỆM"}},
  {{"text": "trào ngược dạ dày thực quản với viêm thực quản", "type": "CHẨN_ĐOÁN", "assertions": []}},
  {{"text": "sốt", "type": "TRIỆU_CHỨNG", "assertions": ["isFamily"]}}
]}}

Note - study every span here:
- `có tiền sử cá nhân có khối u lành tính` -> only `khối u lành tính`. The history framing became `isHistorical`.
- `bện nhân thấy mệt mỏi` -> only `mệt mỏi`. `đi khám phát hiện bị tăng huyết áp` -> only `tăng huyết áp`, and it stays `TRIỆU_CHỨNG` because it is a lay report, not a stated diagnosis.
- `đau bụng thượng vị dữ dội liên lục` -> only `đau bụng`. Location, severity and duration are all stripped.
- `Chỉ số kiểm tra hơi thở UBT` -> only `UBT`. The measurement method is not part of the test name.
- `trào ngược dạ dày thực quản với viêm thực quản` is kept WHOLE - it is one diagnosis under `Chẩn đoán xác định`, not two. Do not split it.
- `Gia đình em có bác ruột dã dùng thuốc điều trị sốt chưa rõ nguyên nhân` -> only `sốt`, marked `isFamily`. `chưa rõ nguyên nhân` is stripped and the type is `TRIỆU_CHỨNG`, not `CHẨN_ĐOÁN`: an unknown cause is the absence of a diagnosis, not a diagnosis.
- Nothing was emitted for `Chào bác sĩ`, `mẹ em`, `Bệnh nhân nhập viện vì` - they name no medical concept."""

PROMPT_PLACEHOLDER = "" #các biến placeholder tạm khác


def prompt_is_filled() -> bool:
    """Prompt trống sẽ throw lỗi"""
    return (
        SYSTEM_PROMPT.strip() != PROMPT_PLACEHOLDER
        and FEW_SHOT_EXAMPLES.strip() != PROMPT_PLACEHOLDER
    )


def build_prompt() -> ChatPromptTemplate:
    """Biến duy nhất: `{input_text}` = văn bản THÔ, chưa đụng vào.

    Không chuẩn hoá input trước khi đưa vào prompt: `position` trả về phải trỏ
    vào chính chuỗi gốc
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT + "\n\n" + FEW_SHOT_EXAMPLES),
            ("human", "{input_text}"),
        ]
    )
