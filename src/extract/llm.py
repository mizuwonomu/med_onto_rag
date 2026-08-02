"""
Client nối vào llama-server tự host (OpenAI-compatible endpoint).

VÌ SAO DÙNG ChatOpenAI CHO MỘT MODEL LOCAL
------------------------------------------
llama-server phơi ra đúng `/v1/chat/completions` của OpenAI, kể cả
`response_format: json_schema`. Dùng client chuẩn nghĩa là toàn bộ tầng này
không biết gì về llama.cpp - đổi sang vLLM/TGI/API thật chỉ là đổi base_url.
`api_key="dummy"` vì server không xác thực; `model="local"` vì nó phục vụ đúng
một model đã nạp sẵn, trường này bị bỏ qua.

CẢNH BÁO FAIL-OPEN (llama.cpp #19051)
-------------------------------------
Khi llama-server dịch JSON Schema sang GBNF thất bại, nó KHÔNG trả lỗi: nó log
`failed to parse grammar` rồi sinh text TỰ DO và vẫn trả HTTP 200. Nghĩa là
"constrained decoding" có thể đã tắt im lặng mà client không hề biết.

Lưới an toàn cuối cùng là Pydantic trong `with_structured_output` - output tự do
gần như chắc chắn trượt validation. Nhưng "gần như" không phải "chắc chắn", nên
mỗi lần sửa `schema.py` PHẢI grep log server tìm `failed to parse grammar` trước
khi tin vào bất kỳ con số nào đo được.
"""

from __future__ import annotations

from langchain_core.runnables import Runnable

from configs.config import (
    NER_LLM_BASE_URL,
    NER_LLM_MAX_TOKENS,
    NER_LLM_TEMPERATURE,
    NER_LLM_REPEAT_PENALTY,
    NER_LLM_REPEAT_LAST_N
)
from extract.prompt import build_prompt, prompt_is_filled
from extract.schema import NerOutput


def build_ner_chain(
    base_url: str = NER_LLM_BASE_URL,
    temperature: float = NER_LLM_TEMPERATURE,
    max_tokens: int = NER_LLM_MAX_TOKENS,
    repeat_penalty: float = NER_LLM_REPEAT_PENALTY,
    repeat_last_n: int = NER_LLM_REPEAT_LAST_N
) -> Runnable:
    """prompt -> LLM ràng buộc grammar -> `NerOutput` đã validate.

    Chặn sớm nếu prompt còn là khung rỗng: chạy tiếp chỉ tạo ra một bảng điểm
    trông có vẻ hợp lệ nhưng đo đúng con số 0 của một prompt không tồn tại.
    """
    if not prompt_is_filled():
        raise RuntimeError(
            "PROMPT TRỐNG! "
        )

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        base_url=base_url,
        api_key="dummy",
        model="local",
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={
            # Lớp phòng thủ thứ hai cho thinking mode. Server đã có
            # `--reasoning off` (xem scripts/ner/serve_llm.sh), nhưng client tự
            # khẳng định lại để chạy đúng cả trên server do người khác dựng.
            # Thiếu nó, Qwen3.5 đốt trọn max_tokens vào reasoning_content rồi
            # chết với LengthFinishReasonError mà chưa viết được ký tự JSON nào.
            "chat_template_kwargs": {"enable_thinking": False},
            # Phá vòng lặp thoái hoá của greedy decoding. llama.cpp mặc định
            # repeat_penalty = 1.0 (TẮT), và với temperature = 0 thì model không
            # có đường nào thoát khi hai nhánh token gần hoà nhau: đo được cảnh
            # model sinh đúng 3 khái niệm rồi lặp `sốt cao -> ho -> không ho`
            # cho tới hết token. 1.05 đủ nhẹ để không méo JSON (dấu ngoặc, tên
            # field lặp lại là chuyện bình thường trong JSON nên phạt mạnh sẽ
            # phá cấu trúc), nhưng đủ để đẩy model ra khỏi điểm kẹt.
            "repeat_penalty": repeat_penalty,
            # Cửa sổ phạt: chỉ nhìn lại 256 token gần nhất. Rộng hơn thì các
            # khái niệm lặp lại HỢP LỆ ở đầu document (cùng một thuốc nhắc nhiều
            # lần) cũng bị phạt oan.
            "repeat_last_n": repeat_last_n,
        },
    )
    return build_prompt() | llm.with_structured_output(NerOutput, method="json_schema")
