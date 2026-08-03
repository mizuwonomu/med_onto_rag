"""
Cấu hình các tham số cho parser, retriever,... 
"""

from pathlib import Path

ICD_PDF_PATH = Path("data/raw/ICD_10.pdf")
ICD_OUTPUT_PATH = Path("data/processed/icd_concepts.jsonl")

#độ rộng mỗi bucket histogram (điểm PDF), lấy quy ước inch sang point (pt) khi dò khoảng chia cột theo tọa độ x0
#trong pdf, a4 của pdf xấp xỉ 595 point ngang x 824 point dọc
#chia trục ngang thành các ô rộng 10 point mỗi ô -> tại mỗi ô, đếm có bao nhiêu từ bắt đầu trong đó
#nếu một ô nhiều tù = có cột chữ, ô nào trống = khe giữa 2 cột
COLUMN_BUCKET_WIDTH = 10.0

# Ngưỡng "dày" (dense) của một bucket: bucket có số từ <= max_count * ratio bị coi
# là khoảng trống. Đặt thấp để bỏ qua nhiễu lác đác (chữ tràn dòng, tham chiếu chéo)
COLUMN_GAP_THRESHOLD_RATIO = 0.05 #tính trên 5% của mật độ từ đông nhất

# Chiều cao vùng footer (điểm PDF) cắt bỏ từ đáy trang để loại số trang "- N -"
# vốn nằm đè lên khe giữa hai cột
FOOTER_MARGIN = 45.0

# Vector index (index/)
ICD_JSONL_PATH = ICD_OUTPUT_PATH  # semantic alias for the read side
RXNORM_CONCEPTS_PATH = Path("data/processed/rxnorm_concepts.jsonl")
RXNORM_ALIASES_PATH = Path("data/processed/rxnorm_aliases.jsonl")
ICD_DB_DIR = Path("data/index/icd_db")
RXNORM_DB_DIR = Path("data/index/rxnorm_db")
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBED_BATCH_SIZE = 64  # GPU encode batch
CHROMA_ADD_BATCH = 1024  # add_documents batch into Chroma

# Retrieval layer (retrieval/)
BM25_ICD_DIR = Path("data/index/bm25_icd")
BM25_RXNORM_DIR = Path("data/index/bm25_rxnorm")
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_BATCH_SIZE = 8
# k mỗi nhánh trước khi hợp nhất (union) rồi đưa toàn bộ qua reranker
TOP_K_DENSE = 30
TOP_K_BM25 = 30
SYNTHETIC_EVAL_PATH = Path("data/test/rxnorm_icd_synthetic_400-labeled.jsonl")
EVAL_OUTPUT_DIR = Path("data/eval/retrieval")

# Ngưỡng floor/margin + biến thể query, TÁCH RIÊNG THEO KB.
#
# Tách theo KB không phải để cho gọn: điểm reranker của hai KB không nằm trên
# cùng một thang (mô tả chẩn đoán dài vs tên hoạt chất ngắn), nên một cặp ngưỡng
# chung luôn phải hy sinh một bên. Biến thể query cũng hoá ra phải tách nốt - đo
# trên 399 dòng cho thấy mỗi KB thắng ở một biến thể khác nhau.
#
# CHỈ có nghĩa với reranker + instruction hiện tại (retrieval/reranker.py). Đổi
# model hoặc sửa một chữ trong instruction là đổi thang điểm -> phải sweep lại.
#
# PHẠM VI ĐO: toàn bộ các số dưới đây hiệu chỉnh trên 399 dòng SYNTHETIC do LLM
# sinh (data/test/, xem SYNTHETIC_EVAL_PATH), CHƯA hề thấy bệnh án thật và chưa
# nối với NER. Mention thật sẽ có lỗi biên, nhiều khái niệm trong một span, và
# phân bố khác - đó là những thứ tập synthetic không mô phỏng được. Coi đây là
# điểm khởi đầu có căn cứ, không phải ngưỡng đã nghiệm thu.
#
# rxnorm: CHỐT. margin=0 tái lập qua 2 vòng đo độc lập, và nó đúng vì lý do CẤU
#   TRÚC chứ không phải thống kê: một alias doc mở thẳng ra trọn bộ RXCUI qua
#   resolve_in, nên doc hạng 1 đã phủ hết gold (multi_solving recall = 1.000).
#   floor hội tụ 2.021 -> 2.157 giữa hai vòng, bootstrap 300/300 lần ra cùng một
#   giá trị, chênh lệch sweep-dev +0.0004. Biến thể strip_dose thắng vì liều/
#   đường dùng/tần suất ("625mg", "po", "bid") không hề có trong text KB ở mức
#   ingredient - giữ lại chỉ làm nhiễu.
#
# icd: TẠM. Nút thắt là RERANKER, không phải cỡ mẫu - thêm data synthetic sẽ chỉ
#   cho một con số floor chính xác hơn về cùng một hiệu năng kém. Bằng chứng:
#   distractor p75 (3.477) CAO HƠN gold p25 (3.156), tức hai phân bố chồng nhau ở
#   giữa thang chứ không phải ở đuôi; không floor nào cắt được vùng đó. Bootstrap
#   đã ổn định ([1.490, 1.890], vòng trước là [0.014, 3.807] khi chỉ có 10 cụm)
#   nên vấn đề còn lại không nằm ở việc chia tập. Hướng gỡ: reranker lớn hơn, hoặc
#   đưa tín hiệu block ICD vào text cấp cho reranker để tách sibling.
#   Biến thể raw thắng vì mention chẩn đoán không có liều - strip_dose chỉ có cơ
#   hội cắt nhầm (bootstrap floor nới ra [-0.906, 1.890]).
RETRIEVAL_THRESHOLDS = {
    "icd": {"floor": 1.490, "margin": 0.399, "query_variant": "raw", "status": "tentative"},
    "rxnorm": {"floor": 2.157, "margin": 0.0, "query_variant": "strip_dose", "status": "frozen"},
}

# NER extraction layer (extract/)
#
# Mô hình chạy tự host bằng llama.cpp (llama-server)
NER_LLM_HF_REPO = "unsloth/Qwen3.5-9B-GGUF:Q4_K_M"
NER_LLM_BASE_URL = "http://127.0.0.1:8080/v1"
NER_LLM_TEMPERATURE = 0.0  # trích xuất là tác vụ tất định, không sampling
NER_LLM_MAX_TOKENS = 3072 #max completion tokens
NER_LLM_REPEAT_PENALTY = 1.05 #penalty cho các token lặp lại
NER_LLM_REPEAT_LAST_N = 256 #số token nhìn lại 
# Trần số khái niệm mỗi document, ép ở tầng GRAMMAR (extract/schema.py dịch nó
# thành `maxItems` -> GBNF). Đây là thứ DUY NHẤT khiến grammar tự đóng mảng khi
# model kẹt trong vòng lặp thoái hoá - max_tokens chỉ giới hạn thiệt hại, không
# ngăn được vòng lặp.
#
# Số khái niệm TỐI ĐA cho phép sinh ra trong MỘT document. Đặt đủ rộng để không
# cắt cụt một bệnh án dài thật. Chạm trần là DẤU HIỆU BẤT THƯỜNG (nhiều khả năng
# đang lặp), không phải giới hạn cần nới - đọc output trước khi tăng.
MAX_ENTITIES_PER_DOC = 100
NER_SYNTHETIC_PATH = Path("data/test/ner/input_synthetic_10.jsonl")
NER_EVAL_OUTPUT_DIR = Path("data/eval/ner")

# Thư mục .txt của BTC - giữ nguyên cấu trúc `input/` bên trong file test.zip.
# Tên file quyết định tên file nộp: `1.txt` -> `1.json`.
NER_INPUT_DIR = Path("data/input/input")
NER_OUTPUT_DIR = Path("data/output/ner")
