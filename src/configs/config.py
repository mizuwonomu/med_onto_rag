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
