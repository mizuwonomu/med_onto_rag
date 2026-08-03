#!/usr/bin/env bash
# Khởi động llama-server cho tầng NER.
#
#   bash scripts/ner/serve_llm.sh                            # dùng NER_LLM_HF_REPO
#   bash scripts/ner/serve_llm.sh unsloth/other-GGUF:Q4_K_M  # ghi đè nhanh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}" #trường hợp build llama.cpp ngoài repo (ví dụ home/)

find_llama_server() {
    if command -v llama-server >/dev/null 2>&1; then
        command -v llama-server
        return 0
    fi
    for candidate in \
        "$LLAMA_CPP_DIR/build/bin/llama-server" \
        "$LLAMA_CPP_DIR/llama-server"; do
        if [[ -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

if ! LLAMA_SERVER="$(find_llama_server)"; then
    echo "LỖI: không tìm thấy llama-server." >&2
    echo "  đã dò: \$PATH, $LLAMA_CPP_DIR/build/bin/, $LLAMA_CPP_DIR/" >&2
    echo "  repo llama.cpp ở chỗ khác thì: LLAMA_CPP_DIR=/duong/dan bash $0" >&2
    exit 1
fi

MODEL_HF="${1:-$(cd "$REPO_ROOT" && python -c 'import sys; sys.path.insert(0, "src"); from configs.config import NER_LLM_HF_REPO; print(NER_LLM_HF_REPO)')}"


NGL="${LLAMA_NGL:-99}"
CTX="${LLAMA_CTX:-10132}"
PORT="${LLAMA_PORT:-8080}"

echo "binary : ${LLAMA_SERVER}"
echo "model  : ${MODEL_HF}"
echo "cấu hình: -ngl ${NGL}  -c ${CTX}  --parallel 1  --reasoning off  -> http://127.0.0.1:${PORT}"
echo "cache  : ${LLAMA_CACHE:-$HOME/.cache/llama.cpp} (nơi -hf tải GGUF về)"

if [[ "$PORT" != "8080" ]]; then
    echo "CẢNH BÁO: port khác 8080 - nhớ sửa NER_LLM_BASE_URL trong src/configs/config.py" >&2
fi

#--jinja: bắt buộc để server dùng chat template thật của model
#--reasoning off: Qwen3.5 là reasoning model, mặc định `auto` dò từ template -> bataj thinking
#-ngl 99: đẩy toàn bộ layer lên GPU
exec "$LLAMA_SERVER" -hf "$MODEL_HF" \
    --jinja \
    --reasoning off \
    --parallel 1 \
    -ngl "$NGL" \
    -c "$CTX" \
    --host 127.0.0.1 \
    --port "$PORT"