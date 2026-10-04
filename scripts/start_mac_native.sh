#!/usr/bin/env bash
# Run cascade-llm natively on Apple Silicon. MLX/llama.cpp/Inferencer remain
# separate host processes; this script starts only the gateway.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This launcher targets macOS; use Docker Compose on Linux." >&2
  exit 2
fi
if [[ -z "${CASCADE_ADMIN_KEY:-}" ]]; then
  echo "Set CASCADE_ADMIN_KEY before starting (used by the admin API/UI)." >&2
  echo "  export CASCADE_ADMIN_KEY=\"$(openssl rand -hex 32)\"" >&2
  exit 2
fi

export CASCADE_HOST_GATEWAY=false
export CASCADE_DB="${CASCADE_DB:-$ROOT/data/cascade.db}"
mkdir -p "$(dirname "$CASCADE_DB")"

# OpenAI-compatible native backends. Override individual values in the shell
# for a different port/provider; nothing here downloads or starts model servers.
export INFERENCE_URL="${INFERENCE_URL:-http://localhost:8080}"
export LARGE_MLLM_URL="${LARGE_MLLM_URL:-http://localhost:8080/v1/chat/completions}"
export LARGE_TEXT_URL="${LARGE_TEXT_URL:-http://localhost:8080/v1/chat/completions}"
export SMALL_MLLM_URL="${SMALL_MLLM_URL:-http://localhost:8081/v1/chat/completions}"
export OCR_URL="${OCR_URL:-http://localhost:8081/v1/chat/completions}"
export MULTIMODAL_CAPABILITY_URL="${MULTIMODAL_CAPABILITY_URL:-http://localhost:8081/v1/models}"
export LARGE_MODEL_MULTIMODAL="${LARGE_MODEL_MULTIMODAL:-false}"
export IMAGE_GENERATION_URL="${IMAGE_GENERATION_URL:-http://localhost:8082/v1/images/generations}"
export EMBEDDINGS_URL="${EMBEDDINGS_URL:-http://localhost:8081}"
export RERANK_URL="${RERANK_URL:-${EMBEDDINGS_URL}}"

echo "Starting cascade-llm natively on http://127.0.0.1:3000"
echo "Native endpoints default to MLX :8080, auxiliary/vision :8081, sd.cpp :8082."
exec cargo run --release
