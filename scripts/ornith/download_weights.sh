#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ORNITH_ROOT="$(cd "${IK_ROOT}/.." && pwd)"

mkdir -p "${ORNITH_ROOT}/weights"

echo "=========================================="
echo "  Downloading Q6_K Base Weights from Hugging Face"
echo "  Repo: skinnyctax/Ornith-1.0-35B-Q6_K-Frankenstein-MTP-GGUF"
echo "  File: ornith-1.0-35b-Q6_K-MTP-final.gguf"
echo "=========================================="

huggingface-cli download skinnyctax/Ornith-1.0-35B-Q6_K-Frankenstein-MTP-GGUF \
  ornith-1.0-35b-Q6_K-MTP-final.gguf \
  --local-dir "${ORNITH_ROOT}/weights"

echo "=========================================="
echo "  下载完成！"
echo "  权重已存放在: weights/ornith-1.0-35b-Q6_K-MTP-final.gguf"
echo "=========================================="
