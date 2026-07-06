#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ORNITH_ROOT="$(cd "${IK_ROOT}/.." && pwd)"
QUANT_BIN="${IK_ROOT}/build-ser-cuda/bin/llama-quantize"

INP="${ORNITH_ROOT}/weights/ornith-1.0-35b-Q6_K-MTP-final.gguf"
OUT="${ORNITH_ROOT}/weights/ornith-1.0-35b-Q4_0-IQ2K-R4-hybrid-hadamard.gguf"

if [ ! -f "${INP}" ]; then
    # 兼容回退检查不带 final 后缀的文件
    if [ -f "${ORNITH_ROOT}/weights/ornith-1.0-35b-Q6_K.gguf" ]; then
        INP="${ORNITH_ROOT}/weights/ornith-1.0-35b-Q6_K.gguf"
    else
        echo "错误: 未在以下路径找到 Q6_K 输入权重: ${INP}"
        echo "请先运行 ./download_weights.sh 下载原始权重。"
        exit 1
    fi
fi

if [ ! -f "${QUANT_BIN}" ]; then
    echo "错误: 未找到 llama-quantize 二进制文件。请先运行 ./build-cuda.sh 编译项目。"
    exit 1
fi

echo "=========================================="
echo "  Convert Weights to Q4_0_HADAMARD + IQ2_K_R4 (Hybrid)"
echo "  输入: ${INP}"
echo "  输出: ${OUT}"
echo "  线程数: 32"
echo "=========================================="

T_EXP="iq2_k_r4"
CQ="blk\\..*\\.ffn_gate_exps=${T_EXP},blk\\..*\\.ffn_up_exps=${T_EXP},blk\\..*\\.ffn_down_exps=${T_EXP}"

exec "${QUANT_BIN}" --allow-requantize \
  --output-tensor-type q4_0_hadamard \
  --token-embedding-type q8_0 \
  --attn-k-type q4_0_hadamard \
  --attn-v-type q4_0_hadamard \
  --custom-q "$CQ" \
  "$INP" "$OUT" Q4_0_HADAMARD 32
