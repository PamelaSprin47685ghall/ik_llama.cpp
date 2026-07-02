#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=_env.sh
source "$(dirname "$0")/_env.sh"
cd "${ORNITH_ROOT}"
INP="${1:-${ORNITH_ROOT}/ornith-1.0-35b-Q6_K-MTP-final.gguf}"
OUT="${2:-${ORNITH_ROOT}/ornith-1.0-35b-IQ4KS-IQ2K-R4-hybrid.gguf}"
T=iq2_k_r4
CQ="blk\\..*\\.ffn_gate_exps=${T},blk\\..*\\.ffn_up_exps=${T},blk\\..*\\.ffn_down_exps=${T}"
exec "${QUANT_BIN}" --allow-requantize \
  --output-tensor-type q6_k --token-embedding-type q8_0 --custom-q "$CQ" \
  "$INP" "$OUT" IQ4_KS_R4 16