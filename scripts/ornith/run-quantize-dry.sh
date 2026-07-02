#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=_env.sh
source "$(dirname "$0")/_env.sh"
cd "${ORNITH_ROOT}"
INP="${1:-${ORNITH_ROOT}/ornith-1.0-35b-Q6_K-MTP-final.gguf}"
OUT=/tmp/ornith-hybrid-dry.gguf
T=iq2_k_r4
CQ="blk\\..*\\.ffn_gate_exps=${T},blk\\..*\\.ffn_up_exps=${T},blk\\..*\\.ffn_down_exps=${T}"
exec "${QUANT_BIN}" --dry-run --allow-requantize \
  --output-tensor-type q6_k --token-embedding-type q8_0 --custom-q "$CQ" \
  "$INP" "$OUT" IQ4_KS_R4 2>&1 | head -30