#!/usr/bin/env bash
# ik llama-server: exps=CPU, IQ4KS/IQ2K hybrid — see header comments in repo doc
set -euo pipefail
# shellcheck source=_env.sh
source "$(dirname "$0")/_env.sh"
MODEL="${ORNITH_HYBRID_GGUF:-${ORNITH_ROOT}/ornith-1.0-35b-IQ4KS-IQ2K-R4-hybrid.gguf}"
SERVER="${IK_SERVER:-${IK_ROOT}/build-ser-cuda/bin/llama-server}"
exec "${SERVER}" \
  -m "${MODEL}" --ctx-size 256000 -ngl 999 -ot "exps=CPU" \
  -fa on -fdn 512 -khad -vhad -b 2048 -ub 2048 -tb 16 -t 16 \
  -ctk q4_0 -ctv q4_0 "$@"