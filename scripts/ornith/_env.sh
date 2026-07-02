# shellcheck shell=bash
_ornith_ik_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IK_ROOT="$(cd -- "${_ornith_ik_script_dir}/../.." && pwd)"
ORNITH_ROOT="${ORNITH_ROOT:-$(cd -- "${IK_ROOT}/.." && pwd)}"
BUILD="${IK_BUILD:-${IK_ROOT}/build-ser-cuda}"
export LD_LIBRARY_PATH="${BUILD}/src:${BUILD}/ggml/src${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
QUANT_BIN="${BUILD}/bin/llama-quantize"