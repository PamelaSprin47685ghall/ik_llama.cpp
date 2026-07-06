#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Ornith-1.0-35B 独立自包含测速脚本 (已移至 scripts/ornith/)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8080
MAX_TOKENS=2048

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)     PORT="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        *)          echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "  Ornith-1.0-35B 独立测速 (自包含)"
echo "  端口: ${PORT}"
echo "  Max Tokens: ${MAX_TOKENS}"
echo "=========================================="

# 杀掉残留服务
echo "[1/3] 清理残留..."
pkill -f "llama-server.*${PORT}" 2>/dev/null || true
pkill -f "sglang.launch_server.*${PORT}" 2>/dev/null || true
sleep 2

# 启动服务 (使用当前项目下编译的 ik_llama.cpp 与本地权重)
echo "[2/3] 启动 llama-server..."
IK_DIR="${SCRIPT_DIR}/../../build-ser-cuda"
export LD_LIBRARY_PATH="${IK_DIR}/src:${IK_DIR}/ggml/src:${IK_DIR}/examples/mtmd"
MODEL_NAME="ornith-1.0-35b-Q4_0-IQ2K-R4-hybrid-hadamard.gguf"

numactl --interleave=all taskset -c 0-15 "${IK_DIR}/bin/llama-server" \
    -m "${SCRIPT_DIR}/../../../weights/${MODEL_NAME}" \
    --ctx-size 256000 \
    -ngl 999 \
    -ot "exps=CPU" \
    -fa on \
    -fdn 512 \
    -b 3072 \
    -ub 3072 \
    -t 16 \
    -ctk q4_0 -ctv q4_0 \
    -muge \
    -sas \
    --numa numactl \
    --port "${PORT}" \
    --host 127.0.0.1 &
SERVER_PID=$!

# 等待服务就绪
echo "等待服务就绪 (最长 120s)..."
for i in $(seq 1 120); do
    if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "服务就绪 (${i}s)"
        break
    fi
    sleep 1
    if [ "$i" -eq 120 ]; then
        echo "错误: 服务启动超时"
        kill "${SERVER_PID}" 2>/dev/null || true
        exit 1
    fi
done

# 执行测速
echo ""
echo "[3/3] 开始自包含测速..."

python3 - "${PORT}" "${MAX_TOKENS}" << 'EOF'
import urllib.request
import urllib.error
import json
import sys
import time

def main():
    port = int(sys.argv[1])
    max_tokens = int(sys.argv[2])
    prd_path = "/home/kunweiz/Desktop/vibe/wanxiangzhen/PRD/PRD.md"
    
    # Load PRD
    try:
        with open(prd_path, "r", encoding="utf-8") as f:
            prd = f.read()
    except Exception as e:
        print(f"Error loading PRD: {e}")
        sys.exit(1)
        
    url = f"http://127.0.0.1:{port}/completion"
    
    # Construct payload
    payload = {
        "prompt": prd + "\n\n请用中文简要总结以上PRD的主要内容，用一句话概括：",
        "n_predict": max_tokens,
        "temperature": 0.2,
        "stream": True
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"发送请求到 {url} (max_tokens={max_tokens})...")
    t0 = time.time()
    content = ""
    res = {}
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            while True:
                line = response.readline()
                if not line:
                    break
                line_str = line.decode("utf-8", errors="replace").strip()
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        c = chunk.get("content", "")
                        content += c
                        print(c, end="", flush=True)
                        if "timings" in chunk:
                            res = chunk
                    except Exception:
                        pass
            print()
    except urllib.error.HTTPError as e:
        print(f"HTTP Error during request: {e.code} {e.reason}")
        try:
            print(e.read().decode("utf-8"))
        except Exception:
            pass
        sys.exit(1)
    except Exception as e:
        print(f"Error during request: {e}")
        sys.exit(1)
    t1 = time.time()
    
    # Get timings
    timings = res.get("timings", {}) if res else {}
    prompt_n = timings.get("prompt_n", 0)
    prompt_ms = timings.get("prompt_ms", 0.0)
    predicted_n = timings.get("predicted_n", 0)
    predicted_ms = timings.get("predicted_ms", 0.0)
    
    prefill_tps = (prompt_n / (prompt_ms / 1000.0)) if prompt_ms > 0 else 0
    decode_tps = (predicted_n / (predicted_ms / 1000.0)) if predicted_ms > 0 else 0
    
    # Print a small sample of content to show it works
    sample = content[:150].replace("\n", " ")
    print(f"[SAMPLE OUTPUT] {sample}...")
    
    print("=" * 60)
    print("  /completion 服务端精确计时 (自包含)")
    print("=" * 60)
    print(f"  输入 tokens    = {prompt_n}")
    print(f"  输入耗时       = {prompt_ms:.1f} ms ({prompt_ms / 1000.0:.2f} s)")
    print(f"  Prefill 速度   = {prefill_tps:.2f} t/s")
    print(f"  输出 tokens    = {predicted_n}")
    print(f"  输出耗时       = {predicted_ms:.1f} ms ({predicted_ms / 1000.0:.2f} s)")
    print(f"  Decode 速度    = {decode_tps:.2f} t/s")
    print(f"  客户端总耗时   = {t1 - t0:.2f} s")
    print("=" * 60)

if __name__ == "__main__":
    main()
EOF

# 清理
echo ""
echo "测速完成, 清理服务..."
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true
echo "完成。"
