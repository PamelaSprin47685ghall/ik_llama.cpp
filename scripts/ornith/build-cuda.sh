#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=========================================="
echo "  Build C++ Inference Server with CUDA"
echo "  Target: build-ser-cuda"
echo "=========================================="

cd "${IK_ROOT}"

# 清理旧的 CMake 缓存
echo "[1/2] 清除旧构建缓存..."
rm -rf build-ser-cuda/CMakeCache.txt build-ser-cuda/CMakeFiles

# 配置并进行 32 线程并行编译
echo "[2/2] 重新配置并编译项目..."
cmake -B build-ser-cuda -S . -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -C build-ser-cuda -j32

echo "=========================================="
echo "  编译完成！"
echo "  可执行文件位于: build-ser-cuda/bin/llama-server"
echo "=========================================="
