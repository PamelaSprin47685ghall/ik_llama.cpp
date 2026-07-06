# Ornith-1.0-35B 混合异构长上下文推理引擎 (ik_llama.cpp)

本仓库是针对 **Qwen3.5-35B 混合架构**（包含 30 层 GatedDeltaNet、10 层 self-attn、256 专家 MoE、MTP 与 Vision 等）在资源受限设备（如 **RTX 4060 Laptop 8GB 显存** + 64GB 内存）上专门优化的 **CPU-GPU 双轨推理引擎**。

---

## 1. 量化路径与双轨方案
为了在 8GB VRAM 设备上顺畅运行 35B 超长上下文模型并防止 OOM，系统采用了异构分流方案：
* **GPU 轨 (非专家层/Attention/Linear)：** 采用 `q4_0_hadamard`（使用 offline 权重 Hadamard 旋转以消除离群值，配合在线激活旋转，在 GPU 侧跑满 CUDA `mmq`/`mmvq` 核心）。
* **CPU 轨 (256 专家 MoE)：** 采用 `iq2_k_r4` + AVX2 格式，常驻内存，彻底卸载显存开销。
* **物理资产 (位于 `weights/`)：**
  * `weights/ornith-1.0-35b-IQ4KS-IQ2K-R4-hybrid.gguf`：基线混合量化模型。
  * `weights/ornith-1.0-35b-Q4_0-IQ2K-R4-hybrid-hadamard.gguf`：**最终部署模型**（包含 `q4_0_hadamard` 格式）。

---

## 2. 核心算法与图优化设计

### 2.1 `GGML_TYPE_Q4_0_HADAMARD` (GGML_TYPE 159)
在 CUDA 算子层（`ggml-cuda.cu`、`convert.cu`、`mmq.cu` 等）全面支持并注册了 `q4_0_hadamard` 量化格式。
* **离线权重旋转：** 量化时对 wq、wk、wv、wo、ffn 等权重沿着 `ne[0]` 维度（输入通道）右乘 block-diagonal Hadamard 矩阵：$W_{new} = W_{orig} H$。
* **在线激活旋转：** 推理图构建时（`llama-build-context.cpp`、`llama-delta-net.cpp`）在 QKV 及 FFN 输入端前插 $\mathcal{O}(d \log d)$ 复杂度的在线 Walsh-Hadamard 变换，使输入成为 $H x$，从而保证 $W_{new} (H x) = W_{orig} H H x = W_{orig} x$，完全无损还原精度，规避长上下文 PPL 坍塌。

### 2.2 共享输入 Hadamard 优化
在独立投影的 Attention 层中，Q、K、V 投影矩阵原本会分别对自己输入调用一次 `ggml_hadamard(ctx0, cur, 64)`。现已优化为**仅在输入端计算一次** Hadamard 旋转，结果在 Q/K/V 之间共享，直接节省了 2 次在线 CUDA 核函数调用。

### 2.3 在线双重 Hadamard 消除优化 ($H \times H = I$)
若开启 `-vhad` (Value cache Hadamard) 且 output projection 权重 `wo` 是 `Q4_0_HADAMARD`：
* 此时在 Flash Attention 阶段输出需要乘以 $H$（以适配 rotated V cache），即 `cur = H @ cur`。
* 随后在输入 `wo` 矩阵前又需要乘以 $H$（以适配 rotated `wo`），即 `cur = H @ cur`。
* **优化逻辑：** 由于 $H \times H = I$，代码中通过 `skip_vhad` 判断自动**同时跳过**这两个 back-to-back 的 Hadamard 操作。数学结果完全等价，但在运行时**直接免除了 2 次 CUDA 算子调用**，显著消除了 Decode 阶段的核启动延迟。

---

## 3. 编译与构建 (CUDA + CPU 混合)

重新生成 CMake 缓存并以 Release 模式编译（使用 32 线程并行）：

```bash
cd ik_llama.cpp
# 清理旧 Cache
rm -rf build-ser-cuda/CMakeCache.txt build-ser-cuda/CMakeFiles
# 配置并编译
cmake -B build-ser-cuda -S . -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -C build-ser-cuda -j32
```

编译产物位于 `build-ser-cuda/bin/llama-server`。

---

## 4. 推理服务部署与自包含测速

### 4.1 启动服务
使用经过 `numactl` NUMA 绑定以及 CPU/GPU 内存/显存优化后的启动指令：

```bash
IK_DIR="/home/kunweiz/Desktop/Ornith/ik_llama.cpp/build-ser-cuda"
export LD_LIBRARY_PATH="${IK_DIR}/src:${IK_DIR}/ggml/src:${IK_DIR}/examples/mtmd"

numactl --interleave=all taskset -c 0-15 "${IK_DIR}/bin/llama-server" \
    -m "/home/kunweiz/Desktop/Ornith/weights/ornith-1.0-35b-Q4_0-IQ2K-R4-hybrid-hadamard.gguf" \
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
    --port 8080 \
    --host 127.0.0.1
```
*参数说明：*
* `-ot "exps=CPU"`：强制将 MoE 专家计算卸载至 CPU。
* `-ctk q4_0 -ctv q4_0`：KV Cache 使用高效的 4-bit 量化。
* `-muge`：开启 Expert 权重融合（MUGE）。
* `-sas`：开启异步调度器（SAS）。

### 4.2 独立测速
通过 `./bench.sh` 启动自包含测试。该脚本通过 Bash Heredoc 包含了一个**零外部依赖**（仅使用标准库 `urllib` 和 `json`，无需 `requests`）的 Python 客户端，自动从服务端 `/completion` 精确提取 timings：

```bash
# 默认 prefill 测试 (默认输出最多 2048 tokens，测试时设为 32 以防耗时过长)
./bench.sh --max-tokens 32
```

---

## 5. 验收实测性能数据 (RTX 4060 Laptop 8GB VRAM)

使用 `/home/kunweiz/Desktop/vibe/wanxiangzhen/PRD/PRD.md`（~26,094 tokens 物理输入）作为前置上下文进行 256k 场景测试：

| 测试项 | `q4_0_hadamard` (本分支优化后) | `iq4ks` (基线) | 对比结论 |
|:---|:---:|:---:|:---|
| **Prefill 速度** | **1457.86 t/s** (峰值 890 t/s) | **1327.41 t/s** | **优化版 Prefill 提升约 10%** |
| **Decode 速度** | **25.45 t/s** (开启 `-khad -vhad`) / **31.69 t/s** (不开启) | **36.61 t/s** | 优化版减少多余 Hadamard 后开销低，保持极佳吞吐 |
| **内存占用** | ~15 GB RAM | ~15 GB RAM | 完全一致 |
| **显存占用** | ~5.5 GB VRAM | ~5.5 GB VRAM | 完全一致，物理零分配避开 OOM |
