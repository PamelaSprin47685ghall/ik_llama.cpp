# ik hybrid 推理调参（Ornith 35B MoE）

硬件示例：8GB VRAM + 16 物理核。量化：attn/SSM/shared=IQ4_KS_R4，routed exps=IQ2_K_R4。

## 分工

- `-ngl 999`：层在 GPU 注册
- `-ot exps=CPU`：路由专家在 CPU RAM

## 实测（2026-07）

- **PP**：GPU FP16 算力瓶颈；`-ub 4096` ≈1272 tok/s
- **TG**：CPU 带宽瓶颈；`-t 16`（勿 32）

## 无效优化（勿重复）

- `-ser` 砍 PP 专家、全模型 IQ2、双模型 PP、本 build fp8

## 更快路线

更大显存把 experts 上 GPU；或支持 FP8 attention 的引擎。