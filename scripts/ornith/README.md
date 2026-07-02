# Ornith + ik_llama.cpp

`ORNITH_ROOT` = 含 `ktransformers/` 与 `ik_llama.cpp/` 的工作区（默认 `ik_llama.cpp/..`）。

| 脚本 | 作用 |
|------|------|
| `run-quantize-dry.sh` | hybrid 量化 dry-run |
| `run-quantize-hybrid.sh` | IQ4KS_R4 + exps IQ2_K_R4 |
| `run-llama-server-hybrid.sh` | `-ot exps=CPU` 起服 |

需先 `cmake --build build-ser-cuda`。调参笔记见 `docs/ornith-ik-hybrid-tuning.md`。