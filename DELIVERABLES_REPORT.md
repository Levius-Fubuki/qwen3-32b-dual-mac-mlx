# Qwen3-32B 双 Mac 优化项目产出报告

- 交付日期：2026-09-01
- 交付分支：`feature/qwen3-32b-dual-mac`
- 交付性质：技术实验产物，不是已上线生产服务

## 公开发布地址

- GitHub 项目仓库：<https://github.com/Levius-Fubuki/qwen3-32b-dual-mac-mlx>
- Hugging Face 4-bit rank packs：<https://huggingface.co/levius-f/Qwen3-32B-4bit-Dual-Mac-MLX>
- Hugging Face 3-bit rank packs：<https://huggingface.co/levius-f/Qwen3-32B-3bit-Dual-Mac-MLX>

Hugging Face 仅发布双机运行所需的最终 rank-local packs；未重复上传上游完整原始模型。

## 1. 交付概览

本次交付保留两个最终模型方案、一套 Qwen3 流水线适配器、一套无反量化 rank pack 工具、最终配置、benchmark runner 和复现文档。

| 交付项 | 状态 | 说明 |
|---|---|---|
| 4-bit 2K 双机方案 | 已产出 | 历史完整通过，清理后动态复验待重启 |
| 3-bit 8K 双机方案 | 已产出 | 完成一次 8188+4 实机运行 |
| Qwen3 pipeline adapter | 已产出 | 支持 weighted split、KV 依赖和长上下文算子拆分 |
| Rank-local packer | 已产出 | raw-byte、layer-aligned、确定性、带 manifest |
| Benchmark runner | 已产出 | 支持 prompt/generation/prefill/KV 参数 |
| 最终配置和运行说明 | 已产出 | 只保留两个最终 Profile |
| 生产 API 服务 | 未产出 | 不在当前最终文件中 |
| 正式多轮 acceptance report | 未产出 | 尚未执行 7936+256 × 3 |

## 2. 代码产出

### 2.1 仓库状态

- 分支：`feature/qwen3-32b-dual-mac`
- 长上下文优化检查点：`557e672`
- 清理检查点：`a8cb1ae`
- GitHub 公开仓库的 `main` 分支由本功能分支发布。

### 2.2 核心文件

| 文件 | 用途 |
|---|---|
| `src/qwen32_cluster/qwen3_pipeline.py` | 两级流水线、分层、KV-cache 依赖、Attention/MLP 拆分 |
| `src/qwen32_cluster/rank_pack.py` | rank-local 权重规划、打包、manifest 和事务恢复 |
| `src/qwen32_cluster/safetensor_raw.py` | safetensor header 校验和 raw payload 搬运 |
| `src/qwen32_cluster/contracts.py` | 集群、Ring、端口和 JSON 契约 |
| `src/qwen32_cluster/profiles.py` | 最终 Profile 加载与校验 |
| `experiments/qwen32_pipeline_benchmark.py` | 双机 prompt/decode benchmark |
| `config/cluster.json` | 最终 M3/M4 地址和 Ring 配置 |
| `config/profiles.json` | `final-4bit-2k`、`final-3bit-8k` |
| `README.md` | 最短运行说明 |
| `EXPERIMENT_REPORT.md` | 完整实验复盘 |
| `DELIVERABLES_REPORT.md` | 本交付清单 |

测试文件根据最终清理要求已删除。需要审查或恢复时，可从 `557e672` 取回当时 297 项通过的测试树。

## 3. 模型产出

### 3.1 最终统一别名

相同别名在两台机器上指向各自的 rank-local 模型：

```text
/Users/Shared/mlx-cluster/models/Qwen3-32B-4bit-final
/Users/Shared/mlx-cluster/models/Qwen3-32B-3bit-final
```

### 3.2 MacBook / Rank 0

| 资产 | 路径 | 逻辑大小/用途 |
|---|---|---|
| 原始 4-bit | `/Users/Shared/mlx-cluster/models/Qwen3-32B-4bit` | 约 17GB，重打包源 |
| 原始 3-bit | `/Users/Shared/mlx-cluster/models/Qwen3-32B-3bit` | 约 13GB，重打包源 |
| 4-bit rank0 | `Qwen3-32B-pipeline-4bit-balanced-barrier4-rank0` | 9,652,414,464 payload bytes，层 32–63 |
| 3-bit rank0 | `Qwen3-32B-pipeline-3bit-performance-mlpsplit-rank0` | 5,800,859,648 payload bytes，层 40–63 |

### 3.3 Mac mini / Rank 1

| 资产 | 路径 | 逻辑大小/用途 |
|---|---|---|
| 4-bit rank1 | `Qwen3-32B-pipeline-4bit-balanced-barrier4-rank1` | 9,652,414,464 payload bytes，层 0–31 |
| 3-bit rank1 | `Qwen3-32B-pipeline-3bit-performance-mlpsplit-rank1` | 9,214,310,400 payload bytes，层 0–39 |

Mac mini 不再保留完整原始 32B 模型，只保留运行所需的两个 rank1 pack。

## 4. 产物标识与完整性

### 4.1 4-bit

| 项目 | Rank 0 | Rank 1 |
|---|---|---|
| 层范围 | `[32,64)` | `[0,32)` |
| Plan ID | `f963cf92a98c9d14bbbb1062f7994767c02adfb794eef3b837d16a385b1a4faa` | `1ba40427b404bf77de79acd354bfe6012368a8b0ff17d333a6e6bd934eaf8ecb` |
| Adapter SHA-256 | `ebc1278e2d25ba6dcc4ffda106d3ebab1851e6c65214aaa1be48b572e06fd64b` | 同 Rank 0 |

### 4.2 3-bit

| 项目 | Rank 0 | Rank 1 |
|---|---|---|
| 层范围 | `[40,64)` | `[0,40)` |
| Plan ID | `44cc87a6efdc738f471567abc05915865d4a4e6acae65658ecad49e25a7552ef` | `4baace15489e999cbe7618d929632d0093ac74eb4ccb05469169f878b13377f6` |
| Adapter SHA-256 | `f3c4c8a1eb8b76d16fb196d5b0b214d69885883105fe347037a760e59f96e83a` | 同 Rank 0 |

清理后静态检查结果：四个最终 pack 的 index 引用 shard 均存在，`config.json`、adapter 和 rank manifest 均完整，同方案两端 adapter hash 一致。

## 5. 性能产出

| Profile | 总上下文 | Prompt tok/s | Decode tok/s | Rank 0 峰值 | 证据等级 |
|---|---:|---:|---:|---:|---|
| `final-4bit-2k` | 2,048 | 64.29 | 6.18 | 10.01GB | warm-up + trial 历史通过 |
| `final-3bit-8k` | 8,192 | 46.45 | 4.35 | 6.65GB | 单次 8188+4 可行性通过 |

4-bit 没有通过 4K；3-bit 是当前唯一有 8K 成功记录的方案。

## 6. 运行环境产出

| 项目 | 最终值 |
|---|---|
| Python | `/Users/Shared/mlx-cluster/.venv/bin/python` |
| mlx.launch | `/Users/Shared/mlx-cluster/.venv/bin/mlx.launch` |
| Hostfile | `/Users/Shared/mlx-cluster/hosts.json` |
| Benchmark | `/Users/Shared/mlx-cluster/qwen32_pipeline_benchmark.py` |
| Backend | Ring |
| Connections/IP | 1 |
| Ports | 33323–33324 |
| M3 IP | `169.254.252.127` |
| M4 IP | `169.254.82.82` |

## 7. 复现命令

### 7.1 4-bit 2K

```bash
/Users/Shared/mlx-cluster/.venv/bin/mlx.launch \
  --backend ring \
  --hostfile /Users/Shared/mlx-cluster/hosts.json \
  --connections-per-ip 1 \
  --starting-port 33323 -- \
  /Users/Shared/mlx-cluster/.venv/bin/python \
  /Users/Shared/mlx-cluster/qwen32_pipeline_benchmark.py \
  --model /Users/Shared/mlx-cluster/models/Qwen3-32B-4bit-final \
  --prompt-tokens 2044 \
  --generation-tokens 4 \
  --prefill-step-size 64
```

### 7.2 3-bit 8K

```bash
/Users/Shared/mlx-cluster/.venv/bin/mlx.launch \
  --backend ring \
  --hostfile /Users/Shared/mlx-cluster/hosts.json \
  --connections-per-ip 1 \
  --starting-port 33323 -- \
  /Users/Shared/mlx-cluster/.venv/bin/python \
  /Users/Shared/mlx-cluster/qwen32_pipeline_benchmark.py \
  --model /Users/Shared/mlx-cluster/models/Qwen3-32B-3bit-final \
  --prompt-tokens 8188 \
  --generation-tokens 4 \
  --prefill-step-size 32
```

## 8. 清理产出

### 8.1 已删除

- 两端 Qwen3-14B-4bit；
- `com.codex.mlx-cluster` LaunchAgent；
- 14B 启动脚本和日志；
- 全部被替代的 balanced/adaptive/cachebarrier/attnsplit rank pack；
- MacBook 上所有冗余 rank1 pack；
- 项目测试文件、pytest 配置、缓存、临时 `.venv`；
- 过期设计/计划草案。

这些删除不可从文件系统直接恢复；仓库内历史代码和测试可从 Git 历史恢复，模型需要重新下载或重新打包。

### 8.2 磁盘结果

| 节点 | 清理后可用空间 |
|---|---:|
| MacBook | 约 242GiB |
| Mac mini | 约 97GiB |

## 9. 当前运行状态

- 没有 14B 常驻服务；
- 没有 32B 常驻服务；
- 旧 LaunchAgent 已卸载并删除；
- 8080、33323、33324 没有最终服务占用；
- 最终模型文件处于静态保留状态。

清理后的 4-bit 短 canary 在当前 Mac mini boot session 触发 Metal timeout。该 pack 的静态完整性正常，且此前有 2K 成功记录；3-bit 未在该 timeout 后继续加载。动态交付确认需在两台 Mac 重启后进行。

## 10. 未交付项

| 项目 | 状态 | 完成条件 |
|---|---|---|
| 重启后 canary | 未完成 | 两端重启，4-bit/3-bit 各跑一次短请求 |
| 正式 8K acceptance | 未完成 | 7936+256，warm-up + 3 次 measured |
| Rank 1 内存和 swap 报告 | 未完成 | 两端同步采样并写报告 |
| 8K API | 未完成 | 真实 tokenizer/chat template、SSE、8192/8193 边界 |
| 并发保护 | 未完成 | 单活请求与第二请求 429 |
| 常驻服务 | 未完成 | 受控启动、健康检查、停止 |
| Promotion/rollback | 未完成 | 可验证的版本切换和回滚演练 |
| 3-bit 质量评估 | 未完成 | 与 4-bit 做固定题集对照 |

## 11. 建议的验收顺序

1. 重启两台 Mac，确认 Thunderbolt IP 和端口状态。
2. 执行 3-bit 短 canary；若出现 Metal timeout，立即停止后续模型加载。
3. 执行 4-bit 2K canary，确认历史基线可复现。
4. 对 3-bit 执行一次 7936+256 warm-up 和三次 measured run。
5. 同时记录两端 MLX peak、memory pressure 和 swap delta。
6. 达到三次中位数 `>=4 tok/s` 后，再建设 API 和常驻服务。
7. 最后决定是否把 `feature/qwen3-32b-dual-mac` 合并到主分支。
