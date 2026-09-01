# Qwen3-32B 双 16GB Mac 分布式推理优化实验报告

- 实验周期：2026-08-31 至 2026-09-01
- 报告日期：2026-09-01
- 项目状态：8K 技术可行性已证明，正式生产验收未完成
- 最终代码分支：`feature/qwen3-32b-dual-mac`

## 1. 摘要

本项目验证了在一台 16GB MacBook Air M3 与一台 16GB Mac mini M4 上，通过 Thunderbolt 40Gb/s 链路和 MLX Ring/TCP 后端协同运行 Qwen3-32B 的可行性。

初始方案使用 mlx-lm 原生张量并行。模型能够启动并响应 `/v1/models`，但第一次真实生成即在 Mac mini 上触发 Metal GPU command-buffer watchdog；随后出现的 Ring socket 断开是节点退出后的次生错误。实验据此放弃逐层高频同步的张量并行，改造出 Qwen3 专用的两级流水线并行适配器，并将模型权重按节点拥有的连续层范围重新打包。

实验最终形成两个可保留方案：

| 方案 | 层切分（M4/M3） | 已验证上下文 | Prefill step | Prompt | Decode | Rank 0 峰值 |
|---|---:|---:|---:|---:|---:|---:|
| 4-bit 基线 | 32/32 | 2,048 | 64 | 64.29 tok/s | 6.18 tok/s | 10.01 GB |
| 3-bit 最终方案 | 40/24 | 8,192 | 32 | 46.45 tok/s | 4.35 tok/s | 6.65 GB |

3-bit 最终方案完成了一次 `8188 prompt + 4 generation = 8192 tokens` 的端到端双机生成，达到预设的 `>= 4 tok/s` 可行性目标。该结果证明了 8K 能运行，但不等价于正式生产验收：原定的 `7936 prompt + 256 generation`、一次 warm-up 加三次测量、双端内存与 swap 监控尚未完成。

## 2. 实验目标与边界

### 2.1 主要目标

1. 在两台各 16GB 统一内存的 Apple Silicon Mac 上运行 Qwen3-32B。
2. 使用 Thunderbolt IP 网络和 MLX Ring/TCP 完成双机协同。
3. 将总上下文推进到 8,192 tokens。
4. 在 8K 条件下获得至少 4 tok/s 的解码速度。
5. 同时保留质量更高的 4-bit 短上下文方案和可达 8K 的 3-bit 方案。

### 2.2 不在本轮完成范围内的目标

- 生产级常驻 OpenAI API 服务。
- 三次正式 acceptance run 的统计稳定性。
- 3-bit 与 4-bit 的回答质量对照评测。
- 多请求并发或批处理。
- 自动 promotion、rollback、GPU 健康门禁和通信压力门禁。

## 3. 实验环境

### 3.1 硬件

| 节点 | 设备 | 芯片 | CPU 核心 | 统一内存 | 流水线角色 |
|---|---|---|---:|---:|---|
| Rank 0 | MacBook Air | Apple M3 | 8（4P+4E） | 16GB | 后段层、LM Head、客户端输出 |
| Rank 1 | Mac mini | Apple M4 | 10（4P+6E） | 16GB | 前段层、Embedding、向 Rank 0 发送 hidden state |

两台机器通过 Thunderbolt/USB4 直连，系统报告物理链路最高 40Gb/s。最终 Ring 地址为：

- Rank 0：`169.254.252.127`
- Rank 1：`169.254.82.82`

MacBook 的链路本地地址曾为 `169.254.217.74`，实验过程中发生变化，因此最终配置同步更新为 `169.254.252.127`。

### 3.2 软件

| 组件 | 版本/路径 |
|---|---|
| Python | 3.12，`/Users/Shared/mlx-cluster/.venv/bin/python` |
| MLX | 0.32.2 |
| mlx-lm | 0.31.3 |
| 分布式后端 | MLX Ring/TCP |
| Ring 连接数 | 每 IP 1 条连接 |
| 起始端口 | 33323 |
| 4-bit 原始模型 | `mlx-community/Qwen3-32B-4bit` 的本地副本 |
| 3-bit 原始模型 | `mlx-community/Qwen3-32B-3bit`，实验时固定到 `b3304de15a278747adbfcf2a2713565e65baba23` |

本地 3-bit `config.json` 可确认模型为 Qwen3、64 层、hidden size 5120、group size 64、3-bit affine quantization；现存 manifest 没有写入 `source_revision`，这是产物可追溯性的已知缺口。

### 3.3 整体实验流程

| 阶段 | 主要工作 | 阶段结论 |
|---|---|---|
| 双机基础环境 | 建立 SSH、Thunderbolt IP、Ring hostfile；验证 14B/小模型基础链路 | 双机和 Ring 基础连接可用 |
| 32B 4-bit 获取与核验 | 恢复完整模型、核对配置、index、shard 和两端文件 | 模型文件不是失败原因 |
| 原生张量并行 | 启动服务并发起首次真实生成 | Mac mini Metal watchdog，Ring 错误为次生故障 |
| 自定义流水线工程 | 实现 adapter、两 Rank 正确性、raw safetensor 和 rank packer | 32B 能按连续层分别加载到两端 |
| 4-bit 调优 | 测试 36/28、32/32、固定和自适应 barrier | 2K 通过，4K/8K 未通过 |
| 3-bit fallback | 固定 3-bit 模型，测试 40/24、KV4 和算子拆分 | 4K 通过，8K 仍需缩小 prefill chunk |
| 8K 突破 | Attention/MLP 拆分，prefill step 从 64 降到 32 | 8188+4 通过，decode 4.35 tok/s |
| 清理交付 | 删除 14B、失败模型包、测试/缓存和旧服务 | 只保留最终 4-bit/3-bit 路线 |

## 4. 初始方案与第一次失败

### 4.1 原生张量并行

最初使用 mlx-lm Qwen3 的 `shard()` 能力进行张量并行。该模式会把线性层分片，并在 Transformer 层内部频繁执行 collective。它在低延迟 GPU 互联上有优势，但当前环境是：

- 两台独立 Mac，各自有独立的 16GB 统一内存；
- Thunderbolt 物理链路承载 IP/TCP；
- MLX Ring 而非 GPU 间共享内存或直接 DMA；
- M3 与 M4 为异构节点。

两台 16GB Mac 并不等价于一台连续 32GB 机器。每台机器仍要独立容纳本地权重、Metal 工作区、激活、KV cache、通信缓冲、Python 和 macOS。

### 4.2 失败链路

第一次 32B 服务的实际链路为：

```text
模型/服务启动成功
  -> /v1/models 返回 200
  -> 首次生成请求
  -> Mac mini Metal GPU Timeout
  -> Rank 1 退出
  -> Rank 0 Ring socket closed / peer lost
```

根因日志是：

```text
RuntimeError: [METAL] Command buffer execution failed:
Caused GPU Timeout Error
(kIOGPUCommandBufferCallbackErrorTimeout)
```

Ring 的 `socket closed by peer` 和 `connection to a peer was lost` 都发生在 Metal timeout 之后。因此通信错误是后果，不是最初原因。

### 4.3 初始瓶颈判断

1. 32B 4-bit 权重约 17GB，节点内运行余量有限。
2. 张量并行在每层引入多次同步，TCP/Ring 延迟被放大。
3. MLX 使用 lazy evaluation，长计算图可能在 `mx.eval()` 或 collective 同步点集中提交。
4. 报错栈显示的同步行不一定是最重算子；它可能只是异步 GPU 错误首次被观察的位置。
5. 更换连接数或只调整 Ring 参数不能消除逐层 collective，也不能从根本上缩短 Metal command buffer。

## 5. 架构调整：从张量并行转向流水线并行

### 5.1 设计原则

实验实现了 Qwen3 专用的两级流水线适配器：

```text
输入 token
  -> Rank 1 / M4：Embedding + 前 N 层
  -> Ring send：hidden state
  -> Rank 0 / M3：后 64-N 层 + Norm + LM Head
  -> final all_gather
  -> 输出 logits
```

流水线的主要收益不是让通信消失，而是把逐层张量并行的高频 collective 收敛为每次 forward 的一次 stage send 和一次最终 gather。

### 5.2 反向 Rank 映射

mlx-lm 的 Ring 世界中 Rank 0 负责最终客户端输出，因此实现采用：

- Rank 1：前段连续层；
- Rank 0：后段连续层。

例如 3-bit 40/24：

- Rank 1 / M4：`model.layers.0` 至 `model.layers.39`；
- Rank 0 / M3：`model.layers.40` 至 `model.layers.63`。

Embedding、Norm 和 LM Head 为兼容 mlx-lm 加载流程在两端保留副本，但只有正确阶段参与最终数据流。

### 5.3 KV-cache 依赖

Rank 1 在发送 hidden state 后，用 `mx.depends` 将最后一层 cache 更新与 send 完成绑定，避免 lazy execution 下 cache 更新被错误裁剪或重排。随后两端执行相同顺序的 final gather。

## 6. 模型打包优化

### 6.1 为什么必须制作 rank-local pack

如果两端都指向完整 32B 模型目录，mlx-lm 即使稍后裁剪层，也可能先 glob 或建立完整权重映射，造成：

- 瞬时加载峰值过高；
- 本地保留不属于该 rank 的权重；
- 原始 safetensor shard 同时含有本地层和远端层，无法只复制部分文件。

### 6.2 Raw safetensor 重打包

项目实现了不反量化的 safetensor reader/writer 和 rank packer：

1. 读取 safetensor header 和 tensor byte range；
2. 按 `model.layers.N` 把一整层视为不可拆分单元；
3. 只选择当前 rank 的连续层和共享模块；
4. 原样复制量化 payload，不调用 `mx.load`，不构造完整模型；
5. 生成 layer-aligned 的新 shard；
6. 写入派生 `config.json`、`model_file` 和 `pipeline_stage_layers`；
7. 生成 index 与 rank manifest；
8. 通过临时文件、fsync、原子 rename、staging marker 和恢复校验避免半成品被误认为有效 pack。

### 6.3 打包器工程加固

Task 5 期间针对以下问题做了多轮修复：

- manifest 发布事务；
- interrupted pack 恢复；
- inode/ownership 绑定；
- staging marker 所有权；
- source snapshot 和 metadata 发布一致性；
- 官方 3-bit index 中可选 `total_parameters` 字段的兼容。

清理前完整单元测试达到 297 项通过；随后用户要求删除测试文件，历史测试仍可从 Git 提交 `557e672` 恢复。

## 7. Metal watchdog 调优策略

### 7.1 固定 hidden-state barrier

MLX 的 lazy graph 会跨多个 Transformer block 累积。首先在每 4 个本地层后执行 `mx.eval(h)`，将一个长 command buffer 切成较短批次。

效果：4-bit 32/32 从 2K 必现 timeout 改为 2K 完整通过，但 4K 仍失败。

### 7.2 上下文自适应 barrier

尝试过：

- 2K 以下每 4 层提交；
- 2K 以上每 2 层或每层提交；
- 同时 materialize hidden state 与所在层组的 KV-cache state。

这些尝试改变了超时发生位置，却未让 4-bit 通过 4K，说明长上下文单个 block 内部的算子仍可能形成过长 GPU 工作。

### 7.3 Attention 与 MLP 分离

对于 `context_tokens > 2048`，适配器不再一次调用完整 `TransformerBlock`，而是数学等价地拆成：

```text
RMSNorm -> Attention -> residual add -> mx.eval(hidden + KV)
        -> post-attention RMSNorm
        -> gate/up projection -> mx.eval(gate, up)
        -> SwiGLU -> down projection -> residual add
```

该变化不修改权重、不近似计算，并通过同权重、同 2048-token KV 状态的上游 Qwen3 数值等价测试。

### 7.4 Prefill chunk 调度

Attention/MLP 拆分后，4K 已通过，但 8K 在 prefill step 64 下仍出现 timeout。最终把 prefill step 从 64 降至 32：

- 单次 attention 覆盖的 query token 数减半；
- command buffer 的工作规模降低；
- prefill chunk 数翻倍，因此首 token 延迟上升；
- 解码阶段的模型质量和上下文长度不变。

这是最终跨过 8K watchdog 阈值的关键调度参数。

## 8. 实验迭代记录

### 8.1 4-bit 路线

| 版本 | 切分 | 测试 | 结果 | Prompt | Decode | Rank 0 峰值 | 结论 |
|---|---:|---:|---|---:|---:|---:|---|
| 原始 pipeline | 36/28 | 8+4 | 通过 | 11.35 | 6.64 | 8.60GB | 基础通信正确 |
| 原始 pipeline | 36/28 | 512+16 | 通过 | 65.47 | 5.34 | 8.79GB | 中短 prompt 可用 |
| 原始 pipeline API | 36/28 | 30+23 | 通过 | — | — | — | 中文请求 5.723 秒完成 |
| 原始 pipeline | 36/28 | 2K | 失败 | — | — | — | Mac mini Metal timeout |
| balanced，无 barrier | 32/32 | 短请求 | 通过 | — | 6.84 | 9.70GB | 负载更均衡 |
| balanced，无 barrier | 32/32 | 2K | 失败 | — | — | — | 仍触发 watchdog |
| 每 4 层 barrier | 32/32 | 短请求 | 通过 | — | 6.53 | 9.69GB | barrier 成本可接受 |
| 每 4 层 barrier | 32/32 | 2K | 通过 | 64.29 | 6.18 | 10.01GB | 4-bit 最终保留基线 |
| 每 4 层 barrier | 32/32 | 4K | 失败 | — | — | — | cache/decode 阶段 timeout |
| 自适应/每层/cache barrier | 32/32 | 4K | 失败 | — | — | — | layer 内部算子仍过重 |

结论：在这组硬件和软件版本上，4-bit 方案有可靠的 2K 成功记录，但没有完成 4K/8K。最终保留 `final-4bit-2k`，不把它描述为 8K 方案。

### 8.2 3-bit 路线

| 版本 | 切分 | Prefill step | 测试 | 结果 | Prompt | Decode | Rank 0 峰值 |
|---|---:|---:|---:|---|---:|---:|---:|
| 基础 3-bit | 40/24 | 64 | 8+4 | 通过 | 11.96 | 8.41 | 5.83GB |
| 基础 3-bit | 40/24 | 64 | 4K | 失败 | — | — | — |
| 3-bit + KV4 | 40/24 | 64 | 4K | 失败 | — | — | — |
| Attention split | 40/24 | 64 | 8+4 | 通过 | 8.11 | 8.08 | 5.83GB |
| Attention split | 40/24 | 64 | 4K+4 | 通过 | 70.76 | 5.14 | 6.23GB |
| Attention split | 40/24 | 64 | 8188+4 | 失败 | — | — | — |
| Attention + MLP split | 40/24 | 64 | 8188+4 | 失败 | — | — | — |
| Attention + MLP split | 40/24 | 32 | 8188+4 | 通过 | 46.45 | 4.35 | 6.65GB |

`KV4` 没有解决 watchdog，说明主要矛盾不是 KV 占用本身，而是长上下文下算子和 cache 更新形成的 GPU 提交规模。

## 9. 主要瓶颈与对应优化

| 瓶颈 | 证据 | 优化 | 效果 |
|---|---|---|---|
| 张量并行 collective 过多 | 初次生成 Metal timeout，Ring 后续断开 | 改为两级流水线 | 每 forward 只保留 stage send 和 final gather |
| 每节点只有独立 16GB | 32B 权重、工作区、KV、系统争用 | rank-local pack、3-bit、40/24 不均衡切分 | 降低每端权重和加载峰值 |
| 完整 shard 混合远端层 | 即使 prune 也可能映射多余权重 | 按层 raw-byte 重打包 | 两端只保留各自连续层 |
| MLX lazy graph 过长 | 错误集中在 `mx.eval`/同步点暴露 | 固定/自适应 barrier | 4-bit 从 2K 失败推进到 2K 通过 |
| 长上下文单层 Attention 过重 | 每层 barrier 仍无法通过 4K | Attention 后独立 materialize hidden+KV | 3-bit 4K 通过 |
| MLP gate/up/down 聚合提交 | 8K 在 MLP 后 barrier 失败 | gate/up projection 独立 materialize | 将失败点推进到 cache/prefill 调度 |
| 64-token prefill chunk 过大 | 8K step64 仍 timeout | prefill step 32 | 8188+4 完整通过 |
| Ring 错误容易误诊 | socket 错误总在 Metal timeout 之后 | 按时间顺序区分根因和次生故障 | 避免无效的网络参数调优 |
| 官方 3-bit metadata 差异 | index 含 `total_parameters` | packer 接受并校验可选字段 | 官方 3-bit 成功打包 |

## 10. 最终方案

### 10.1 4-bit：质量优先的 2K 基线

- Profile：`final-4bit-2k`
- M4/M3 层数：32/32
- 上下文：已验证 2,048
- Prefill step：64
- Decode：6.18 tok/s
- 适用：质量优先、短上下文验证、后续与 3-bit 做质量对照。

### 10.2 3-bit：容量优先的 8K 可行性方案

- Profile：`final-3bit-8k`
- M4/M3 层数：40/24
- 总上下文：8,192
- Prefill step：32
- Prompt：46.45 tok/s
- Decode：4.35 tok/s
- Rank 0 峰值：6.65GB
- 适用：需要完整 8K 上下文的单请求推理实验。

## 11. 正确性与验证证据

清理前的自动测试覆盖：

- 分层连续性、无重叠和 reverse-rank mapping；
- 非法 world size、stage split、cache 形态；
- 单机 tiny Qwen3 与上游 logits 数值一致；
- 长上下文 Attention/MLP 拆分与上游数值一致；
- 两 Rank 本地 Ring forward、cache dependency 和 collective 顺序；
- safetensor header、dtype、shape、payload SHA-256 不变；
- packer 确定性、事务发布、恢复、marker/inode ownership；
- 官方 index metadata 兼容。

最后一次清理前结果为 `297 passed`。根据用户清理要求，测试文件已从最终工作树删除，但可从 Git 历史恢复。

最终模型清理后完成了静态完整性验证：

- 两端别名均解析到正确 rank；
- index 引用的所有 safetensor shard 均存在；
- `config.json`、`qwen3_pipeline.py`、`rank-manifest.json` 均存在；
- 同一方案两端 adapter SHA-256 相同；
- 14B 模型和 LaunchAgent 已删除。

## 12. 已知限制与风险

1. 8K 结果只完成一次 `8188+4`，4-token decode 样本过短，速度可能有较大波动。
2. 没有完成 `7936+256` 的长生成；KV 增长和 steady decode 尚未得到充分覆盖。
3. 没有执行 warm-up 后三次正式测量，因此不能给出统计中位数或稳定性结论。
4. 只记录了 Rank 0 的 MLX peak，缺少 Rank 1 峰值与两端 swap 趋势。
5. 3-bit 的语言质量、代码能力和长上下文准确度没有与 4-bit 做系统比较。
6. 没有完成 8K 真实聊天模板/API/SSE/并发拒绝测试。
7. 现存 3-bit manifest 的 `source_revision` 为空，尽管下载过程固定过 upstream revision。
8. 清理后执行 4-bit 短 canary 时，Mac mini 当前 boot session 再次出现 Metal timeout；静态文件完整性正常。为避免在疑似受污染 GPU 会话中继续加载模型，没有再运行 3-bit。两端重启后的动态复验仍是必要步骤。
9. 当前没有 32B 常驻服务，也没有自动重启、健康检查或 rollback 流程。

## 13. 清理与最终状态

实验完成后删除了：

- 两端 14B 模型；
- 14B LaunchAgent、启动脚本和日志；
- 所有失败或被替代的 4-bit/3-bit rank pack；
- MacBook 上多余的 rank1 包；
- pytest 测试目录、缓存、项目临时 `.venv`；
- 过期实验设计/计划文档和临时链接。

最终磁盘状态：

| 节点 | 清理前可用 | 清理后可用 | 约释放 |
|---|---:|---:|---:|
| MacBook | 128GiB | 242–244GiB | 114–116GiB |
| Mac mini | 16GiB | 97GiB | 81GiB |

## 14. 结论

本实验最重要的结论是：在两台 16GB Mac 上运行 32B 的主要障碍不是模型文件大小或 Thunderbolt 是否连通，而是节点内存余量、逐层同步和 MLX lazy execution 共同造成的 Metal command-buffer watchdog。

仅优化网络参数不足以解决该问题。真正有效的组合是：

1. 用两级流水线替代逐层张量并行；
2. 用 rank-local、layer-aligned pack 降低权重映射和加载峰值；
3. 采用 3-bit 40/24 不均衡切分匹配 M4/M3 能力；
4. 在长上下文下拆分 Attention、MLP 和 gate/up 投影；
5. 将 prefill step 降至 32，控制单次 GPU 提交规模。

该组合已把系统从“32B 首次生成即失败”推进到“完成一次 8,192-token 双机生成，并达到 4.35 tok/s”。这证明了技术可行性，但生产化仍需要重启后的复验、`7936+256` 多轮验收、双端资源监控和 API/运维层建设。

## 附录 A：最终运行命令

### A.1 4-bit / 2K

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

### A.2 3-bit / 8K

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

执行前应重启两台 Mac，并确认端口 33323–33324 空闲、Thunderbolt IP 与 hostfile 一致。
