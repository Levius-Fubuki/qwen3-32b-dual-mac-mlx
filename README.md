# Qwen3-32B Dual-Mac Final Configurations

- [完整实验报告](EXPERIMENT_REPORT.md)
- [项目产出报告](DELIVERABLES_REPORT.md)
- [Hugging Face 4-bit model card](huggingface/4bit/README.md)
- [Hugging Face 3-bit model card](huggingface/3bit/README.md)

This repository retains only the two configurations that completed real two-host inference on the 16 GB M3 MacBook and 16 GB M4 Mac mini.

## Cluster

- Rank 0: M3 MacBook, Ring IP `169.254.252.127`
- Rank 1: M4 Mac mini, Ring IP `169.254.82.82`
- Backend: MLX Ring/TCP, one connection per IP, starting port `33323`
- Python: `/Users/Shared/mlx-cluster/.venv/bin/python`
- Launcher hostfile: `/Users/Shared/mlx-cluster/hosts.json`
- Benchmark runner: `/Users/Shared/mlx-cluster/qwen32_pipeline_benchmark.py`

The same final model alias exists on both hosts and points to that host's rank-local pack.

## Final 4-bit configuration

- Alias: `/Users/Shared/mlx-cluster/models/Qwen3-32B-4bit-final`
- Split: M4 first 32 layers, M3 final 32 layers
- Supported tested budget: 2,048 total tokens
- Prefill step: 64
- Measured decode: 6.18 tok/s

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

## Final 3-bit configuration

- Alias: `/Users/Shared/mlx-cluster/models/Qwen3-32B-3bit-final`
- Split: M4 first 40 layers, M3 final 24 layers
- Supported tested budget: 8,192 total tokens
- Prefill step: 32
- Measured prefill: 46.45 tok/s
- Measured decode: 4.35 tok/s
- Rank 0 measured peak: 6.65 GB

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

The 8K result is a successful feasibility run, not a multi-run production acceptance test. The former 14B model and its LaunchAgent have been removed.
