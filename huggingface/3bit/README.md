---
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
base_model: mlx-community/Qwen3-32B-3bit
tags:
  - mlx
  - qwen3
  - distributed-inference
  - pipeline-parallelism
  - apple-silicon
---

# Qwen3-32B 3-bit Dual-Mac MLX Pipeline

This repository contains the final two-rank, layer-local MLX pipeline package used to complete an 8,192-token Qwen3-32B inference run on two 16 GB Apple Silicon Macs.

It is not a single-host model. Both rank directories are required:

- `rank0/`: final 24 layers on the M3 MacBook, layers `[40, 64)`
- `rank1/`: first 40 layers on the M4 Mac mini, layers `[0, 40)`

Each rank package includes its local safetensor shards, tokenizer assets, rank manifest, derived config, and the exact custom adapter with long-context Attention and MLP submission splitting.

## Tested configuration

- Source: `mlx-community/Qwen3-32B-3bit`, experiment revision `b3304de15a278747adbfcf2a2713565e65baba23`
- MLX 0.32.2
- mlx-lm 0.31.3
- M4/M3 split: 40/24 layers
- Ring/TCP over a 40 Gb/s Thunderbolt IP link
- One connection per IP
- Prefill step 32
- Total context: 8,188 prompt + 4 generation = 8,192 tokens
- Measured prompt throughput: 46.45 tok/s
- Measured decode throughput: 4.35 tok/s
- Measured Rank 0 peak: 6.65 GB

## Usage

Install the matching code and benchmark runner from [qwen3-32b-dual-mac-mlx](https://github.com/Levius-Fubuki/qwen3-32b-dual-mac-mlx). Place `rank0/` on the Rank 0 host and `rank1/` at the same logical model path on Rank 1, then launch with `mlx.launch --backend ring --connections-per-ip 1` and prefill step 32.

See the GitHub experiment and deliverables reports for exact commands, implementation details, integrity identifiers, and the full sequence of failed and successful experiments.

## Limitations

- The 8K result is one feasibility run with four generated tokens, not the planned `7,936 + 256` warm-up plus three measured runs.
- Rank 1 peak memory and per-run swap growth were not captured in the final feasibility run.
- 3-bit response quality was not systematically compared with the retained 4-bit baseline.
- A fresh-boot dynamic revalidation is still required after the final cleanup.
- This is an experimental two-host package, not a production server deployment.
- Use is subject to the upstream Apache-2.0 license and model terms.
