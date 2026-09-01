---
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
base_model: mlx-community/Qwen3-32B-4bit
tags:
  - mlx
  - qwen3
  - distributed-inference
  - pipeline-parallelism
  - apple-silicon
---

# Qwen3-32B 4-bit Dual-Mac MLX Pipeline

This repository contains a two-rank, layer-local MLX pipeline package derived from `mlx-community/Qwen3-32B-4bit` for two 16 GB Apple Silicon Macs.

It is not a single-host model. Both rank directories are required:

- `rank0/`: final 32 layers on the M3 MacBook, layers `[32, 64)`
- `rank1/`: first 32 layers on the M4 Mac mini, layers `[0, 32)`

Each rank package includes its local safetensor shards, tokenizer assets, rank manifest, derived config, and the exact custom Qwen3 pipeline adapter used in the experiment.

## Tested configuration

- MLX 0.32.2
- mlx-lm 0.31.3
- Ring/TCP over a 40 Gb/s Thunderbolt IP link
- One connection per IP
- Prefill step 64
- Tested total context: 2,048 tokens
- Measured prompt throughput: 64.29 tok/s
- Measured decode throughput: 6.18 tok/s
- Measured Rank 0 peak: 10.01 GB

The package historically completed the 2K benchmark. A post-cleanup canary encountered a Metal timeout in the then-current Mac mini boot session, so a fresh-boot revalidation is recommended before use.

## Usage

Install the matching code and benchmark runner from [qwen3-32b-dual-mac-mlx](https://github.com/Levius-Fubuki/qwen3-32b-dual-mac-mlx). Place `rank0/` on the Rank 0 host and `rank1/` at the same logical model path on Rank 1, then launch with `mlx.launch --backend ring --connections-per-ip 1`.

See the GitHub experiment and deliverables reports for the complete architecture, command line, limitations, and integrity identifiers.

## Limitations

- 4K and 8K did not pass on this 4-bit profile.
- This is an experimental two-host package, not a production server deployment.
- The package has not completed a multi-run acceptance test after cleanup.
- Use is subject to the upstream Apache-2.0 license and model terms.
