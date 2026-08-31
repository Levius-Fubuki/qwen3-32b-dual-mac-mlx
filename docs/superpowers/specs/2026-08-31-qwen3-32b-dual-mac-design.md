# Qwen3-32B Dual-Mac Inference Optimization Design

## 1. Objective

Build a separate, reversible MLX inference path that runs Qwen3-32B across:

- MacBook Air M3, 16 GB unified memory, macOS 15.1
- Mac mini M4, 16 GB unified memory, macOS 26.5.2
- Direct 40 Gb/s Thunderbolt Bridge using the MLX Ring/TCP backend
- MLX 0.32.2 and mlx-lm 0.31.3 on both nodes

The accepted success target is:

- Qwen3-32B, preferring the existing 4-bit model and permitting a 3-bit fallback
- At least 8,192 total active tokens per request, measured as a 7,936-token prompt followed by 256 generated tokens
- Median decode throughput of at least 4.0 tokens/second after warm-up
- Three consecutive successful benchmark runs without a Metal GPU timeout, lost Ring peer, process crash, or corrupt output
- No sustained swap growth during steady-state decode

The existing Qwen3-14B service and the original Qwen3-32B-4bit model directory remain untouched and available for rollback.

## 2. Baseline Evidence and Root Cause

The existing tensor-parallel implementation was measured before this design:

| Configuration | Prompt throughput | Decode throughput | Result |
| --- | ---: | ---: | --- |
| M4 mini, Qwen3-14B 4-bit, single node | 96.878 tok/s | 12.857 tok/s | Stable |
| M3 Air, Qwen3-14B 4-bit, single node under current desktop load | 14.715 tok/s | 2.221 tok/s | Stable but memory pressured |
| Two-node Ring tensor parallel, Qwen3-14B 4-bit | 26.121 tok/s | 1.837 tok/s | Stable with one TCP connection |
| Two-node Ring tensor parallel, Qwen3-32B 4-bit | Not completed | Not completed | Metal GPU timeout during a 32-token warm-up |

The 32B failure coincided with approximately 23 GB of additional cumulative swap-out on the MacBook and 3.3 GB on the Mac mini. Increasing `--connections-per-ip` from one to two subsequently caused even the 14B distributed warm-up to time out.

The installed Qwen3 tensor-parallel implementation performs two distributed `all_sum` operations per Transformer layer: one after the attention output projection and one after the MLP down projection. Qwen3-32B has 64 layers, so autoregressive decode requires approximately 128 dependency-bound collectives per token. The direct-link ping round-trip averages about 0.37 ms before MLX scheduling and synchronization overhead. This makes Ring latency, not link bandwidth, the dominant decode bottleneck.

The failure has two distinct causes:

1. Tensor-parallel model loading and sharding creates insufficient transient memory headroom on the 16 GB nodes, causing aggressive compression and swap.
2. Tensor parallelism leaves a high-frequency collective on the critical path of every layer, preventing the current architecture from reaching the 4 tok/s target even when the model fits.

The fix therefore changes the parallelism architecture rather than masking the Metal timeout.

## 3. Considered Approaches

### 3.1 Continue optimizing tensor parallelism

Offline rank-specific weight packing could reduce the transient loading peak. More TCP connections or larger network MTUs could improve bulk prefill bandwidth. Neither removes the 128 collectives per decode token. This approach is retained only as a benchmark control.

### 3.2 Uneven two-stage pipeline parallelism — selected

Each node owns a contiguous range of complete Transformer layers. Rank 1 computes the first stage and sends one hidden-state tensor to Rank 0. Rank 0 computes the second stage. The final hidden state is synchronized once so both ranks remain compatible with mlx-lm's distributed generation loop.

This reduces inter-node operations from approximately 128 collectives per token to one point-to-point stage transfer plus one final synchronization. It also avoids tensor slicing during model initialization because each rank loads complete local layers.

The split is deliberately uneven because the M4 mini has materially higher observed sustained throughput and more recommended Metal working-set memory than the M3 Air.

### 3.3 Hybrid pipeline and tensor parallelism

With only one Apple GPU per node, local tensor parallelism is unavailable. Tensor-sharding layers across the same two nodes would reintroduce the per-layer collectives that pipeline parallelism removes. This approach is out of scope.

## 4. Architecture

The hostfile preserves the current rank ordering so Rank 0 remains on the MacBook and continues to own the local API endpoint:

```text
Client/API on MacBook
        |
        v
Rank 1: Mac mini M4
  embedding lookup
  first contiguous Transformer stage
        |
        | Ring send(hidden_state)
        v
Rank 0: MacBook Air M3
  second contiguous Transformer stage
        |
        | final hidden-state synchronization
        v
Both ranks: final norm and LM head
Rank 0: HTTP response
```

Both ranks initially retain the embedding, final norm, and LM head because mlx-lm's distributed generation loop expects compatible logits and sampling state on every rank. Removing those replicated modules would require a separate token-broadcast generation loop and would trade about 0.9 GB of memory for a substantially larger per-token logits transfer. That optimization is deferred unless measured memory proves it necessary.

The stock distributed server listens on an internal localhost port. A thin localhost-only admission proxy exposes the user-facing OpenAI-compatible port. It loads only the tokenizer, applies the same chat template, counts prompt tokens, and rejects a request before forwarding when `prompt_tokens + max_tokens` exceeds 8,192. Successful requests, including streaming responses, pass through unchanged. This guard is necessary because mlx-lm 0.31.3 has no distributed-server `--max-kv-size` or active-KV memory limit.

The first production candidates are:

| Profile | Quantization | M3 Rank 0, final layers | M4 Rank 1, first layers | Purpose |
| --- | --- | ---: | ---: | --- |
| `balanced-4bit` | Existing affine 4-bit, group 64 | 32 | 32 | Memory-safe 4-bit correctness baseline |
| `quality-4bit` | Existing affine 4-bit, group 64 | 28 | 36 | Faster-node bias; allowed only if measured peak stays below the guardrail |
| `performance-3bit` | MLX affine 3-bit | 24 | 40 | Expected default for the 8K/4 tok/s target |
| `aggressive-3bit` | MLX affine 3-bit | 20 | 44 | Throughput fallback if 24/40 is below 4 tok/s |
| `balanced-3bit` | MLX affine 3-bit | 28 | 36 | Diagnostic comparison point |

The benchmark harness may test neighboring splits in two-layer increments, but it must reject a split before the 8K benchmark if either node exceeds its memory guardrail. Because a single autoregressive request executes the two stages serially, a numerically balanced split is not assumed to be fastest. Subject to the M4 memory guardrail, the search favors assigning more layers to the faster M4.

## 5. Custom Qwen3 Pipeline Adapter

A model-local Python adapter will be installed beside a derived model configuration using mlx-lm's `model_file` mechanism. Because mlx-lm loads this file as a standalone `custom_model` module rather than as part of the `mlx_lm.models` package, every mlx-lm import is absolute; package-relative imports would fail. The adapter imports the stable attention, RoPE, cache, and quantized-linear primitives from the pinned mlx-lm environment instead of patching `site-packages`.

The adapter provides:

- A Qwen3-compatible `ModelArgs` data class
- The unmodified Qwen3 attention, MLP, and block computation
- A pipeline-aware language model with `pipeline(group)`
- Validation of the requested split against world size and the model's 64 layers
- A `pipeline_layers` view containing only the local stage
- A `layers` property that exposes only local layers to cache creation
- A `make_cache()` method returning one `KVCache` per local layer
- The reference MLX pipeline flow: receive from the preceding stage, compute local layers, send to the following stage, then synchronize the final hidden state
- Cache dependencies attached to the send operation so lazy evaluation cannot reorder communication ahead of KV-cache updates

The split point is stored in the derived profile's `config.json`, which is identical on both ranks and is loaded twice during mlx-lm's probe/load sequence. For two ranks, a cut of `36` means Rank 1 owns `[0, 36)` and Rank 0 owns `[36, 64)`. Invalid cuts and any world size other than two fail before weights are evaluated. Adapter construction and partitioning must be deterministic and idempotent because `sharded_load` executes the custom model file and constructs the model more than once per rank.

No existing model configuration is edited. A streaming offline packer creates a rank-local derived model directory on each host. It reads the original safetensor index, copies packed tensors without dequantizing them, and writes new layer-aligned safetensor shards containing only:

- the local contiguous Transformer layers for that rank
- the shared embedding, final norm, and LM head
- the tokenizer, custom adapter, profile configuration, and a rank-local index

This is required because mlx-lm's local-path `sharded_load` still globs every `model*.safetensors` file in the directory. Pointing it at the existing complete 32B directory would recreate the full-model lazy mapping and transient memory pressure before pipeline pruning. Repacking by layer boundary also avoids the file-granularity problem where a nominally local shard contains weights owned by the other rank.

The repacker processes one original shard at a time, keeps packed 3-bit/4-bit arrays unchanged, limits output shard size, and never materializes a dequantized or full 32B model. Before launch, each rank enumerates its live parameter keys and asserts that every key exists in its local index and that no non-local Transformer-layer key is present. The two hosts receive the same adapter/config checksum and different, explicitly expected rank-local weight manifests.

## 6. Memory Budget

Qwen3-32B uses 64 layers, 8 KV heads, and a head dimension of 128. With pipeline parallelism, KV heads are not tensor-sharded, but KV state is allocated only for the local layers. FP16 KV-cache memory is therefore:

```text
local_layers * 8 heads * 128 dimensions * 2 (K and V) * 2 bytes * tokens
```

At 8,192 active tokens:

| Local layers | KV cache per rank |
| ---: | ---: |
| 24 | 768 MiB |
| 28 | 896 MiB |
| 32 | 1,024 MiB |
| 36 | 1,152 MiB |
| 40 | 1,280 MiB |

The embedding and LM head are untied in Qwen3-32B and remain replicated. Exact inspection of the local 4-bit tensors gives 417.305 MiB for each of those modules, or about 0.815 GiB together per rank. The projected 3-bit packed versions occupy about 324.570 MiB each. Runtime allocations, Metal command buffers, Python, and the operating system require additional reserve.

Exact 4-bit packed-weight inspection, projected 3-bit packed sizes, and the KV formula above produce these pre-runtime estimates:

| Profile | Rank | Packed weights | 8K KV | Weight + KV total |
| --- | --- | ---: | ---: | ---: |
| 4-bit 32/32 | either | 8.989 GiB | 1.000 GiB | 9.989 GiB |
| 4-bit 28/36 | M3 / 28 layers | 7.968 GiB | 0.875 GiB | 8.843 GiB |
| 4-bit 28/36 | M4 / 36 layers | 10.011 GiB | 1.125 GiB | 11.136 GiB |
| 3-bit 24/40 | M3 / 24 layers | 5.402 GiB | 0.750 GiB | 6.152 GiB |
| 3-bit 24/40 | M4 / 40 layers | 8.581 GiB | 1.250 GiB | 9.831 GiB |
| 3-bit 20/44 | M3 / 20 layers | about 4.608 GiB | 0.625 GiB | about 5.233 GiB |
| 3-bit 20/44 | M4 / 44 layers | about 9.376 GiB | 1.375 GiB | about 10.751 GiB |

The 4-bit 28/36 profile leaves only about 0.70 GiB below the M4's recommended working set before temporary allocations and is therefore a yellow-light experiment, not the expected production profile. The 3-bit 24/40 profile leaves about 2.0 GiB of recommended-working-set headroom and is the expected stable profile.

The initial MLX peak-memory guardrails are:

- MacBook M3 Rank 0: no more than 10.1 GiB
- Mac mini M4 Rank 1: no more than 11.3 GiB

These preserve at least about 512 MiB below the devices' reported maximum recommended Metal working sets of 10.667 GiB and 11.839 GiB. A profile that crosses a guardrail is not allowed to proceed to the full 8K test.

The server is configured for a single active request and no retained prompt cache:

```text
decode_concurrency = 1
prompt_concurrency = 1
prompt_cache_size = 0
prompt_cache_bytes = 0
```

Persistent prompt caching is disabled so a completed 8K cache cannot overlap transiently with the next request. These settings still do not cap an active request. The API layer and benchmark harness therefore tokenize requests before admission and reject `prompt_tokens + max_tokens > 8,192`.

KV-cache quantization is not part of the first implementation. On the pinned mlx-lm version it is not exposed by the distributed server CLI, and current upstream reports show that quantized KV cache can increase peak memory. It may be evaluated only after the unquantized 8K profile is stable.

## 7. Communication and Scheduling

The Ring hostfile uses only the direct Thunderbolt Bridge addresses. Wi-Fi and Ethernet addresses are excluded from the collective path.

The launch policy is:

- Ring backend
- One TCP connection per IP
- No `MLX_METAL_FAST_SYNCH`
- No JACCL/RDMA, because both nodes expose 40 Gb/s Thunderbolt rather than Thunderbolt 5 RDMA
- `caffeinate` on both ranks for the lifetime of the server or benchmark

Before loading model weights, a communication gate runs 10,000 two-rank `send`/`recv`/`all_gather` iterations with the Qwen3 hidden-state shape. It verifies rank ordering, collective ordering, loss-free direct-link operation, and stable latency using one TCP connection. The model benchmark cannot start if this gate fails.

Prefill is chunked so an 8K prompt does not create a single long Metal command buffer. Candidate step sizes are tested in this order:

1. 256 tokens
2. 128 tokens if 256 times out or crosses a memory guardrail
3. 512 tokens only if 256 is stable and profiling shows avoidable scheduling overhead

Decode remains batch size one. Speculative decoding is excluded because mlx-lm 0.31.3 rejects draft-model loading in distributed mode.

## 8. Runtime Selection

The optimizer does not silently change quantization during a live request. A deterministic calibration command evaluates profiles in this order:

1. `balanced-4bit`, split 32/32, to prove 4-bit pipeline correctness with safe static memory
2. `quality-4bit`, split 28/36, only if its short preflight remains below the M4 guardrail
3. Other memory-safe 4-bit splits between 32/32 and 28/36
4. `performance-3bit`, split 24/40
5. `aggressive-3bit`, split 20/44, if 24/40 is stable but below 4 tok/s
6. Other memory-safe 3-bit splits between those points

Each profile receives a short correctness warm-up before any long benchmark. A Metal timeout, lost peer, invalid output, or guardrail violation immediately disqualifies that profile for the current boot session. The selected profile and all measured values are written to a machine-readable benchmark report. The start command uses only a previously passing profile. Throughput is an empirical gate rather than a promise inferred from layer count: stock MLX pipeline execution is serial for a single request and does not overlap the two stages.

The 3-bit model is downloaded only after the 4-bit pipeline path has been proven correct but fails either the stability or throughput target. It is stored independently on both nodes and never replaces the 4-bit files.

## 9. Failure Handling

The launcher terminates the remaining rank when either rank exits. It records:

- rank and host
- model profile and layer split
- prefill step size
- prompt and generation token counts
- elapsed prefill and decode time
- MLX peak memory
- memory pressure and swap usage before and after the run
- the final Ring or Metal error

After a Metal GPU timeout, no further model benchmark is launched automatically. A minimal independent `mx.eval` GPU health probe runs once on each host. If either probe fails or hangs, the tool marks the current boot session as contaminated and instructs the operator to restart both Macs before continuing. It does not call `sudo purge`, disable swap, raise private IOGPU limits, update macOS, or reboot either machine automatically.

All deployed files are checksummed on both nodes before launch. A checksum or version mismatch fails before distributed initialization.

## 10. Test Strategy

### 10.1 Unit tests

Unit tests run without loading 32B weights and cover:

- split parsing and validation
- correct contiguous layer ownership for both ranks
- reverse pipeline ordering: Rank 1 owns the first stage and Rank 0 the final stage
- local cache count equals local layer count
- the derived rank-local index contains every live local parameter and no remote-layer parameter
- invalid split, wrong world size, and missing configuration fail early
- benchmark result classification and profile selection
- context-budget enforcement

### 10.2 Two-rank lightweight integration test

A tiny local test model exercises the same pipeline adapter on two local ranks. It verifies:

- send/receive ordering
- identical final logits across ranks
- cache offsets advancing equally through prefill and decode
- deterministic generated tokens relative to an unpartitioned reference
- 10,000 ordered hidden-state send/receive/final-gather iterations without deadlock or mismatch

### 10.3 Real-model staged tests

Tests escalate one variable at a time:

1. Load-only and one-token forward pass
2. 32-token prompt and 16-token generation
3. 512-token prompt and 64-token generation
4. 2,048-token prompt and 128-token generation
5. 7,936-token prompt and 256-token generation

Every stage records per-rank memory and swap deltas. A failed stage prevents escalation.

### 10.4 Performance acceptance

The final profile must pass three consecutive 7,936-plus-256 runs after one warm-up. The report uses:

- median decode tokens/second across the three measured runs
- median prefill tokens/second
- maximum per-rank MLX peak memory
- total swap delta per rank
- output token hashes for consistency

Acceptance requires:

- median decode throughput at least 4.0 tok/s
- all three outputs contain exactly 256 generated tokens
- no Metal, Ring, SSH, or server error
- no rank exceeds its memory guardrail
- aggregate swap growth during each measured run remains below 512 MiB per node and does not continue increasing during steady-state decode

### 10.5 API acceptance

After benchmark acceptance, the selected profile starts an OpenAI-compatible server bound to `127.0.0.1`. A client smoke test verifies `/v1/models` and a streamed `/v1/chat/completions` response. The test confirms the server refuses a request whose requested prompt-plus-completion budget exceeds 8,192 tokens.

## 11. Deployment and Rollback

All implementation files live in this repository and deploy under a new versioned directory beneath `/Users/Shared/mlx-cluster`. The existing `run-server.sh`, launch agent, hostfile, 14B model, and original 32B model are not modified during development.

Promotion occurs only after the final profile passes. Promotion updates a single optimized-service entry point to reference the selected profile. Rollback stops the optimized service and starts the unchanged existing 14B service.

No model or environment file is deleted as part of rollback. Derived model directories can be removed later because they contain adapter/config files and rank-local repacked copies rather than the only copy of any weight.

## 12. Explicit Non-Goals

- Training or fine-tuning Qwen3-32B
- Supporting more than two Macs
- Multi-request throughput optimization
- Exposing the inference endpoint beyond localhost
- Enabling unsupported private Metal or IOGPU settings
- Automatically updating macOS or rebooting a host
- Claiming that a profile is successful without the complete 8K/256-token acceptance run
