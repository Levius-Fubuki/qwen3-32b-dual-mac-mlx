# Qwen3-32B Dual-Mac Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Qwen3-32B across the 16 GB M3 MacBook and 16 GB M4 Mac mini at an 8,192-token request budget, with median decode throughput of at least 4.0 tok/s and three consecutive stable 7,936-prompt-plus-256-generation acceptance runs.

**Architecture:** Replace per-layer tensor parallelism with an uneven two-stage pipeline: Rank 1 on the M4 owns the first contiguous layer range and Rank 0 on the M3 owns the final range. Repack existing quantized safetensor payloads into rank-local, layer-aligned model directories; retain duplicated embedding, norm, and LM head for mlx-lm compatibility; run one Ring send and one final gather per forward; protect the service with memory, communication, GPU-health, context-budget, benchmark, promotion, and rollback gates.

**Tech Stack:** Python 3.12, MLX 0.32.2, mlx-lm 0.31.3, pytest 9.1.1, safetensors format, FastAPI 0.141.1, httpx 0.28.1, uvicorn 0.52.4, MLX Ring/TCP, launchd-compatible shell entry points.

---

## 0. Fixed Inputs, Safety Invariants, and Definition of Done

The implementation must use these fixed cluster inputs unless a later task explicitly validates and records a replacement:

```text
Repository:        /Users/levius/Desktop/Idea/projects/Qwen3 32B 4-bit optimize
Python:            /Users/Shared/mlx-cluster/.venv/bin/python
Hostfile:          /Users/Shared/mlx-cluster/hosts.json
Rank 0 / API:      MacBook Air M3, Thunderbolt IP 169.254.217.74
Rank 1 / stage 1:  Mac mini M4, SSH kelly@169.254.82.82
4-bit source:      /Users/Shared/mlx-cluster/models/Qwen3-32B-4bit
Deploy root:       /Users/Shared/mlx-cluster/qwen3-32b-opt
Existing service:  /Users/Shared/mlx-cluster/run-server.sh
Internal API:      127.0.0.1:18081
Guarded API:       127.0.0.1:18080 during canary, then 127.0.0.1:8080 on promotion
Optimized Ring:    starting port 33323 (legacy 14B keeps MLX default 32323)
Context budget:    8,192 total tokens
Acceptance input:  7,936 prompt tokens + 256 generated tokens
Decode target:     median >= 4.0 tok/s over three measured runs
```

Safety invariants:

- The existing 14B launch configuration and both original model directories remain unmodified. The 14B service is stopped only inside an explicitly recorded 32B maintenance window and is restored on every exit that does not complete promotion; 14B and 32B models are never resident together.
- No broad `pkill`, automatic reboot, `sudo purge`, swap disabling, private Metal settings, or macOS update is permitted.
- The rank packer copies quantized safetensor bytes. It must not import `mlx.nn` or `mlx_lm`, call `mx.load`, dequantize, or construct Qwen3.
- Rank 1 always owns the first stage; Rank 0 always owns the final stage and the client-facing launcher.
- Ring uses one TCP connection per IP, only Thunderbolt addresses, and no `MLX_METAL_FAST_SYNCH`.
- A Metal timeout contaminates the current boot session until both independent GPU probes pass. A failed probe stops all MLX model loads, leaves both 14B and 32B stopped, and requires both Macs to restart; restoring 14B is forbidden on a contaminated boot because that is another real GPU workload.
- No profile may be selected without a complete machine-readable acceptance report.
- Every implementation behavior starts with a failing test, then receives the minimum implementation, a targeted pass, full regression, and a commit.

The work is complete only when all of the following are true:

- Unit and local-integration suites pass.
- Rank manifests prove exact local key ownership and unchanged tensor payloads.
- Both hosts pass version, checksum, memory, GPU, and 10,000-iteration communication gates.
- One profile completes warm-up plus three consecutive 7,936+256 runs.
- Median decode is at least 4.0 tok/s; both ranks stay below 10.1 GiB (M3) and 11.3 GiB (M4) MLX peak; swap growth is below 512 MiB per measured run and does not trend upward during steady decode.
- The guarded OpenAI endpoint accepts exactly-budgeted requests, rejects over-budget requests, streams correctly, and limits active generation to one.
- Promotion and rollback are both verified without modifying or deleting the previous 14B deployment.

## Task 1: Create the Package, Test Markers, Contracts, and Validated Profiles

**Files:**

- Create: `pyproject.toml`
- Create: `pytest.ini`
- Create: `.gitignore`
- Create: `src/qwen32_cluster/__init__.py`
- Create: `src/qwen32_cluster/contracts.py`
- Create: `src/qwen32_cluster/profiles.py`
- Create: `config/cluster.json`
- Create: `config/profiles.json`
- Create: `tests/conftest.py`
- Create: `tests/unit/test_profiles.py`
- Create: `tests/unit/test_contracts.py`

- [ ] Write `pyproject.toml` with package version `0.1.0`, a pinned setuptools build backend (`setuptools==80.9.0`), src-layout package discovery, package name `qwen32-cluster`, Python `>=3.12,<3.13`, runtime dependencies `mlx==0.32.2`, `mlx-lm==0.31.3`, `fastapi==0.141.1`, `httpx==0.28.1`, and `uvicorn==0.52.4`, a `dev` extra containing `pytest==9.1.1`, and this console entry point:

```toml
[build-system]
requires = ["setuptools==80.9.0"]
build-backend = "setuptools.build_meta"

[project]
name = "qwen32-cluster"
version = "0.1.0"
requires-python = ">=3.12,<3.13"

[project.scripts]
qwen32-cluster = "qwen32_cluster.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] Register the `local_integration`, `model_metadata`, `cluster`, and `live_api` pytest markers in `pytest.ini`, make unregistered markers errors, and set `pythonpath = src` so the src-layout package is importable before editable installation.

- [ ] Add `tests/conftest.py` with `--hostfile`, `--profile-file`, and `--base-url` options. Cluster-marked tests skip unless a validated hostfile is explicitly provided; real-model stage tests also require a canonical profile file; live-API tests skip unless a loopback `http://127.0.0.1:<port>` base URL is explicitly provided. This prevents accidental hardware/model runs during normal pytest and keeps post-promotion tests from launching another Ring world.

- [ ] Add `.gitignore` entries for `reports/`, `.pytest_cache/`, `__pycache__/`, `*.pyc`, `.coverage`, `build/`, `dist/`, `*.egg-info/`, and local virtual environments. Rank packs and model weights live outside the repository and must never be added through a broad `git add`.

- [ ] Create `src/qwen32_cluster/__init__.py` with only a package version constant and no eager MLX or control-plane imports, so editable installation works before the RED implementation modules exist.

- [ ] Verify the pinned environment imports every declared dependency. The initial environment lacks pip, pytest, FastAPI, and uvicorn but has `/opt/homebrew/bin/uv`; install this repository editable with its development extra through uv, then record the resolved package versions:

```bash
/opt/homebrew/bin/uv pip install \
  --python /Users/Shared/mlx-cluster/.venv/bin/python -e '.[dev]'
/Users/Shared/mlx-cluster/.venv/bin/python -c \
  'from importlib.metadata import version; print(*(version(name) for name in ("mlx", "mlx-lm", "fastapi", "httpx", "uvicorn", "pytest")))'
```

Repeat editable installation only after project dependency metadata changes.

- [ ] Write failing contract tests for deterministic JSON serialization and strict enum values. Use these public types:

```python
class RunStatus(str, Enum):
    PASS = "PASS"
    OUTPUT_FAIL = "OUTPUT_FAIL"
    TIMEOUT = "TIMEOUT"
    PEER_LOST = "PEER_LOST"
    MEMORY_GUARD = "MEMORY_GUARD"
    SWAP_GUARD = "SWAP_GUARD"
    GPU_UNHEALTHY = "GPU_UNHEALTHY"

@dataclass(frozen=True)
class ClusterHost:
    rank: int
    name: str
    ssh: str
    thunderbolt_ip: str
    mlx_peak_guardrail_bytes: int

@dataclass(frozen=True)
class Profile:
    name: str
    quantization_bits: int
    stage_layers: tuple[int, int]  # forward order: M4 first, M3 final
    prefill_step_size: int
    context_limit: int = 8192
```

- [ ] Write failing profile tests asserting the exact initial candidates and reverse rank mapping:

```python
EXPECTED = {
    "balanced-4bit": (4, (32, 32), 256),
    "quality-4bit": (4, (36, 28), 256),
    "performance-3bit": (3, (40, 24), 256),
    "aggressive-3bit": (3, (44, 20), 256),
    "balanced-3bit": (3, (36, 28), 256),
}
```

The serialized list is in forward stage order. Therefore `stage_layers[0]` belongs to Rank 1 and `stage_layers[1]` belongs to Rank 0.

- [ ] Add and test `derive_profile(base, stage_layers, prefill_step_size)`. It returns a new immutable calibration profile named `calibration-${bits}bit-m4-${stage0}-m3-${stage1}-p${prefill}` without editing `config/profiles.json`; it applies every normal profile validation rule. Task 10 serializes both built-in and derived profiles into the immutable run directory beside their benchmark report.

- [ ] Add failure cases for duplicate ranks, invalid `ClusterHost.thunderbolt_ip` values, unexpected or reordered Ring `hosts[*].ips`, totals other than 64 layers, non-positive stage sizes, quantization other than 3 or 4, context limits other than 8192, and prefill steps outside `{128, 256, 512}`. Keep control-plane SSH endpoints distinct from tensor-plane addresses: `127.0.0.1` and `kelly@169.254.82.82` are the allowed `ssh` values, while the ordered Ring IPs must be exactly `169.254.217.74` then `169.254.82.82`.

- [ ] Run the RED tests and confirm the failure is import/config absence, not a syntax or collection error:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest \
  tests/unit/test_contracts.py tests/unit/test_profiles.py -vv
```

Expected: tests collect and fail because `qwen32_cluster.contracts` and `qwen32_cluster.profiles` do not exist.

- [ ] Implement strict dataclass parsing, `to_dict()`, canonical JSON (`sort_keys=True`, compact separators, newline at EOF), `load_cluster()`, and `load_profiles()`.

- [ ] Populate `config/cluster.json` with the fixed ranks, SSH values, Thunderbolt addresses, guardrails `10844792422` bytes (10.1 GiB) and `12133282611` bytes (11.3 GiB), hostfile path, optimized Ring starting port 33323, and deployment ports.

- [ ] Populate `config/profiles.json` with the five candidates above and static server settings `decode_concurrency=1`, `prompt_concurrency=1`, `prompt_cache_size=0`, and `prompt_cache_bytes=0`.

- [ ] Run the targeted tests, then the current full suite:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest \
  tests/unit/test_contracts.py tests/unit/test_profiles.py -vv
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest -q
```

Expected: all current tests pass.

- [ ] Commit the scaffold and contracts:

```bash
git add .gitignore pyproject.toml pytest.ini src/qwen32_cluster config \
  tests/conftest.py tests/unit
git commit -m "feat: define dual-mac cluster profiles"
```

## Task 2: Implement Weighted Reverse-Rank Partitioning and the Standalone Qwen3 Adapter

**Files:**

- Create: `src/qwen32_cluster/qwen3_pipeline.py`
- Create: `tests/support/__init__.py`
- Create: `tests/support/model_fixtures.py`
- Create: `tests/unit/test_qwen3_pipeline.py`
- Create: `tests/integration/test_custom_model_file.py`

`qwen3_pipeline.py` is both an installed source module and the exact file copied beside each derived model. It must be self-contained and use only standard-library imports plus absolute `mlx`, `mlx.nn`, and `mlx_lm...` imports. No relative import is allowed.

- [ ] Write failing pure tests for this interface:

```python
@dataclass(frozen=True)
class PipelinePartition:
    rank: int
    world_size: int
    stage_index: int
    start: int
    end: int

def partition_layers(
    num_layers: int,
    rank: int,
    world_size: int,
    stage_layers: Optional[Sequence[int]] = None,
) -> PipelinePartition:
    ...
```

- [ ] Assert `partition_layers(64, rank=1, world_size=2, stage_layers=[40, 24])` returns `[0, 40)` and Rank 0 returns `[40, 64)`. Assert balanced fallback covers every layer once, with contiguous, non-overlapping ranges.

- [ ] Add RED cases for rank out of range, world size not equal to one or two, wrong number of stage entries, non-positive entries, and a stage total other than `num_layers`.

- [ ] Run the pure RED test:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest \
  tests/unit/test_qwen3_pipeline.py::test_weighted_partition_maps_forward_stages_to_reverse_ranks -vv
```

Expected: import or attribute failure for `partition_layers`.

- [ ] Implement `PipelinePartition` and `partition_layers()` with `stage_index = world_size - 1 - rank`. Preserve single-rank behavior as `[0, num_layers)`.

- [ ] Add failing model-construction tests for these interfaces:

```python
@dataclass
class ModelArgs(UpstreamModelArgs):
    pipeline_stage_layers: Optional[List[int]] = None

class Qwen3PipelineModel(UpstreamQwen3Model):
    def pipeline(self, group) -> None: ...
    @property
    def pipeline_layers(self): ...

class Model(nn.Module):
    @property
    def layers(self): ...
    def make_cache(self): ...
```

Tests must prove that Rank 1 keeps original layer keys `model.layers.0...39`, Rank 0 retains a `None` prefix and original keys `model.layers.40...63`, `Model.layers` exposes only local non-`None` layers, and `make_cache()` creates one `KVCache` per local layer.

- [ ] Test model-level pipeline validation separately from the pure partition helper: `Qwen3PipelineModel.pipeline()` accepts exactly two ranks, rejects any other group size before pruning, and is idempotent when called twice with the same group. A second call with a different rank or split must fail rather than repartition an already-pruned model. The ordinary unpartitioned call remains usable for the single-rank numerical reference test.

- [ ] Add a source-level test that rejects `from .` imports and rejects constructing an upstream top-level `Model` before replacing it. One custom `Qwen3PipelineModel` construction is allowed; a double model construction is not.

- [ ] Add a single-rank numerical test: copy one tiny upstream Qwen3 model's weights into the custom model, evaluate fixed token IDs, and assert logits agree within `rtol=1e-5, atol=1e-5`.

- [ ] Add compatibility tests for both `tie_word_embeddings=False` and `True`: the former owns `lm_head`; the latter uses `embed_tokens.as_linear()` and `sanitize()` removes an unexpected `lm_head.weight`, matching upstream Qwen3 behavior.

- [ ] Implement the minimal adapter using these absolute imports:

```python
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3 import (
    ModelArgs as UpstreamModelArgs,
    Qwen3Model as UpstreamQwen3Model,
)
```

The top-level `Model` must directly instantiate `Qwen3PipelineModel`, the appropriate LM head, and no second upstream model.

- [ ] Implement the pipeline call in this exact order:

```python
h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)
cache = [None] * len(self.pipeline_layers) if cache is None else cache
mask = create_attention_mask(h, cache[0])
if self.pipeline_rank < self.pipeline_size - 1:
    h = mx.distributed.recv_like(h, self.pipeline_rank + 1)
for layer, layer_cache in zip(self.pipeline_layers, cache):
    h = layer(h, mask, layer_cache)
if self.pipeline_rank != 0:
    sent_h = mx.distributed.send(h, self.pipeline_rank - 1)
    if cache[-1] is not None:
        cache[-1].keys = mx.depends(cache[-1].keys, sent_h)
    h = sent_h
if self.pipeline_size > 1:
    h = mx.distributed.all_gather(h)[: h.shape[0]]
return self.norm(h)
```

Do not remove embedding, norm, or LM head from either rank in this implementation.

- [ ] Build a temporary tiny model directory in `tests/support/model_fixtures.py`: save deterministic upstream Qwen3 weights, tokenizer-independent config, copied `qwen3_pipeline.py`, and `model_file: "qwen3_pipeline.py"`.

- [ ] Add a real `mlx_lm.utils.load_model(..., lazy=True)` integration test proving the custom `model_file` executes twice without global-state leakage and `pipeline_stage_layers` is parsed.

- [ ] Run targeted and full tests:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest \
  tests/unit/test_qwen3_pipeline.py tests/integration/test_custom_model_file.py -vv
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest -q
```

Expected: all tests pass; no 32B weights are loaded.

- [ ] Commit the adapter:

```bash
git add src/qwen32_cluster/qwen3_pipeline.py tests/support \
  tests/unit/test_qwen3_pipeline.py tests/integration/test_custom_model_file.py
git commit -m "feat: add weighted Qwen3 pipeline adapter"
```

## Task 3: Prove the Adapter on Two Local Ring Ranks, Including Lazy Cache Dependencies

**Files:**

- Create: `tests/integration/two_rank_pipeline_case.py`
- Create: `tests/integration/test_two_rank_pipeline.py`

- [ ] Write a two-rank worker that initializes a local Ring group, asserts `group.size() == 2` before model construction, constructs the same deterministic two-layer tiny Qwen3 on both ranks, applies `pipeline_stage_layers=[1, 1]`, and writes one JSON result per rank into a pytest temporary directory.

- [ ] Add a failing wrapper test that launches the worker with a 90-second hard deadline:

```bash
/Users/Shared/mlx-cluster/.venv/bin/mlx.launch \
  --backend ring --repeat-hosts 2 --connections-per-ip 1 --starting-port 33323 -- \
  /Users/Shared/mlx-cluster/.venv/bin/python tests/integration/two_rank_pipeline_case.py \
  --case forward --output-dir "$TMPDIR/qwen32-forward"
```

The wrapper must first verify that test ports 33323–33324 are free, must never fall back to legacy ports 32323–32324, and must kill the process group on timeout. It fails if a configured port is occupied or either rank result is missing, malformed, or non-zero. Test every local two-rank launch builder for the same fixed connection count and optimized starting port.

- [ ] Assert final logits on both ranks match the unpartitioned tiny reference within `rtol=1e-5, atol=1e-5`, and both ranks report the same finite checksum.

- [ ] Add a RED `cache_dependency` case that performs chunked prefill, discards the returned logits, evaluates only each cache's state, and then decodes one token. Compare cache offsets, cache hashes, and logits with an unpartitioned reference.

- [ ] Confirm that removing `mx.depends(cache[-1].keys, sent_h)` makes the cache test fail or hang, then restore the dependency.

- [ ] Add sequence-order corruption in the test worker and assert the wrapper classifies it as failure rather than silently passing.

- [ ] Run the local integration test three times to catch intermittent ordering errors:

```bash
for run_id in 1 2 3; do
  /Users/Shared/mlx-cluster/.venv/bin/python -m pytest \
    -m local_integration tests/integration/test_two_rank_pipeline.py -vv || exit 1
done
```

Expected: all three iterations pass within 90 seconds each.

- [ ] Run the full non-cluster suite and commit:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest -q -m "not cluster"
git add tests/integration/two_rank_pipeline_case.py tests/integration/test_two_rank_pipeline.py
git commit -m "test: verify two-rank Qwen3 pipeline ordering"
```

## Task 4: Build a Raw Safetensor Reader and Byte-Preserving Shard Writer

**Files:**

- Create: `src/qwen32_cluster/safetensor_raw.py`
- Create: `tests/unit/test_safetensor_raw.py`

- [ ] Write failing tests for these immutable types and functions:

```python
@dataclass(frozen=True)
class TensorRecord:
    name: str
    dtype: str
    shape: tuple[int, ...]
    source_file: Path
    start: int
    end: int
    nbytes: int

@dataclass(frozen=True)
class SourceShard:
    path: Path
    header_length: int
    data_start: int
    tensors: tuple[TensorRecord, ...]

def read_header(path: Path) -> SourceShard: ...
def copy_payload(record: TensorRecord, dst_fd: int, chunk_bytes: int = 8 << 20) -> str: ...
def write_shard(records: Sequence[TensorRecord], output_tmp: Path) -> ShardResult: ...
```

- [ ] Generate a small fixture with `mx.save_safetensors` containing U32 packed weights, BF16 scales/biases, and norms. Reorder it through `write_shard()` and assert every tensor's name, dtype, shape, byte count, and raw payload SHA-256 are identical.

- [ ] Monkeypatch `os.pread` to assert every payload read is at most 8 MiB and peak RSS does not scale with total fixture payload size.

- [ ] Add malformed-input cases: safetensor header length beyond EOF, negative/overlapping/out-of-range offsets, duplicate JSON keys, tensor byte size inconsistent with dtype and shape, and destination path already existing.

- [ ] Add an import test that fails if `safetensor_raw.py` imports `mlx.nn`, `mlx_lm`, NumPy, or calls `mx.load`/`mx.save_safetensors`.

- [ ] Run the RED test:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest tests/unit/test_safetensor_raw.py -vv
```

Expected: import or missing-function failures.

- [ ] Implement direct safetensors parsing: read the little-endian `u64` header length, parse JSON with duplicate-key detection, calculate absolute payload offsets from `8 + header_length`, and validate all records before opening an output file.

- [ ] Implement deterministic output headers and raw payload copying with `os.pread`/`os.write`. Encode compact sorted JSON, pad the header with ASCII spaces to the safetensors-required 8-byte alignment, and calculate every output data offset before writing. Never materialize a tensor array. Write only to an explicitly named `.tmp` path and `fsync` before returning.

- [ ] Run the targeted test, inspect imports, then full regression:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest tests/unit/test_safetensor_raw.py -vv
rg -n "mlx\.nn|mlx_lm|numpy|mx\.load|dequant" src/qwen32_cluster/safetensor_raw.py
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest -q -m "not cluster"
```

Expected: pytest passes; `rg` returns no matches.

- [ ] Commit the raw copier:

```bash
git add src/qwen32_cluster/safetensor_raw.py tests/unit/test_safetensor_raw.py
git commit -m "feat: copy quantized safetensors without decoding"
```

## Task 5: Plan and Write Deterministic Rank-Local Model Packs

**Files:**

- Create: `src/qwen32_cluster/rank_pack.py`
- Create: `tests/unit/test_rank_pack.py`

- [ ] Write failing tests for the pack-planning interface:

```python
def module_unit(key: str) -> str: ...

def select_rank_keys(
    weight_map: Mapping[str, str],
    *,
    rank: int,
    world_size: int,
    stage_layers: Sequence[int],
    shared_prefixes: Sequence[str],
) -> frozenset[str]: ...

def plan_rank_pack(
    source_dir: Path,
    output_dir: Path,
    profile: Profile,
    rank: int,
    max_shard_bytes: int = 768 << 20,
) -> RankPackPlan: ...

def pack_rank(plan: RankPackPlan, *, force: bool = False) -> RankManifest: ...
```

- [ ] Assert the shared prefixes are exactly `model.embed_tokens.`, `model.norm.`, and `lm_head.`. For the 40/24 profile, Rank 1 gets layers `[0, 40)` and Rank 0 gets `[40, 64)`; for 36/28, Rank 1 gets `[0, 36)` and Rank 0 gets `[36, 64)`.

- [ ] Assert `module_unit()` groups all tensors for one `model.layers.N` together even when their source tensors span multiple original shards. The output bin-packer must not split one layer across destination shards. A single module unit larger than `max_shard_bytes` fails before writing.

- [ ] Assert deterministic planning and filenames `model-00001-of-000NN.safetensors`, stable ordering by module unit and tensor key, no remote-layer key, all shared keys, and complete local layers.

- [ ] Add filesystem safety tests: non-empty destination fails without `force`; `force` permits only a directory carrying this tool's matching staging marker; every shard is written as `.tmp`, fsynced, and atomically renamed; an interrupted pack never leaves a valid manifest.

- [ ] Run RED:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest tests/unit/test_rank_pack.py -vv
```

- [ ] Implement planning on top of `read_header()` and `write_shard()`. Compute the complete destination shard count before writing the first file.

- [ ] Copy non-weight assets by an explicit allowlist: `tokenizer*.json`, `tokenizer.model`, `*.tiktoken`, `merges.txt`, `vocab.json`, `special_tokens_map.json`, `added_tokens.json`, `chat_template.jinja`, `generation_config.json`, and `README.md`. Do not recursively copy unknown files.

- [ ] Derive `config.json` without editing the source: preserve upstream model fields, add `model_file: "qwen3_pipeline.py"` and `pipeline_stage_layers` in forward stage order, and copy the exact adapter from `src/qwen32_cluster/qwen3_pipeline.py`.

- [ ] Write `model.safetensors.index.json` whose `weight_map` exactly covers output tensors once and whose `metadata.total_size` equals the sum of tensor payload bytes.

- [ ] Run targeted tests twice and compare directory hashes for determinism:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest tests/unit/test_rank_pack.py -vv
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest tests/unit/test_rank_pack.py -vv
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest -q -m "not cluster"
```

- [ ] Commit rank packing:

```bash
git add src/qwen32_cluster/rank_pack.py tests/unit/test_rank_pack.py
git commit -m "feat: build deterministic rank-local model packs"
```

## Task 6: Add Tensor-Level Manifests, Pair Validation, and Memory Guards

**Files:**

- Create: `src/qwen32_cluster/manifest.py`
- Create: `src/qwen32_cluster/memory_guard.py`
- Create: `tests/unit/test_manifest.py`
- Create: `tests/unit/test_memory_guard.py`
- Create: `tests/integration/test_real_index_contract.py`

- [ ] Write failing manifest tests requiring these canonical fields:

```text
schema_version, profile, source_revision, source_config_sha256,
source_index_sha256, adapter_sha256, derived_config_sha256,
rank, world_size, stage_layers, layer_start, layer_end, quantization,
packed_weight_bytes, shared_prefixes,
shards[{file,size,sha256}],
tensors[{key,dtype,shape,nbytes,payload_sha256,source_basename,output_file}]
```

Absolute paths must never appear in canonical manifests.

- [ ] Test `validate_rank_pack()` against the actual destination safetensor headers: key sets equal expected keys, payload hashes match, index entries cover all tensors once, `metadata.total_size` is exact, and the local layer range is contiguous.

- [ ] Test `validate_pair()` proves the two layer sets are disjoint, their union is layers 0 through 63, their only tensor-key intersection is the three shared module families, and config/adapter/source checksums match.

- [ ] Add mutation tests that flip one payload byte, delete a key, add a remote key, alter a cut, change one adapter byte, or point an index key at the wrong shard. Each must fail validation.

- [ ] Add a live-model contract test: call `mlx_lm.utils.load_model(derived_path, lazy=True, strict=False)` so the config applies the same quantization transform as production, apply a fake two-rank group, flatten model parameter names after `pipeline()`, and assert the live keys exactly equal the rank manifest keys without evaluating tensor payloads.

- [ ] Write failing exact memory tests around:

```python
def kv_bytes(
    local_layers: int,
    tokens: int,
    kv_heads: int = 8,
    head_dim: int = 128,
    dtype_bytes: int = 2,
) -> int: ...

def projected_bytes(manifest: RankManifest, tokens: int, runtime_reserve: int) -> int: ...

def calibrated_runtime_reserve(
    observed_peak: int,
    packed_weight_bytes: int,
    observed_tokens: int,
    local_layers: int,
    safety_margin: int = 256 << 20,
) -> int: ...

def evaluate_preflight(
    device_limit: int,
    guardrail: int,
    projection: int,
) -> GuardDecision: ...
```

- [ ] Assert 8,192-token KV values are exactly 768, 896, 1,024, 1,152, and 1,280 MiB for 24, 28, 32, 36, and 40 layers. Assert the design totals: 4-bit 28/36 gives 8.843/11.136 GiB; 3-bit 24/40 gives 6.152/9.831 GiB.

- [ ] Reject more than 8,192 tokens, packed bytes inconsistent with the manifest, projections above the profile's host guardrail, and guardrails greater than the host's reported recommended working set. Test calibrated reserve as `max(0, observed_peak - packed_weight_bytes - kv_bytes(local_layers, observed_tokens)) + 256 MiB`.

- [ ] Add runtime sample tests:

```python
@dataclass(frozen=True)
class MemorySample:
    timestamp: float
    active_bytes: int
    peak_bytes: int
    cache_bytes: int
    swapouts_bytes: int
    phase: str

def parse_vm_stat(text: str, page_size: int) -> VmCounters: ...
def evaluate_runtime(
    samples: Sequence[MemorySample],
    peak_guardrail: int,
    swap_limit: int = 512 << 20,
) -> GuardDecision: ...
```

Test immediate peak failure, swap delta at or above 512 MiB, monotonic steady-decode swap growth, counter reset/wrap treated as invalid data, and correct `vm_stat` page-size conversion.

- [ ] Run the RED tests:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest \
  tests/unit/test_manifest.py tests/unit/test_memory_guard.py -vv
```

- [ ] Implement canonical manifest serialization, validation reports with explicit reasons, static KV/weight projections, `mx.get_active_memory()`/`mx.get_peak_memory()` sampling behind a lazy import, and `vm_stat` parsing. Set `source_revision` to the source config's immutable commit hash when present; otherwise use `local:` followed by `source_index_sha256`, so a local model never has an ambiguous revision.

- [ ] Add a read-only, `model_metadata`-marked contract against `/Users/Shared/mlx-cluster/models/Qwen3-32B-4bit`. Scan only the index and safetensor headers. Assert:

```text
total packed payload:      18,429,667,328 bytes
Transformer layers:       64
payload per layer:         274,289,152 bytes
embedding payload:         437,575,680 bytes
LM-head payload:           437,575,680 bytes
4-bit 28-layer rank pack:  8,555,257,856 bytes
4-bit 36-layer rank pack: 10,749,571,072 bytes
```

- [ ] Run targeted, real-metadata, and full regression tests:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest \
  tests/unit/test_manifest.py tests/unit/test_memory_guard.py -vv
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest \
  -m model_metadata tests/integration/test_real_index_contract.py -vv
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest -q -m "not cluster"
```

- [ ] Commit validation and memory guards:

```bash
git add src/qwen32_cluster/manifest.py src/qwen32_cluster/memory_guard.py \
  tests/unit/test_manifest.py tests/unit/test_memory_guard.py \
  tests/integration/test_real_index_contract.py
git commit -m "feat: validate rank packs and memory budgets"
```

## Task 7: Implement Fail-Fast Process Supervision and Independent GPU Health Probes

**Files:**

- Create: `src/qwen32_cluster/supervisor.py`
- Create: `src/qwen32_cluster/gpu_health.py`
- Create: `src/qwen32_cluster/_gpu_probe_child.py`
- Create: `tests/support/fake_rank.py`
- Create: `tests/unit/test_supervisor.py`
- Create: `tests/unit/test_gpu_health.py`
- Create: `tests/integration/test_supervisor_processes.py`

- [ ] Write RED tests for precise process identity:

```python
@dataclass(frozen=True)
class ManagedProcess:
    pid: int
    start_time: int
    command_sha256: str
    host: str
    process_group_id: int

def start_world(commands: Sequence[RankCommand], deadline_s: float) -> WorldHandle: ...
def terminate_world(handle: WorldHandle, grace_s: float = 3.0) -> None: ...
```

Tests must cover one rank exiting, one rank hanging, stale heartbeat, PID reuse, TERM escalation to KILL, SSH child cleanup, idempotent termination, and refusal to signal a process whose current start time or command hash differs from the recorded identity.

- [ ] Use real `tests/support/fake_rank.py` subprocesses for normal, early-exit, hang, stale-heartbeat, and child-process cases. Do not add test-only branches to production supervision.

- [ ] Run supervisor RED tests:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest \
  tests/unit/test_supervisor.py tests/integration/test_supervisor_processes.py -vv
```

- [ ] Implement process groups, JSONL heartbeats, monotonic deadlines, fail-fast peer shutdown, exact process verification, and TERM/grace/KILL cleanup. Never search or kill by an unscoped command substring.

- [ ] Write RED GPU tests around:

```python
def probe_gpu(argv: Sequence[str], timeout_s: float) -> GpuHealthResult: ...
def probe_cluster(hosts: Sequence[ClusterHost], timeout_s: float) -> ClusterGpuHealth: ...
```

The child process must perform deterministic small MLX matrix multiplication and reduction, call `mx.eval()` and `mx.synchronize()`, validate a checksum, and exit. Test healthy, wrong checksum, Python exception, hang, remote SSH timeout, and orphan cleanup.

- [ ] Implement `_gpu_probe_child.py` as a fresh-process workload and `gpu_health.py` as its hard-timeout parent. When a post-failure probe times out or errors, return `boot_contaminated=true` and do not retry automatically.

- [ ] Run targeted tests and regression:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest \
  tests/unit/test_supervisor.py tests/unit/test_gpu_health.py \
  tests/integration/test_supervisor_processes.py -vv
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest -q -m "not cluster"
```

- [ ] Commit supervision and GPU health:

```bash
git add src/qwen32_cluster/supervisor.py src/qwen32_cluster/gpu_health.py \
  src/qwen32_cluster/_gpu_probe_child.py tests/support/fake_rank.py \
  tests/unit/test_supervisor.py tests/unit/test_gpu_health.py \
  tests/integration/test_supervisor_processes.py
git commit -m "feat: supervise ranks and detect contaminated GPUs"
```

## Task 8: Build the Ordered Communication Gate

**Files:**

- Create: `src/qwen32_cluster/comm_probe.py`
- Create: `tests/integration/two_rank_comm_case.py`
- Create: `tests/integration/test_local_two_rank_probe.py`
- Create: `tests/cluster/test_thunderbolt_comm.py`

- [ ] Write failing configuration validation for:

```python
@dataclass(frozen=True)
class CommProbeConfig:
    iterations: int = 10_000
    hidden_size: int = 5_120
    dtype: str = "bfloat16"
    prefill_rows: tuple[int, ...] = (128, 256, 512)
    connections_per_ip: int = 1
    starting_port: int = 33_323

def run_rank_probe(config: CommProbeConfig, group) -> RankCommMetrics: ...
```

Reject world size other than two, a backend other than `ring`, any presence of `MLX_METAL_FAST_SYNCH` in the environment (including an empty or `0` value), any JACCL/RDMA selection, a connection count other than one, a cluster starting port other than 33323, any `hosts[*].ips` value or order other than the exact pair `169.254.217.74`, `169.254.82.82`, and iterations below 10,000 for a promotion gate. Validate `hosts[*].ssh` separately and allow the fixed control endpoints `127.0.0.1` and `kelly@169.254.82.82`; never mistake the local SSH endpoint for a Ring data address.

- [ ] Write a two-rank local worker with the same operation order as the model: Rank 1 produces a GPU tensor and sends it; Rank 0 receives and applies a lightweight GPU operation; both ranks perform `all_gather`. Bind send completion through `mx.depends`.

- [ ] Encode each sequence number into several bf16-exact digits below 128, so all 10,000 iterations can be checked without relying on bf16 representation of large integers.

- [ ] Emit a JSONL heartbeat every 100 iterations containing rank, sequence, last completed operation, and elapsed latency. Rank 0 reports p50/p95/p99, mismatch count, completed count, and payload hash.

- [ ] Add RED tests for swapped collective order, stale payload, corrupted payload, one rank hanging, and one rank exiting. The parent must enforce a hard deadline and terminate both ranks.

- [ ] Add short prefill-shape probes for `[1, 128, 5120]`, `[1, 256, 5120]`, and `[1, 512, 5120]`; do not repeat these large payloads 10,000 times.

- [ ] Run local RED, implement the worker and aggregation, then run local tests:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest \
  -m local_integration tests/integration/test_local_two_rank_probe.py -vv
```

- [ ] Add a cluster-marked wrapper using exactly:

```bash
VERSION_ID="$(git rev-parse HEAD)"
/Users/Shared/mlx-cluster/.venv/bin/mlx.launch \
  --backend ring \
  --hostfile /Users/Shared/mlx-cluster/hosts.json \
  --connections-per-ip 1 \
  --cwd "/Users/Shared/mlx-cluster/qwen3-32b-opt/versions/${VERSION_ID}" \
  --env "PYTHONPATH=/Users/Shared/mlx-cluster/qwen3-32b-opt/versions/${VERSION_ID}/src" \
  --starting-port 33323 -- \
  /Users/Shared/mlx-cluster/.venv/bin/python -m qwen32_cluster.cli comm-probe \
  --iterations 10000 --hidden-size 5120 \
  --output /Users/Shared/mlx-cluster/qwen3-32b-opt/reports/comm.json
```

- [ ] Do not run the real cluster marker yet. Run full local regression and commit:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest -q -m "not cluster"
git add src/qwen32_cluster/comm_probe.py tests/integration/two_rank_comm_case.py \
  tests/integration/test_local_two_rank_probe.py tests/cluster/test_thunderbolt_comm.py
git commit -m "feat: gate model launch on ordered Ring traffic"
```

## Task 9: Enforce the 8K Budget with a Single-Active-Request Streaming Proxy

**Files:**

- Create: `src/qwen32_cluster/admission.py`
- Create: `src/qwen32_cluster/proxy.py`
- Create: `tests/unit/test_admission.py`
- Create: `tests/integration/test_proxy.py`

- [ ] Write pure RED tests for:

```python
def resolve_max_tokens(payload: Mapping[str, Any], default: int = 512) -> int: ...
def count_chat_tokens(
    payload: Mapping[str, Any],
    tokenizer,
    server_template_kwargs: Optional[Mapping[str, Any]] = None,
) -> int: ...
def check_budget(prompt_tokens: int, max_tokens: int, limit: int = 8192) -> AdmissionDecision: ...
```

Mirror mlx-lm 0.31.3 precedence exactly: a non-null `max_completion_tokens` wins, otherwise use `max_tokens`, otherwise use default 512. Accept a total of exactly 8,192 and reject 8,193. Reject negative, zero, boolean, float, and string values under either field; quickly reject the resolved maximum when it exceeds 8,192 before tokenization; fail closed on tokenization/template errors. Add the bypass regression `max_tokens=1, max_completion_tokens=8192` and prove it is rejected when the prompt is non-empty.

- [ ] Assert token counting deep-copies messages, applies pinned mlx-lm's `process_message_content()` normalization, and then uses the same tokenizer chat template with `add_generation_prompt=True`. Include system messages, structured text content, `tools`, and request-level `chat_template_kwargs` such as `enable_thinking`. Merge request kwargs over configured server CLI `--chat-template-args` defaults exactly as mlx-lm does. Match upstream behavior by ignoring `tool_choice` for tokenization while forwarding it unchanged; add parity tests that would fail if `tool_choice` were inserted into the template.

- [ ] Add a real-tokenizer parity test using the existing local Qwen3 tokenizer. Compare `count_chat_tokens()` with the exact IDs produced by mlx-lm's tokenizer path.

- [ ] Add `tests/support/exact_prompt.py` with a bounded search helper that builds a real chat payload whose fully rendered Qwen template has an exact requested token count. It must verify the final count with the real tokenizer and fail if it cannot hit the target; do not assume characters, words, or repeated strings map one-to-one to tokens. Reuse this helper for the exact 8,192/8,193 proxy boundaries and Task 14's API acceptance.

- [ ] Write HTTP RED tests for `create_app(tokenizer_path, upstream, limit=8192, max_active=1, max_body_bytes=4 << 20)`: exact-budget success, over-budget OpenAI-compatible 400 error, body above 4 MiB returning 413, concurrent second request 429, upstream disconnect 502, readiness downgrade, and client cancellation closing the upstream response.

- [ ] Add model-identity RED tests. The public model alias is exactly `qwen3-32b`; `/v1/chat/completions` accepts that alias or an omitted `model`, rewrites it to upstream `default_model`, and rejects every other model string or type. Reject any non-null `adapters` or `draft_model` field before tokenization. This prevents mlx-lm 0.31.3 from interpreting client input as a local path/Hugging Face repository and unloading the accepted rank-local model.

- [ ] Make `/v1/models` a proxy-owned response rather than an upstream pass-through. It exposes one OpenAI-compatible entry with id `qwen3-32b`, contains no absolute path or Hugging Face cache name, and cannot be used to select another upstream model. Add tests proving upstream `/v1/models` is never called and a malicious path/repository model ID receives 400.

- [ ] Assert `stream=true` forwards SSE status, safe response headers, and byte chunks without response buffering. Only guarded `/v1/chat/completions` reaches upstream; proxy-owned `/v1/models` is local, and unknown generation endpoints return 404.

- [ ] Run RED:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest \
  tests/unit/test_admission.py tests/integration/test_proxy.py -vv
```

- [ ] Implement the pure validator and localhost-only FastAPI/httpx streaming proxy. Use one non-blocking semaphore slot; return 429 instead of queueing another active KV cache. Configure a 5-second upstream connect timeout and no short default httpx read timeout. Allow up to 1,800 seconds from forwarding through the first response byte so 8K prefill cannot hit the 300-second idle rule; only after the first SSE byte arrives, apply a 300-second inter-chunk idle deadline. Keep an independent 1,800-second hard request deadline.

- [ ] Run targeted and full tests, then commit:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest \
  tests/unit/test_admission.py tests/integration/test_proxy.py -vv
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest -q -m "not cluster"
git add src/qwen32_cluster/admission.py src/qwen32_cluster/proxy.py \
  tests/support/exact_prompt.py tests/unit/test_admission.py tests/integration/test_proxy.py
git commit -m "feat: guard 8K requests and stream one completion"
```

## Task 10: Implement the Staged Benchmark and Profile Selection State Machine

**Files:**

- Create: `src/qwen32_cluster/benchmark.py`
- Create: `tests/unit/test_benchmark.py`
- Create: `tests/integration/test_benchmark_processes.py`

- [ ] Write RED tests for these stages, in this fixed order:

```python
ACCEPTANCE_PLAN = (
    StageSpec("load-forward", 1, 1, repetitions=1, deadline_s=240),
    StageSpec("short", 32, 16, repetitions=1, deadline_s=180),
    StageSpec("medium", 512, 64, repetitions=1, deadline_s=300),
    StageSpec("long", 2048, 128, repetitions=1, deadline_s=600),
    StageSpec("acceptance-warmup", 7936, 256, repetitions=1, deadline_s=1800),
    StageSpec("acceptance", 7936, 256, repetitions=3, deadline_s=1800),
)
```

- [ ] Test failure classification exactly as `PASS`, `OUTPUT_FAIL`, `TIMEOUT`, `PEER_LOST`, `MEMORY_GUARD`, `SWAP_GUARD`, or `GPU_UNHEALTHY`. Any failed stage prevents every later stage from launching.

- [ ] Test that `TIMEOUT`, `PEER_LOST`, or a Metal-error signature stops the plan, terminates the peer, and invokes one fresh-process GPU probe per host. An unhealthy probe records `GPU_UNHEALTHY` plus `boot_contaminated=true`; a healthy probe still does not auto-launch the next benchmark stage—the next run must be an explicit resume with the preserved report.

- [ ] Use real fake-rank subprocesses to test Rank 1 early exit, Rank 0 hang, invalid token count, NaN/invalid output, peak memory violation, swap growth, and the second of three final runs failing.

- [ ] Write a deterministic prompt-builder test that produces an `mx.uint32` vector of exactly the requested prompt length by cycling a fixed, non-special tokenizer ID sequence. The real performance worker passes those IDs directly to `mlx_lm.generate.generate_step`; it must not call `stream_generate`, because `stream_generate` may stop on EOS before the required 256 measured decode steps.

- [ ] Test the real worker consumes exactly 7,936 prompt IDs and exactly 256 `generate_step` results with greedy sampling, validates every token is in vocabulary and every log-probability tensor is finite, and compares the generated-token hash across both ranks. Keep API/chat-template acceptance separate in Task 14.

- [ ] Lock the real worker's distributed load path and test it with a tiny rank-local model plus import spies:

```python
group = mx.distributed.init()
model, tokenizer, config = mlx_lm.utils.sharded_load(
    rank_local_path,
    pipeline_group=group,
    tensor_group=None,
    return_config=True,
)
prompt_cache = mlx_lm.models.cache.make_prompt_cache(model)
```

Assert group size is two, `model.model.pipeline(group)` has run, cache count equals local layer count, and ordinary `load()`/`load_model()` is never the real benchmark entry point. Pass the explicit local cache to `generate_step()`.

- [ ] Match the stock server's wired-memory behavior. After device/guardrail preflight, capture the old wired limit, call `mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])`, and run the stage inside `try/finally`. Before every measured run call `mx.reset_peak_memory()`; on every exit call `mx.synchronize()` and restore the prior wired limit. Tests must cover restoration after success, Python error, timeout-supervisor termination, and guard failure.

- [ ] Define timing without ambiguity: record `time_to_first_token_s` from generator start through the first yielded token; record `steady_decode_s` from the first yield through the 256th yield; report both mlx-lm-comparable `256 / steady_decode_s` and conservative `255 / steady_decode_s`. Gate the 4.0 tok/s target on the conservative value from Rank 0's end-to-end wall clock, not on summed rank time or GPU kernel time.

- [ ] Give every explicit benchmark invocation a unique, newly created `reports/benchmarks/${RUN_ID}/` directory containing canonical `profile.json` and `report.json`. Resolve a built-in or derived profile once, write the exact profile sidecar before ranks start, and make the report record its sibling path plus SHA-256. Never overwrite or reconstruct a winning profile from its name. Require every report to contain schema version, both boot IDs, MLX/mlx-lm versions, git revision, adapter/config/manifest checksums, profile, split, prefill step, profile sidecar path/hash, per-rank peak/active/cache/RSS/wired/swap samples, process exit codes, prompt and generated token counts, output hashes, prompt tok/s, and decode tok/s.

- [ ] Test the real benchmark launch builder pins `--cwd /Users/Shared/mlx-cluster/qwen3-32b-opt/versions/${VERSION_ID}` and `--env PYTHONPATH=.../versions/${VERSION_ID}/src` on `mlx.launch`. Both paths must exist on both hosts and match the report's code checksum before a rank starts.

- [ ] Test selection logic: a profile is eligible only if all stages pass; acceptance has exactly three passing measured runs after one warm-up; median decode is at least 4.0 tok/s; all guardrails pass; each run's swap delta is below 512 MiB with no steady-decode trend. Selection writes an immutable record containing the exact run directory plus report/profile hashes, then atomically updates ignored `reports/selected.json`. A verifier rejects path traversal, a mutable/missing run, or either hash mismatch. Its default output follows the JSON exit contract; `--format path` writes only the one absolute canonical profile-sidecar path plus a newline to stdout and sends diagnostics to stderr. Test both formats.

- [ ] Test prefill-step policy: start at 256; permit 128 only after 256 times out or violates memory; permit 512 only after 256 passes and an explicit comparison is requested.

- [ ] Before escalating from the 2,048-token stage to 7,936+256, derive each rank's runtime reserve from its highest observed peak at the shorter stages, add the fixed 256 MiB safety margin, and project packed weights plus 8,192-token KV plus that reserve. Classify the profile `MEMORY_GUARD` without starting the 8K run when either projection reaches its host guardrail.

- [ ] Run RED, implement the state machine and canonical report writer, and rerun:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest \
  tests/unit/test_benchmark.py tests/integration/test_benchmark_processes.py -vv
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest -q -m "not cluster"
```

- [ ] Commit benchmark orchestration:

```bash
git add src/qwen32_cluster/benchmark.py tests/unit/test_benchmark.py \
  tests/integration/test_benchmark_processes.py
git commit -m "feat: stage and classify 8K model benchmarks"
```

## Task 11: Implement CLI, Deployment, Promotion, and Idempotent Rollback

**Files:**

- Create: `src/qwen32_cluster/deployment.py`
- Create: `src/qwen32_cluster/cli.py`
- Create: `scripts/install-version.sh`
- Create: `scripts/start-optimized.sh`
- Create: `scripts/stop-optimized.sh`
- Create: `tests/unit/test_deployment.py`
- Create: `tests/integration/test_cli.py`

- [ ] Write CLI RED tests for these commands and JSON exit contracts:

```text
profile validate
profile derive
pack-rank
validate-pack
validate-pair
memory-preflight
gpu-health
comm-probe
benchmark
benchmark selected-profile [--format json|path]
proxy
deploy preflight
release create
service start
service status --deep
service promote
service rollback
service stop
```

- [ ] Write deployment-state RED tests for:

```text
PREFLIGHT -> LEGACY_STOPPED -> GPU_OK -> COMM_OK -> WORLD_READY -> CANARY_OK -> PROXY_READY -> PROMOTED
```

Any failure before `LEGACY_STOPPED` leaves the existing 14B service running. Any ordinary later failure stops only exact new PIDs, restores the recorded 14B LaunchAgent state, verifies its one-token health, and records a failure report. `GPU_UNHEALTHY` or `boot_contaminated=true` is the exception: keep every model stopped and require both Macs to restart. `service rollback` is idempotent, but it refuses to start a model on a contaminated boot.

- [ ] Assert `PREFLIGHT` checks both hosts' OS/boot IDs, Python/MLX/mlx-lm versions, source/adapter/config/manifest checksums, model directory presence, direct hostfile addresses, one connection, `MLX_METAL_FAST_SYNCH` unset, optimized Ring ports 33323–33324 free before launch, available disk, and memory projection. It validates communication configuration but does not require a pre-existing report. `COMM_OK` requires a newly passing report bound to the current boot IDs, hostfile/code checksums, starting port, and maintenance window.

- [ ] Assert world readiness requires both rank heartbeats plus a successful one-token `/v1/chat/completions` canary. `/v1/models` alone is insufficient.

- [ ] Encode and test this deployment layout:

```text
/Users/Shared/mlx-cluster/qwen3-32b-opt/
  versions/${VERSION_ID}/                  immutable code installation
  packs/${PROFILE_NAME}/model/             immutable rank-local model pack
  releases/${RELEASE_ID}/
    code -> ../../versions/${VERSION_ID}
    model -> ../../packs/${PROFILE_NAME}/model
    profile.json
    acceptance.sha256
    release-manifest.json
  current -> releases/${RELEASE_ID}
  previous -> releases/${PREVIOUS_RELEASE_ID}
```

Set `VERSION_ID` to the full `git rev-parse HEAD` value and `RELEASE_ID` to `${VERSION_ID}-${PROFILE_NAME}-${ACCEPTANCE_SHA256:0:12}`. Construct a release completely in `releases/.${RELEASE_ID}.tmp`, fsync it, then atomically rename it. Code versions, model packs, and completed releases are immutable.

- [ ] Assert promotion requires a complete eligible acceptance JSON and an existing validated release on both hosts. Atomically replace `previous` with the old `current`, then replace `current.new -> releases/${RELEASE_ID}` with `current`; a failure between those operations restores the original links.

- [ ] Cover first promotion separately: when no optimized `current` exists, leave `previous` absent and record `previous_service.kind="legacy-14b"` with the verified existing PID identity and unchanged `/Users/Shared/mlx-cluster/run-server.sh` entry point. Rollback chooses that legacy record; later optimized-to-optimized promotions use the `previous` release symlink.

- [ ] Model the actual legacy service manager: `/Users/levius/Library/LaunchAgents/com.codex.mlx-cluster.plist` has label `com.codex.mlx-cluster` and `KeepAlive=true`. Tests must prove that entering `LEGACY_STOPPED` records whether this exact job is loaded, uses `launchctl bootout gui/${UID} /Users/levius/Library/LaunchAgents/com.codex.mlx-cluster.plist` rather than killing a process that KeepAlive would restart, verifies local port 8080 and both legacy ranks are gone, and restores the prior loaded state with `launchctl bootstrap gui/${UID} ...` on failure or rollback. Never edit, unload by wildcard, disable, or delete the plist.

- [ ] If `bootout` leaves a remote legacy rank alive, compare its PID, start time, and full command to the preflight record before sending TERM and then KILL after the normal grace period. A mismatch aborts promotion and requests operator review; it never broad-matches or kills the newly started optimized rank.

- [ ] Store state in `/Users/Shared/mlx-cluster/qwen3-32b-opt/state.json` with exact PIDs, start times, command hashes, current/previous release IDs, code version, selected profile, report hash, and boot IDs. Do not store secrets or absolute source model paths in portable manifests.

- [ ] Run RED:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest \
  tests/unit/test_deployment.py tests/integration/test_cli.py -vv
```

- [ ] Implement argparse CLI wiring, state transitions, precise supervision, remote SSH commands with `ConnectTimeout`, and JSON outputs. All mutating subcommands support `--dry-run`; no action is inferred from a status command.

- [ ] Keep `qwen32_cluster.cli` imports subcommand-local. Add a subprocess test that runs non-proxy commands with FastAPI/uvicorn imports blocked and proves rank workers still start. Remote ranks use the source tree installed under the immutable version and must not depend on an editable install in the remote account.

- [ ] Test canary start against `releases/${RELEASE_ID}/model` before `current` changes and only in `LEGACY_STOPPED`. The command builder accepts an explicit validated release path for canary and uses `current/model` only after promotion. Refuse to load any 32B rank while a recorded 14B rank remains resident.

- [ ] Implement service start with the complete internal mlx-lm launch command pinned to:

```bash
/Users/Shared/mlx-cluster/.venv/bin/mlx.launch \
  --backend ring \
  --hostfile /Users/Shared/mlx-cluster/hosts.json \
  --connections-per-ip 1 \
  --cwd "${CODE_DIR}" \
  --env "PYTHONPATH=${CODE_DIR}/src" \
  --starting-port 33323 -- \
  /usr/bin/caffeinate -dims \
  /Users/Shared/mlx-cluster/.venv/bin/python -m mlx_lm server \
  --model "${MODEL_DIR}" \
  --pipeline \
  --host 127.0.0.1 \
  --port 18081 \
  --temp 0 \
  --max-tokens 512 \
  --decode-concurrency 1 \
  --prompt-concurrency 1 \
  --prompt-cache-size 0 \
  --prompt-cache-bytes 0 \
  --prefill-step-size "${PREFILL_STEP_SIZE}"
```

The command builder sets both paths explicitly on both hosts: canary uses `CODE_DIR=releases/${RELEASE_ID}/code` and `MODEL_DIR=releases/${RELEASE_ID}/model`, while the promoted service uses `CODE_DIR=current/code` and `MODEL_DIR=current/model`. It resolves both to absolute paths, verifies their code and rank-local model-manifest checksums against the candidate release, and refuses a missing, stale, or mixed pair. Tests cover first promotion with no `current`, canary while an older `current` exists, and post-promotion startup, proving each command contains the intended `--model` path. The builder reads `PREFILL_STEP_SIZE` from the accepted release's canonical `profile.json`, requires it to be exactly 128, 256, or 512, and requires the acceptance report to carry the same value; 256 is the initial candidate, not a hard-coded deployment value. Before implementing the builder, snapshot `/Users/Shared/mlx-cluster/.venv/bin/mlx.launch --help` and `python -m mlx_lm server --help` in a test fixture and assert every flag above is supported by the pinned versions. Both ranks run under `caffeinate` for the process lifetime.

- [ ] Launch the proxy from the same immutable release code, not the developer checkout or editable-package path:

```bash
PYTHONPATH="${CODE_DIR}/src" \
  /Users/Shared/mlx-cluster/.venv/bin/python -m qwen32_cluster.cli proxy \
  --listen "127.0.0.1:${PROXY_PORT}" \
  --upstream http://127.0.0.1:18081 \
  --tokenizer "${MODEL_DIR}" \
  --public-model qwen3-32b \
  --context-limit 8192 --max-active 1
```

Validate `CODE_DIR`, `MODEL_DIR`, and their release checksums before starting; `PROXY_PORT` is 18080 for canary or 8080 for production.

- [ ] Implement the proxy on the canary port first; only after deep canary and acceptance may `service promote` expose the guarded endpoint on `127.0.0.1:8080`. The legacy LaunchAgent is already stopped before `WORLD_READY`; if the new proxy fails to bind or pass its one-token check, stop the optimized world, restore that exact job with `bootstrap`, and leave `current` unchanged.

- [ ] Test that promotion first terminates and identity-verifies the 18080 canary proxy, confirms its semaphore-holding process and port are gone, and only then starts the 8080 production proxy. Canary and production proxies must never overlap, because two process-local semaphores would permit two active KV caches. If the production proxy fails, stop the optimized world and restore healthy 14B rather than leaving two proxy variants.

- [ ] Run targeted and full local tests:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest \
  tests/unit/test_deployment.py tests/integration/test_cli.py -vv
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest -q -m "not cluster"
```

- [ ] Run shell syntax checks and ensure scripts contain no broad process kill or source-service edit:

```bash
zsh -n scripts/install-version.sh scripts/start-optimized.sh scripts/stop-optimized.sh
rg -n "pkill|killall|sudo purge|swapoff|run-server\.sh.*>" scripts src/qwen32_cluster
```

Expected: syntax passes; `rg` finds no prohibited action.

- [ ] Commit control-plane implementation:

```bash
git add src/qwen32_cluster/deployment.py src/qwen32_cluster/cli.py scripts \
  tests/unit/test_deployment.py tests/integration/test_cli.py
git commit -m "feat: deploy and rollback guarded 32B service"
```

## Task 12: Install the Version and Create Validated 4-Bit Rank Packs

**Files:**

- Create during execution: `/Users/Shared/mlx-cluster/qwen3-32b-opt/versions/${VERSION_ID}/`, where `VERSION_ID=$(git rev-parse HEAD)`
- Create during execution on each host: `/Users/Shared/mlx-cluster/qwen3-32b-opt/packs/balanced-4bit/model/`
- Create during execution: `reports/install-${VERSION_ID}.json`
- Create: `tests/cluster/test_real_model_stages.py`

- [ ] Before any deployment write, record a read-only snapshot of both hosts: boot ID, macOS version, available disk, current memory pressure, swap counters, MLX versions, hostfile checksum, LaunchAgent loaded state, local launcher plus both current 14B rank PIDs/start times/commands, and original 32B directory checksum metadata.

- [ ] Write `tests/cluster/test_real_model_stages.py` against the already-tested benchmark state machine, verify it skips without explicit cluster options, and commit the test before capturing the deploy revision:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest -q \
  tests/cluster/test_real_model_stages.py
git add tests/cluster/test_real_model_stages.py
git commit -m "test: stage real dual-mac 32B validation"
```

Expected: the test is collected and skipped because no `--hostfile`/`--profile-file` was supplied.

- [ ] Run all local gates from a clean worktree:

```bash
git status --short
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest -q -m "not cluster"
```

Expected: empty `git status --short`; all tests pass.

- [ ] Set `VERSION_ID=$(git rev-parse HEAD)` and install that same immutable revision on both hosts under `versions/${VERSION_ID}` using `scripts/install-version.sh`. Compare a canonical file manifest across hosts before proceeding.

- [ ] Dry-run the balanced 4-bit packs on both ranks:

```bash
PYTHONPATH="/Users/Shared/mlx-cluster/qwen3-32b-opt/versions/${VERSION_ID}/src" \
  /Users/Shared/mlx-cluster/.venv/bin/python -m qwen32_cluster.cli pack-rank \
  --source /Users/Shared/mlx-cluster/models/Qwen3-32B-4bit \
  --output /Users/Shared/mlx-cluster/qwen3-32b-opt/packs/balanced-4bit/model \
  --profile balanced-4bit --rank 0 --world-size 2 \
  --max-shard-size 768MiB --dry-run
```

Run Rank 1 from the immutable remote source tree, not from an assumed editable install:

```bash
ssh -o ConnectTimeout=5 kelly@169.254.82.82 \
  "cd /Users/Shared/mlx-cluster/qwen3-32b-opt/versions/${VERSION_ID} && \
   PYTHONPATH=/Users/Shared/mlx-cluster/qwen3-32b-opt/versions/${VERSION_ID}/src \
   /Users/Shared/mlx-cluster/.venv/bin/python -m qwen32_cluster.cli pack-rank \
   --source /Users/Shared/mlx-cluster/models/Qwen3-32B-4bit \
   --output /Users/Shared/mlx-cluster/qwen3-32b-opt/packs/balanced-4bit/model \
   --profile balanced-4bit --rank 1 --world-size 2 \
   --max-shard-size 768MiB --dry-run"
```

Keep all non-rank arguments identical. Verify expected keys and bytes before real writes, then remove `--dry-run` for the actual remote pack.

- [ ] Pack Rank 0 locally and Rank 1 remotely. Do not run both packers against one shared directory; each host has the same logical path but a distinct rank-local content manifest.

- [ ] Validate each pack locally, copy only the two manifest JSON files into the repository's ignored `reports/` workspace, and run pair validation:

```bash
PYTHONPATH="/Users/Shared/mlx-cluster/qwen3-32b-opt/versions/${VERSION_ID}/src" \
  /Users/Shared/mlx-cluster/.venv/bin/python -m qwen32_cluster.cli validate-pack \
  --pack /Users/Shared/mlx-cluster/qwen3-32b-opt/packs/balanced-4bit/model
PYTHONPATH="/Users/Shared/mlx-cluster/qwen3-32b-opt/versions/${VERSION_ID}/src" \
  /Users/Shared/mlx-cluster/.venv/bin/python -m qwen32_cluster.cli validate-pair \
  --rank0-manifest reports/balanced-4bit-rank0.json \
  --rank1-manifest reports/balanced-4bit-rank1.json \
  --source-index /Users/Shared/mlx-cluster/models/Qwen3-32B-4bit/model.safetensors.index.json
```

- [ ] Run header-only checksum, manifest/pair, and 8K static memory preflight on both packs. Do not construct a 32B model or evaluate any tensor while 14B is still resident; abort before the maintenance window if any static gate fails.

- [ ] Enter a recorded maintenance window before any 32B load: exact-`bootout` the legacy LaunchAgent, verify both 14B ranks exited, verify ports 32323–32324 and 8080 are free, and wait for memory pressure/swap activity to settle. A remaining or mismatched legacy process aborts the 32B load.

- [ ] Because the earlier 32B tensor-parallel attempt produced a Metal GPU timeout, run one fresh-process GPU health probe on each host after the 14B ranks are gone and before the new model load, binding results to current boot IDs. If either probe errors or times out, keep every model stopped and require both Macs to be restarted; do not bootstrap 14B on that contaminated boot.

- [ ] Run the real 10,000-iteration Thunderbolt communication gate after both GPUs are healthy. If it fails, do not load model weights and preserve the report. Restore 14B only for a clean protocol/link failure with healthy post-failure probes; a Metal/timeout failure that makes either probe unhealthy keeps all models stopped.

- [ ] Only after the communication, maintenance, and GPU gates pass, run the Task 6 live-parameter contract in bounded fresh processes against both rank-local packs. It may construct each quantized rank model with `lazy=True` to compare parameter keys but must not evaluate tensor payloads; require both processes to exit and memory to settle before continuing. Treat timeout/Metal errors with the same GPU-health rules as a model-stage failure.

- [ ] Run the already committed cluster test for `load-forward` on `balanced-4bit`; do not add or modify the test after `VERSION_ID` was captured. Enforce the benchmark deadline and GPU-health behavior. Never overlap the 14B and 32B model working sets.

- [ ] If load-forward fails, preserve reports and run each host's GPU probe once. Stop the current task regardless; if either host is unhealthy, keep both models stopped and require the user to restart both Macs, and if both are healthy, restore 14B and resume only through a new explicit benchmark invocation.

- [ ] If load-forward passes, run the 32/16, 512/64, and 2,048/128 stages. Stop at the first failure. Do not yet run 7,936/256 if a memory sample is at or above a guardrail.

- [ ] Test `quality-4bit` 36/28 only when its pack preflight and short stages remain under the M4 guardrail. Neighboring 4-bit splits may be tried in two-layer increments between 32/32 and 36/28, with a fresh immutable report per profile.

- [ ] Run 7,936/256 only for a 4-bit candidate that passed all earlier stages. If it produces median decode at least 4.0 tok/s across the required three runs, skip Task 13 and proceed to Task 14. If it is stable but slower, retain it as the correctness baseline and continue to 3-bit.

- [ ] At every healthy Task 12 exit, stop exact 32B processes, restore the previously loaded 14B LaunchAgent state, and verify its one-token completion. This includes success/checkpoint exits as well as ordinary failures; a contaminated-boot exit keeps all models stopped. Final promotion happens only in Task 14.

- [ ] Preserve only ignored machine reports after the real runs; do not commit large weights, raw machine logs containing user data, or deployment state. Keep using the `VERSION_ID` captured before installation even if later documentation-only commits advance repository HEAD.

## Task 13: Acquire, Pin, Pack, and Tune the 3-Bit Fallback

**Files:**

- Create during execution on both hosts: `/Users/Shared/mlx-cluster/models/Qwen3-32B-3bit/`
- Create during execution on both hosts: `/Users/Shared/mlx-cluster/qwen3-32b-opt/packs/performance-3bit/model/`
- Create during execution only if 40/24 is stable but misses throughput: `/Users/Shared/mlx-cluster/qwen3-32b-opt/packs/aggressive-3bit/model/`
- Create during execution: `reports/download-3bit.json`

This task runs only when no 4-bit profile passes both stability and 4.0 tok/s.

All local and remote Task 13 CLI invocations use `PYTHONPATH=/Users/Shared/mlx-cluster/qwen3-32b-opt/versions/${VERSION_ID}/src`, and distributed launches also set the same version directory as `--cwd`; do not fall back to either account's editable checkout.

- [ ] Pin the upstream model to `mlx-community/Qwen3-32B-3bit` revision `b3304de15a278747adbfcf2a2713565e65baba23`. Capture `RUN_UTC=$(date -u +%Y%m%dT%H%M%SZ)`, verify neither staging path exists, run `hf download --dry-run --format json`, then download once on the M4 and copy the verified staging tree to the M3 across the Thunderbolt address:

```bash
RUN_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
ssh -o ConnectTimeout=5 kelly@169.254.82.82 \
  "/Users/Shared/mlx-cluster/.venv/bin/hf download \
   mlx-community/Qwen3-32B-3bit \
   --revision b3304de15a278747adbfcf2a2713565e65baba23 \
   --local-dir /Users/Shared/mlx-cluster/models/.Qwen3-32B-3bit.staging-${RUN_UTC} \
   --dry-run --format json"
ssh -o ConnectTimeout=5 kelly@169.254.82.82 \
  "/Users/Shared/mlx-cluster/.venv/bin/hf download \
   mlx-community/Qwen3-32B-3bit \
   --revision b3304de15a278747adbfcf2a2713565e65baba23 \
   --local-dir /Users/Shared/mlx-cluster/models/.Qwen3-32B-3bit.staging-${RUN_UTC} \
   --format json"
rsync -a --exclude=.cache/ \
  kelly@169.254.82.82:/Users/Shared/mlx-cluster/models/.Qwen3-32B-3bit.staging-${RUN_UTC}/ \
  /Users/Shared/mlx-cluster/models/.Qwen3-32B-3bit.staging-${RUN_UTC}/
```

Never write into the 4-bit model directory. Do not add `--delete`; both staging targets must be newly created and empty.

- [ ] Before promotion into the local model path, verify repository revision, every downloaded file size/hash, config quantization `bits=3` and `group_size=64`, Qwen3 architecture fields, 64 layers, 5,120 hidden size, tokenizer assets, and index coverage. Atomically rename the verified staging directory.

- [ ] Run a read-only 3-bit format check against config, index, and safetensor headers without constructing or loading the full model. Do not attempt a full single-rank 32B load on either 16 GB Mac; the first evaluated 3-bit tensors are the validated rank-local packs after the maintenance/GPU/communication gates.

- [ ] Dry-run, create, and validate rank-local `performance-3bit` packs: Rank 1 gets layers `[0, 40)` and Rank 0 gets `[40, 64)`. Validate the pair and 8K memory budgets before launch.

- [ ] Immediately before the first real 3-bit load, enter the same recorded maintenance window: stop and verify both 14B ranks are gone, wait for memory to settle, run both fresh-process GPU probes, then pass a boot-bound 10,000-iteration communication gate. Never load the 3-bit and 14B models together.

- [ ] Execute the same staged sequence: load-forward, 32/16, 512/64, 2,048/128, one 7,936/256 warm-up, and three 7,936/256 measured runs. Begin with prefill step 256.

- [ ] If 256 times out, stop and run health probes; only after both are healthy may a new explicit invocation retry the same profile with prefill step 128. A pure memory-guard failure needs no GPU probe but still starts 128 as a separate invocation. Do not treat 128 as a new quantization/profile result; record it as a configuration variant.

- [ ] If 40/24 is stable but median decode is below 4.0 tok/s and the M4 retains memory headroom, create and validate `aggressive-3bit` with Rank 1 `[0, 44)` and Rank 0 `[44, 64)`, then repeat the staged sequence.

- [ ] Search neighboring memory-safe splits in two-layer increments only if neither fixed 3-bit profile meets the target. Stop any candidate before the full 8K stage when projected or observed peak crosses a guardrail.

- [ ] Select the fastest profile that passes every acceptance condition; do not select a faster run with missing memory/swap samples, a failed repetition, or an unhealthy post-timeout GPU check.

- [ ] At every healthy Task 13 exit, stop exact 32B processes, restore the previously loaded 14B LaunchAgent state, and verify its one-token completion. A contaminated-boot exit keeps all models stopped until restart.

## Task 14: Final Acceptance, Guarded API Promotion, Rollback Drill, and Operator Notes

**Files:**

- Create: `docs/operations/qwen3-32b-dual-mac-runbook.md`
- Create: `tests/cluster/test_api_8k.py`
- Create during execution: `reports/acceptance-${PROFILE_NAME}-${RUN_UTC}.json`, where `RUN_UTC=$(date -u +%Y%m%dT%H%M%SZ)` is captured once
- Create during execution: `reports/rollback-drill-${RUN_UTC}.json`

- [ ] On the selected immutable profile and a clean boot state, enter the final recorded maintenance window: stop and verify both 14B ranks are gone, wait for memory to settle, and run both GPU health probes. Record checksums and boot IDs. An ordinary failure restores 14B and ends the attempt; `GPU_UNHEALTHY` keeps all models stopped until both Macs restart.

- [ ] While no real model world is resident, run the complete local repository suite, then the boot-bound communication gate on its own so it must pass before any real-model cluster test:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest -q -m "not cluster and not live_api"
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest -q \
  -m cluster tests/cluster/test_thunderbolt_comm.py \
  --hostfile /Users/Shared/mlx-cluster/hosts.json
```

Neither command may leave a Ring process alive. A communication failure follows the GPU-health/rollback rules before any 32B load.

- [ ] Run `tests/cluster/test_real_model_stages.py` for the selected profile. Resolve the exact sidecar through the selection verifier, which validates the immutable run, report, and profile hashes before printing one absolute path; do not infer a path from `PROFILE_NAME`. The test executes exactly one 7,936+256 warm-up followed by three consecutive measured runs with deterministic sampling settings (`temperature=0`), resets MLX peak memory before each run, and samples swap/memory throughout prefill and steady decode:

```bash
SELECTION_RECORD="reports/selected.json"
SELECTED_PROFILE_FILE="$(
  PYTHONPATH="/Users/Shared/mlx-cluster/qwen3-32b-opt/versions/${VERSION_ID}/src" \
    /Users/Shared/mlx-cluster/.venv/bin/python -m qwen32_cluster.cli \
    benchmark selected-profile --selection "${SELECTION_RECORD}" --format path
)"
test -n "${SELECTED_PROFILE_FILE}" && test -f "${SELECTED_PROFILE_FILE}"
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest -q \
  -m cluster tests/cluster/test_real_model_stages.py \
  --hostfile /Users/Shared/mlx-cluster/hosts.json \
  --profile-file "${SELECTED_PROFILE_FILE}"
```

- [ ] Verify the acceptance report programmatically:

```text
3/3 measured runs PASS
exactly 256 generated tokens per run
median decode >= 4.0 tok/s
finite valid outputs and stable output hashes
M3 MLX peak < 10.1 GiB
M4 MLX peak < 11.3 GiB
swap delta < 512 MiB per host per run
no sustained steady-decode swap growth
no Ring, SSH, Metal, or peer error
matching boot/version/code/config/adapter/manifest checksums
```

- [ ] Run `release create` on both hosts from the selected code version, profile pack, and acceptance hash. Validate the shared release ID, common code/profile/acceptance checksums, and the explicitly different expected rank-local model manifest before starting any service.

- [ ] Start the optimized server with `service start --release ${RELEASE_ID}` on internal port 18081 and proxy on canary port 18080, while leaving `current` unchanged. The 14B endpoint is intentionally unavailable during this bounded maintenance window because its model is not co-resident. Run deep one-token health, `/v1/models`, streamed `/v1/chat/completions`, client cancellation, exact 8,192 acceptance, 8,193 rejection, and a concurrent-second-request 429 test. An ordinary failure restores 14B immediately; a GPU/Metal health failure keeps every model stopped until restart.

- [ ] Add `tests/cluster/test_api_8k.py`, mark it `live_api` rather than `cluster`, and run it against the canary endpoint with `--base-url http://127.0.0.1:18080`. It must never launch Ring or load a model.

- [ ] Promote only after the acceptance report and API test pass: stop and verify the 18080 canary proxy, atomically update the release links, then start one guarded proxy on `127.0.0.1:8080`. Record previous/current targets plus exact process identities and assert only one proxy owns the single-active-request gate.

- [ ] Perform a rollback drill: stop only the optimized PIDs, restore/start the unchanged 14B LaunchAgent, verify its one-token completion, exact-`bootout` 14B again and wait for both ranks to exit, then restore the accepted 32B service and rerun its one-token completion. At no point are both models resident. Ensure neither model directory changed.

- [ ] Write `docs/operations/qwen3-32b-dual-mac-runbook.md` with exact start, status, deep-health, benchmark, stop, rollback, report locations, selected profile, measured performance, known limits, and post-Metal-timeout restart instructions.

- [ ] After promotion, run only live API verification and static repository hygiene; do not invoke cluster, local-integration, communication, or model-stage tests while the accepted server owns Ring ports and GPU memory:

```bash
/Users/Shared/mlx-cluster/.venv/bin/python -m pytest -q -m live_api \
  tests/cluster/test_api_8k.py --base-url http://127.0.0.1:8080
git diff --check
git status --short
```

Expected: the live endpoint passes without starting another model, `git diff --check` is silent, and only deliberately generated ignored reports/deployment files remain outside version control. The earlier local/cluster reports are referenced by checksum instead of being rerun.

- [ ] Commit the runbook and final cluster API tests:

```bash
git add docs/operations/qwen3-32b-dual-mac-runbook.md tests/cluster/test_api_8k.py
git commit -m "docs: operate accepted dual-mac 32B service"
```

## Execution Checkpoints

Stop for review at these checkpoints even when using subagent-driven execution:

1. After Task 3: tiny-model forward and cache behavior prove the custom pipeline protocol.
2. After Task 6: byte-preserving rank packs, exact manifests, and memory projections are proven locally.
3. After Task 11: all control-plane behavior, canary promotion, and rollback are proven with fake processes.
4. After Task 12: 4-bit real-model outcome is recorded; decide whether 3-bit is necessary from evidence.
5. After Task 13: a profile either meets the 4.0 tok/s target or the report establishes the measured hardware ceiling.
6. After Task 14: final acceptance and rollback drill are complete.

If the measured ceiling is below 4.0 tok/s after all memory-safe 3-bit splits, preserve the reports and stop. Do not weaken the 8K, stability, memory, or swap criteria to declare success. The next architectural experiment would be a custom token-broadcast generation loop that removes duplicated endpoint modules and the final all-gather, and it requires a separate approved design.
