from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3 import Model as UpstreamModel
from mlx_lm.models.qwen3 import ModelArgs as UpstreamModelArgs

from qwen32_cluster.qwen3_pipeline import Model as PipelineModel
from qwen32_cluster.qwen3_pipeline import ModelArgs as PipelineModelArgs


CASES = (
    "forward",
    "cache_dependency",
    "cache_dependency_bypassed",
    "sequence_corruption",
)
NEGATIVE_MARKER_TIMEOUT_SECONDS = 5


def _model_config() -> dict[str, Any]:
    return {
        "model_type": "qwen3",
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "intermediate_size": 24,
        "num_attention_heads": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": 32,
        "num_key_value_heads": 1,
        "max_position_embeddings": 128,
        "rope_theta": 10_000.0,
        "head_dim": 8,
        "tie_word_embeddings": False,
    }


def _build_models(group):
    upstream_args = UpstreamModelArgs.from_dict(_model_config())
    pipeline_args = PipelineModelArgs.from_dict(
        _model_config() | {"pipeline_stage_layers": [1, 1]}
    )
    mx.random.seed(1729)
    reference = UpstreamModel(upstream_args)
    pipeline = PipelineModel(pipeline_args)
    pipeline.load_weights(list(tree_flatten(reference.parameters())), strict=True)
    pipeline.model.pipeline(group)
    return reference, pipeline


def _array_hash(value: mx.array) -> str:
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def _cache_tensors(caches: list[KVCache]) -> list[mx.array]:
    return [tensor for cache in caches for tensor in cache.state]


def _cache_record(
    layer: int,
    local_cache: KVCache,
    reference_cache: KVCache,
) -> dict[str, Any]:
    return {
        "layer": layer,
        "offset": local_cache.offset,
        "reference_offset": reference_cache.offset,
        "keys_hash": _array_hash(local_cache.state[0]),
        "reference_keys_hash": _array_hash(reference_cache.state[0]),
        "values_hash": _array_hash(local_cache.state[1]),
        "reference_values_hash": _array_hash(reference_cache.state[1]),
    }


def _base_result(rank: int, sequence: list[str]) -> dict[str, Any]:
    return {
        "rank": rank,
        "world_size": 2,
        "status": "ok",
        "exit_code": 0,
        "error": None,
        "sequence": sequence,
        "local_layers": [1] if rank == 0 else [0],
        "batch_size": 2,
    }


def _forward_case(
    rank: int,
    reference: UpstreamModel,
    pipeline: PipelineModel,
    sequence: list[str],
) -> dict[str, Any]:
    tokens = mx.array([[1, 7, 3, 9], [2, 6, 4, 8]])
    reference_logits = reference(tokens)
    logits = pipeline(tokens)
    mx.eval(reference_logits, logits)
    sequence.extend((["receive"] if rank == 0 else ["send"]) + ["all_gather"])
    result = _base_result(rank, sequence + ["result"])
    result.update(
        {
            "logits": logits.tolist(),
            "reference_logits": reference_logits.tolist(),
            "checksum": float(mx.sum(logits.astype(mx.float32)).item()),
        }
    )
    return result


def _cache_dependency_case(
    rank: int,
    reference: UpstreamModel,
    pipeline: PipelineModel,
    sequence: list[str],
) -> dict[str, Any]:
    reference_caches = [KVCache(), KVCache()]
    pipeline_caches = pipeline.make_cache()
    tokens = mx.array([[1, 7, 3, 9], [2, 6, 4, 8]])

    for start in (0, 2):
        reference(tokens[:, start : start + 2], cache=reference_caches)
        pipeline(tokens[:, start : start + 2], cache=pipeline_caches)
        mx.eval(*_cache_tensors(reference_caches))
        mx.eval(*_cache_tensors(pipeline_caches))

    layer = 1 if rank == 0 else 0
    local_cache = pipeline_caches[0]
    reference_cache = reference_caches[layer]
    prefill_cache = _cache_record(layer, local_cache, reference_cache)

    decode_tokens = mx.array([[10], [11]])
    reference_decode_logits = reference(decode_tokens, cache=reference_caches)
    decode_logits = pipeline(decode_tokens, cache=pipeline_caches)
    mx.eval(
        reference_decode_logits,
        decode_logits,
        *_cache_tensors(reference_caches),
        *_cache_tensors(pipeline_caches),
    )
    sequence.extend((["receive"] if rank == 0 else ["send"]) + ["all_gather"])

    result = _base_result(rank, sequence + ["result"])
    result.update(
        {
            "decode_logits": decode_logits.tolist(),
            "reference_decode_logits": reference_decode_logits.tolist(),
            "checksum": float(mx.sum(decode_logits.astype(mx.float32)).item()),
            "prefill_caches": [prefill_cache],
            "caches": [_cache_record(layer, local_cache, reference_cache)],
        }
    )
    return result


def _negative_cache_dependency_case(
    rank: int,
    pipeline: PipelineModel,
    output_dir: Path,
) -> None:
    pipeline_caches = pipeline.make_cache()
    tokens = mx.array([[1, 7], [2, 6]])
    pipeline(tokens, cache=pipeline_caches)
    marker = output_dir / "rank-1-cache-evaluated-without-send.marker"

    if rank == 1:
        mx.eval(*_cache_tensors(pipeline_caches))
        output_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text("cache-only evaluation completed", encoding="utf-8")
        raise RuntimeError(
            "cache-only evaluation completed without the required Ring send dependency"
        )

    deadline = time.monotonic() + NEGATIVE_MARKER_TIMEOUT_SECONDS
    while time.monotonic() < deadline and not marker.is_file():
        time.sleep(0.01)
    if not marker.is_file():
        raise TimeoutError("rank 1 cache-only evaluation did not produce its marker")
    raise RuntimeError(
        "rank 1 cache-only evaluation bypassed the required Ring send dependency"
    )


def _write_result(output_dir: Path, rank: int, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"rank-{rank}.json"
    temporary = output_dir / f".rank-{rank}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    os.replace(temporary, destination)


def _run(case: str, output_dir: Path) -> None:
    group = mx.distributed.init(strict=True, backend="ring")
    if group.size() != 2:
        raise AssertionError(f"expected exactly two local Ring ranks, got {group.size()}")
    rank = group.rank()
    sequence = ["group_ready"]

    reference, pipeline = _build_models(group)
    sequence.append("model_partitioned")

    if case == "cache_dependency_bypassed":
        mx.depends = lambda value, *dependencies: value
        _negative_cache_dependency_case(rank, pipeline, output_dir)

    if case in ("forward", "sequence_corruption"):
        result = _forward_case(rank, reference, pipeline, sequence)
        if case == "sequence_corruption" and rank == 1:
            result["sequence"][2:4] = reversed(result["sequence"][2:4])
    else:
        result = _cache_dependency_case(rank, reference, pipeline, sequence)

    _write_result(output_dir, rank, result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        _run(args.case, args.output_dir)
    except BaseException as exc:
        rank = int(os.environ.get("MLX_RANK", "-1"))
        error_result = {
            "rank": rank,
            "world_size": None,
            "status": "error",
            "exit_code": 1,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "sequence": [],
        }
        try:
            _write_result(args.output_dir, rank, error_result)
        finally:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
