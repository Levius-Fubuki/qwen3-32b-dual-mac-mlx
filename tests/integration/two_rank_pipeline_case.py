from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

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


class DistributedCallTrace:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._call: str | None = None
        distributed = mx.distributed
        self._recv_like = distributed.recv_like
        self._send = distributed.send
        self._all_gather = distributed.all_gather

        def recv_like(value, source, *args, **kwargs):
            self._record("recv", peer=source)
            return self._recv_like(value, source, *args, **kwargs)

        def send(value, destination, *args, **kwargs):
            self._record("send", peer=destination)
            return self._send(value, destination, *args, **kwargs)

        def all_gather(value, *args, **kwargs):
            self._record("all_gather")
            return self._all_gather(value, *args, **kwargs)

        distributed.recv_like = recv_like
        distributed.send = send
        distributed.all_gather = all_gather

    def _record(self, event: str, **details: Any) -> None:
        if self._call is None:
            raise RuntimeError(f"distributed {event} constructed outside a traced call")
        self.events.append({"call": self._call, "event": event} | details)

    def construct(self, call: str, function: Callable[[], mx.array]) -> mx.array:
        if self._call is not None:
            raise RuntimeError("nested traced calls are not supported")
        self._call = call
        try:
            return function()
        finally:
            self._call = None

    def eval_complete(self, call: str, target: str) -> None:
        self.events.append(
            {"call": call, "event": "eval_complete", "target": target}
        )


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


def _base_result(
    case: str,
    group,
    pipeline: PipelineModel,
    trace: DistributedCallTrace,
) -> dict[str, Any]:
    local_layers = [
        index
        for index, layer in enumerate(pipeline.model.layers)
        if layer is not None
    ]
    return {
        "case": case,
        "rank": group.rank(),
        "world_size": group.size(),
        "status": "ok",
        "exit_code": 0,
        "error": None,
        "events": trace.events,
        "local_layers": local_layers,
        "partition": {
            "start": pipeline.model.start_idx,
            "end": pipeline.model.end_idx,
        },
    }


def _forward_case(
    case: str,
    group,
    reference: UpstreamModel,
    pipeline: PipelineModel,
    trace: DistributedCallTrace,
) -> dict[str, Any]:
    tokens = mx.array([[1, 7, 3, 9], [2, 6, 4, 8]])
    reference_logits = reference(tokens)
    logits = trace.construct("forward", lambda: pipeline(tokens))
    mx.eval(reference_logits, logits)
    trace.eval_complete("forward", "logits")
    result = _base_result(case, group, pipeline, trace)
    result.update(
        {
            "input_shape": list(tokens.shape),
            "logits_shape": list(logits.shape),
            "reference_logits_shape": list(reference_logits.shape),
            "logits": logits.tolist(),
            "reference_logits": reference_logits.tolist(),
            "checksum": float(mx.sum(logits.astype(mx.float32)).item()),
        }
    )
    return result


def _cache_dependency_case(
    case: str,
    group,
    reference: UpstreamModel,
    pipeline: PipelineModel,
    trace: DistributedCallTrace,
) -> dict[str, Any]:
    reference_caches = [KVCache(), KVCache()]
    pipeline_caches = pipeline.make_cache()
    tokens = mx.array([[1, 7, 3, 9], [2, 6, 4, 8]])
    prefill_input_shapes = []

    for chunk_index, start in enumerate((0, 2)):
        chunk = tokens[:, start : start + 2]
        prefill_input_shapes.append(list(chunk.shape))
        reference(chunk, cache=reference_caches)
        call = f"prefill_{chunk_index}"
        trace.construct(call, lambda chunk=chunk: pipeline(chunk, cache=pipeline_caches))
        mx.eval(*_cache_tensors(reference_caches))
        mx.eval(*_cache_tensors(pipeline_caches))
        trace.eval_complete(call, "cache_state")

    local_layers = [
        index
        for index, layer in enumerate(pipeline.model.layers)
        if layer is not None
    ]
    if len(local_layers) != 1:
        raise AssertionError(f"expected one local layer, got {local_layers}")
    layer = local_layers[0]
    local_cache = pipeline_caches[0]
    reference_cache = reference_caches[layer]
    prefill_cache = _cache_record(layer, local_cache, reference_cache)

    decode_tokens = mx.array([[10], [11]])
    reference_decode_logits = reference(decode_tokens, cache=reference_caches)
    decode_logits = trace.construct(
        "decode", lambda: pipeline(decode_tokens, cache=pipeline_caches)
    )
    mx.eval(
        reference_decode_logits,
        decode_logits,
        *_cache_tensors(reference_caches),
        *_cache_tensors(pipeline_caches),
    )
    trace.eval_complete("decode", "logits_and_cache_state")

    result = _base_result(case, group, pipeline, trace)
    result.update(
        {
            "prefill_input_shapes": prefill_input_shapes,
            "decode_input_shape": list(decode_tokens.shape),
            "decode_logits_shape": list(decode_logits.shape),
            "reference_decode_logits_shape": list(reference_decode_logits.shape),
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
    trace: DistributedCallTrace,
    output_dir: Path,
) -> None:
    pipeline_caches = pipeline.make_cache()
    tokens = mx.array([[1, 7], [2, 6]])
    trace.construct(
        "negative_prefill", lambda: pipeline(tokens, cache=pipeline_caches)
    )
    marker = output_dir / "rank-1-cache-evaluated-without-send.marker"

    if rank == 1:
        mx.eval(*_cache_tensors(pipeline_caches))
        trace.eval_complete("negative_prefill", "cache_state")
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


def _run(case: str, output_dir: Path, observation: dict[str, Any]) -> None:
    group = mx.distributed.init(strict=True, backend="ring")
    if group.size() != 2:
        raise AssertionError(f"expected exactly two local Ring ranks, got {group.size()}")
    rank = group.rank()
    observation.update({"rank": rank, "world_size": group.size()})

    reference, pipeline = _build_models(group)
    trace = DistributedCallTrace()
    observation["events"] = trace.events

    if case == "cache_dependency_bypassed":
        mx.depends = lambda value, *dependencies: value
        _negative_cache_dependency_case(rank, pipeline, trace, output_dir)

    if case in ("forward", "sequence_corruption"):
        result = _forward_case(case, group, reference, pipeline, trace)
        if case == "sequence_corruption" and rank == 1:
            result["events"][0:2] = reversed(result["events"][0:2])
    else:
        result = _cache_dependency_case(case, group, reference, pipeline, trace)

    _write_result(output_dir, rank, result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    observation: dict[str, Any] = {"events": []}
    try:
        _run(args.case, args.output_dir, observation)
    except BaseException as exc:
        file_rank = observation.get("rank", int(os.environ.get("MLX_RANK", "-1")))
        error_result = {
            "case": args.case,
            "rank": observation.get("rank"),
            "world_size": observation.get("world_size"),
            "status": "error",
            "exit_code": 1,
            "error": f"{type(exc).__name__}: {exc}",
            "events": observation["events"],
        }
        try:
            _write_result(args.output_dir, file_rank, error_result)
        finally:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
