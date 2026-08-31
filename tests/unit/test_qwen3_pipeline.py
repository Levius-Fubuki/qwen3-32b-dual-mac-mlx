from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3 import Model as UpstreamModel
from mlx_lm.models.qwen3 import ModelArgs as UpstreamModelArgs


class FakeGroup:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def qwen3_pipeline():
    try:
        return import_module("qwen32_cluster.qwen3_pipeline")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Qwen3 pipeline adapter is not implemented: {exc}")


def model_args(
    *,
    num_layers: int = 4,
    stage_layers: list[int] | None = None,
    tie_word_embeddings: bool = False,
):
    module = qwen3_pipeline()
    return module.ModelArgs(
        model_type="qwen3",
        hidden_size=16,
        num_hidden_layers=num_layers,
        intermediate_size=24,
        num_attention_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=32,
        num_key_value_heads=1,
        max_position_embeddings=128,
        rope_theta=10_000.0,
        head_dim=8,
        tie_word_embeddings=tie_word_embeddings,
        pipeline_stage_layers=stage_layers,
    )


def layer_numbers(model) -> set[int]:
    keys = (name for name, _ in tree_flatten(model.parameters()))
    return {
        int(name.split(".")[2])
        for name in keys
        if name.startswith("model.layers.")
    }


def test_weighted_partition_maps_forward_stages_to_reverse_ranks() -> None:
    module = qwen3_pipeline()
    first = module.partition_layers(64, rank=1, world_size=2, stage_layers=[40, 24])
    last = module.partition_layers(64, rank=0, world_size=2, stage_layers=[40, 24])

    assert (first.stage_index, first.start, first.end) == (0, 0, 40)
    assert (last.stage_index, last.start, last.end) == (1, 40, 64)


def test_balanced_partition_is_contiguous_nonoverlapping_and_complete() -> None:
    module = qwen3_pipeline()
    partitions = sorted(
        (
            module.partition_layers(65, rank=rank, world_size=2)
            for rank in range(2)
        ),
        key=lambda partition: partition.start,
    )

    assert [(part.start, part.end) for part in partitions] == [(0, 33), (33, 65)]
    assert [layer for part in partitions for layer in range(part.start, part.end)] == list(
        range(65)
    )


def test_single_rank_partition_preserves_the_whole_model() -> None:
    module = qwen3_pipeline()
    partition = module.partition_layers(7, rank=0, world_size=1)
    assert (partition.stage_index, partition.start, partition.end) == (0, 0, 7)


@pytest.mark.parametrize(
    ("num_layers", "rank", "world_size", "stage_layers"),
    [
        (4, -1, 2, None),
        (4, 2, 2, None),
        (4, 0, 0, None),
        (4, 0, 3, None),
        (4, 0, 2, [4]),
        (4, 0, 2, [2, 0]),
        (4, 0, 2, [2, -1]),
        (4, 0, 2, [3, 2]),
    ],
)
def test_partition_rejects_invalid_topologies(
    num_layers: int,
    rank: int,
    world_size: int,
    stage_layers: list[int] | None,
) -> None:
    module = qwen3_pipeline()
    with pytest.raises(ValueError):
        module.partition_layers(num_layers, rank, world_size, stage_layers)


@pytest.mark.parametrize(
    ("rank", "expected_range"),
    [(1, range(0, 40)), (0, range(40, 64))],
)
def test_pipeline_pruning_retains_original_layer_keys(
    rank: int, expected_range: range
) -> None:
    module = qwen3_pipeline()
    model = module.Model(model_args(num_layers=64, stage_layers=[40, 24]))

    model.model.pipeline(FakeGroup(rank, 2))

    assert layer_numbers(model) == set(expected_range)
    assert len(model.layers) == len(expected_range)
    assert all(layer is not None for layer in model.layers)
    if rank == 0:
        assert model.model.layers[:40] == [None] * 40
        assert len(model.model.layers) == 64
    else:
        assert len(model.model.layers) == 40


def test_pipeline_keeps_embedding_norm_and_lm_head_on_both_ranks() -> None:
    module = qwen3_pipeline()
    for rank in (0, 1):
        model = module.Model(model_args(num_layers=4, stage_layers=[3, 1]))
        model.model.pipeline(FakeGroup(rank, 2))
        keys = {name for name, _ in tree_flatten(model.parameters())}
        assert "model.embed_tokens.weight" in keys
        assert "model.norm.weight" in keys
        assert "lm_head.weight" in keys


def test_make_cache_returns_exactly_one_kv_cache_per_local_layer() -> None:
    module = qwen3_pipeline()
    model = module.Model(model_args(num_layers=7, stage_layers=[5, 2]))
    model.model.pipeline(FakeGroup(1, 2))

    cache = model.make_cache()

    assert len(cache) == 5
    assert all(isinstance(item, KVCache) for item in cache)


def test_pipeline_rejects_non_two_rank_group_before_pruning() -> None:
    module = qwen3_pipeline()
    model = module.Model(model_args(num_layers=4, stage_layers=[3, 1]))
    original_layers = tuple(model.model.layers)

    with pytest.raises(ValueError, match="exactly two ranks"):
        model.model.pipeline(FakeGroup(0, 1))

    assert tuple(model.model.layers) == original_layers


def test_pipeline_is_idempotent_for_the_same_rank_and_split() -> None:
    module = qwen3_pipeline()
    model = module.Model(model_args(num_layers=4, stage_layers=[3, 1]))
    group = FakeGroup(0, 2)
    model.model.pipeline(group)
    retained = tuple(model.model.layers)

    model.model.pipeline(group)

    assert tuple(model.model.layers) == retained


def test_pipeline_refuses_repartitioning_with_a_different_rank() -> None:
    module = qwen3_pipeline()
    model = module.Model(model_args(num_layers=4, stage_layers=[3, 1]))
    model.model.pipeline(FakeGroup(1, 2))

    with pytest.raises(RuntimeError, match="already partitioned"):
        model.model.pipeline(FakeGroup(0, 2))


def test_pipeline_refuses_repartitioning_after_the_split_changes() -> None:
    module = qwen3_pipeline()
    model = module.Model(model_args(num_layers=4, stage_layers=[3, 1]))
    group = FakeGroup(1, 2)
    model.model.pipeline(group)
    model.args.pipeline_stage_layers = [2, 2]

    with pytest.raises(RuntimeError, match="already partitioned"):
        model.model.pipeline(group)


def test_unpartitioned_single_rank_logits_match_upstream_qwen3() -> None:
    module = qwen3_pipeline()
    custom_args = model_args(num_layers=3)
    upstream_args = UpstreamModelArgs.from_dict(vars(custom_args))
    mx.random.seed(11)
    upstream = UpstreamModel(upstream_args)
    custom = module.Model(custom_args)
    custom.load_weights(list(tree_flatten(upstream.parameters())), strict=True)
    tokens = mx.array([[1, 7, 3, 9]])

    upstream_logits = upstream(tokens)
    custom_logits = custom(tokens)
    mx.eval(upstream_logits, custom_logits)

    assert mx.allclose(
        upstream_logits, custom_logits, rtol=1e-5, atol=1e-5
    ).item()


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_lm_head_and_sanitize_match_upstream_qwen3(
    tie_word_embeddings: bool,
) -> None:
    module = qwen3_pipeline()
    custom_args = model_args(tie_word_embeddings=tie_word_embeddings)
    upstream_args = UpstreamModelArgs.from_dict(vars(custom_args))
    custom = module.Model(custom_args)
    upstream = UpstreamModel(upstream_args)

    assert hasattr(custom, "lm_head") is (not tie_word_embeddings)
    tokens = mx.array([[2, 4]])
    if tie_word_embeddings:
        hidden = custom.model(tokens)
        assert mx.allclose(custom(tokens), custom.model.embed_tokens.as_linear(hidden)).item()

    custom_weights = {"lm_head.weight": mx.ones((32, 16)), "keep": mx.ones((1,))}
    upstream_weights = dict(custom_weights)
    assert set(custom.sanitize(custom_weights)) == set(upstream.sanitize(upstream_weights))


def test_adapter_source_is_standalone_and_constructs_one_qwen3_body() -> None:
    source_path = (
        Path(__file__).parents[2]
        / "src"
        / "qwen32_cluster"
        / "qwen3_pipeline.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert not any(
        isinstance(node, ast.ImportFrom) and node.level
        for node in ast.walk(tree)
    ), "custom model_file cannot use relative imports"
    assert "from mlx_lm.models.base import create_attention_mask" in source
    assert "from mlx_lm.models.cache import KVCache" in source
    assert "from mlx_lm.models.qwen3 import (" in source
    assert "ModelArgs as UpstreamModelArgs" in source
    assert "Qwen3Model as UpstreamQwen3Model" in source

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Qwen3PipelineModel"
    ]
    assert len(calls) == 1
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "mlx_lm.models.qwen3"
        and any(alias.name == "Model" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "UpstreamModel"
        for node in ast.walk(tree)
    )
