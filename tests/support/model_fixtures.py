from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3 import Model as UpstreamModel
from mlx_lm.models.qwen3 import ModelArgs as UpstreamModelArgs


def tiny_qwen3_config(
    *,
    num_hidden_layers: int = 4,
    tie_word_embeddings: bool = False,
    pipeline_stage_layers: list[int] | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model_type": "qwen3",
        "hidden_size": 16,
        "num_hidden_layers": num_hidden_layers,
        "intermediate_size": 24,
        "num_attention_heads": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": 32,
        "num_key_value_heads": 1,
        "max_position_embeddings": 128,
        "rope_theta": 10_000.0,
        "head_dim": 8,
        "tie_word_embeddings": tie_word_embeddings,
        "model_file": "qwen3_pipeline.py",
    }
    if pipeline_stage_layers is not None:
        config["pipeline_stage_layers"] = pipeline_stage_layers
    return config


def build_tiny_qwen3_model_directory(
    destination: Path,
    *,
    tie_word_embeddings: bool = False,
    pipeline_stage_layers: list[int] | None = None,
) -> Path:
    """Create tokenizer-independent deterministic Qwen3 weights and adapter."""
    destination.mkdir()
    config = tiny_qwen3_config(
        tie_word_embeddings=tie_word_embeddings,
        pipeline_stage_layers=pipeline_stage_layers,
    )

    args = UpstreamModelArgs.from_dict(config)
    mx.random.seed(1729)
    upstream = UpstreamModel(args)
    weights = dict(tree_flatten(upstream.parameters()))
    mx.save_safetensors(str(destination / "model.safetensors"), weights)

    source = (
        Path(__file__).parents[2]
        / "src"
        / "qwen32_cluster"
        / "qwen3_pipeline.py"
    )
    shutil.copy2(source, destination / "qwen3_pipeline.py")
    (destination / "config.json").write_text(
        json.dumps(config, sort_keys=True), encoding="utf-8"
    )
    return destination
