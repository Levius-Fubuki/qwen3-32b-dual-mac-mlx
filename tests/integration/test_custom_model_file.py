from __future__ import annotations

from mlx_lm.utils import load_model

from tests.support.model_fixtures import build_tiny_qwen3_model_directory


def test_custom_model_file_loads_twice_without_global_state_leakage(tmp_path) -> None:
    model_path = build_tiny_qwen3_model_directory(
        tmp_path / "tiny-qwen3",
        pipeline_stage_layers=[3, 1],
    )

    first, first_config = load_model(model_path, lazy=True)
    second, second_config = load_model(model_path, lazy=True)

    assert first.args.pipeline_stage_layers == [3, 1]
    assert second.args.pipeline_stage_layers == [3, 1]
    assert first_config["pipeline_stage_layers"] == [3, 1]
    assert second_config["pipeline_stage_layers"] == [3, 1]
    assert type(first) is not type(second)
    assert len(first.layers) == len(second.layers) == 4
