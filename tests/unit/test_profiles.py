from __future__ import annotations

import dataclasses
import json
from importlib import import_module
from pathlib import Path

import pytest


EXPECTED = {
    "balanced-4bit": (4, (32, 32), 256),
    "quality-4bit": (4, (36, 28), 256),
    "performance-3bit": (3, (40, 24), 256),
    "aggressive-3bit": (3, (44, 20), 256),
    "balanced-3bit": (3, (36, 28), 256),
}


def profiles():
    try:
        return import_module("qwen32_cluster.profiles")
    except ModuleNotFoundError as exc:
        pytest.fail(f"profiles module is not implemented: {exc}")


def test_repository_profiles_match_the_canonical_matrix() -> None:
    module = profiles()
    config = module.load_profiles(Path(__file__).parents[2] / "config" / "profiles.json")
    actual = {
        profile.name: (profile.quantization_bits, profile.stage_layers, profile.prefill_step_size)
        for profile in config.profiles
    }
    assert actual == EXPECTED
    assert config.server.to_dict() == {
        "decode_concurrency": 1,
        "prompt_concurrency": 1,
        "prompt_cache_size": 0,
        "prompt_cache_bytes": 0,
    }


def test_profile_is_frozen_and_stage_order_is_preserved() -> None:
    module = profiles()
    profile = module.Profile("quality-4bit", 4, (36, 28), 256)
    with pytest.raises(dataclasses.FrozenInstanceError):
        profile.stage_layers = (32, 32)
    assert profile.to_dict()["stage_layers"] == [36, 28]
    assert module.canonical_json(profile) == (
        '{"context_limit":8192,"name":"quality-4bit","prefill_step_size":256,'
        '"quantization_bits":4,"stage_layers":[36,28]}\n'
    )


@pytest.mark.parametrize(
    ("bits", "layers", "context", "prefill"),
    [
        (4, (32, 31), 8192, 256),
        (4, (0, 64), 8192, 256),
        (4, (-1, 65), 8192, 256),
        (2, (32, 32), 8192, 256),
        (5, (32, 32), 8192, 256),
        (4, (32, 32), 4096, 256),
        (4, (32, 32), 8192, 64),
        (4, (32, 32), 8192, 1024),
    ],
)
def test_profile_rejects_invalid_values(bits, layers, context, prefill) -> None:
    module = profiles()
    with pytest.raises(ValueError):
        module.Profile("invalid", bits, layers, prefill, context)


@pytest.mark.parametrize("context", [8192.0, True, "8192"])
def test_profile_rejects_non_integer_context_limit(context) -> None:
    module = profiles()
    with pytest.raises(ValueError, match="context_limit must be an integer"):
        module.Profile("invalid-context", 4, (32, 32), 256, context)


def test_profile_strict_parser_rejects_unknown_fields() -> None:
    module = profiles()
    with pytest.raises(ValueError, match="unexpected"):
        module.Profile.from_dict(
            {
                "name": "balanced-4bit",
                "quantization_bits": 4,
                "stage_layers": [32, 32],
                "prefill_step_size": 256,
                "context_limit": 8192,
                "extra": 1,
            }
        )


def test_derive_profile_returns_calibration_profile_without_editing_source(tmp_path: Path) -> None:
    module = profiles()
    source = tmp_path / "profiles.json"
    original = {
        "profiles": [module.Profile("balanced-4bit", 4, (32, 32), 256).to_dict()],
        "server": {
            "decode_concurrency": 1,
            "prompt_concurrency": 1,
            "prompt_cache_size": 0,
            "prompt_cache_bytes": 0,
        },
    }
    source.write_text(json.dumps(original, sort_keys=True), encoding="utf-8")
    base = module.load_profiles(source).profiles[0]
    before = source.read_bytes()
    derived = module.derive_profile(base, (40, 24), 512)
    assert derived == module.Profile("calibration-4bit-m4-40-m3-24-p512", 4, (40, 24), 512)
    assert source.read_bytes() == before


def test_derive_profile_applies_normal_validation() -> None:
    module = profiles()
    base = module.Profile("balanced-3bit", 3, (36, 28), 256)
    with pytest.raises(ValueError):
        module.derive_profile(base, (63, 0), 256)
