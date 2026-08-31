from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

from .contracts import _load_json, _require_exact_keys, _require_int, _require_string, canonical_json


VALID_QUANTIZATION_BITS = frozenset({3, 4})
VALID_PREFILL_STEP_SIZES = frozenset({128, 256, 512})
MODEL_LAYER_COUNT = 64
CANONICAL_CONTEXT_LIMIT = 8192


@dataclass(frozen=True)
class Profile:
    name: str
    quantization_bits: int
    stage_layers: tuple[int, int]  # Forward stage order: Rank1 M4, then Rank0 M3.
    prefill_step_size: int
    context_limit: int = CANONICAL_CONTEXT_LIMIT

    def __post_init__(self) -> None:
        _require_string(self.name, "profile name")
        _require_int(self.quantization_bits, "quantization_bits")
        if self.quantization_bits not in VALID_QUANTIZATION_BITS:
            raise ValueError("quantization_bits must be 3 or 4")
        if not isinstance(self.stage_layers, tuple) or len(self.stage_layers) != 2:
            raise ValueError("stage_layers must be a two-item tuple")
        for index, layers in enumerate(self.stage_layers):
            _require_int(layers, f"stage_layers[{index}]", minimum=1)
        if sum(self.stage_layers) != MODEL_LAYER_COUNT:
            raise ValueError(f"stage_layers must total {MODEL_LAYER_COUNT}")
        _require_int(self.prefill_step_size, "prefill_step_size")
        if self.prefill_step_size not in VALID_PREFILL_STEP_SIZES:
            raise ValueError(f"prefill_step_size must be one of {sorted(VALID_PREFILL_STEP_SIZES)}")
        _require_int(self.context_limit, "context_limit")
        if self.context_limit != CANONICAL_CONTEXT_LIMIT:
            raise ValueError(f"context_limit must be {CANONICAL_CONTEXT_LIMIT}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Profile:
        keys = {"name", "quantization_bits", "stage_layers", "prefill_step_size", "context_limit"}
        _require_exact_keys(data, keys, "profile")
        stage_layers = data["stage_layers"]
        if not isinstance(stage_layers, list) or len(stage_layers) != 2:
            raise ValueError("stage_layers must be a two-item JSON list")
        return cls(
            name=data["name"],
            quantization_bits=data["quantization_bits"],
            stage_layers=(stage_layers[0], stage_layers[1]),
            prefill_step_size=data["prefill_step_size"],
            context_limit=data["context_limit"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "quantization_bits": self.quantization_bits,
            "stage_layers": list(self.stage_layers),
            "prefill_step_size": self.prefill_step_size,
            "context_limit": self.context_limit,
        }


@dataclass(frozen=True)
class ServerSettings:
    decode_concurrency: int
    prompt_concurrency: int
    prompt_cache_size: int
    prompt_cache_bytes: int

    def __post_init__(self) -> None:
        _require_int(self.decode_concurrency, "decode_concurrency", minimum=0)
        _require_int(self.prompt_concurrency, "prompt_concurrency", minimum=0)
        _require_int(self.prompt_cache_size, "prompt_cache_size", minimum=0)
        _require_int(self.prompt_cache_bytes, "prompt_cache_bytes", minimum=0)
        if self.decode_concurrency != 1:
            raise ValueError("decode_concurrency must be 1")
        if self.prompt_concurrency != 1:
            raise ValueError("prompt_concurrency must be 1")
        if self.prompt_cache_size != 0:
            raise ValueError("prompt_cache_size must be 0")
        if self.prompt_cache_bytes != 0:
            raise ValueError("prompt_cache_bytes must be 0")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ServerSettings:
        keys = {"decode_concurrency", "prompt_concurrency", "prompt_cache_size", "prompt_cache_bytes"}
        _require_exact_keys(data, keys, "server settings")
        for key in keys:
            _require_int(data[key], key, minimum=0)
        return cls(**{key: data[key] for key in keys})

    def to_dict(self) -> dict[str, int]:
        return {
            "decode_concurrency": self.decode_concurrency,
            "prompt_concurrency": self.prompt_concurrency,
            "prompt_cache_size": self.prompt_cache_size,
            "prompt_cache_bytes": self.prompt_cache_bytes,
        }


@dataclass(frozen=True)
class ProfilesConfig(Mapping[str, Profile]):
    profiles: tuple[Profile, ...]
    server: ServerSettings

    def __post_init__(self) -> None:
        names = tuple(profile.name for profile in self.profiles)
        if not names:
            raise ValueError("profiles must not be empty")
        if len(set(names)) != len(names):
            raise ValueError("profile names must be unique")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProfilesConfig:
        _require_exact_keys(data, {"profiles", "server"}, "profiles configuration")
        raw_profiles = data["profiles"]
        if not isinstance(raw_profiles, list):
            raise ValueError("profiles must be a JSON list")
        return cls(
            profiles=tuple(Profile.from_dict(profile) for profile in raw_profiles),
            server=ServerSettings.from_dict(data["server"]),
        )

    def __iter__(self) -> Iterator[str]:
        return (profile.name for profile in self.profiles)

    def __len__(self) -> int:
        return len(self.profiles)

    @overload
    def __getitem__(self, key: str) -> Profile: ...

    @overload
    def __getitem__(self, key: int) -> Profile: ...

    def __getitem__(self, key: str | int) -> Profile:
        if isinstance(key, int):
            return self.profiles[key]
        for profile in self.profiles:
            if profile.name == key:
                return profile
        raise KeyError(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profiles": [profile.to_dict() for profile in self.profiles],
            "server": self.server.to_dict(),
        }


def load_profiles(path: str | Path) -> ProfilesConfig:
    return ProfilesConfig.from_dict(_load_json(path))


def derive_profile(
    base: Profile,
    stage_layers: tuple[int, int],
    prefill_step_size: int,
) -> Profile:
    if not isinstance(base, Profile):
        raise TypeError("base must be a Profile")
    if not isinstance(stage_layers, tuple) or len(stage_layers) != 2:
        raise ValueError("stage_layers must be a two-item tuple")
    name = (
        f"calibration-{base.quantization_bits}bit-"
        f"m4-{stage_layers[0]}-m3-{stage_layers[1]}-p{prefill_step_size}"
    )
    return Profile(
        name=name,
        quantization_bits=base.quantization_bits,
        stage_layers=stage_layers,
        prefill_step_size=prefill_step_size,
        context_limit=base.context_limit,
    )


__all__ = [
    "Profile",
    "ProfilesConfig",
    "ServerSettings",
    "canonical_json",
    "derive_profile",
    "load_profiles",
]
