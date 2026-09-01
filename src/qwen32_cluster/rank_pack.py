from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import canonical_json
from .profiles import MODEL_LAYER_COUNT, Profile
from .safetensor_raw import TensorRecord, read_header, write_shard


WORLD_SIZE = 2
SHARED_PREFIXES = ("model.embed_tokens.", "model.norm.", "lm_head.")
ADAPTER_NAME = "qwen3_pipeline.py"
CONFIG_NAME = "config.json"
INDEX_NAME = "model.safetensors.index.json"
FINAL_MANIFEST_NAME = "rank-manifest.json"
STAGING_MANIFEST_NAME = "rank-manifest.json.staging"
DEFAULT_MAX_SHARD_BYTES = 768 << 20
FORMAT_VERSION = 1

_LAYER_KEY = re.compile(r"^model\.layers\.(0|[1-9][0-9]*)\.(.+)$")
_TENSOR_SUFFIX = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*$")
_SOURCE_SHARD = re.compile(r"^model-([0-9]{5})-of-([0-9]{5})\.safetensors$")
_EXACT_ASSETS = frozenset(
    {
        "tokenizer.model",
        "merges.txt",
        "vocab.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "generation_config.json",
        "README.md",
    }
)
_MAX_METADATA_BYTES = 64 << 20


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{label} contains a lone UTF-16 surrogate")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_bytes(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label}: {exc.msg}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} root must be a JSON object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected {sorted(unexpected)}")
        raise ValueError(f"invalid {label}: {', '.join(details)}")


def _is_asset_name(name: str) -> bool:
    return name in _EXACT_ASSETS or (
        name.startswith("tokenizer") and name.endswith(".json")
    ) or name.endswith(".tiktoken")


def module_unit(key: str) -> str:
    """Return the indivisible module unit for one supported Qwen3 tensor key."""

    _require_string(key, "tensor key")
    layer = _LAYER_KEY.fullmatch(key)
    if layer is not None:
        suffix = layer.group(2)
        if _TENSOR_SUFFIX.fullmatch(suffix) is None:
            raise ValueError(f"malformed layer tensor key: {key!r}")
        return f"model.layers.{int(layer.group(1))}"
    if key.startswith("model.layers."):
        raise ValueError(f"malformed layer tensor key: {key!r}")
    for prefix in SHARED_PREFIXES:
        if key.startswith(prefix):
            suffix = key[len(prefix) :]
            if _TENSOR_SUFFIX.fullmatch(suffix) is None:
                raise ValueError(f"malformed shared tensor key: {key!r}")
            return prefix[:-1]
    raise ValueError(f"unclassified tensor key: {key!r}")


def _validated_partition(
    rank: int,
    world_size: int,
    stage_layers: Sequence[int],
    shared_prefixes: Sequence[str],
) -> tuple[int, int]:
    _require_int(world_size, "world_size")
    if world_size != WORLD_SIZE:
        raise ValueError(f"world_size must be {WORLD_SIZE}")
    _require_int(rank, "rank")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in range(world_size)")
    if not isinstance(stage_layers, tuple) or len(stage_layers) != WORLD_SIZE:
        raise ValueError("stage_layers must be a two-item tuple")
    for index, count in enumerate(stage_layers):
        _require_int(count, f"stage_layers[{index}]", minimum=1)
    if sum(stage_layers) != MODEL_LAYER_COUNT:
        raise ValueError(f"stage_layers must total {MODEL_LAYER_COUNT}")
    if not isinstance(shared_prefixes, tuple) or shared_prefixes != SHARED_PREFIXES:
        raise ValueError(f"shared_prefixes must be exactly {SHARED_PREFIXES!r}")
    stage_index = world_size - 1 - rank
    start = sum(stage_layers[:stage_index])
    return start, start + stage_layers[stage_index]


def select_rank_keys(
    weight_map: Mapping[str, str],
    *,
    rank: int,
    world_size: int,
    stage_layers: Sequence[int],
    shared_prefixes: Sequence[str],
) -> frozenset[str]:
    """Select shared tensors and the reverse-rank-owned contiguous layer range."""

    start, end = _validated_partition(rank, world_size, stage_layers, shared_prefixes)
    if not isinstance(weight_map, Mapping):
        raise ValueError("weight_map must be a mapping")
    layers: dict[int, int] = {}
    shared_seen = {prefix: False for prefix in SHARED_PREFIXES}
    selected: set[str] = set()
    for key, source_name in weight_map.items():
        _require_string(key, "weight_map tensor key")
        _require_string(source_name, f"weight_map value for {key!r}")
        unit = module_unit(key)
        match = _LAYER_KEY.fullmatch(key)
        if match is None:
            for prefix in SHARED_PREFIXES:
                if key.startswith(prefix):
                    shared_seen[prefix] = True
            selected.add(key)
            continue
        layer = int(match.group(1))
        if layer >= MODEL_LAYER_COUNT:
            raise ValueError(f"remote or out-of-range layer tensor key: {key!r}")
        layers[layer] = layers.get(layer, 0) + 1
        if start <= layer < end:
            selected.add(key)
        if unit != f"model.layers.{layer}":
            raise ValueError(f"malformed layer ownership for {key!r}")
    missing = set(range(MODEL_LAYER_COUNT)) - set(layers)
    if missing:
        raise ValueError(f"missing layer ownership for layers {sorted(missing)}")
    missing_shared = [prefix for prefix, present in shared_seen.items() if not present]
    if missing_shared:
        raise ValueError(f"missing shared module ownership for prefixes {missing_shared}")
    return frozenset(selected)


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("snapshot path must be an absolute Path")
        for field_name in ("device", "inode", "size", "mtime_ns", "ctime_ns"):
            _require_int(getattr(self, field_name), f"snapshot {field_name}", minimum=0)
        if self.sha256 is not None and (
            not isinstance(self.sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None
        ):
            raise ValueError("snapshot sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class AssetPlan:
    name: str
    snapshot: FileSnapshot
    payload: bytes

    def __post_init__(self) -> None:
        _require_string(self.name, "asset name")
        if Path(self.name).name != self.name or not _is_asset_name(self.name):
            raise ValueError("asset name is not root-local and allowlisted")
        if not isinstance(self.snapshot, FileSnapshot) or not isinstance(self.payload, bytes):
            raise ValueError("asset snapshot and payload are invalid")
        if self.snapshot.size != len(self.payload):
            raise ValueError("asset payload size differs from snapshot")
        if self.snapshot.sha256 != hashlib.sha256(self.payload).hexdigest():
            raise ValueError("asset payload hash differs from snapshot")


@dataclass(frozen=True)
class PlannedShard:
    filename: str
    units: tuple[str, ...]
    tensors: tuple[TensorRecord, ...]
    payload_bytes: int

    def __post_init__(self) -> None:
        _require_string(self.filename, "planned shard filename")
        if _SOURCE_SHARD.fullmatch(self.filename) is None:
            raise ValueError("planned shard filename is not canonical")
        if not isinstance(self.units, tuple) or not self.units:
            raise ValueError("planned shard units must be a non-empty tuple")
        if len(set(self.units)) != len(self.units):
            raise ValueError("planned shard units must be unique")
        if not isinstance(self.tensors, tuple) or not self.tensors:
            raise ValueError("planned shard tensors must be a non-empty tuple")
        expected_order = tuple(
            name
            for unit in self.units
            for name in sorted(
                record.name for record in self.tensors if module_unit(record.name) == unit
            )
        )
        if expected_order != tuple(record.name for record in self.tensors):
            raise ValueError("planned shard tensors must be ordered by module unit then key")
        if set(map(module_unit, (record.name for record in self.tensors))) != set(self.units):
            raise ValueError("planned shard units do not match its tensors")
        _require_int(self.payload_bytes, "planned shard payload_bytes", minimum=0)
        if self.payload_bytes != sum(record.nbytes for record in self.tensors):
            raise ValueError("planned shard payload_bytes is inconsistent")


@dataclass(frozen=True)
class RankPackPlan:
    source_dir: Path
    output_dir: Path
    profile: Profile
    rank: int
    world_size: int
    stage_layers: tuple[int, int]
    layer_start: int
    layer_end: int
    shared_prefixes: tuple[str, str, str]
    max_shard_bytes: int
    shards: tuple[PlannedShard, ...]
    selected_keys: tuple[str, ...]
    total_size: int
    assets: tuple[AssetPlan, ...]
    config_snapshot: FileSnapshot
    index_snapshot: FileSnapshot
    source_shard_snapshots: tuple[FileSnapshot, ...]
    adapter_snapshot: FileSnapshot
    config_bytes: bytes
    index_bytes: bytes
    adapter_bytes: bytes
    plan_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_dir, Path) or not self.source_dir.is_absolute():
            raise ValueError("source_dir must be an absolute Path")
        if not isinstance(self.output_dir, Path) or not self.output_dir.is_absolute():
            raise ValueError("output_dir must be an absolute Path")
        if not isinstance(self.profile, Profile):
            raise ValueError("profile must be a Profile")
        start, end = _validated_partition(
            self.rank, self.world_size, self.stage_layers, self.shared_prefixes
        )
        if (self.layer_start, self.layer_end) != (start, end):
            raise ValueError("plan layer ownership is inconsistent")
        if self.profile.stage_layers != self.stage_layers:
            raise ValueError("plan stage_layers differ from profile")
        _require_int(self.max_shard_bytes, "max_shard_bytes", minimum=1)
        if not isinstance(self.shards, tuple) or not self.shards:
            raise ValueError("plan shards must be a non-empty tuple")
        names = tuple(shard.filename for shard in self.shards)
        expected_names = tuple(
            f"model-{number:05d}-of-{len(self.shards):05d}.safetensors"
            for number in range(1, len(self.shards) + 1)
        )
        if names != expected_names:
            raise ValueError("plan shard filenames are not a complete canonical sequence")
        tensor_keys = tuple(record.name for shard in self.shards for record in shard.tensors)
        if len(set(tensor_keys)) != len(tensor_keys) or tuple(sorted(tensor_keys)) != self.selected_keys:
            raise ValueError("plan selected_keys do not exactly match shard tensors")
        units = [unit for shard in self.shards for unit in shard.units]
        if len(set(units)) != len(units):
            raise ValueError("a module unit is split or duplicated across planned shards")
        if tuple(units) != tuple(sorted(units, key=_unit_sort_key)):
            raise ValueError("planned module units are not in stable canonical order")
        expected_units = {
            "model.embed_tokens",
            "model.norm",
            "lm_head",
            *(f"model.layers.{layer}" for layer in range(self.layer_start, self.layer_end)),
        }
        if set(units) != expected_units:
            raise ValueError("plan module units do not exactly cover shared and local layers")
        if any(shard.payload_bytes > self.max_shard_bytes for shard in self.shards):
            raise ValueError("planned shard exceeds max_shard_bytes")
        _require_int(self.total_size, "total_size", minimum=0)
        if self.total_size != sum(shard.payload_bytes for shard in self.shards):
            raise ValueError("plan total_size is inconsistent")
        asset_names = tuple(asset.name for asset in self.assets) if isinstance(self.assets, tuple) else ()
        if (
            not isinstance(self.assets, tuple)
            or asset_names != tuple(sorted(asset_names))
            or len(set(asset_names)) != len(asset_names)
        ):
            raise ValueError("plan assets must be a sorted tuple")
        for snapshot in (
            self.config_snapshot,
            self.index_snapshot,
            *self.source_shard_snapshots,
            self.adapter_snapshot,
        ):
            if not isinstance(snapshot, FileSnapshot):
                raise ValueError("plan contains an invalid snapshot")
        for payload_name in ("config_bytes", "index_bytes", "adapter_bytes"):
            if not isinstance(getattr(self, payload_name), bytes):
                raise ValueError(f"{payload_name} must be bytes")
        if hashlib.sha256(self.adapter_bytes).hexdigest() != self.adapter_snapshot.sha256:
            raise ValueError("adapter bytes do not match the bound adapter hash")
        if re.fullmatch(r"[0-9a-f]{64}", self.plan_id) is None:
            raise ValueError("plan_id must be a lowercase SHA-256 digest")
        for key in self.selected_keys:
            unit = module_unit(key)
            match = re.fullmatch(r"model\.layers\.(0|[1-9][0-9]*)", unit)
            if match is not None and not self.layer_start <= int(match.group(1)) < self.layer_end:
                raise ValueError("plan contains a remote layer tensor")
        source_paths = {snapshot.path for snapshot in self.source_shard_snapshots}
        if not source_paths or any(path.parent != self.source_dir for path in source_paths):
            raise ValueError("plan source shard snapshots must be root-local")
        if any(record.source_file not in source_paths for shard in self.shards for record in shard.tensors):
            raise ValueError("plan tensor source is not a snapshotted source shard")
        if self.config_snapshot.path != self.source_dir / CONFIG_NAME:
            raise ValueError("plan config snapshot path is invalid")
        if self.index_snapshot.path != self.source_dir / INDEX_NAME:
            raise ValueError("plan index snapshot path is invalid")
        if any(asset.snapshot.path != self.source_dir / asset.name for asset in self.assets):
            raise ValueError("plan asset snapshot path is invalid")
        expected_adapter_path = Path(__file__).with_name(ADAPTER_NAME).resolve(strict=True)
        if self.adapter_snapshot.path != expected_adapter_path:
            raise ValueError("plan adapter snapshot path is invalid")
        if any(
            snapshot.sha256 is None
            for snapshot in (self.config_snapshot, self.index_snapshot, self.adapter_snapshot)
        ):
            raise ValueError("plan metadata and adapter snapshots must be hash-bound")
        if (
            self.output_dir == self.source_dir
            or self.output_dir.is_relative_to(self.source_dir)
            or self.source_dir.is_relative_to(self.output_dir)
        ):
            raise ValueError("plan source and output directories must be separate and non-nested")
        config = _load_json_bytes(self.config_bytes, "planned config")
        if config.get("model_file") != ADAPTER_NAME or config.get("pipeline_stage_layers") != list(
            self.stage_layers
        ):
            raise ValueError("planned config does not bind the adapter and forward stage order")
        index = _load_json_bytes(self.index_bytes, "planned index")
        _require_exact_keys(index, {"metadata", "weight_map"}, "planned index")
        if index != {
            "metadata": {"total_size": self.total_size},
            "weight_map": {
                record.name: shard.filename for shard in self.shards for record in shard.tensors
            },
        }:
            raise ValueError("planned index does not exactly describe planned shards")
        expected_plan_id = hashlib.sha256(
            canonical_json(
                _plan_identity_payload(
                    self.profile,
                    self.rank,
                    self.max_shard_bytes,
                    self.shards,
                    self.assets,
                    self.config_bytes,
                    self.index_bytes,
                    self.adapter_snapshot.sha256 or "",
                )
            ).encode("utf-8")
        ).hexdigest()
        if self.plan_id != expected_plan_id:
            raise ValueError("plan_id does not match the immutable plan contents")


@dataclass(frozen=True)
class ManifestShard:
    filename: str
    tensor_keys: tuple[str, ...]
    module_units: tuple[str, ...]
    payload_bytes: int

    def __post_init__(self) -> None:
        if _SOURCE_SHARD.fullmatch(self.filename) is None:
            raise ValueError("manifest shard filename is not canonical")
        if not isinstance(self.tensor_keys, tuple) or not self.tensor_keys:
            raise ValueError("manifest tensor_keys must be a non-empty tuple")
        if len(set(self.tensor_keys)) != len(self.tensor_keys):
            raise ValueError("manifest tensor_keys must be unique")
        if not isinstance(self.module_units, tuple) or not self.module_units:
            raise ValueError("manifest module_units must be a non-empty tuple")
        if len(set(self.module_units)) != len(self.module_units):
            raise ValueError("manifest module_units must be unique")
        expected_order = tuple(
            key
            for unit in self.module_units
            for key in sorted(key for key in self.tensor_keys if module_unit(key) == unit)
        )
        if expected_order != self.tensor_keys:
            raise ValueError("manifest tensors must be ordered by module unit then key")
        _require_int(self.payload_bytes, "manifest payload_bytes", minimum=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "module_units": list(self.module_units),
            "payload_bytes": self.payload_bytes,
            "tensor_keys": list(self.tensor_keys),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ManifestShard:
        _require_exact_keys(
            value, {"filename", "module_units", "payload_bytes", "tensor_keys"}, "manifest shard"
        )
        if not isinstance(value["module_units"], list) or not isinstance(value["tensor_keys"], list):
            raise ValueError("manifest shard sequences must be JSON lists")
        return cls(
            value["filename"],
            tuple(value["tensor_keys"]),
            tuple(value["module_units"]),
            value["payload_bytes"],
        )


@dataclass(frozen=True)
class RankManifest:
    format_version: int
    state: str
    plan_id: str
    profile_name: str
    rank: int
    world_size: int
    stage_layers: tuple[int, int]
    layer_start: int
    layer_end: int
    shared_prefixes: tuple[str, str, str]
    shards: tuple[ManifestShard, ...]
    assets: tuple[str, ...]
    total_size: int
    adapter_sha256: str
    managed_files: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise ValueError("manifest format_version is unsupported")
        if self.state != "complete":
            raise ValueError("final manifest state must be complete")
        if re.fullmatch(r"[0-9a-f]{64}", self.plan_id) is None:
            raise ValueError("manifest plan_id is invalid")
        _require_string(self.profile_name, "manifest profile_name")
        start, end = _validated_partition(
            self.rank, self.world_size, self.stage_layers, self.shared_prefixes
        )
        if (start, end) != (self.layer_start, self.layer_end):
            raise ValueError("manifest layer range is inconsistent")
        if not isinstance(self.shards, tuple) or not self.shards:
            raise ValueError("manifest shards must be a non-empty tuple")
        shard_names = tuple(shard.filename for shard in self.shards)
        expected_shard_names = tuple(
            f"model-{number:05d}-of-{len(self.shards):05d}.safetensors"
            for number in range(1, len(self.shards) + 1)
        )
        if shard_names != expected_shard_names:
            raise ValueError("manifest shard filenames are not a complete canonical sequence")
        tensor_keys = [key for shard in self.shards for key in shard.tensor_keys]
        module_units = [unit for shard in self.shards for unit in shard.module_units]
        if len(set(tensor_keys)) != len(tensor_keys):
            raise ValueError("manifest duplicates a tensor key across shards")
        if len(set(module_units)) != len(module_units):
            raise ValueError("manifest splits or duplicates a module unit across shards")
        if tuple(module_units) != tuple(sorted(module_units, key=_unit_sort_key)):
            raise ValueError("manifest module units are not in stable canonical order")
        expected_units = {
            "model.embed_tokens",
            "model.norm",
            "lm_head",
            *(f"model.layers.{layer}" for layer in range(self.layer_start, self.layer_end)),
        }
        if set(module_units) != expected_units:
            raise ValueError("manifest units do not exactly cover shared and local layers")
        asset_names = self.assets if isinstance(self.assets, tuple) else ()
        if (
            not isinstance(self.assets, tuple)
            or tuple(sorted(asset_names)) != asset_names
            or len(set(asset_names)) != len(asset_names)
            or any(Path(name).name != name or not _is_asset_name(name) for name in asset_names)
        ):
            raise ValueError("manifest assets must be unique, sorted, root-local, and allowlisted")
        _require_int(self.total_size, "manifest total_size", minimum=0)
        if self.total_size != sum(shard.payload_bytes for shard in self.shards):
            raise ValueError("manifest total_size is inconsistent")
        if re.fullmatch(r"[0-9a-f]{64}", self.adapter_sha256) is None:
            raise ValueError("manifest adapter_sha256 is invalid")
        if not isinstance(self.managed_files, tuple) or len(set(self.managed_files)) != len(
            self.managed_files
        ):
            raise ValueError("manifest managed_files must be a unique tuple")
        for name in self.managed_files:
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("manifest managed file names must be root-local")
        final_names = [
            *shard_names,
            *asset_names,
            ADAPTER_NAME,
            CONFIG_NAME,
            INDEX_NAME,
            FINAL_MANIFEST_NAME,
            STAGING_MANIFEST_NAME,
        ]
        expected_managed = tuple(
            sorted(
                (
                    *final_names,
                    *(f"{name}.tmp" for name in final_names if name != FINAL_MANIFEST_NAME),
                )
            )
        )
        if self.managed_files != expected_managed:
            raise ValueError("manifest managed_files do not exactly match its output inventory")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_sha256": self.adapter_sha256,
            "assets": list(self.assets),
            "format_version": self.format_version,
            "layer_end": self.layer_end,
            "layer_start": self.layer_start,
            "managed_files": list(self.managed_files),
            "plan_id": self.plan_id,
            "profile_name": self.profile_name,
            "rank": self.rank,
            "shared_prefixes": list(self.shared_prefixes),
            "shards": [shard.to_dict() for shard in self.shards],
            "stage_layers": list(self.stage_layers),
            "state": self.state,
            "total_size": self.total_size,
            "world_size": self.world_size,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RankManifest:
        fields = {
            "adapter_sha256",
            "assets",
            "format_version",
            "layer_end",
            "layer_start",
            "managed_files",
            "plan_id",
            "profile_name",
            "rank",
            "shared_prefixes",
            "shards",
            "stage_layers",
            "state",
            "total_size",
            "world_size",
        }
        _require_exact_keys(value, fields, "rank manifest")
        for name in ("assets", "managed_files", "shared_prefixes", "shards", "stage_layers"):
            if not isinstance(value[name], list):
                raise ValueError(f"manifest {name} must be a JSON list")
        return cls(
            format_version=value["format_version"],
            state=value["state"],
            plan_id=value["plan_id"],
            profile_name=value["profile_name"],
            rank=value["rank"],
            world_size=value["world_size"],
            stage_layers=tuple(value["stage_layers"]),
            layer_start=value["layer_start"],
            layer_end=value["layer_end"],
            shared_prefixes=tuple(value["shared_prefixes"]),
            shards=tuple(ManifestShard.from_dict(shard) for shard in value["shards"]),
            assets=tuple(value["assets"]),
            total_size=value["total_size"],
            adapter_sha256=value["adapter_sha256"],
            managed_files=tuple(value["managed_files"]),
        )


def _snapshot_from_stat(path: Path, value: os.stat_result, digest: str | None) -> FileSnapshot:
    return FileSnapshot(
        path=path,
        device=value.st_dev,
        inode=value.st_ino,
        size=value.st_size,
        mtime_ns=value.st_mtime_ns,
        ctime_ns=value.st_ctime_ns,
        sha256=digest,
    )


def _read_regular(path: Path, label: str) -> tuple[bytes, FileSnapshot]:
    try:
        before = os.lstat(path)
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label}: {path.name}") from exc
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{label} must not be a symlink")
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if before.st_size > _MAX_METADATA_BYTES:
        raise ValueError(f"{label} exceeds the metadata size limit")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise ValueError(f"{label} changed while opening")
        chunks = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                raise ValueError(f"{label} was truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError(f"{label} changed while reading")
    finally:
        os.close(fd)
    payload = b"".join(chunks)
    return payload, _snapshot_from_stat(path, opened, hashlib.sha256(payload).hexdigest())


def _snapshot_regular(path: Path, label: str) -> FileSnapshot:
    value = os.lstat(path)
    if stat.S_ISLNK(value.st_mode):
        raise ValueError(f"{label} must not be a symlink")
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return _snapshot_from_stat(path, value, None)


def _validate_snapshot(snapshot: FileSnapshot, label: str) -> None:
    try:
        value = os.lstat(snapshot.path)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} changed since planning") from exc
    actual = (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    expected = (
        snapshot.device,
        snapshot.inode,
        snapshot.size,
        snapshot.mtime_ns,
        snapshot.ctime_ns,
    )
    if not stat.S_ISREG(value.st_mode) or actual != expected:
        raise ValueError(f"{label} changed since planning")
    if snapshot.sha256 is not None:
        payload, current = _read_regular(snapshot.path, label)
        del payload
        if current != snapshot:
            raise ValueError(f"{label} changed since planning")


def _unit_sort_key(unit: str) -> tuple[int, int, str]:
    if unit == "model.embed_tokens":
        return (0, 0, unit)
    match = re.fullmatch(r"model\.layers\.(0|[1-9][0-9]*)", unit)
    if match:
        return (1, int(match.group(1)), unit)
    if unit == "model.norm":
        return (2, 0, unit)
    if unit == "lm_head":
        return (3, 0, unit)
    raise ValueError(f"invalid module unit: {unit!r}")


def _canonical_output_path(source: Path, output: Path) -> Path:
    raw = Path(output)
    if raw.exists() or raw.is_symlink():
        value = os.lstat(raw)
        if stat.S_ISLNK(value.st_mode):
            raise ValueError("output directory must not be a symlink")
        if not stat.S_ISDIR(value.st_mode):
            raise ValueError("output path must be a directory")
        resolved = raw.resolve(strict=True)
    else:
        parent = raw.parent.resolve(strict=True)
        resolved = parent / raw.name
    if resolved == source or resolved.is_relative_to(source) or source.is_relative_to(resolved):
        raise ValueError("source and output directories must be separate and non-nested")
    return resolved


def _plan_identity_payload(
    profile: Profile,
    rank: int,
    max_shard_bytes: int,
    shards: tuple[PlannedShard, ...],
    assets: tuple[AssetPlan, ...],
    config_bytes: bytes,
    index_bytes: bytes,
    adapter_hash: str,
) -> dict[str, Any]:
    return {
        "adapter_sha256": adapter_hash,
        "assets": [
            {"name": asset.name, "sha256": asset.snapshot.sha256, "size": asset.snapshot.size}
            for asset in assets
        ],
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "format_version": FORMAT_VERSION,
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "max_shard_bytes": max_shard_bytes,
        "profile": profile.to_dict(),
        "rank": rank,
        "shards": [
            {
                "filename": shard.filename,
                "payload_bytes": shard.payload_bytes,
                "tensors": [
                    {
                        "dtype": record.dtype,
                        "end": record.end,
                        "name": record.name,
                        "nbytes": record.nbytes,
                        "shape": list(record.shape),
                        "source": record.source_file.name,
                        "start": record.start,
                    }
                    for record in shard.tensors
                ],
                "units": list(shard.units),
            }
            for shard in shards
        ],
        "world_size": WORLD_SIZE,
    }


def plan_rank_pack(
    source_dir: Path,
    output_dir: Path,
    profile: Profile,
    rank: int,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> RankPackPlan:
    if not isinstance(source_dir, Path) or not isinstance(output_dir, Path):
        raise ValueError("source_dir and output_dir must be Path values")
    if not isinstance(profile, Profile):
        raise ValueError("profile must be a Profile")
    _require_int(max_shard_bytes, "max_shard_bytes", minimum=1)
    source = source_dir.resolve(strict=True)
    if not source.is_dir():
        raise ValueError("source_dir must be a directory")
    output = _canonical_output_path(source, output_dir)
    start, end = _validated_partition(rank, WORLD_SIZE, profile.stage_layers, SHARED_PREFIXES)

    config_payload, config_snapshot = _read_regular(source / CONFIG_NAME, "config")
    config = dict(_load_json_bytes(config_payload, CONFIG_NAME))
    if config.get("model_type") != "qwen3":
        raise ValueError("config model_type must be qwen3")
    if type(config.get("num_hidden_layers")) is not int or config["num_hidden_layers"] != MODEL_LAYER_COUNT:
        raise ValueError(f"config num_hidden_layers must be {MODEL_LAYER_COUNT}")

    index_payload, index_snapshot = _read_regular(source / INDEX_NAME, "safetensor index")
    index = _load_json_bytes(index_payload, INDEX_NAME)
    _require_exact_keys(index, {"metadata", "weight_map"}, "safetensor index")
    metadata = index["metadata"]
    weight_map = index["weight_map"]
    if not isinstance(metadata, Mapping):
        raise ValueError("safetensor index metadata must be an object")
    _require_exact_keys(metadata, {"total_size"}, "safetensor index metadata")
    _require_int(metadata["total_size"], "safetensor index total_size", minimum=0)
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("safetensor index weight_map must be a non-empty object")
    for key, filename in weight_map.items():
        _require_string(key, "weight_map tensor key")
        _require_string(filename, f"weight_map source for {key!r}")
        if Path(filename).name != filename or _SOURCE_SHARD.fullmatch(filename) is None:
            raise ValueError(f"weight_map source shard is not allowed: {filename!r}")

    selected = select_rank_keys(
        weight_map,
        rank=rank,
        world_size=WORLD_SIZE,
        stage_layers=profile.stage_layers,
        shared_prefixes=SHARED_PREFIXES,
    )
    source_names = tuple(sorted(set(weight_map.values())))
    parsed_names = [_SOURCE_SHARD.fullmatch(name) for name in source_names]
    totals = {int(match.group(2)) for match in parsed_names if match is not None}
    ordinals = {int(match.group(1)) for match in parsed_names if match is not None}
    if len(totals) != 1 or totals != {len(source_names)} or ordinals != set(range(1, len(source_names) + 1)):
        raise ValueError("source shard names do not form one complete canonical sequence")
    actual_safetensors = {
        entry.name for entry in os.scandir(source) if entry.name.endswith(".safetensors")
    }
    if actual_safetensors != set(source_names):
        raise ValueError("source directory contains missing or unindexed safetensor shards")

    records: dict[str, TensorRecord] = {}
    source_snapshots: list[FileSnapshot] = []
    for name in source_names:
        path = source / name
        snapshot = _snapshot_regular(path, f"source shard {name}")
        header = read_header(path)
        after = _snapshot_regular(path, f"source shard {name}")
        if after != snapshot:
            raise ValueError(f"source shard {name} changed during planning")
        source_snapshots.append(snapshot)
        for record in header.tensors:
            if record.name in records:
                raise ValueError(f"duplicate tensor across source shard headers: {record.name!r}")
            records[record.name] = record
    if set(records) != set(weight_map):
        raise ValueError("safetensor index and source headers do not cover the same tensors")
    for key, filename in weight_map.items():
        if records[key].source_file.name != filename:
            raise ValueError(f"safetensor index source disagrees with header for tensor {key!r}")
    source_total = sum(record.nbytes for record in records.values())
    if metadata["total_size"] != source_total:
        raise ValueError("safetensor index metadata total_size is not exact")

    grouped: dict[str, list[TensorRecord]] = {}
    for key in selected:
        grouped.setdefault(module_unit(key), []).append(records[key])
    ordered_units = tuple(sorted(grouped, key=_unit_sort_key))
    bins: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for unit in ordered_units:
        size = sum(record.nbytes for record in grouped[unit])
        if size > max_shard_bytes:
            raise ValueError(
                f"module unit {unit!r} size {size} exceeds max_shard_bytes {max_shard_bytes}"
            )
        if current and current_size + size > max_shard_bytes:
            bins.append(current)
            current = []
            current_size = 0
        current.append(unit)
        current_size += size
    if current:
        bins.append(current)
    count = len(bins)
    shards = tuple(
        PlannedShard(
            filename=f"model-{number:05d}-of-{count:05d}.safetensors",
            units=tuple(units),
            tensors=tuple(
                record
                for unit in units
                for record in sorted(grouped[unit], key=lambda record: record.name)
            ),
            payload_bytes=sum(record.nbytes for unit in units for record in grouped[unit]),
        )
        for number, units in enumerate(bins, 1)
    )

    assets: list[AssetPlan] = []
    with os.scandir(source) as entries:
        for entry in entries:
            if not _is_asset_name(entry.name):
                continue
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise ValueError(f"allowlisted asset {entry.name!r} must be a regular non-symlink file")
            payload, snapshot = _read_regular(source / entry.name, f"asset {entry.name}")
            assets.append(AssetPlan(entry.name, snapshot, payload))
    asset_tuple = tuple(sorted(assets, key=lambda asset: asset.name))

    adapter_path = Path(__file__).with_name(ADAPTER_NAME).resolve(strict=True)
    adapter_payload, adapter_snapshot = _read_regular(adapter_path, "pipeline adapter")
    derived_config = dict(config)
    derived_config["model_file"] = ADAPTER_NAME
    derived_config["pipeline_stage_layers"] = list(profile.stage_layers)
    config_bytes = canonical_json(derived_config).encode("utf-8")
    output_weight_map = {
        record.name: shard.filename for shard in shards for record in shard.tensors
    }
    output_index = {
        "metadata": {"total_size": sum(shard.payload_bytes for shard in shards)},
        "weight_map": output_weight_map,
    }
    index_bytes = canonical_json(output_index).encode("utf-8")
    identity_payload = _plan_identity_payload(
        profile,
        rank,
        max_shard_bytes,
        shards,
        asset_tuple,
        config_bytes,
        index_bytes,
        adapter_snapshot.sha256 or "",
    )
    plan_id = hashlib.sha256(canonical_json(identity_payload).encode("utf-8")).hexdigest()
    return RankPackPlan(
        source_dir=source,
        output_dir=output,
        profile=profile,
        rank=rank,
        world_size=WORLD_SIZE,
        stage_layers=profile.stage_layers,
        layer_start=start,
        layer_end=end,
        shared_prefixes=SHARED_PREFIXES,
        max_shard_bytes=max_shard_bytes,
        shards=shards,
        selected_keys=tuple(sorted(selected)),
        total_size=sum(shard.payload_bytes for shard in shards),
        assets=asset_tuple,
        config_snapshot=config_snapshot,
        index_snapshot=index_snapshot,
        source_shard_snapshots=tuple(source_snapshots),
        adapter_snapshot=adapter_snapshot,
        config_bytes=config_bytes,
        index_bytes=index_bytes,
        adapter_bytes=adapter_payload,
        plan_id=plan_id,
    )


def _managed_files(plan: RankPackPlan) -> tuple[str, ...]:
    final_names = [
        *(shard.filename for shard in plan.shards),
        *(asset.name for asset in plan.assets),
        ADAPTER_NAME,
        CONFIG_NAME,
        INDEX_NAME,
        FINAL_MANIFEST_NAME,
        STAGING_MANIFEST_NAME,
    ]
    temp_names = [f"{name}.tmp" for name in final_names if name != FINAL_MANIFEST_NAME]
    names = tuple(sorted((*final_names, *temp_names)))
    if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
        raise ValueError("plan produces duplicate or unsafe output paths")
    return names


def _manifest_for(plan: RankPackPlan) -> RankManifest:
    return RankManifest(
        format_version=FORMAT_VERSION,
        state="complete",
        plan_id=plan.plan_id,
        profile_name=plan.profile.name,
        rank=plan.rank,
        world_size=plan.world_size,
        stage_layers=plan.stage_layers,
        layer_start=plan.layer_start,
        layer_end=plan.layer_end,
        shared_prefixes=plan.shared_prefixes,
        shards=tuple(
            ManifestShard(
                shard.filename,
                tuple(record.name for record in shard.tensors),
                shard.units,
                shard.payload_bytes,
            )
            for shard in plan.shards
        ),
        assets=tuple(asset.name for asset in plan.assets),
        total_size=plan.total_size,
        adapter_sha256=plan.adapter_snapshot.sha256 or "",
        managed_files=_managed_files(plan),
    )


def _manifest_bytes(manifest: RankManifest, state: str) -> bytes:
    value = manifest.to_dict()
    value["state"] = state
    return canonical_json(value).encode("utf-8")


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        try:
            count = os.write(fd, view[offset:])
        except InterruptedError:
            continue
        if count <= 0:
            raise OSError("short write while publishing rank pack metadata")
        offset += count


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_bytes(path: Path, payload: bytes, *, replace: bool = False) -> None:
    if not isinstance(payload, bytes):
        raise ValueError("published payload must be bytes")
    temporary = path.with_name(path.name + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd: int | None = None
    try:
        fd = os.open(temporary, flags, 0o600)
        _write_all(fd, payload)
        os.fsync(fd)
        os.close(fd)
        fd = None
        if not replace and path.exists():
            raise FileExistsError(f"refusing to overwrite {path.name}")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _validate_plan_sources(plan: RankPackPlan) -> None:
    with os.scandir(plan.source_dir) as entries:
        names = tuple(entry.name for entry in entries)
    current_assets = {name for name in names if _is_asset_name(name)}
    expected_assets = {asset.name for asset in plan.assets}
    current_shards = {name for name in names if name.endswith(".safetensors")}
    expected_shards = {snapshot.path.name for snapshot in plan.source_shard_snapshots}
    if current_assets != expected_assets or current_shards != expected_shards:
        raise ValueError("source allowlisted asset or safetensor inventory changed since planning")
    _validate_snapshot(plan.config_snapshot, "config")
    _validate_snapshot(plan.index_snapshot, "safetensor index")
    for snapshot in plan.source_shard_snapshots:
        _validate_snapshot(snapshot, f"source shard {snapshot.path.name}")
    for asset in plan.assets:
        _validate_snapshot(asset.snapshot, f"asset {asset.name}")
    _validate_snapshot(plan.adapter_snapshot, "pipeline adapter")


def _parse_manifest(path: Path) -> RankManifest:
    payload, _ = _read_regular(path, "rank manifest")
    return RankManifest.from_dict(_load_json_bytes(payload, FINAL_MANIFEST_NAME))


def _expected_shard_geometry(shard: PlannedShard) -> tuple[int, int, int]:
    offset = 0
    header: dict[str, dict[str, Any]] = {}
    for record in sorted(shard.tensors, key=lambda item: item.name):
        end = offset + record.nbytes
        header[record.name] = {
            "data_offsets": [offset, end],
            "dtype": record.dtype,
            "shape": list(record.shape),
        }
        offset = end
    compact = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    header_length = len(compact) + (-len(compact) % 8)
    data_start = 8 + header_length
    return header_length, data_start, data_start + shard.payload_bytes


def _expected_shard_descriptors(
    shard: PlannedShard,
) -> dict[str, tuple[str, tuple[int, ...], int, int, int]]:
    offset = 0
    descriptors: dict[str, tuple[str, tuple[int, ...], int, int, int]] = {}
    for record in sorted(shard.tensors, key=lambda item: item.name):
        end = offset + record.nbytes
        descriptors[record.name] = (
            record.dtype,
            record.shape,
            record.nbytes,
            offset,
            end,
        )
        offset = end
    return descriptors


def _validate_completed_output(plan: RankPackPlan, expected: RankManifest) -> None:
    output = plan.output_dir
    actual_names = {entry.name for entry in os.scandir(output)}
    expected_names = {
        *(shard.filename for shard in plan.shards),
        *(asset.name for asset in plan.assets),
        ADAPTER_NAME,
        CONFIG_NAME,
        INDEX_NAME,
        FINAL_MANIFEST_NAME,
    }
    if actual_names != expected_names:
        raise ValueError("completed rank pack has an unexpected file inventory")
    _validate_payload_files(plan, expected_names - {FINAL_MANIFEST_NAME})
    parsed = _parse_manifest(output / FINAL_MANIFEST_NAME)
    if parsed != expected:
        raise ValueError("completed rank pack manifest differs from plan")


def _validate_payload_files(plan: RankPackPlan, expected_names: set[str]) -> None:
    output = plan.output_dir
    for name in expected_names:
        value = os.lstat(output / name)
        if not stat.S_ISREG(value.st_mode):
            raise ValueError(f"rank pack path {name!r} is not a regular file")
    if (output / CONFIG_NAME).read_bytes() != plan.config_bytes:
        raise ValueError("completed rank pack config differs from plan")
    if (output / INDEX_NAME).read_bytes() != plan.index_bytes:
        raise ValueError("completed rank pack index differs from plan")
    if (output / ADAPTER_NAME).read_bytes() != plan.adapter_bytes:
        raise ValueError("completed rank pack adapter differs from plan")
    for asset in plan.assets:
        if (output / asset.name).read_bytes() != asset.payload:
            raise ValueError(f"completed rank pack asset {asset.name!r} differs from plan")
    actual_keys: set[str] = set()
    for shard in plan.shards:
        header = read_header(output / shard.filename)
        actual_descriptors = {
            record.name: (
                record.dtype,
                record.shape,
                record.nbytes,
                record.start - header.data_start,
                record.end - header.data_start,
            )
            for record in header.tensors
        }
        expected_descriptors = _expected_shard_descriptors(shard)
        keys = set(actual_descriptors)
        if (
            actual_descriptors != expected_descriptors
            or actual_keys.intersection(keys)
            or (
                header.header_length,
                header.data_start,
                (output / shard.filename).stat().st_size,
            )
            != _expected_shard_geometry(shard)
        ):
            raise ValueError("completed rank pack shard descriptors differ from plan")
        actual_keys.update(keys)
    if actual_keys != set(plan.selected_keys):
        raise ValueError("rank pack tensor coverage is incomplete")


def _validate_staged_output(plan: RankPackPlan, expected: RankManifest) -> None:
    output = plan.output_dir
    expected_names = {
        *(shard.filename for shard in plan.shards),
        *(asset.name for asset in plan.assets),
        ADAPTER_NAME,
        CONFIG_NAME,
        INDEX_NAME,
        STAGING_MANIFEST_NAME,
    }
    actual_names = {entry.name for entry in os.scandir(output)}
    if actual_names != expected_names:
        raise ValueError("staged rank pack output inventory differs from plan")
    _validate_payload_files(plan, expected_names - {STAGING_MANIFEST_NAME})
    marker_payload, _ = _read_regular(
        output / STAGING_MANIFEST_NAME, "staging rank manifest"
    )
    marker = _load_json_bytes(marker_payload, STAGING_MANIFEST_NAME)
    if marker != expected.to_dict():
        raise ValueError("complete staging marker differs from plan")


def _path_identifies_snapshot(path: Path, snapshot: FileSnapshot) -> bool:
    try:
        value = os.lstat(path)
    except FileNotFoundError:
        return False
    return stat.S_ISREG(value.st_mode) and (
        value.st_dev,
        value.st_ino,
    ) == (
        snapshot.device,
        snapshot.inode,
    )


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Move a regular file while atomically refusing an existing destination."""

    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = library.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux"):
        try:
            rename = library.renameat2
        except AttributeError as exc:
            raise OSError(
                errno.ENOTSUP, "atomic no-replace rename is unavailable"
            ) from exc
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 0x00000001)
    else:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename is unavailable")
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination)


def _unlink_snapshot_path(path: Path, snapshot: FileSnapshot, error: BaseException) -> None:
    if not _path_identifies_snapshot(path, snapshot):
        return
    try:
        os.unlink(path)
    except BaseException as cleanup_error:
        error.add_note(f"failed to remove the tool-owned final manifest: {cleanup_error}")


def _rollback_final_manifest(
    staging: Path,
    final: Path,
    staging_snapshot: FileSnapshot,
    error: BaseException,
) -> None:
    final_is_owned = False
    try:
        final_is_owned = _path_identifies_snapshot(final, staging_snapshot)
    except BaseException as inspection_error:
        error.add_note(f"failed to inspect final manifest rollback state: {inspection_error}")

    staging_exists = True
    try:
        staging_exists = staging.exists() or staging.is_symlink()
    except BaseException as inspection_error:
        error.add_note(f"failed to inspect staging manifest rollback state: {inspection_error}")

    if final_is_owned and not staging_exists:
        try:
            os.replace(final, staging)
        except BaseException as rollback_error:
            error.add_note(
                f"failed to restore the complete staging manifest: {rollback_error}"
            )

    try:
        if _path_identifies_snapshot(final, staging_snapshot):
            _unlink_snapshot_path(final, staging_snapshot, error)
    except BaseException as inspection_error:
        error.add_note(f"failed to inspect final manifest cleanup state: {inspection_error}")
    try:
        if not _path_identifies_snapshot(staging, staging_snapshot):
            error.add_note(
                "the complete staging manifest could not be restored; no tool-owned "
                "valid final manifest was retained"
            )
    except BaseException as inspection_error:
        error.add_note(f"failed to inspect restored staging manifest: {inspection_error}")
    try:
        _fsync_directory(final.parent)
    except BaseException as fsync_error:
        error.add_note(f"failed to fsync final manifest rollback state: {fsync_error}")


def _prepare_output(plan: RankPackPlan, expected: RankManifest, force: bool) -> RankManifest | None:
    output = plan.output_dir
    if output.exists() or output.is_symlink():
        value = os.lstat(output)
        if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
            raise ValueError("output path must remain a real directory")
    else:
        os.mkdir(output, 0o700)
        _fsync_directory(output.parent)
    final = output / FINAL_MANIFEST_NAME
    if final.exists() or final.is_symlink():
        if final.is_symlink():
            raise ValueError("final rank manifest must not be a symlink")
        parsed = _parse_manifest(final)
        if parsed != expected:
            raise FileExistsError("completed output belongs to a different rank pack plan")
        _validate_completed_output(plan, expected)
        return parsed
    entries = {entry.name for entry in os.scandir(output)}
    if not entries:
        return None
    if not force:
        raise FileExistsError("nonempty destination requires force and a matching staging marker")
    staging = output / STAGING_MANIFEST_NAME
    if not staging.exists() or staging.is_symlink():
        raise ValueError("force requires a matching regular staging marker")
    marker_payload, _ = _read_regular(staging, "staging rank manifest")
    marker = _load_json_bytes(marker_payload, STAGING_MANIFEST_NAME)
    expected_value = expected.to_dict()
    marker_state = marker.get("state")
    if marker_state not in {"staging", "complete"}:
        raise ValueError("staging marker has an invalid state")
    expected_value["state"] = marker_state
    if marker != expected_value:
        raise ValueError("staging marker does not match this plan")
    allowed = set(expected.managed_files)
    unknown = entries - allowed
    if unknown:
        raise ValueError(f"staging directory contains unknown files: {sorted(unknown)}")
    for name in sorted(entries):
        path = output / name
        value = os.lstat(path)
        if not stat.S_ISREG(value.st_mode):
            raise ValueError(f"owned staging path {name!r} is not a regular file")
    for name in sorted(entries):
        os.unlink(output / name)
    _fsync_directory(output)
    return None


def pack_rank(plan: RankPackPlan, *, force: bool = False) -> RankManifest:
    if not isinstance(plan, RankPackPlan):
        raise ValueError("plan must be a RankPackPlan")
    if type(force) is not bool:
        raise ValueError("force must be a boolean")
    _validate_plan_sources(plan)
    expected = _manifest_for(plan)
    existing = _prepare_output(plan, expected, force)
    if existing is not None:
        return existing
    output = plan.output_dir
    staging = output / STAGING_MANIFEST_NAME
    _publish_bytes(staging, _manifest_bytes(expected, "staging"))
    for shard in plan.shards:
        temporary = output / f"{shard.filename}.tmp"
        write_shard(shard.tensors, temporary)
        final = output / shard.filename
        if final.exists() or final.is_symlink():
            raise FileExistsError(f"refusing to overwrite {shard.filename}")
        os.replace(temporary, final)
        _fsync_directory(output)
    for asset in plan.assets:
        _publish_bytes(output / asset.name, asset.payload)
    _publish_bytes(output / ADAPTER_NAME, plan.adapter_bytes)
    _publish_bytes(output / CONFIG_NAME, plan.config_bytes)
    _publish_bytes(output / INDEX_NAME, plan.index_bytes)
    _validate_plan_sources(plan)
    _publish_bytes(staging, _manifest_bytes(expected, "complete"), replace=True)
    _validate_staged_output(plan, expected)
    final = output / FINAL_MANIFEST_NAME
    if final.exists() or final.is_symlink():
        raise FileExistsError("refusing to overwrite final rank manifest")
    _, staging_snapshot = _read_regular(staging, "complete staging rank manifest")
    try:
        _rename_no_replace(staging, final)
        if not _path_identifies_snapshot(final, staging_snapshot):
            raise RuntimeError("final manifest identity changed during publication")
        _fsync_directory(output)
        if not _path_identifies_snapshot(final, staging_snapshot):
            raise RuntimeError("final manifest identity changed during publication fsync")
    except BaseException as error:
        _rollback_final_manifest(staging, final, staging_snapshot, error)
        raise
    return expected


__all__ = [
    "ADAPTER_NAME",
    "AssetPlan",
    "DEFAULT_MAX_SHARD_BYTES",
    "FINAL_MANIFEST_NAME",
    "FileSnapshot",
    "ManifestShard",
    "PlannedShard",
    "RankManifest",
    "RankPackPlan",
    "SHARED_PREFIXES",
    "STAGING_MANIFEST_NAME",
    "module_unit",
    "pack_rank",
    "plan_rank_pack",
    "select_rank_keys",
]
