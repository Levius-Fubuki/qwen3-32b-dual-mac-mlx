from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import os
import struct
from pathlib import Path

import pytest

from qwen32_cluster.profiles import Profile


SHARED_PREFIXES = ("model.embed_tokens.", "model.norm.", "lm_head.")


def rank_pack():
    try:
        return importlib.import_module("qwen32_cluster.rank_pack")
    except ModuleNotFoundError as exc:
        pytest.fail(f"rank_pack module is not implemented: {exc}")


def _write_safetensors(path: Path, tensors: dict[str, bytes]) -> None:
    offset = 0
    header = {}
    payload = bytearray()
    for key, value in tensors.items():
        header[key] = {
            "dtype": "U8",
            "shape": [len(value)],
            "data_offsets": [offset, offset + len(value)],
        }
        payload.extend(value)
        offset += len(value)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _source_model(root: Path, *, unknown_asset: bool = True) -> Path:
    root.mkdir()
    first: dict[str, bytes] = {
        "model.embed_tokens.weight": b"embed",
        "model.norm.weight": b"norm",
        "lm_head.weight": b"head",
    }
    second: dict[str, bytes] = {}
    weight_map: dict[str, str] = {}
    for layer in range(64):
        left = f"model.layers.{layer}.self_attn.q_proj.weight"
        right = f"model.layers.{layer}.self_attn.q_proj.scales"
        first[left] = bytes((layer, layer + 1))
        second[right] = bytes((255 - layer,))
    for key in first:
        weight_map[key] = "model-00001-of-00002.safetensors"
    for key in second:
        weight_map[key] = "model-00002-of-00002.safetensors"
    _write_safetensors(root / "model-00001-of-00002.safetensors", first)
    _write_safetensors(root / "model-00002-of-00002.safetensors", second)
    total_size = sum(map(len, first.values())) + sum(map(len, second.values()))
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "num_hidden_layers": 64,
                "hidden_size": 16,
                "model_file": "upstream.py",
                "upstream_field": {"preserved": True},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_bytes(b'{"tokenizer":true}\n')
    (root / "README.md").write_bytes(b"synthetic fixture\n")
    if unknown_asset:
        (root / "notes.txt").write_text("do not copy", encoding="utf-8")
        nested = root / "nested"
        nested.mkdir()
        (nested / "tokenizer.json").write_text("do not recurse", encoding="utf-8")
    return root


def _profile(split: tuple[int, int] = (40, 24)) -> Profile:
    return Profile("test-4bit", 4, split, 256)


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.iterdir())
        if path.is_file()
    }


def test_module_units_and_reverse_rank_selection_are_exact() -> None:
    module = rank_pack()
    assert module.module_unit("model.layers.12.self_attn.q_proj.weight") == "model.layers.12"
    assert module.module_unit("model.layers.12.mlp.down_proj.scales") == "model.layers.12"
    assert module.module_unit("model.embed_tokens.weight") == "model.embed_tokens"
    assert module.module_unit("model.norm.weight") == "model.norm"
    assert module.module_unit("lm_head.weight") == "lm_head"
    with pytest.raises(ValueError, match="unclassified|malformed"):
        module.module_unit("model.rotary_emb.inv_freq")
    with pytest.raises(ValueError, match="malformed"):
        module.module_unit("model.layers.01.weight")

    weight_map = {
        **{f"model.layers.{layer}.weight": "source.safetensors" for layer in range(64)},
        "model.embed_tokens.weight": "source.safetensors",
        "model.norm.weight": "source.safetensors",
        "lm_head.weight": "source.safetensors",
    }
    rank1 = module.select_rank_keys(
        weight_map,
        rank=1,
        world_size=2,
        stage_layers=(40, 24),
        shared_prefixes=SHARED_PREFIXES,
    )
    rank0 = module.select_rank_keys(
        weight_map,
        rank=0,
        world_size=2,
        stage_layers=(36, 28),
        shared_prefixes=SHARED_PREFIXES,
    )
    assert "model.layers.0.weight" in rank1
    assert "model.layers.39.weight" in rank1
    assert "model.layers.40.weight" not in rank1
    assert "model.layers.35.weight" not in rank0
    assert "model.layers.36.weight" in rank0
    assert all(shared in rank1 and shared in rank0 for shared in weight_map if not shared.startswith("model.layers."))


def test_plan_accepts_upstream_total_parameters_metadata(tmp_path: Path) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["metadata"]["total_parameters"] = 32_768_000_000
    index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")

    plan = module.plan_rank_pack(
        source,
        tmp_path / "output",
        _profile(),
        rank=1,
        max_shard_bytes=512,
    )

    assert plan.shards


@pytest.mark.parametrize(
    ("rank", "world_size", "stage_layers", "shared_prefixes"),
    [
        (True, 2, (40, 24), SHARED_PREFIXES),
        (0, True, (40, 24), SHARED_PREFIXES),
        (2, 2, (40, 24), SHARED_PREFIXES),
        (0, 3, (40, 24), SHARED_PREFIXES),
        (0, 2, [40, 24], SHARED_PREFIXES),
        (0, 2, (40, True), SHARED_PREFIXES),
        (0, 2, (40, 23), SHARED_PREFIXES),
        (0, 2, (40, 24), tuple(reversed(SHARED_PREFIXES))),
    ],
)
def test_selection_rejects_invalid_production_partition_types(
    rank, world_size, stage_layers, shared_prefixes
) -> None:
    module = rank_pack()
    weight_map = {
        **{f"model.layers.{layer}.weight": "source.safetensors" for layer in range(64)},
        "model.embed_tokens.weight": "source.safetensors",
        "model.norm.weight": "source.safetensors",
        "lm_head.weight": "source.safetensors",
    }
    with pytest.raises(ValueError):
        module.select_rank_keys(
            weight_map,
            rank=rank,
            world_size=world_size,
            stage_layers=stage_layers,
            shared_prefixes=shared_prefixes,
        )


def test_selection_rejects_missing_layers_unknown_keys_and_bad_map_values() -> None:
    module = rank_pack()
    base = {
        **{f"model.layers.{layer}.weight": "source.safetensors" for layer in range(64)},
        "model.embed_tokens.weight": "source.safetensors",
        "model.norm.weight": "source.safetensors",
        "lm_head.weight": "source.safetensors",
    }
    for changed in (
        {key: value for key, value in base.items() if key != "model.layers.8.weight"},
        {key: value for key, value in base.items() if key != "model.norm.weight"},
        {**base, "model.rotary_emb.weight": "source.safetensors"},
        {**base, "model.layers.0.extra": 7},
    ):
        with pytest.raises(ValueError):
            module.select_rank_keys(
                changed,
                rank=1,
                world_size=2,
                stage_layers=(40, 24),
                shared_prefixes=SHARED_PREFIXES,
            )


def test_plan_is_frozen_complete_deterministic_and_never_splits_units(tmp_path: Path) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    first = module.plan_rank_pack(source, tmp_path / "first", _profile(), rank=1, max_shard_bytes=31)
    second = module.plan_rank_pack(source, tmp_path / "second", _profile(), rank=1, max_shard_bytes=31)

    assert first.plan_id == second.plan_id
    assert first.selected_keys == second.selected_keys
    assert tuple(shard.filename for shard in first.shards) == tuple(
        f"model-{number:05d}-of-{len(first.shards):05d}.safetensors"
        for number in range(1, len(first.shards) + 1)
    )
    planned = [record.name for shard in first.shards for record in shard.tensors]
    assert set(planned) == set(first.selected_keys)
    assert len(planned) == len(set(planned))
    for unit in {module.module_unit(key) for key in planned}:
        assert sum(unit in shard.units for shard in first.shards) == 1
    assert all(shard.payload_bytes <= 31 for shard in first.shards)
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.rank = 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.shards[0].filename = "changed"
    changed_config = json.loads(first.config_bytes)
    changed_config["upstream_field"] = {"preserved": False}
    with pytest.raises(ValueError, match="plan_id"):
        dataclasses.replace(
            first,
            config_bytes=(
                json.dumps(changed_config, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode(),
        )
    original_shard = first.shards[0]
    changed_record = dataclasses.replace(
        original_shard.tensors[0],
        start=original_shard.tensors[0].start + 1,
        end=original_shard.tensors[0].end + 1,
    )
    changed_shard = dataclasses.replace(
        original_shard,
        tensors=(changed_record, *original_shard.tensors[1:]),
    )
    with pytest.raises(ValueError, match="plan_id"):
        dataclasses.replace(first, shards=(changed_shard, *first.shards[1:]))


@pytest.mark.parametrize("mutation", ["missing", "extra", "mismatch", "order"])
def test_manifest_shard_units_exactly_match_tensor_derived_units(mutation: str) -> None:
    module = rank_pack()
    shard = module.ManifestShard(
        "model-00001-of-00001.safetensors",
        (
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ),
        ("model.embed_tokens", "model.layers.0"),
        2,
    )
    if mutation == "missing":
        changed = shard.module_units[:-1]
    elif mutation == "extra":
        changed = (*shard.module_units, "model.layers.1")
    elif mutation == "mismatch":
        changed = ("model.embed_tokens", "model.layers.1")
    else:
        changed = tuple(reversed(shard.module_units))

    with pytest.raises(ValueError, match="module unit|units"):
        dataclasses.replace(shard, module_units=changed)


def test_oversized_unit_fails_during_planning_without_writing_output(tmp_path: Path) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="module unit.*exceeds"):
        module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=2)
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate-index", "duplicate"),
        ("unknown-index-field", "unexpected"),
        ("missing-index-tensor", "index|header"),
        ("unindexed-shard", "unindexed"),
        ("unknown-tensor", "unclassified"),
    ],
)
def test_plan_strictly_audits_index_headers_and_tensor_names(
    tmp_path: Path, mutation: str, message: str
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    index_path = source / "model.safetensors.index.json"
    if mutation == "duplicate-index":
        index_path.write_text('{"metadata":{"total_size":1},"metadata":{"total_size":1},"weight_map":{}}')
    elif mutation == "unknown-index-field":
        value = json.loads(index_path.read_text())
        value["extra"] = 1
        index_path.write_text(json.dumps(value))
    elif mutation == "missing-index-tensor":
        value = json.loads(index_path.read_text())
        value["weight_map"].pop("model.layers.0.self_attn.q_proj.weight")
        index_path.write_text(json.dumps(value))
    elif mutation == "unindexed-shard":
        _write_safetensors(source / "model-00003-of-00003.safetensors", {"model.layers.0.extra": b"x"})
    else:
        value = json.loads(index_path.read_text())
        value["weight_map"]["model.rotary_emb.weight"] = value["weight_map"].pop(
            "model.layers.0.self_attn.q_proj.weight"
        )
        index_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=message):
        module.plan_rank_pack(source, tmp_path / "output", _profile(), rank=1, max_shard_bytes=64)


def test_plan_rejects_duplicate_config_keys(tmp_path: Path) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    (source / "config.json").write_text(
        '{"model_type":"qwen3","model_type":"qwen3","num_hidden_layers":64}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        module.plan_rank_pack(source, tmp_path / "output", _profile(), rank=1)


def test_pack_copies_exact_adapter_assets_and_canonical_metadata(tmp_path: Path) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile((36, 28)), rank=0, max_shard_bytes=40)
    manifest = module.pack_rank(plan)

    assert manifest.plan_id == plan.plan_id
    with pytest.raises(ValueError, match="shard|sequence"):
        dataclasses.replace(manifest, shards=tuple(reversed(manifest.shards)))
    with pytest.raises(ValueError, match="managed_files"):
        dataclasses.replace(manifest, managed_files=manifest.managed_files[:-1])
    assert not (output / module.STAGING_MANIFEST_NAME).exists()
    assert (output / module.FINAL_MANIFEST_NAME).is_file()
    assert (output / "qwen3_pipeline.py").read_bytes() == (
        Path(module.__file__).with_name("qwen3_pipeline.py").read_bytes()
    )
    config = json.loads((output / "config.json").read_text())
    assert config["upstream_field"] == {"preserved": True}
    assert config["model_file"] == "qwen3_pipeline.py"
    assert config["pipeline_stage_layers"] == [36, 28]
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == sum(shard.payload_bytes for shard in plan.shards)
    assert set(index["weight_map"]) == set(plan.selected_keys)
    assert set(index["weight_map"].values()) == {shard.filename for shard in plan.shards}
    assert (output / "tokenizer.json").read_bytes() == (source / "tokenizer.json").read_bytes()
    assert (output / "README.md").read_bytes() == (source / "README.md").read_bytes()
    assert not (output / "notes.txt").exists()
    assert not (output / "nested").exists()
    for json_name in ("config.json", "model.safetensors.index.json", module.FINAL_MANIFEST_NAME):
        assert (output / json_name).read_bytes().endswith(b"\n")


def test_fresh_packs_are_byte_deterministic_and_completed_pack_is_idempotent(tmp_path: Path) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    outputs = (tmp_path / "one", tmp_path / "two")
    manifests = []
    for output in outputs:
        plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=27)
        manifests.append(module.pack_rank(plan))
    assert manifests[0] == manifests[1]
    assert _tree_hashes(outputs[0]) == _tree_hashes(outputs[1])
    before = _tree_hashes(outputs[0])
    plan = module.plan_rank_pack(source, outputs[0], _profile(), rank=1, max_shard_bytes=27)
    assert module.pack_rank(plan, force=True) == manifests[0]
    assert _tree_hashes(outputs[0]) == before


def test_completed_output_requires_force_for_idempotent_reuse(tmp_path: Path) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=512)
    manifest = module.pack_rank(plan)

    with pytest.raises(FileExistsError, match="force|nonempty|completed"):
        module.pack_rank(plan)
    assert module.pack_rank(plan, force=True) == manifest

    config = output / module.CONFIG_NAME
    payload = config.read_bytes()
    os.replace(config, tmp_path / "displaced-owned-config")
    config.write_bytes(payload)
    replacement = os.lstat(config)
    with pytest.raises(ValueError, match="owned|ownership|transaction"):
        module.pack_rank(plan, force=True)
    current = os.lstat(config)
    assert (current.st_dev, current.st_ino) == (replacement.st_dev, replacement.st_ino)


def test_idempotence_rejects_changed_output_tensor_descriptors(tmp_path: Path) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=32)
    module.pack_rank(plan)
    shard = output / plan.shards[0].filename
    raw = shard.read_bytes()
    (header_length,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_length])
    first_key = next(iter(header))
    header[first_key]["shape"] = [1, *header[first_key]["shape"]]
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + header_length :])

    with pytest.raises(ValueError, match="descriptor|shard"):
        module.pack_rank(plan)


def test_idempotence_rejects_swapped_same_size_output_offsets(tmp_path: Path) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=32)
    module.pack_rank(plan)
    shard = output / plan.shards[0].filename
    raw = shard.read_bytes()
    (header_length,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_length])
    equal_size = [
        key
        for key, value in header.items()
        if value["data_offsets"][1] - value["data_offsets"][0] == 1
    ]
    assert len(equal_size) >= 2
    first, second = equal_size[:2]
    header[first]["data_offsets"], header[second]["data_offsets"] = (
        header[second]["data_offsets"],
        header[first]["data_offsets"],
    )
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    assert len(encoded) == header_length
    shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + header_length :])

    with pytest.raises(ValueError, match="descriptor|shard"):
        module.pack_rank(plan)


def test_destination_and_force_marker_safety(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    output.mkdir()
    (output / "user.txt").write_text("keep", encoding="utf-8")
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=64)
    with pytest.raises(FileExistsError, match="nonempty"):
        module.pack_rank(plan)
    with pytest.raises(ValueError, match="staging"):
        module.pack_rank(plan, force=True)
    assert (output / "user.txt").read_text() == "keep"

    (output / "user.txt").unlink()
    (output / module.STAGING_MANIFEST_NAME).write_text(
        json.dumps({"plan_id": "wrong"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="staging|plan"):
        module.pack_rank(plan, force=True)

    original_read_bytes = Path.read_bytes

    def reject_unsafe_marker_read(path: Path) -> bytes:
        if path.name == module.STAGING_MANIFEST_NAME:
            raise AssertionError("staging marker must use bounded no-follow reader")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_unsafe_marker_read)
    with pytest.raises(ValueError, match="staging|plan"):
        module.pack_rank(plan, force=True)


@pytest.mark.parametrize("boundary", ["shard", "asset", "config", "index", "manifest"])
def test_interruption_leaves_matching_staging_marker_and_no_final_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=32)
    failure = RuntimeError(f"fault at {boundary}")

    if boundary == "shard":
        original = module.write_shard
        calls = 0

        def fail_second_shard(records, path):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise failure
            return original(records, path)

        monkeypatch.setattr(module, "write_shard", fail_second_shard)
    else:
        original = module._publish_bytes

        def fail_boundary(path, payload, *, replace=False, owner_token):
            if boundary == "manifest" and path.name == module.STAGING_MANIFEST_NAME:
                original(path, payload, replace=replace, owner_token=owner_token)
                raise failure
            if (
                (boundary == "asset" and path.name == "tokenizer.json")
                or (boundary == "config" and path.name == "config.json")
                or (boundary == "index" and path.name == "model.safetensors.index.json")
            ):
                raise failure
            return original(path, payload, replace=replace, owner_token=owner_token)

        monkeypatch.setattr(module, "_publish_bytes", fail_boundary)

    with pytest.raises(RuntimeError, match=f"fault at {boundary}"):
        module.pack_rank(plan)
    assert not (output / module.FINAL_MANIFEST_NAME).exists()
    marker = json.loads((output / module.STAGING_MANIFEST_NAME).read_text())
    assert marker["plan_id"] == plan.plan_id
    assert marker["state"] in {"staging", "complete"}


def test_matching_force_marker_recovers_only_owned_partial_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=32)
    original = module.write_shard
    calls = 0

    def fail_second(records, path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return original(records, path)

    monkeypatch.setattr(module, "write_shard", fail_second)
    with pytest.raises(RuntimeError, match="interrupted"):
        module.pack_rank(plan)
    monkeypatch.setattr(module, "write_shard", original)
    (output / "user-file.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        module.pack_rank(plan, force=True)
    assert (output / "user-file.txt").read_text() == "preserve"
    (output / "user-file.txt").unlink()
    manifest = module.pack_rank(plan, force=True)
    assert manifest.plan_id == plan.plan_id
    assert (output / module.FINAL_MANIFEST_NAME).is_file()


@pytest.mark.parametrize(
    "target_name",
    [
        "README.md",
        "config.json",
        "model-00001-of-00001.safetensors",
        "model.safetensors.index.json",
        "qwen3_pipeline.py",
        "tokenizer.json",
        "rank-manifest.json.staging",
    ],
)
@pytest.mark.parametrize("timing", ["before", "after"])
def test_force_cleanup_keeps_bound_marker_until_every_owned_artifact_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
    timing: str,
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=512)
    original_rename = module._rename_no_replace

    def stop_before_final(source_path, destination_path) -> None:
        if Path(destination_path).name == module.FINAL_MANIFEST_NAME:
            raise RuntimeError("leave complete staging output")
        original_rename(source_path, destination_path)

    monkeypatch.setattr(module, "_rename_no_replace", stop_before_final)
    with pytest.raises(RuntimeError, match="leave complete staging output"):
        module.pack_rank(plan)
    monkeypatch.setattr(module, "_rename_no_replace", original_rename)

    target = output / target_name
    assert target.is_file()
    original_unlink = os.unlink
    failure = KeyboardInterrupt(f"cleanup {timing} {target_name}")

    def interrupt_cleanup(path) -> None:
        if Path(path) == target:
            if timing == "after":
                original_unlink(path)
            raise failure
        original_unlink(path)

    monkeypatch.setattr(os, "unlink", interrupt_cleanup)
    with pytest.raises(KeyboardInterrupt, match=f"cleanup {timing}") as caught:
        module.pack_rank(plan, force=True)

    assert caught.value is failure
    assert not (output / module.FINAL_MANIFEST_NAME).exists()
    remaining = {entry.name for entry in os.scandir(output)}
    if target_name == module.STAGING_MANIFEST_NAME and timing == "after":
        assert remaining == set()
    else:
        marker = json.loads((output / module.STAGING_MANIFEST_NAME).read_text())
        assert marker["plan_id"] == plan.plan_id
        assert marker["state"] == "complete"

    monkeypatch.setattr(os, "unlink", original_unlink)
    manifest = module.pack_rank(plan, force=True)
    assert manifest.plan_id == plan.plan_id


def test_force_cleanup_preserves_an_identity_replacement_and_matching_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=512)
    original_rename = module._rename_no_replace

    def stop_before_final(source_path, destination_path) -> None:
        if Path(destination_path).name == module.FINAL_MANIFEST_NAME:
            raise RuntimeError("leave complete staging output")
        original_rename(source_path, destination_path)

    monkeypatch.setattr(module, "_rename_no_replace", stop_before_final)
    with pytest.raises(RuntimeError, match="leave complete staging output"):
        module.pack_rank(plan)
    monkeypatch.setattr(module, "_rename_no_replace", original_rename)

    target = output / "config.json"
    displaced = output / "displaced-tool-config"
    unrelated = b"unrelated replacement config\n"
    original_validate = module._validate_snapshot
    injected = False

    def replace_after_binding(snapshot, label) -> None:
        nonlocal injected
        if not injected and snapshot.path == target:
            injected = True
            os.replace(target, displaced)
            target.write_bytes(unrelated)
        original_validate(snapshot, label)

    monkeypatch.setattr(module, "_validate_snapshot", replace_after_binding)
    with pytest.raises(ValueError, match="changed"):
        module.pack_rank(plan, force=True)

    assert target.read_bytes() == unrelated
    assert displaced.is_file()
    marker = json.loads((output / module.STAGING_MANIFEST_NAME).read_text())
    assert marker["plan_id"] == plan.plan_id
    assert not (output / module.FINAL_MANIFEST_NAME).exists()


@pytest.mark.parametrize(
    "target_name",
    [
        "rank-manifest.json.staging",
        "model-00001-of-00001.safetensors",
        "README.md",
        "qwen3_pipeline.py",
        "config.json",
        "model.safetensors.index.json",
    ],
)
def test_nonfinal_publication_collision_preserves_unrelated_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=512)
    original_rename = module._rename_no_replace
    unrelated = f"unrelated {target_name}\n".encode()

    def collide_before_publication(source_path, destination_path) -> None:
        if Path(destination_path).name == target_name:
            Path(destination_path).write_bytes(unrelated)
        original_rename(source_path, destination_path)

    monkeypatch.setattr(module, "_rename_no_replace", collide_before_publication)
    with pytest.raises(FileExistsError):
        module.pack_rank(plan)

    assert (output / target_name).read_bytes() == unrelated
    assert not (output / module.FINAL_MANIFEST_NAME).exists()
    if target_name != module.STAGING_MANIFEST_NAME:
        marker = json.loads((output / module.STAGING_MANIFEST_NAME).read_text())
        assert marker["plan_id"] == plan.plan_id
        assert marker["state"] == "complete"
        monkeypatch.setattr(module, "_rename_no_replace", original_rename)
        with pytest.raises(ValueError, match="differs|changed"):
            module.pack_rank(plan, force=True)
        assert (output / target_name).read_bytes() == unrelated
        assert (output / module.STAGING_MANIFEST_NAME).is_file()


def test_shard_publication_rejects_a_replaced_writer_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=512)
    original_rename = module._rename_no_replace
    displaced = output / "displaced-tool-shard"
    injected = False

    def replace_shard_temporary(source_path, destination_path) -> None:
        nonlocal injected
        source_value = Path(source_path)
        destination_value = Path(destination_path)
        if not injected and destination_value.suffix == ".safetensors":
            injected = True
            payload = bytearray(source_value.read_bytes())
            payload[-1] ^= 0xFF
            os.replace(source_value, displaced)
            source_value.write_bytes(payload)
        original_rename(source_path, destination_path)

    monkeypatch.setattr(module, "_rename_no_replace", replace_shard_temporary)
    with pytest.raises(ValueError, match="shard.*changed"):
        module.pack_rank(plan)

    assert injected
    assert displaced.is_file()
    assert not (output / module.FINAL_MANIFEST_NAME).exists()
    marker = json.loads((output / module.STAGING_MANIFEST_NAME).read_text())
    assert marker["plan_id"] == plan.plan_id


def test_shard_hash_rejects_a_same_bytes_writer_inode_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=512)
    original_hash = module._hash_regular_file
    displaced = tmp_path / "displaced-tool-shard-tmp"
    replacement_inode: tuple[int, int] | None = None

    def replace_before_shard_hash(path: Path, label: str):
        nonlocal replacement_inode
        if replacement_inode is None and path.name.endswith(".safetensors.tmp"):
            payload = path.read_bytes()
            os.replace(path, displaced)
            path.write_bytes(payload)
            value = os.lstat(path)
            replacement_inode = (value.st_dev, value.st_ino)
        return original_hash(path, label)

    monkeypatch.setattr(module, "_hash_regular_file", replace_before_shard_hash)
    with pytest.raises(ValueError, match="shard|writer|changed"):
        module.pack_rank(plan)

    shard_tmp = output / f"{plan.shards[0].filename}.tmp"
    assert replacement_inode is not None
    current = os.lstat(shard_tmp)
    assert (current.st_dev, current.st_ino) == replacement_inode
    assert displaced.is_file()
    assert not (output / module.FINAL_MANIFEST_NAME).exists()


def test_force_preserves_same_bytes_shard_destination_collision_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=512)
    original_rename = module._rename_no_replace
    collided: Path | None = None
    collision_inode: tuple[int, int] | None = None

    def collide_with_same_bytes(source_path, destination_path) -> None:
        nonlocal collided, collision_inode
        source_value = Path(source_path)
        destination_value = Path(destination_path)
        if destination_value.suffix == ".safetensors":
            destination_value.write_bytes(source_value.read_bytes())
            value = os.lstat(destination_value)
            collided = destination_value
            collision_inode = (value.st_dev, value.st_ino)
        original_rename(source_path, destination_path)

    monkeypatch.setattr(module, "_rename_no_replace", collide_with_same_bytes)
    with pytest.raises(FileExistsError):
        module.pack_rank(plan)

    assert collided is not None and collision_inode is not None
    monkeypatch.setattr(module, "_rename_no_replace", original_rename)
    with pytest.raises(ValueError, match="owned|ownership|transaction"):
        module.pack_rank(plan, force=True)

    current = os.lstat(collided)
    assert (current.st_dev, current.st_ino) == collision_inode
    assert not (output / module.FINAL_MANIFEST_NAME).exists()


def test_metadata_temporary_same_bytes_inode_replacement_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=512)
    original_hash = module._hash_regular_file
    displaced = tmp_path / "displaced-tool-config-tmp"
    replacement_inode: tuple[int, int] | None = None

    def replace_before_metadata_hash(path: Path, label: str):
        nonlocal replacement_inode
        if replacement_inode is None and path.name == "config.json.tmp":
            payload = path.read_bytes()
            os.replace(path, displaced)
            path.write_bytes(payload)
            value = os.lstat(path)
            replacement_inode = (value.st_dev, value.st_ino)
        return original_hash(path, label)

    monkeypatch.setattr(module, "_hash_regular_file", replace_before_metadata_hash)
    with pytest.raises(ValueError, match="metadata|temporary|changed"):
        module.pack_rank(plan)

    config_tmp = output / "config.json.tmp"
    assert replacement_inode is not None
    current = os.lstat(config_tmp)
    assert (current.st_dev, current.st_ino) == replacement_inode
    assert displaced.is_file()
    assert not (output / module.FINAL_MANIFEST_NAME).exists()


def test_force_preserves_an_unrelated_shard_temporary_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=512)
    original_write_shard = module.write_shard
    unrelated = b"unrelated shard temporary\n"
    collided: Path | None = None

    def collide_before_shard_write(records, path):
        nonlocal collided
        collided = Path(path)
        collided.write_bytes(unrelated)
        return original_write_shard(records, path)

    monkeypatch.setattr(module, "write_shard", collide_before_shard_write)
    with pytest.raises(FileExistsError):
        module.pack_rank(plan)

    assert collided is not None
    assert collided.read_bytes() == unrelated
    monkeypatch.setattr(module, "write_shard", original_write_shard)
    with pytest.raises(ValueError, match="differs|changed|plan-owned"):
        module.pack_rank(plan, force=True)

    assert collided.read_bytes() == unrelated
    marker = json.loads((output / module.STAGING_MANIFEST_NAME).read_text())
    assert marker["plan_id"] == plan.plan_id
    assert not (output / module.FINAL_MANIFEST_NAME).exists()


def test_force_promotes_a_complete_staging_temporary_but_preserves_foreign_tmp(
    tmp_path: Path,
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")

    valid_output = tmp_path / "valid-output"
    valid_output.mkdir()
    valid_plan = module.plan_rank_pack(
        source, valid_output, _profile(), rank=1, max_shard_bytes=512
    )
    valid_tmp = valid_output / f"{module.STAGING_MANIFEST_NAME}.tmp"
    valid_tmp.write_bytes(
        module._manifest_bytes(module._manifest_for(valid_plan), "complete")
    )
    valid_snapshot = module._hash_regular_file(valid_tmp, "valid staging temporary")
    module._tag_owned_file(
        valid_tmp,
        valid_snapshot,
        os.urandom(module._OWNER_TOKEN_BYTES),
        "valid staging temporary",
    )

    manifest = module.pack_rank(valid_plan, force=True)
    assert manifest.plan_id == valid_plan.plan_id
    assert not valid_tmp.exists()
    assert (valid_output / module.FINAL_MANIFEST_NAME).is_file()

    foreign_output = tmp_path / "foreign-output"
    foreign_output.mkdir()
    foreign_plan = module.plan_rank_pack(
        source, foreign_output, _profile(), rank=1, max_shard_bytes=512
    )
    foreign_tmp = foreign_output / f"{module.STAGING_MANIFEST_NAME}.tmp"
    foreign = b"foreign staging temporary\n"
    foreign_tmp.write_bytes(foreign)

    with pytest.raises(ValueError, match="staging|manifest|plan"):
        module.pack_rank(foreign_plan, force=True)
    assert foreign_tmp.read_bytes() == foreign
    assert not (foreign_output / module.FINAL_MANIFEST_NAME).exists()


def test_shard_replacement_after_directory_fsync_blocks_final_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=512)
    original_fsync = module._fsync_directory
    shard_name = plan.shards[0].filename
    displaced = tmp_path / "displaced-fsynced-shard"
    replacement: bytes | None = None
    injected = False

    def replace_after_shard_fsync(path: Path) -> None:
        nonlocal injected, replacement
        original_fsync(path)
        shard = path / shard_name
        if not injected and shard.exists():
            injected = True
            changed = bytearray(shard.read_bytes())
            changed[-1] ^= 0xFF
            replacement = bytes(changed)
            os.replace(shard, displaced)
            shard.write_bytes(replacement)

    monkeypatch.setattr(module, "_fsync_directory", replace_after_shard_fsync)
    with pytest.raises(ValueError, match="shard.*changed|differs"):
        module.pack_rank(plan)

    assert injected
    assert replacement is not None
    assert (output / shard_name).read_bytes() == replacement
    assert displaced.is_file()
    assert not (output / module.FINAL_MANIFEST_NAME).exists()
    marker = json.loads((output / module.STAGING_MANIFEST_NAME).read_text())
    assert marker["plan_id"] == plan.plan_id


@pytest.mark.parametrize("target_name", ["config.json", "README.md"])
def test_metadata_replacement_after_staged_validation_blocks_final_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=512)
    original_validate = module._validate_staged_output
    displaced = tmp_path / f"displaced-{target_name}"
    hostile = f"hostile replacement for {target_name}\n".encode()
    injected = False

    def replace_after_validation(plan_value, expected) -> None:
        nonlocal injected
        original_validate(plan_value, expected)
        target = output / target_name
        os.replace(target, displaced)
        target.write_bytes(hostile)
        injected = True

    monkeypatch.setattr(module, "_validate_staged_output", replace_after_validation)
    with pytest.raises(ValueError, match="changed|publication"):
        module.pack_rank(plan)

    assert injected
    assert (output / target_name).read_bytes() == hostile
    assert displaced.is_file()
    assert not (output / module.FINAL_MANIFEST_NAME).exists()


def test_same_bytes_staging_inode_replacement_is_not_adopted_as_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=512)
    original_validate = module._validate_staged_output
    displaced = tmp_path / "displaced-tool-staging"
    replacement_inode: tuple[int, int] | None = None

    def replace_marker_after_validation(plan_value, expected) -> None:
        nonlocal replacement_inode
        original_validate(plan_value, expected)
        marker = output / module.STAGING_MANIFEST_NAME
        payload = marker.read_bytes()
        os.replace(marker, displaced)
        marker.write_bytes(payload)
        value = os.lstat(marker)
        replacement_inode = (value.st_dev, value.st_ino)

    monkeypatch.setattr(module, "_validate_staged_output", replace_marker_after_validation)
    with pytest.raises(ValueError, match="staging|changed|publication"):
        module.pack_rank(plan)

    marker = output / module.STAGING_MANIFEST_NAME
    assert replacement_inode is not None
    current = os.lstat(marker)
    assert (current.st_dev, current.st_ino) == replacement_inode
    assert displaced.is_file()
    assert not (output / module.FINAL_MANIFEST_NAME).exists()


def test_final_directory_fsync_failure_restores_complete_staging_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=32)
    original = module._fsync_directory
    injected = False

    def fail_after_final_rename(path: Path) -> None:
        nonlocal injected
        if not injected and (path / module.FINAL_MANIFEST_NAME).exists():
            injected = True
            raise RuntimeError("final directory fsync failed")
        original(path)

    monkeypatch.setattr(module, "_fsync_directory", fail_after_final_rename)
    with pytest.raises(RuntimeError, match="final directory fsync failed"):
        module.pack_rank(plan)
    assert not (output / module.FINAL_MANIFEST_NAME).exists()
    marker = json.loads((output / module.STAGING_MANIFEST_NAME).read_text())
    assert marker["plan_id"] == plan.plan_id
    assert marker["state"] == "complete"


class _InjectedBaseException(BaseException):
    pass


@pytest.mark.parametrize(
    ("boundary", "failure_type"),
    [
        ("post-rename", KeyboardInterrupt),
        ("post-rename", _InjectedBaseException),
        ("post-rename", OSError),
        ("replace", OSError),
        ("fsync", OSError),
    ],
)
def test_final_manifest_publication_rolls_back_only_its_renamed_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    failure_type: type[BaseException],
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=32)
    failure = failure_type(f"injected {boundary} failure")
    unknown = output / "unknown-user-file.txt"

    if boundary in {"post-rename", "replace"}:
        original_rename = module._rename_no_replace

        def fail_final_rename(source_path, destination_path) -> None:
            source_value = Path(source_path)
            destination_value = Path(destination_path)
            if (
                source_value.name == module.STAGING_MANIFEST_NAME
                and destination_value.name == module.FINAL_MANIFEST_NAME
            ):
                if boundary == "post-rename":
                    original_rename(source_path, destination_path)
                unknown.write_text("preserve me", encoding="utf-8")
                raise failure
            original_rename(source_path, destination_path)

        monkeypatch.setattr(module, "_rename_no_replace", fail_final_rename)
    else:
        original_fsync = module._fsync_directory
        injected = False

        def fail_final_fsync(path: Path) -> None:
            nonlocal injected
            if not injected and (path / module.FINAL_MANIFEST_NAME).exists():
                injected = True
                unknown.write_text("preserve me", encoding="utf-8")
                raise failure
            original_fsync(path)

        monkeypatch.setattr(module, "_fsync_directory", fail_final_fsync)

    with pytest.raises(
        failure_type, match=f"injected {boundary} failure"
    ) as caught:
        module.pack_rank(plan)

    assert caught.value is failure
    assert not (output / module.FINAL_MANIFEST_NAME).exists()
    marker = json.loads((output / module.STAGING_MANIFEST_NAME).read_text())
    assert marker["plan_id"] == plan.plan_id
    assert marker["state"] == "complete"
    assert unknown.read_text() == "preserve me"


def test_final_publication_rollback_inspection_failure_does_not_mask_primary_or_leave_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=32)
    primary = KeyboardInterrupt("publication interrupted")
    original_rename = module._rename_no_replace
    original_exists = Path.exists

    def publish_then_fail(source_path, destination_path) -> None:
        original_rename(source_path, destination_path)
        if (
            Path(source_path).name == module.STAGING_MANIFEST_NAME
            and Path(destination_path).name == module.FINAL_MANIFEST_NAME
        ):
            raise primary

    def fail_rollback_staging_inspection(path: Path) -> bool:
        if (
            path.name == module.STAGING_MANIFEST_NAME
            and original_exists(path.parent / module.FINAL_MANIFEST_NAME)
        ):
            raise OSError("rollback staging inspection failed")
        return original_exists(path)

    monkeypatch.setattr(module, "_rename_no_replace", publish_then_fail)
    monkeypatch.setattr(Path, "exists", fail_rollback_staging_inspection)

    with pytest.raises(KeyboardInterrupt, match="publication interrupted") as caught:
        module.pack_rank(plan)

    assert caught.value is primary
    assert any(
        "rollback staging inspection failed" in note
        for note in getattr(caught.value, "__notes__", ())
    )
    assert not (output / module.FINAL_MANIFEST_NAME).exists()


def test_final_publication_restore_failure_preserves_primary_and_removes_owned_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=32)
    primary = OSError("publication failed after rename")
    restore_failure = OSError("restore rename failed")
    original_rename = module._rename_no_replace

    def publish_then_fail(source_path, destination_path) -> None:
        if (
            Path(source_path).name == module.FINAL_MANIFEST_NAME
            and Path(destination_path).name == module.STAGING_MANIFEST_NAME
        ):
            raise restore_failure
        original_rename(source_path, destination_path)
        if Path(destination_path).name == module.FINAL_MANIFEST_NAME:
            raise primary

    monkeypatch.setattr(module, "_rename_no_replace", publish_then_fail)

    with pytest.raises(OSError, match="publication failed after rename") as caught:
        module.pack_rank(plan)

    assert caught.value is primary
    assert any(
        "restore rename failed" in note for note in getattr(caught.value, "__notes__", ())
    )
    assert not (output / module.FINAL_MANIFEST_NAME).exists()
    assert not (output / module.STAGING_MANIFEST_NAME).exists()


def test_final_rollback_no_clobber_preserves_concurrent_staging_and_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=512)
    primary = OSError("final directory fsync failed")
    concurrent = b"concurrent staging replacement\n"
    original_fsync = module._fsync_directory
    original_exists = Path.exists
    fsync_injected = False
    staging_injected = False

    def fail_final_fsync(path: Path) -> None:
        nonlocal fsync_injected
        if not fsync_injected and (path / module.FINAL_MANIFEST_NAME).exists():
            fsync_injected = True
            raise primary
        original_fsync(path)

    def create_staging_after_absence_check(path: Path) -> bool:
        nonlocal staging_injected
        if (
            not staging_injected
            and path.name == module.STAGING_MANIFEST_NAME
            and original_exists(path.parent / module.FINAL_MANIFEST_NAME)
        ):
            staging_injected = True
            path.write_bytes(concurrent)
            return False
        return original_exists(path)

    monkeypatch.setattr(module, "_fsync_directory", fail_final_fsync)
    monkeypatch.setattr(Path, "exists", create_staging_after_absence_check)
    with pytest.raises(OSError, match="final directory fsync failed") as caught:
        module.pack_rank(plan)

    assert caught.value is primary
    assert staging_injected
    assert (output / module.STAGING_MANIFEST_NAME).read_bytes() == concurrent
    assert not (output / module.FINAL_MANIFEST_NAME).exists()


def test_successful_final_manifest_publication_is_the_last_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=32)
    original_rename = module._rename_no_replace
    original_fsync_directory = module._fsync_directory
    final_events: list[str] = []

    def track_final_rename(source_path, destination_path) -> None:
        original_rename(source_path, destination_path)
        if (
            Path(source_path).name == module.STAGING_MANIFEST_NAME
            and Path(destination_path).name == module.FINAL_MANIFEST_NAME
        ):
            final_events.append("rename")

    def track_final_fsync(path: Path) -> None:
        original_fsync_directory(path)
        if (path / module.FINAL_MANIFEST_NAME).exists():
            final_events.append("fsync")

    monkeypatch.setattr(module, "_rename_no_replace", track_final_rename)
    monkeypatch.setattr(module, "_fsync_directory", track_final_fsync)

    manifest = module.pack_rank(plan)

    assert manifest.plan_id == plan.plan_id
    assert final_events == ["rename", "fsync"]
    assert (output / module.FINAL_MANIFEST_NAME).is_file()
    assert not (output / module.STAGING_MANIFEST_NAME).exists()


def test_final_publication_atomically_refuses_a_concurrent_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=32)
    unrelated = b"concurrent final replacement\n"
    original_rename = module._rename_no_replace

    def create_final_before_rename(source_path, destination_path) -> None:
        if Path(destination_path).name == module.FINAL_MANIFEST_NAME:
            Path(destination_path).write_bytes(unrelated)
        original_rename(source_path, destination_path)

    monkeypatch.setattr(module, "_rename_no_replace", create_final_before_rename)

    with pytest.raises(FileExistsError):
        module.pack_rank(plan)

    assert (output / module.FINAL_MANIFEST_NAME).read_bytes() == unrelated
    marker = json.loads((output / module.STAGING_MANIFEST_NAME).read_text())
    assert marker["state"] == "complete"


def test_final_publication_uses_an_atomic_rename_not_a_hard_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=32)

    def reject_hard_link(*args, **kwargs) -> None:
        raise AssertionError("final publication must not expose a dual-link state")

    monkeypatch.setattr(os, "link", reject_hard_link)

    manifest = module.pack_rank(plan)

    assert manifest.plan_id == plan.plan_id
    assert (output / module.FINAL_MANIFEST_NAME).is_file()
    assert not (output / module.STAGING_MANIFEST_NAME).exists()


def test_final_publication_detects_replacement_during_directory_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=32)
    original_fsync = module._fsync_directory
    displaced = output / "displaced-tool-manifest"
    unrelated = b"replacement during final fsync\n"
    injected = False

    def replace_during_final_fsync(path: Path) -> None:
        nonlocal injected
        original_fsync(path)
        final = path / module.FINAL_MANIFEST_NAME
        if not injected and final.exists():
            injected = True
            os.replace(final, displaced)
            final.write_bytes(unrelated)

    monkeypatch.setattr(module, "_fsync_directory", replace_during_final_fsync)

    with pytest.raises(RuntimeError, match="identity"):
        module.pack_rank(plan)

    assert (output / module.FINAL_MANIFEST_NAME).read_bytes() == unrelated
    assert displaced.is_file()


def test_final_publication_detects_and_preserves_an_unrelated_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=32)
    original_rename = module._rename_no_replace
    displaced = output / "displaced-tool-manifest"
    unrelated = b"unrelated replacement\n"

    def replace_then_substitute(source_path, destination_path) -> None:
        original_rename(source_path, destination_path)
        if (
            Path(source_path).name == module.STAGING_MANIFEST_NAME
            and Path(destination_path).name == module.FINAL_MANIFEST_NAME
        ):
            os.replace(destination_path, displaced)
            Path(destination_path).write_bytes(unrelated)

    monkeypatch.setattr(module, "_rename_no_replace", replace_then_substitute)
    with pytest.raises(RuntimeError, match="identity"):
        module.pack_rank(plan)

    assert (output / module.FINAL_MANIFEST_NAME).read_bytes() == unrelated
    assert displaced.is_file()


def test_unknown_file_race_prevents_final_manifest_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=32)
    original = module._publish_bytes

    def inject_after_index(
        path: Path, payload: bytes, *, replace: bool = False, owner_token: bytes
    ) -> None:
        original(path, payload, replace=replace, owner_token=owner_token)
        if path.name == "model.safetensors.index.json":
            (path.parent / "intruder.txt").write_text("user data", encoding="utf-8")

    monkeypatch.setattr(module, "_publish_bytes", inject_after_index)
    with pytest.raises(ValueError, match="inventory"):
        module.pack_rank(plan)
    assert (output / "intruder.txt").read_text() == "user data"
    assert (output / module.STAGING_MANIFEST_NAME).is_file()
    assert not (output / module.FINAL_MANIFEST_NAME).exists()


def test_plan_rejects_allowed_asset_symlink_and_dangerous_output_paths(tmp_path: Path) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    target = tmp_path / "tokenizer-target.json"
    target.write_text("{}", encoding="utf-8")
    (source / "tokenizer.json").unlink()
    (source / "tokenizer.json").symlink_to(target)
    with pytest.raises(ValueError, match="symlink|regular"):
        module.plan_rank_pack(source, tmp_path / "output", _profile(), rank=1)

    (source / "tokenizer.json").unlink()
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="source|output"):
        module.plan_rank_pack(source, source / "rank-1", _profile(), rank=1)
    with pytest.raises(ValueError, match="source|output"):
        module.plan_rank_pack(source, source, _profile(), rank=1)


def test_source_mutation_after_plan_is_rejected_before_publication(tmp_path: Path) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=64)
    (source / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        module.pack_rank(plan)
    assert not output.exists()


@pytest.mark.parametrize("addition", ["tokenizer_config.json", "model-00003-of-00003.safetensors"])
def test_source_inventory_addition_after_plan_is_rejected(
    tmp_path: Path, addition: str
) -> None:
    module = rank_pack()
    source = _source_model(tmp_path / "source")
    output = tmp_path / "output"
    plan = module.plan_rank_pack(source, output, _profile(), rank=1, max_shard_bytes=64)
    if addition.endswith(".safetensors"):
        _write_safetensors(source / addition, {"model.layers.0.added": b"x"})
    else:
        (source / addition).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory changed"):
        module.pack_rank(plan)
    assert not output.exists()
