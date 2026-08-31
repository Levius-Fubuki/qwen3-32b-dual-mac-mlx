from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import os
import struct
import tracemalloc
from pathlib import Path

import mlx.core as mx
import pytest


MIB = 1 << 20


def safetensor_raw():
    try:
        return importlib.import_module("qwen32_cluster.safetensor_raw")
    except ModuleNotFoundError as exc:
        pytest.fail(f"safetensor_raw module is not implemented: {exc}")


@pytest.fixture
def mlx_safetensor(tmp_path: Path) -> Path:
    path = tmp_path / "quantized.safetensors"
    mx.save_safetensors(
        path,
        {
            "model.layers.0.self_attn.q_proj.weight": mx.array(
                [[0, 1, 2, 4_294_967_295]], dtype=mx.uint32
            ),
            "model.layers.0.self_attn.q_proj.scales": mx.array(
                [[0.5, 1.5]], dtype=mx.bfloat16
            ),
            "model.layers.0.self_attn.q_proj.biases": mx.array(
                [[-0.25, 0.25]], dtype=mx.bfloat16
            ),
            "model.norm.weight": mx.array([1.0, 2.0, 3.0], dtype=mx.bfloat16),
        },
        metadata={"format": "mlx", "fixture": "raw-copy"},
    )
    return path


def _raw_file(path: Path, header: bytes, payload: bytes = b"") -> Path:
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return path


def _json_file(path: Path, header: object, payload: bytes = b"") -> Path:
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return _raw_file(path, encoded, payload)


def _payload_sha(record) -> str:
    with record.source_file.open("rb", buffering=0) as source:
        source.seek(record.start)
        return hashlib.sha256(source.read(record.nbytes)).hexdigest()


def test_records_and_results_are_immutable_and_header_skips_metadata(
    mlx_safetensor: Path,
) -> None:
    module = safetensor_raw()
    shard = module.read_header(mlx_safetensor)

    assert isinstance(shard.tensors, tuple)
    assert shard.data_start == 8 + shard.header_length
    assert {record.dtype for record in shard.tensors} == {"BF16", "U32"}
    assert {record.name for record in shard.tensors} == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.scales",
        "model.layers.0.self_attn.q_proj.biases",
        "model.norm.weight",
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        shard.header_length = 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        shard.tensors[0].name = "changed"


def test_write_shard_reorders_without_changing_tensor_bytes(
    mlx_safetensor: Path, tmp_path: Path
) -> None:
    module = safetensor_raw()
    source = module.read_header(mlx_safetensor)
    expected = {
        record.name: (record.dtype, record.shape, record.nbytes, _payload_sha(record))
        for record in source.tensors
    }
    output = tmp_path / "rank-0.safetensors.tmp"

    result = module.write_shard(tuple(reversed(source.tensors)), output)
    rewritten = module.read_header(output)
    actual = {
        record.name: (record.dtype, record.shape, record.nbytes, _payload_sha(record))
        for record in rewritten.tensors
    }

    assert actual == expected
    assert result.path == output
    assert result.header_length % 8 == 0
    assert result.data_start % 8 == 0
    assert result.file_size == output.stat().st_size
    assert result.sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert dict(result.payload_sha256) == {
        name: details[3] for name, details in expected.items()
    }
    assert result.tensors == rewritten.tensors
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.file_size = 0


def test_writer_is_deterministic_across_input_order(
    mlx_safetensor: Path, tmp_path: Path
) -> None:
    module = safetensor_raw()
    records = module.read_header(mlx_safetensor).tensors
    first = tmp_path / "first.tmp"
    second = tmp_path / "second.tmp"

    module.write_shard(records, first)
    module.write_shard(tuple(reversed(records)), second)

    assert first.read_bytes() == second.read_bytes()


def test_payload_reads_are_bounded_to_eight_mib(
    mlx_safetensor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = safetensor_raw()
    records = module.read_header(mlx_safetensor).tensors
    original_pread = os.pread
    requested_sizes: list[int] = []

    def tracking_pread(fd: int, count: int, offset: int) -> bytes:
        requested_sizes.append(count)
        return original_pread(fd, count, offset)

    monkeypatch.setattr(os, "pread", tracking_pread)
    module.write_shard(records, tmp_path / "bounded.tmp")

    assert requested_sizes
    assert max(requested_sizes) <= 8 * MIB


def test_copy_payload_materialization_stays_bounded_as_payload_grows(
    tmp_path: Path,
) -> None:
    module = safetensor_raw()
    chunk_bytes = 64 * 1024

    def copy_peak(size: int, name: str) -> int:
        source = tmp_path / f"{name}.bin"
        source.write_bytes(b"x" * size)
        record = module.TensorRecord(name, "U8", (size,), source, 0, size, size)
        destination = os.open(os.devnull, os.O_WRONLY)
        try:
            tracemalloc.start()
            module.copy_payload(record, destination, chunk_bytes=chunk_bytes)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return peak
        finally:
            os.close(destination)

    small_peak = copy_peak(4 * chunk_bytes, "small")
    large_peak = copy_peak(64 * chunk_bytes, "large")

    assert large_peak <= small_peak + 4 * chunk_bytes
    assert large_peak < 8 * chunk_bytes


@pytest.mark.parametrize(
    ("header", "payload", "message"),
    [
        (b"not-json", b"", "JSON"),
        (b"[]", b"", "root"),
        (b'{"x":{"dtype":"U8","dtype":"U8","shape":[1],"data_offsets":[0,1]}}', b"x", "duplicate"),
        (b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}', b"x", "duplicate"),
        (b'{"__metadata__":null}', b"", "metadata"),
        (b'{"__metadata__":{"key":1}}', b"", "metadata"),
    ],
    ids=[
        "invalid-json",
        "non-object-root",
        "duplicate-field",
        "duplicate-name",
        "null-metadata",
        "invalid-metadata",
    ],
)
def test_read_header_rejects_invalid_json_and_duplicate_keys(
    tmp_path: Path, header: bytes, payload: bytes, message: str
) -> None:
    module = safetensor_raw()
    path = _raw_file(tmp_path / "bad.safetensors", header, payload)
    with pytest.raises(ValueError, match=message):
        module.read_header(path)


def test_read_header_rejects_header_length_beyond_eof(tmp_path: Path) -> None:
    module = safetensor_raw()
    path = tmp_path / "truncated-header.safetensors"
    path.write_bytes(struct.pack("<Q", 100) + b"{}")
    with pytest.raises(ValueError, match="header length"):
        module.read_header(path)


@pytest.mark.parametrize(
    ("tensor", "payload", "message"),
    [
        (None, b"", "object"),
        ({"dtype": "U8", "shape": [1]}, b"x", "fields"),
        ({"dtype": "U8", "shape": [1], "data_offsets": [0, 1], "extra": 1}, b"x", "fields"),
        ({"dtype": 8, "shape": [1], "data_offsets": [0, 1]}, b"x", "dtype"),
        ({"dtype": "U8", "shape": "1", "data_offsets": [0, 1]}, b"x", "shape"),
        ({"dtype": "U8", "shape": [True], "data_offsets": [0, 1]}, b"x", "shape"),
        ({"dtype": "U8", "shape": [1], "data_offsets": "0,1"}, b"x", "offset"),
        ({"dtype": "U8", "shape": [1], "data_offsets": [False, 1]}, b"x", "offset"),
        ({"dtype": "UNKNOWN", "shape": [1], "data_offsets": [0, 1]}, b"x", "dtype"),
        ({"dtype": "BF16", "shape": [2], "data_offsets": [0, 2]}, b"xx", "nbytes"),
    ],
    ids=[
        "tensor-not-object",
        "missing-field",
        "unknown-field",
        "dtype-not-string",
        "shape-not-list",
        "bool-shape",
        "offsets-not-list",
        "bool-offset",
        "unknown-dtype",
        "size-mismatch",
    ],
)
def test_read_header_rejects_invalid_tensor_fields(
    tmp_path: Path, tensor: object, payload: bytes, message: str
) -> None:
    module = safetensor_raw()
    path = _json_file(tmp_path / "bad-fields.safetensors", {"tensor": tensor}, payload)
    with pytest.raises(ValueError, match=message):
        module.read_header(path)


@pytest.mark.parametrize(
    ("header", "payload", "message"),
    [
        ({"x": {"dtype": "U8", "shape": [4], "data_offsets": [-1, 3]}}, b"xxx", "negative"),
        ({"x": {"dtype": "U8", "shape": [1], "data_offsets": [2, 1]}}, b"xx", "order"),
        ({"x": {"dtype": "U8", "shape": [5], "data_offsets": [0, 5]}}, b"xxxx", "range"),
        (
            {
                "a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
                "b": {"dtype": "U8", "shape": [4], "data_offsets": [2, 6]},
            },
            b"xxxxxx",
            "overlap",
        ),
    ],
    ids=["negative", "reversed", "out-of-range", "overlap"],
)
def test_read_header_rejects_invalid_offsets(
    tmp_path: Path, header: object, payload: bytes, message: str
) -> None:
    module = safetensor_raw()
    path = _json_file(tmp_path / "bad-offset.safetensors", header, payload)
    with pytest.raises(ValueError, match=message):
        module.read_header(path)


def test_read_header_rejects_unsafe_tensor_names(tmp_path: Path) -> None:
    module = safetensor_raw()
    path = _json_file(
        tmp_path / "unsafe-name.safetensors",
        {"bad\u0000name": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}},
        b"x",
    )
    with pytest.raises(ValueError, match="name"):
        module.read_header(path)


@pytest.mark.parametrize("chunk_bytes", [0, -1, True, 1.5, 8 * MIB + 1])
def test_copy_payload_rejects_invalid_chunk_size(
    tmp_path: Path, chunk_bytes: object
) -> None:
    module = safetensor_raw()
    source = tmp_path / "source.bin"
    source.write_bytes(b"x")
    record = module.TensorRecord("x", "U8", (1,), source, 0, 1, 1)
    destination = os.open(os.devnull, os.O_WRONLY)
    try:
        with pytest.raises(ValueError, match="chunk"):
            module.copy_payload(record, destination, chunk_bytes=chunk_bytes)
    finally:
        os.close(destination)


def test_copy_payload_never_requests_more_than_eight_mib_for_large_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = safetensor_raw()
    payload_size = 8 * MIB + 4096
    source = tmp_path / "large-payload.bin"
    with source.open("wb") as source_file:
        source_file.truncate(payload_size)
    record = module.TensorRecord(
        "large", "U8", (payload_size,), source, 0, payload_size, payload_size
    )
    original_pread = os.pread
    requested_sizes: list[int] = []

    def tracking_pread(fd: int, count: int, offset: int) -> bytes:
        requested_sizes.append(count)
        return original_pread(fd, count, offset)

    monkeypatch.setattr(os, "pread", tracking_pread)
    destination = os.open(os.devnull, os.O_WRONLY)
    try:
        module.copy_payload(record, destination)
    finally:
        os.close(destination)

    assert requested_sizes == [8 * MIB, 4096]
    assert max(requested_sizes) <= 8 * MIB


def test_copy_payload_rejects_truncated_reads(
    mlx_safetensor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = safetensor_raw()
    record = module.read_header(mlx_safetensor).tensors[0]
    monkeypatch.setattr(os, "pread", lambda fd, count, offset: b"")
    destination = os.open(os.devnull, os.O_WRONLY)
    try:
        with pytest.raises(OSError, match="truncated"):
            module.copy_payload(record, destination)
    finally:
        os.close(destination)


def test_write_shard_handles_short_writes(
    mlx_safetensor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = safetensor_raw()
    records = module.read_header(mlx_safetensor).tensors
    original_write = os.write

    def short_write(fd: int, data) -> int:
        return original_write(fd, data[: max(1, len(data) // 3)])

    monkeypatch.setattr(os, "write", short_write)
    output = tmp_path / "short-write.tmp"
    module.write_shard(records, output)

    assert {
        record.name: _payload_sha(record) for record in module.read_header(output).tensors
    } == {record.name: _payload_sha(record) for record in records}


def test_write_shard_never_overwrites_existing_output(
    mlx_safetensor: Path, tmp_path: Path
) -> None:
    module = safetensor_raw()
    records = module.read_header(mlx_safetensor).tensors
    output = tmp_path / "existing.tmp"
    output.write_bytes(b"keep-me")

    with pytest.raises(FileExistsError):
        module.write_shard(records, output)

    assert output.read_bytes() == b"keep-me"


def test_write_shard_validates_sources_before_opening_output(tmp_path: Path) -> None:
    module = safetensor_raw()
    source = tmp_path / "source.bin"
    source.write_bytes(b"x")
    invalid = module.TensorRecord("x", "U8", (1,), source, 0, 1, 2)
    output = tmp_path / "must-not-exist.tmp"

    with pytest.raises(ValueError, match="nbytes"):
        module.write_shard((invalid,), output)

    assert not output.exists()


def test_write_shard_requires_explicit_tmp_suffix(
    mlx_safetensor: Path, tmp_path: Path
) -> None:
    module = safetensor_raw()
    records = module.read_header(mlx_safetensor).tensors
    output = tmp_path / "unsafe.safetensors"
    with pytest.raises(ValueError, match=r"\.tmp"):
        module.write_shard(records, output)
    assert not output.exists()


def test_write_shard_cleans_its_partial_file_after_copy_failure(
    mlx_safetensor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = safetensor_raw()
    records = module.read_header(mlx_safetensor).tensors
    output = tmp_path / "partial.tmp"
    monkeypatch.setattr(os, "pread", lambda fd, count, offset: b"")

    with pytest.raises(OSError, match="truncated"):
        module.write_shard(records, output)

    assert not output.exists()


def test_production_module_has_no_tensor_framework_dependencies() -> None:
    module = safetensor_raw()
    source = Path(module.__file__).read_text(encoding="utf-8")
    forbidden = ("mlx.nn", "mlx_lm", "numpy", "mx.load", "mx.save_safetensors", "dequant")
    assert all(token not in source for token in forbidden)
