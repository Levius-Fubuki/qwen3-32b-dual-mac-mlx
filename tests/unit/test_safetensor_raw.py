from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import json
import os
import subprocess
import struct
import sys
import textwrap
from pathlib import Path

import mlx.core as mx
import pytest
import safetensors


MIB = 1 << 20
CANONICAL_SAFETENSORS_VERSION = "0.8.0"
CANONICAL_DTYPES = (
    ("F4", 4, 2),
    ("F6_E2M3", 6, 4),
    ("F6_E3M2", 6, 4),
    ("BOOL", 8, 1),
    ("U8", 8, 1),
    ("I8", 8, 1),
    ("F8_E5M2", 8, 1),
    ("F8_E4M3", 8, 1),
    ("F8_E8M0", 8, 1),
    ("F8_E4M3FNUZ", 8, 1),
    ("F8_E5M2FNUZ", 8, 1),
    ("I16", 16, 1),
    ("U16", 16, 1),
    ("F16", 16, 1),
    ("BF16", 16, 1),
    ("I32", 32, 1),
    ("U32", 32, 1),
    ("F32", 32, 1),
    ("C64", 64, 1),
    ("F64", 64, 1),
    ("I64", 64, 1),
    ("U64", 64, 1),
)


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


def _canonical_json_file(path: Path, header: object, payload: bytes = b"") -> Path:
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
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


def test_copy_payload_rss_high_water_does_not_scale_with_sparse_payload(
    tmp_path: Path,
) -> None:
    script = textwrap.dedent(
        """
        import json
        import os
        import resource
        import sys
        from pathlib import Path
        from qwen32_cluster.safetensor_raw import TensorRecord, copy_payload

        size = int(sys.argv[1])
        source = Path(sys.argv[2])
        output = Path(sys.argv[3])
        with source.open("wb") as source_file:
            source_file.truncate(size)
        record = TensorRecord("large", "U8", (size,), source, 0, size, size)
        destination = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            copy_payload(record, destination)
            os.fsync(destination)
        finally:
            os.close(destination)
        scale = 1 if sys.platform == "darwin" else 1024
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
        print(json.dumps({"peak": peak, "output_size": output.stat().st_size}))
        """
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[2] / "src")

    def peak_for(size: int, label: str) -> dict[str, int]:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(size),
                str(tmp_path / f"{label}.source"),
                str(tmp_path / f"{label}.output"),
            ],
            cwd=Path(__file__).parents[2],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return json.loads(result.stdout)

    small = peak_for(1 * MIB, "small")
    large = peak_for(64 * MIB, "large")

    assert small["output_size"] == 1 * MIB
    assert large["output_size"] == 64 * MIB
    assert large["peak"] <= small["peak"] + 24 * MIB


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


@pytest.mark.parametrize(("dtype", "bits", "elements"), CANONICAL_DTYPES)
def test_dtype_sizes_match_pinned_canonical_safetensors(
    tmp_path: Path, dtype: str, bits: int, elements: int
) -> None:
    assert importlib_metadata.version("safetensors") == CANONICAL_SAFETENSORS_VERSION
    module = safetensor_raw()
    nbytes = elements * bits // 8
    path = _canonical_json_file(
        tmp_path / f"{dtype}.safetensors",
        {dtype: {"dtype": dtype, "shape": [elements], "data_offsets": [0, nbytes]}},
        b"\0" * nbytes,
    )

    record = module.read_header(path).tensors[0]
    canonical = safetensors.deserialize(path.read_bytes())[0]

    assert (record.dtype, record.shape, record.nbytes) == (
        dtype,
        (elements,),
        nbytes,
    )
    assert canonical[1]["dtype"] == dtype


@pytest.mark.parametrize("dtype", ["F8_E4M3FN", "C128"])
def test_noncanonical_dtype_names_are_rejected_by_both_parsers(
    tmp_path: Path, dtype: str
) -> None:
    module = safetensor_raw()
    path = _canonical_json_file(
        tmp_path / f"invalid-{dtype}.safetensors",
        {"x": {"dtype": dtype, "shape": [1], "data_offsets": [0, 1]}},
        b"\0",
    )
    with pytest.raises(ValueError, match="dtype"):
        module.read_header(path)
    with pytest.raises(safetensors.SafetensorError):
        safetensors.deserialize(path.read_bytes())


@pytest.mark.parametrize("dtype", ["F4", "F6_E2M3", "F6_E3M2"])
def test_subbyte_tensors_must_end_on_byte_boundary(
    tmp_path: Path, dtype: str
) -> None:
    module = safetensor_raw()
    path = _canonical_json_file(
        tmp_path / f"unaligned-{dtype}.safetensors",
        {"x": {"dtype": dtype, "shape": [1], "data_offsets": [0, 1]}},
        b"\0",
    )
    with pytest.raises(ValueError, match="byte"):
        module.read_header(path)
    with pytest.raises(safetensors.SafetensorError):
        safetensors.deserialize(path.read_bytes())


def test_tensor_size_calculation_rejects_u64_overflow(tmp_path: Path) -> None:
    module = safetensor_raw()
    path = _json_file(
        tmp_path / "overflow.safetensors",
        {"x": {"dtype": "U64", "shape": [2**63, 2**63], "data_offsets": [0, 0]}},
    )
    with pytest.raises(ValueError, match="overflow"):
        module.read_header(path)


@pytest.mark.parametrize(
    ("header", "payload", "message"),
    [
        ({"x": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]}}, b"xx", "hole"),
        ({"x": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}, b"xx", "trailing"),
    ],
    ids=["leading-hole", "unindexed-trailing-byte"],
)
def test_canonical_buffer_coverage_rejects_holes_and_trailing_bytes(
    tmp_path: Path, header: object, payload: bytes, message: str
) -> None:
    module = safetensor_raw()
    path = _canonical_json_file(tmp_path / f"{message}.safetensors", header, payload)
    with pytest.raises(ValueError, match=message):
        module.read_header(path)
    with pytest.raises(safetensors.SafetensorError):
        safetensors.deserialize(path.read_bytes())


def test_canonical_buffer_coverage_allows_zero_tensors_at_shared_boundaries(
    tmp_path: Path,
) -> None:
    module = safetensor_raw()
    header = {
        "empty-a": {"dtype": "U8", "shape": [0], "data_offsets": [0, 0]},
        "empty-b": {"dtype": "U8", "shape": [0, 4], "data_offsets": [0, 0]},
        "value": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
        "empty-c": {"dtype": "U8", "shape": [0], "data_offsets": [1, 1]},
    }
    path = _canonical_json_file(tmp_path / "zero-boundaries.safetensors", header, b"x")

    records = module.read_header(path).tensors

    assert [(record.name, record.nbytes) for record in records] == [
        ("empty-a", 0),
        ("empty-b", 0),
        ("value", 1),
        ("empty-c", 0),
    ]
    assert len(safetensors.deserialize(path.read_bytes())) == 4


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


@pytest.mark.parametrize("name", ["", "bad\u0000name", "line\nbreak"])
def test_read_header_accepts_canonical_json_tensor_names(
    tmp_path: Path, name: str
) -> None:
    module = safetensor_raw()
    path = _json_file(
        tmp_path / "canonical-name.safetensors",
        {name: {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}},
        b"x",
    )
    shard = module.read_header(path)
    assert shard.tensors[0].name == name
    assert safetensors.deserialize(path.read_bytes())[0][0] == name


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


@pytest.mark.parametrize("alias_kind", ["same-path", "hardlink"])
def test_copy_payload_rejects_regular_destination_alias_before_writing(
    tmp_path: Path, alias_kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = safetensor_raw()
    source = tmp_path / "source.bin"
    source.write_bytes(b"preserve")
    destination_path = source
    if alias_kind == "hardlink":
        destination_path = tmp_path / "destination.bin"
        os.link(source, destination_path)
    record = module.TensorRecord("x", "U8", (8,), source, 0, 8, 8)
    destination = os.open(destination_path, os.O_WRONLY)
    original_write = os.write
    writes = 0

    def tracking_write(fd: int, data) -> int:
        nonlocal writes
        writes += 1
        return original_write(fd, data)

    monkeypatch.setattr(os, "write", tracking_write)
    try:
        with pytest.raises(ValueError, match="same file"):
            module.copy_payload(record, destination)
    finally:
        os.close(destination)

    assert writes == 0
    assert source.read_bytes() == b"preserve"


def test_copy_payload_retries_interrupted_reads_and_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = safetensor_raw()
    source = tmp_path / "source.bin"
    output = tmp_path / "output.bin"
    source.write_bytes(b"payload")
    record = module.TensorRecord("x", "U8", (7,), source, 0, 7, 7)
    original_pread = os.pread
    original_write = os.write
    pread_interrupted = False
    write_interrupted = False

    def interrupted_pread(fd: int, count: int, offset: int) -> bytes:
        nonlocal pread_interrupted
        if not pread_interrupted:
            pread_interrupted = True
            raise InterruptedError
        return original_pread(fd, count, offset)

    def interrupted_write(fd: int, data) -> int:
        nonlocal write_interrupted
        if not write_interrupted:
            write_interrupted = True
            raise InterruptedError
        return original_write(fd, data)

    monkeypatch.setattr(os, "pread", interrupted_pread)
    monkeypatch.setattr(os, "write", interrupted_write)
    destination = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        module.copy_payload(record, destination)
    finally:
        os.close(destination)

    assert pread_interrupted and write_interrupted
    assert output.read_bytes() == b"payload"


def test_read_header_opens_and_fstats_before_reading(
    mlx_safetensor: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = safetensor_raw()
    original_stat = Path.stat

    def forbidden_path_stat(self: Path, *args, **kwargs):
        if self == mlx_safetensor:
            raise AssertionError("read_header must use its opened file descriptor")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", forbidden_path_stat)
    assert module.read_header(mlx_safetensor).tensors


def test_write_shard_rejects_same_size_source_replacement_after_header_read(
    mlx_safetensor: Path, tmp_path: Path
) -> None:
    module = safetensor_raw()
    records = module.read_header(mlx_safetensor).tensors
    original = mlx_safetensor.read_bytes()
    replacement = tmp_path / "replacement.safetensors"
    replacement.write_bytes(original[:-1] + bytes([original[-1] ^ 0xFF]))
    os.replace(replacement, mlx_safetensor)
    output = tmp_path / "replacement-rejected.tmp"

    with pytest.raises(ValueError, match="changed|identity"):
        module.write_shard(records, output)

    assert not output.exists()


def test_write_shard_rejects_source_mutation_after_header_read(
    mlx_safetensor: Path, tmp_path: Path
) -> None:
    module = safetensor_raw()
    records = module.read_header(mlx_safetensor).tensors
    with mlx_safetensor.open("r+b", buffering=0) as source:
        source.seek(records[0].start)
        byte = source.read(1)
        source.seek(records[0].start)
        source.write(bytes([byte[0] ^ 0xFF]))
        os.fsync(source.fileno())
    output = tmp_path / "mutation-rejected.tmp"

    with pytest.raises(ValueError, match="changed|identity"):
        module.write_shard(records, output)

    assert not output.exists()


def test_writer_rejects_path_replacement_during_held_snapshot_copy(
    mlx_safetensor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = safetensor_raw()
    records = module.read_header(mlx_safetensor).tensors
    replacement_bytes = bytearray(mlx_safetensor.read_bytes())
    for record in records:
        if record.nbytes:
            replacement_bytes[record.start] ^= 0xFF
    replacement = tmp_path / "new-source.safetensors"
    replacement.write_bytes(replacement_bytes)
    displaced = tmp_path / "old-source.safetensors"
    original_pread = os.pread
    replaced = False

    def replacing_pread(fd: int, count: int, offset: int) -> bytes:
        nonlocal replaced
        data = original_pread(fd, count, offset)
        if not replaced and offset >= records[0].start:
            replaced = True
            os.replace(mlx_safetensor, displaced)
            os.replace(replacement, mlx_safetensor)
        return data

    monkeypatch.setattr(os, "pread", replacing_pread)
    output = tmp_path / "stable-snapshot.tmp"
    with pytest.raises(ValueError, match="changed|identity"):
        module.write_shard(records, output)

    assert replaced
    assert not output.exists()


def test_writer_rejects_in_place_source_mutation_during_copy(
    mlx_safetensor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = safetensor_raw()
    records = module.read_header(mlx_safetensor).tensors
    original_pread = os.pread
    mutated = False

    def mutating_pread(fd: int, count: int, offset: int) -> bytes:
        nonlocal mutated
        data = original_pread(fd, count, offset)
        if not mutated and offset >= records[0].start:
            mutated = True
            with mlx_safetensor.open("r+b", buffering=0) as source:
                source.seek(records[-1].start)
                byte = source.read(1)
                source.seek(records[-1].start)
                source.write(bytes([byte[0] ^ 0xFF]))
                os.fsync(source.fileno())
        return data

    monkeypatch.setattr(os, "pread", mutating_pread)
    output = tmp_path / "mid-copy-mutation.tmp"

    with pytest.raises(ValueError, match="changed|identity"):
        module.write_shard(records, output)

    assert mutated
    assert not output.exists()


def test_writer_rechecks_source_identity_after_output_fsync(
    mlx_safetensor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = safetensor_raw()
    records = module.read_header(mlx_safetensor).tensors
    original_fsync = os.fsync
    mutated = False

    def mutate_source_during_fsync(fd: int) -> None:
        nonlocal mutated
        with mlx_safetensor.open("r+b", buffering=0) as source:
            source.seek(records[0].start)
            byte = source.read(1)
            source.seek(records[0].start)
            source.write(bytes([byte[0] ^ 0xFF]))
            original_fsync(source.fileno())
        mutated = True
        original_fsync(fd)

    monkeypatch.setattr(os, "fsync", mutate_source_during_fsync)
    output = tmp_path / "fsync-race.tmp"

    with pytest.raises(ValueError, match="changed|identity"):
        module.write_shard(records, output)

    assert mutated
    assert not output.exists()


def test_writer_groups_symlink_and_hardlink_sources_by_inode(tmp_path: Path) -> None:
    module = safetensor_raw()
    source = tmp_path / "source.bin"
    hardlink = tmp_path / "hardlink.bin"
    symlink = tmp_path / "symlink.bin"
    source.write_bytes(b"abc")
    os.link(source, hardlink)
    symlink.symlink_to(source)
    records = (
        module.TensorRecord("a", "U8", (1,), source, 0, 1, 1),
        module.TensorRecord("b", "U8", (1,), hardlink, 1, 2, 1),
        module.TensorRecord("c", "U8", (1,), symlink, 2, 3, 1),
    )
    output = tmp_path / "aliases.tmp"

    module.write_shard(records, output)

    rewritten = module.read_header(output)
    assert [_payload_sha(record) for record in rewritten.tensors] == [
        hashlib.sha256(value).hexdigest() for value in (b"a", b"b", b"c")
    ]


def test_writer_rejects_overlapping_ranges_through_hardlink_alias(
    tmp_path: Path,
) -> None:
    module = safetensor_raw()
    source = tmp_path / "source.bin"
    alias = tmp_path / "alias.bin"
    source.write_bytes(b"x")
    os.link(source, alias)
    records = (
        module.TensorRecord("a", "U8", (1,), source, 0, 1, 1),
        module.TensorRecord("b", "U8", (1,), alias, 0, 1, 1),
    )
    output = tmp_path / "overlapping-alias.tmp"

    with pytest.raises(ValueError, match="overlap"):
        module.write_shard(records, output)

    assert not output.exists()


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


def test_writer_rejects_padded_header_over_safety_limit_before_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = safetensor_raw()
    source = tmp_path / "source.bin"
    source.write_bytes(b"x")
    record = module.TensorRecord("long-name", "U8", (1,), source, 0, 1, 1)
    output = tmp_path / "oversized-header.tmp"
    monkeypatch.setattr(module, "_MAX_HEADER_BYTES", 16)

    with pytest.raises(ValueError, match="header"):
        module.write_shard((record,), output)

    assert not output.exists()


def test_writer_fsync_failure_preserves_error_and_removes_its_partial_file(
    mlx_safetensor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = safetensor_raw()
    records = module.read_header(mlx_safetensor).tensors
    output = tmp_path / "fsync-failure.tmp"
    failure = RuntimeError("fsync failed")

    def fail_fsync(fd: int) -> None:
        raise failure

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(RuntimeError) as raised:
        module.write_shard(records, output)

    assert raised.value is failure
    assert not output.exists()


def test_writer_cleanup_failure_never_masks_original_error(
    mlx_safetensor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = safetensor_raw()
    records = module.read_header(mlx_safetensor).tensors
    output = tmp_path / "unlink-failure.tmp"
    failure = RuntimeError("fsync failed")
    original_unlink = os.unlink

    def fail_fsync(fd: int) -> None:
        raise failure

    def fail_unlink(path, *args, **kwargs) -> None:
        if Path(path) == output:
            raise PermissionError("unlink denied")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "fsync", fail_fsync)
    monkeypatch.setattr(os, "unlink", fail_unlink)
    with pytest.raises(RuntimeError) as raised:
        module.write_shard(records, output)

    assert raised.value is failure
    assert any("unlink denied" in note for note in getattr(raised.value, "__notes__", ()))
    assert output.exists()


def test_writer_cleanup_does_not_unlink_pathname_replacement(
    mlx_safetensor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = safetensor_raw()
    records = module.read_header(mlx_safetensor).tensors
    output = tmp_path / "replaced-output.tmp"
    displaced = tmp_path / "writer-created-partial.tmp"
    failure = RuntimeError("fsync failed after replacement")

    def replace_then_fail(fd: int) -> None:
        os.replace(output, displaced)
        output.write_bytes(b"caller replacement")
        raise failure

    monkeypatch.setattr(os, "fsync", replace_then_fail)
    with pytest.raises(RuntimeError) as raised:
        module.write_shard(records, output)

    assert raised.value is failure
    assert output.read_bytes() == b"caller replacement"
    assert any("not removed" in note for note in getattr(raised.value, "__notes__", ()))


def test_writer_zero_write_closes_all_fds_and_removes_partial_output(
    mlx_safetensor: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = safetensor_raw()
    records = module.read_header(mlx_safetensor).tensors
    output = tmp_path / "zero-write.tmp"
    original_open = os.open
    opened: list[int] = []

    def tracking_open(*args, **kwargs) -> int:
        fd = original_open(*args, **kwargs)
        opened.append(fd)
        return fd

    monkeypatch.setattr(os, "open", tracking_open)
    monkeypatch.setattr(os, "write", lambda fd, data: 0)
    with pytest.raises(OSError, match="short write"):
        module.write_shard(records, output)

    assert not output.exists()
    assert opened
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_production_module_has_no_tensor_framework_dependencies() -> None:
    module = safetensor_raw()
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))

    def dotted_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = dotted_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return None

    forbidden_roots = {"mlx", "mlx_lm", "numpy"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(name.name.split(".", 1)[0] not in forbidden_roots for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".", 1)[0] not in forbidden_roots
        elif isinstance(node, ast.Call):
            name = dotted_name(node.func) or ""
            assert name not in {"mx.load", "mx.save_safetensors"}
            assert not any(part.startswith("dequant") for part in name.split("."))
