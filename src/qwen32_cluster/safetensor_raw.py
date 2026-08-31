from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_DEFAULT_CHUNK_BYTES = 8 << 20
_MAX_HEADER_BYTES = 100_000_000
_TENSOR_FIELDS = frozenset({"dtype", "shape", "data_offsets"})
_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2": 1,
    "F8_E5M2FNUZ": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "C64": 8,
    "C128": 16,
}


@dataclass(frozen=True)
class TensorRecord:
    name: str
    dtype: str
    shape: tuple[int, ...]
    source_file: Path
    start: int
    end: int
    nbytes: int


@dataclass(frozen=True)
class SourceShard:
    path: Path
    header_length: int
    data_start: int
    tensors: tuple[TensorRecord, ...]


@dataclass(frozen=True)
class ShardResult:
    path: Path
    header_length: int
    data_start: int
    file_size: int
    sha256: str
    tensors: tuple[TensorRecord, ...]
    payload_sha256: tuple[tuple[str, str], ...]


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must not be negative")
    return value


def _validate_name(name: Any) -> str:
    if not isinstance(name, str) or not name or name == "__metadata__":
        raise ValueError("tensor name must be a non-empty, non-reserved string")
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise ValueError(f"tensor name contains an unsafe control character: {name!r}")
    return name


def _tensor_nbytes(
    dtype: Any, shape: Any, label: str
) -> tuple[str, tuple[int, ...], int]:
    if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
        raise ValueError(f"{label} dtype is not supported")
    if not isinstance(shape, list):
        raise ValueError(f"{label} shape must be a JSON list")
    dimensions: list[int] = []
    elements = 1
    for index, dimension in enumerate(shape):
        value = _require_int(dimension, f"{label} shape[{index}]", minimum=0)
        dimensions.append(value)
        elements *= value
    return dtype, tuple(dimensions), elements * _DTYPE_BYTES[dtype]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _pread_exact(fd: int, count: int, offset: int, label: str) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    position = offset
    while remaining:
        try:
            chunk = os.pread(fd, remaining, position)
        except InterruptedError:
            continue
        if not chunk:
            raise OSError(f"truncated {label}")
        if len(chunk) > remaining:
            raise OSError(f"invalid oversized read while reading {label}")
        chunks.append(chunk)
        position += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _validate_offsets(
    records: Sequence[TensorRecord], file_size: int, data_start: int
) -> None:
    previous_end = data_start
    for record in sorted(records, key=lambda item: (item.start, item.end, item.name)):
        if record.start < data_start or record.end > file_size:
            raise ValueError(f"tensor {record.name!r} offsets are out of range")
        if record.start > record.end:
            raise ValueError(f"tensor {record.name!r} offset order is invalid")
        if record.start < previous_end:
            raise ValueError(f"tensor {record.name!r} payload overlaps another tensor")
        previous_end = max(previous_end, record.end)


def read_header(path: Path) -> SourceShard:
    source_path = Path(path)
    file_size = source_path.stat().st_size
    if file_size < 8:
        raise ValueError("safetensor file is too short for its header length")

    fd = os.open(source_path, os.O_RDONLY)
    try:
        prefix = _pread_exact(fd, 8, 0, "safetensor header length")
        (header_length,) = struct.unpack("<Q", prefix)
        if header_length > _MAX_HEADER_BYTES:
            raise ValueError("safetensor header length exceeds the safety limit")
        data_start = 8 + header_length
        if data_start > file_size:
            raise ValueError("safetensor header length extends beyond EOF")
        header_bytes = _pread_exact(fd, header_length, 8, "safetensor header")
    finally:
        os.close(fd)

    try:
        header_text = header_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("safetensor header is not valid UTF-8 JSON") from exc
    try:
        header = json.loads(header_text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid safetensor header JSON: {exc.msg}") from exc
    if not isinstance(header, Mapping):
        raise ValueError("safetensor header root must be a JSON object")

    metadata = header.get("__metadata__")
    if metadata is not None:
        if not isinstance(metadata, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        ):
            raise ValueError("safetensor metadata must map strings to strings")

    records: list[TensorRecord] = []
    for raw_name, value in header.items():
        if raw_name == "__metadata__":
            continue
        name = _validate_name(raw_name)
        if not isinstance(value, Mapping):
            raise ValueError(f"tensor {name!r} must be a JSON object")
        if set(value) != _TENSOR_FIELDS:
            raise ValueError(f"tensor {name!r} has invalid fields")
        dtype, shape, expected_nbytes = _tensor_nbytes(
            value["dtype"], value["shape"], f"tensor {name!r}"
        )
        offsets = value["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"tensor {name!r} offsets must be a two-item JSON list")
        relative_start = _require_int(
            offsets[0], f"tensor {name!r} start offset", minimum=0
        )
        relative_end = _require_int(
            offsets[1], f"tensor {name!r} end offset", minimum=0
        )
        if relative_start > relative_end:
            raise ValueError(f"tensor {name!r} offset order is invalid")
        actual_nbytes = relative_end - relative_start
        if actual_nbytes != expected_nbytes:
            raise ValueError(
                f"tensor {name!r} nbytes {actual_nbytes} does not match dtype and shape ({expected_nbytes})"
            )
        records.append(
            TensorRecord(
                name=name,
                dtype=dtype,
                shape=shape,
                source_file=source_path,
                start=data_start + relative_start,
                end=data_start + relative_end,
                nbytes=actual_nbytes,
            )
        )

    _validate_offsets(records, file_size, data_start)
    return SourceShard(source_path, header_length, data_start, tuple(records))


def _validate_record(record: TensorRecord) -> None:
    if not isinstance(record, TensorRecord):
        raise ValueError("records must contain TensorRecord values")
    _validate_name(record.name)
    if not isinstance(record.source_file, Path):
        raise ValueError(f"tensor {record.name!r} source_file must be a Path")
    dtype, shape, expected_nbytes = _tensor_nbytes(
        record.dtype,
        list(record.shape) if isinstance(record.shape, tuple) else record.shape,
        f"tensor {record.name!r}",
    )
    if dtype != record.dtype or shape != record.shape:
        raise ValueError(f"tensor {record.name!r} has invalid dtype or shape")
    start = _require_int(record.start, f"tensor {record.name!r} start", minimum=0)
    end = _require_int(record.end, f"tensor {record.name!r} end", minimum=0)
    nbytes = _require_int(record.nbytes, f"tensor {record.name!r} nbytes", minimum=0)
    if start > end:
        raise ValueError(f"tensor {record.name!r} offset order is invalid")
    if end - start != nbytes or nbytes != expected_nbytes:
        raise ValueError(f"tensor {record.name!r} nbytes is inconsistent with offsets, dtype, or shape")
    source_stat = record.source_file.stat()
    if not stat.S_ISREG(source_stat.st_mode):
        raise ValueError(f"tensor {record.name!r} source is not a regular file")
    if end > source_stat.st_size:
        raise ValueError(f"tensor {record.name!r} source range extends beyond EOF")


def _validate_records(records: Sequence[TensorRecord]) -> tuple[TensorRecord, ...]:
    values = tuple(records)
    names: set[str] = set()
    by_source: dict[Path, list[TensorRecord]] = {}
    for record in values:
        _validate_record(record)
        if record.name in names:
            raise ValueError(f"duplicate tensor name: {record.name}")
        names.add(record.name)
        by_source.setdefault(record.source_file, []).append(record)
    for source_records in by_source.values():
        ordered = sorted(source_records, key=lambda item: (item.start, item.end, item.name))
        previous_end = 0
        for record in ordered:
            if record.start < previous_end:
                raise ValueError(f"tensor {record.name!r} source range overlaps another tensor")
            previous_end = max(previous_end, record.end)
    return values


def _write_all(fd: int, data: bytes | memoryview) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        try:
            count = os.write(fd, view[written:])
        except InterruptedError:
            continue
        if count <= 0:
            raise OSError("short write while writing safetensor shard")
        if count > len(view) - written:
            raise OSError("invalid oversized write while writing safetensor shard")
        written += count


def _copy_payload_validated(
    record: TensorRecord,
    dst_fd: int,
    chunk_bytes: int,
    shard_hasher: Any | None = None,
) -> str:
    payload_hasher = hashlib.sha256()
    source_fd = os.open(record.source_file, os.O_RDONLY)
    try:
        if record.end > os.fstat(source_fd).st_size:
            raise OSError(f"truncated payload source for tensor {record.name!r}")
        offset = record.start
        remaining = record.nbytes
        while remaining:
            request = min(chunk_bytes, remaining)
            try:
                chunk = os.pread(source_fd, request, offset)
            except InterruptedError:
                continue
            if not chunk:
                raise OSError(f"truncated payload read for tensor {record.name!r}")
            if len(chunk) > request:
                raise OSError(f"oversized payload read for tensor {record.name!r}")
            _write_all(dst_fd, chunk)
            payload_hasher.update(chunk)
            if shard_hasher is not None:
                shard_hasher.update(chunk)
            offset += len(chunk)
            remaining -= len(chunk)
    finally:
        os.close(source_fd)
    return payload_hasher.hexdigest()


def copy_payload(
    record: TensorRecord,
    dst_fd: int,
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
) -> str:
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    if type(dst_fd) is not int or dst_fd < 0:
        raise ValueError("dst_fd must be a non-negative integer file descriptor")
    _validate_record(record)
    return _copy_payload_validated(record, dst_fd, chunk_bytes)


def _build_output_layout(
    records: Sequence[TensorRecord], output_path: Path
) -> tuple[bytes, int, tuple[TensorRecord, ...]]:
    header: dict[str, dict[str, Any]] = {}
    relative_offset = 0
    for record in records:
        relative_end = relative_offset + record.nbytes
        header[record.name] = {
            "data_offsets": [relative_offset, relative_end],
            "dtype": record.dtype,
            "shape": list(record.shape),
        }
        relative_offset = relative_end
    compact = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    header_bytes = compact + (b" " * (-len(compact) % 8))
    data_start = 8 + len(header_bytes)

    output_records: list[TensorRecord] = []
    relative_offset = 0
    for record in records:
        start = data_start + relative_offset
        end = start + record.nbytes
        output_records.append(
            TensorRecord(
                record.name,
                record.dtype,
                record.shape,
                output_path,
                start,
                end,
                record.nbytes,
            )
        )
        relative_offset += record.nbytes
    return header_bytes, data_start, tuple(output_records)


def write_shard(records: Sequence[TensorRecord], output_tmp: Path) -> ShardResult:
    output_path = Path(output_tmp)
    if output_path.suffix != ".tmp":
        raise ValueError("output path must have an explicit .tmp suffix")

    validated = _validate_records(records)
    ordered = tuple(sorted(validated, key=lambda record: record.name))
    header_bytes, data_start, output_records = _build_output_layout(ordered, output_path)
    prefix = struct.pack("<Q", len(header_bytes))
    file_hasher = hashlib.sha256()
    payload_hashes: list[tuple[str, str]] = []
    output_fd: int | None = None
    created = False
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC

    try:
        output_fd = os.open(output_path, flags, 0o600)
        created = True
        _write_all(output_fd, prefix)
        file_hasher.update(prefix)
        _write_all(output_fd, header_bytes)
        file_hasher.update(header_bytes)
        for record in ordered:
            digest = _copy_payload_validated(
                record,
                output_fd,
                _DEFAULT_CHUNK_BYTES,
                file_hasher,
            )
            payload_hashes.append((record.name, digest))
        os.fsync(output_fd)
        os.close(output_fd)
        output_fd = None
    except BaseException:
        if output_fd is not None:
            try:
                os.close(output_fd)
            except OSError:
                pass
        if created:
            try:
                output_path.unlink()
            except FileNotFoundError:
                pass
        raise

    file_size = data_start + sum(record.nbytes for record in ordered)
    return ShardResult(
        path=output_path,
        header_length=len(header_bytes),
        data_start=data_start,
        file_size=file_size,
        sha256=file_hasher.hexdigest(),
        tensors=output_records,
        payload_sha256=tuple(payload_hashes),
    )
