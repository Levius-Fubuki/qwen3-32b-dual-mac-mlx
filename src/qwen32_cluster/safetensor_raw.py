from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MAX_COPY_CHUNK_BYTES = 8 << 20
_MAX_HEADER_BYTES = 100_000_000
_MAX_U64 = (1 << 64) - 1
_TENSOR_FIELDS = frozenset({"dtype", "shape", "data_offsets"})
_DTYPE_BITS = {
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E5M2": 8,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "C64": 64,
    "F64": 64,
    "I64": 64,
    "U64": 64,
}


@dataclass(frozen=True)
class _SourceIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class TensorRecord:
    name: str
    dtype: str
    shape: tuple[int, ...]
    source_file: Path
    start: int
    end: int
    nbytes: int
    _source_identity: _SourceIdentity | None = field(
        default=None, repr=False, compare=False
    )


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


@dataclass
class _SourceHandle:
    fd: int
    identity: _SourceIdentity


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must not be negative")
    return value


def _validate_unicode_scalar(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{label} contains a lone UTF-16 surrogate")
    return value


def _validate_name(name: Any) -> str:
    value = _validate_unicode_scalar(name, "tensor name")
    if value == "__metadata__":
        raise ValueError("tensor name must be a JSON string other than __metadata__")
    return value


def _tensor_nbytes(
    dtype: Any, shape: Any, label: str
) -> tuple[str, tuple[int, ...], int]:
    dtype = _validate_unicode_scalar(dtype, f"{label} dtype")
    if dtype not in _DTYPE_BITS:
        raise ValueError(f"{label} dtype is not supported")
    if not isinstance(shape, list):
        raise ValueError(f"{label} shape must be a JSON list")
    dimensions: list[int] = []
    elements = 1
    for index, dimension in enumerate(shape):
        value = _require_int(dimension, f"{label} shape[{index}]", minimum=0)
        if value > _MAX_U64:
            raise ValueError(
                f"{label} shape[{index}] dimension exceeds unsigned 64-bit range"
            )
        dimensions.append(value)
        if value and elements > _MAX_U64 // value:
            raise ValueError(f"{label} shape product overflow")
        elements *= value
    bits = _DTYPE_BITS[dtype]
    if elements and elements > _MAX_U64 // bits:
        raise ValueError(f"{label} bit size overflow")
    total_bits = elements * bits
    if total_bits % 8:
        raise ValueError(f"{label} size does not end on a byte boundary")
    return dtype, tuple(dimensions), total_bits // 8


def _identity_from_stat(source_stat: os.stat_result) -> _SourceIdentity:
    return _SourceIdentity(
        device=source_stat.st_dev,
        inode=source_stat.st_ino,
        size=source_stat.st_size,
        mtime_ns=source_stat.st_mtime_ns,
        ctime_ns=source_stat.st_ctime_ns,
    )


def _require_stable_identity(
    expected: _SourceIdentity, actual_stat: os.stat_result, label: str
) -> None:
    if _identity_from_stat(actual_stat) != expected:
        raise ValueError(f"{label} source identity changed")


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
        if record.start > previous_end:
            raise ValueError(f"tensor {record.name!r} leaves a hole in the payload")
        previous_end = record.end
    if previous_end != file_size:
        raise ValueError("safetensor payload has unindexed trailing bytes")


def read_header(path: Path) -> SourceShard:
    source_path = Path(path)
    fd = os.open(source_path, os.O_RDONLY)
    try:
        initial_stat = os.fstat(fd)
        if not stat.S_ISREG(initial_stat.st_mode):
            raise ValueError("safetensor source is not a regular file")
        identity = _identity_from_stat(initial_stat)
        file_size = identity.size
        if file_size < 8:
            raise ValueError("safetensor file is too short for its header length")
        prefix = _pread_exact(fd, 8, 0, "safetensor header length")
        (header_length,) = struct.unpack("<Q", prefix)
        if header_length > _MAX_HEADER_BYTES:
            raise ValueError("safetensor header length exceeds the safety limit")
        data_start = 8 + header_length
        if data_start > file_size:
            raise ValueError("safetensor header length extends beyond EOF")
        header_bytes = _pread_exact(fd, header_length, 8, "safetensor header")
        shard = _parse_header(
            header_bytes,
            source_path,
            header_length,
            data_start,
            file_size,
            identity,
        )
        _require_stable_identity(identity, os.fstat(fd), "safetensor header")
        return shard
    finally:
        os.close(fd)


def _parse_header(
    header_bytes: bytes,
    source_path: Path,
    header_length: int,
    data_start: int,
    file_size: int,
    identity: _SourceIdentity,
) -> SourceShard:
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

    if "__metadata__" in header:
        metadata = header["__metadata__"]
        if not isinstance(metadata, Mapping):
            raise ValueError("safetensor metadata must map strings to strings")
        for key, value in metadata.items():
            _validate_unicode_scalar(key, "metadata key")
            _validate_unicode_scalar(value, "metadata value")

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
                _source_identity=identity,
            )
        )

    _validate_offsets(records, file_size, data_start)
    return SourceShard(source_path, header_length, data_start, tuple(records))


def _validate_record_structure(record: TensorRecord) -> None:
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


def _validate_record_source(record: TensorRecord, identity: _SourceIdentity) -> None:
    if record._source_identity is not None and record._source_identity != identity:
        raise ValueError(f"tensor {record.name!r} source identity changed")
    if record.end > identity.size:
        raise ValueError(f"tensor {record.name!r} source range extends beyond EOF")


def _validate_record_structures(
    records: Sequence[TensorRecord],
) -> tuple[TensorRecord, ...]:
    values = tuple(records)
    names: set[str] = set()
    for record in values:
        _validate_record_structure(record)
        if record.name in names:
            raise ValueError(f"duplicate tensor name: {record.name}")
        names.add(record.name)
    return values


def _open_source_handles(
    records: Sequence[TensorRecord],
) -> tuple[dict[Path, _SourceHandle], dict[tuple[int, int], _SourceHandle]]:
    by_path: dict[Path, _SourceHandle] = {}
    by_inode: dict[tuple[int, int], _SourceHandle] = {}
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        for source_path in dict.fromkeys(record.source_file for record in records):
            source_fd = os.open(source_path, flags)
            try:
                source_stat = os.fstat(source_fd)
                if not stat.S_ISREG(source_stat.st_mode):
                    raise ValueError(f"source {source_path} is not a regular file")
                identity = _identity_from_stat(source_stat)
                inode_key = (identity.device, identity.inode)
                handle = by_inode.get(inode_key)
                if handle is None:
                    handle = _SourceHandle(source_fd, identity)
                    by_inode[inode_key] = handle
                    source_fd = -1
                elif handle.identity != identity:
                    raise ValueError(f"source {source_path} changed while aliases were opened")
                by_path[source_path] = handle
            finally:
                if source_fd >= 0:
                    os.close(source_fd)

        ranges_by_inode: dict[tuple[int, int], list[TensorRecord]] = {}
        for record in records:
            handle = by_path[record.source_file]
            _validate_record_source(record, handle.identity)
            inode_key = (handle.identity.device, handle.identity.inode)
            ranges_by_inode.setdefault(inode_key, []).append(record)
        for source_records in ranges_by_inode.values():
            ordered = sorted(
                source_records, key=lambda item: (item.start, item.end, item.name)
            )
            previous_end = 0
            for record in ordered:
                if record.start < previous_end:
                    raise ValueError(
                        f"tensor {record.name!r} source range overlaps another tensor"
                    )
                previous_end = max(previous_end, record.end)
        return by_path, by_inode
    except BaseException:
        for handle in by_inode.values():
            try:
                os.close(handle.fd)
            except OSError:
                pass
        raise


def _validate_open_sources(handles: Mapping[tuple[int, int], _SourceHandle]) -> None:
    for handle in handles.values():
        _require_stable_identity(
            handle.identity, os.fstat(handle.fd), "safetensor payload"
        )


def _close_source_handles(
    handles: Mapping[tuple[int, int], _SourceHandle], error: BaseException | None = None
) -> None:
    first_error: OSError | None = None
    for handle in handles.values():
        try:
            os.close(handle.fd)
        except OSError as exc:
            if error is not None:
                error.add_note(f"failed to close source fd {handle.fd}: {exc}")
            elif first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


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


def _copy_payload_from_fd(
    record: TensorRecord,
    source_fd: int,
    dst_fd: int,
    chunk_bytes: int,
    shard_hasher: Any | None = None,
) -> str:
    payload_hasher = hashlib.sha256()
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
    return payload_hasher.hexdigest()


def copy_payload(
    record: TensorRecord,
    dst_fd: int,
    chunk_bytes: int = MAX_COPY_CHUNK_BYTES,
) -> str:
    if (
        type(chunk_bytes) is not int
        or chunk_bytes <= 0
        or chunk_bytes > MAX_COPY_CHUNK_BYTES
    ):
        raise ValueError(
            f"chunk_bytes must be an integer from 1 through {MAX_COPY_CHUNK_BYTES}"
        )
    if type(dst_fd) is not int or dst_fd < 0:
        raise ValueError("dst_fd must be a non-negative integer file descriptor")
    _validate_record_structure(record)
    source_fd = os.open(record.source_file, os.O_RDONLY)
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError(f"tensor {record.name!r} source is not a regular file")
        identity = _identity_from_stat(source_stat)
        _validate_record_source(record, identity)
        destination_stat = os.fstat(dst_fd)
        if stat.S_ISREG(destination_stat.st_mode) and (
            destination_stat.st_dev,
            destination_stat.st_ino,
        ) == (identity.device, identity.inode):
            raise ValueError("source and destination refer to the same file")
        digest = _copy_payload_from_fd(record, source_fd, dst_fd, chunk_bytes)
        _require_stable_identity(identity, os.fstat(source_fd), "tensor payload")
        return digest
    finally:
        os.close(source_fd)


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
    if len(header_bytes) > _MAX_HEADER_BYTES:
        raise ValueError("padded safetensor header exceeds the safety limit")
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


def _close_fd_with_note(fd: int, label: str, error: BaseException) -> None:
    try:
        os.close(fd)
    except OSError as close_error:
        error.add_note(f"failed to close {label} fd {fd}: {close_error}")


def _cleanup_created_output(
    output_path: Path, created_identity: _SourceIdentity, error: BaseException
) -> None:
    try:
        current_stat = os.lstat(output_path)
    except FileNotFoundError:
        error.add_note("partial output not removed because its pathname no longer exists")
        return
    except OSError as stat_error:
        error.add_note(f"could not inspect partial output before cleanup: {stat_error}")
        return
    if (current_stat.st_dev, current_stat.st_ino) != (
        created_identity.device,
        created_identity.inode,
    ):
        error.add_note(
            "partial output not removed because the pathname no longer identifies "
            "the writer-created inode"
        )
        return
    try:
        os.unlink(output_path)
    except OSError as unlink_error:
        error.add_note(f"failed to remove writer-created partial output: {unlink_error}")


def write_shard(records: Sequence[TensorRecord], output_tmp: Path) -> ShardResult:
    output_path = Path(output_tmp)
    if output_path.suffix != ".tmp":
        raise ValueError("output path must have an explicit .tmp suffix")

    validated = _validate_record_structures(records)
    ordered = tuple(sorted(validated, key=lambda record: record.name))
    header_bytes, data_start, output_records = _build_output_layout(ordered, output_path)
    prefix = struct.pack("<Q", len(header_bytes))
    file_hasher = hashlib.sha256()
    payload_hashes: list[tuple[str, str]] = []
    output_fd: int | None = None
    created_identity: _SourceIdentity | None = None
    sources_by_path: dict[Path, _SourceHandle] = {}
    sources_by_inode: dict[tuple[int, int], _SourceHandle] = {}
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC

    try:
        sources_by_path, sources_by_inode = _open_source_handles(ordered)
        output_fd = os.open(output_path, flags, 0o600)
        output_stat = os.fstat(output_fd)
        created_identity = _identity_from_stat(output_stat)
        if (output_stat.st_dev, output_stat.st_ino) in sources_by_inode:
            raise ValueError("output and source refer to the same file")
        _write_all(output_fd, prefix)
        file_hasher.update(prefix)
        _write_all(output_fd, header_bytes)
        file_hasher.update(header_bytes)
        for record in ordered:
            digest = _copy_payload_from_fd(
                record,
                sources_by_path[record.source_file].fd,
                output_fd,
                MAX_COPY_CHUNK_BYTES,
                file_hasher,
            )
            payload_hashes.append((record.name, digest))
        os.fsync(output_fd)
        _validate_open_sources(sources_by_inode)
        os.close(output_fd)
        output_fd = None
        _close_source_handles(sources_by_inode)
        sources_by_inode = {}
    except BaseException as error:
        if output_fd is not None:
            _close_fd_with_note(output_fd, "output", error)
        _close_source_handles(sources_by_inode, error)
        if created_identity is not None:
            _cleanup_created_output(output_path, created_identity, error)
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
