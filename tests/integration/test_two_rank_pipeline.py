from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import time
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest


ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "tests" / "integration" / "two_rank_pipeline_case.py"
LAUNCHER = Path("/Users/Shared/mlx-cluster/.venv/bin/mlx.launch")
PYTHON = Path("/Users/Shared/mlx-cluster/.venv/bin/python")
RING_PORTS = (33323, 33324)
LAUNCH_TIMEOUT_SECONDS = 90
TERM_GRACE_SECONDS = 2
PROCESS_CLEANUP_RESERVE_SECONDS = 4
LOCK_POLL_SECONDS = 0.01
MAX_DIAGNOSTIC_CHARS = 4_000
RING_LOCK_PATH = Path("/tmp/qwen3-two-rank-ring-33323.lock")
_ACTIVE_DEADLINE: _Deadline | None = None


class RingLaunchError(AssertionError):
    pass


class RingResultError(AssertionError):
    pass


class _Deadline:
    def __init__(self, seconds: float) -> None:
        if type(seconds) not in (int, float) or seconds <= 0:
            raise ValueError("deadline seconds must be positive")
        self._expires = time.monotonic() + float(seconds)

    def remaining(self, *, reserve: float = 0.0) -> float:
        return max(0.0, self._expires - time.monotonic() - reserve)

    def require(self, operation: str, *, reserve: float = 0.0) -> float:
        remaining = self.remaining(reserve=reserve)
        if remaining <= 0:
            raise RingLaunchError(
                f"overall {LAUNCH_TIMEOUT_SECONDS}s local-integration budget "
                f"exhausted before {operation}"
            )
        return remaining


@pytest.fixture(scope="module", autouse=True)
def _overall_module_deadline() -> Iterator[None]:
    global _ACTIVE_DEADLINE
    _ACTIVE_DEADLINE = _Deadline(LAUNCH_TIMEOUT_SECONDS)
    try:
        yield
    finally:
        _ACTIVE_DEADLINE = None


def _module_deadline() -> _Deadline:
    if _ACTIVE_DEADLINE is None:
        raise RuntimeError("module deadline fixture is not active")
    return _ACTIVE_DEADLINE


@contextmanager
def _ring_launch_lock(deadline: _Deadline, lock_path: Path) -> Iterator[None]:
    lock_file = lock_path.open("a+", encoding="utf-8")
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                remaining = deadline.remaining(
                    reserve=PROCESS_CLEANUP_RESERVE_SECONDS
                )
                if remaining <= 0:
                    raise RingLaunchError(
                        "Ring launch lock acquisition exceeded the overall budget"
                    )
                time.sleep(min(LOCK_POLL_SECONDS, remaining))
        yield
    finally:
        if acquired:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _assert_ring_ports_free() -> None:
    occupied = [port for port in RING_PORTS if not _port_is_free(port)]
    if occupied:
        raise RingLaunchError(
            f"configured MLX Ring port(s) occupied: {occupied}; "
            "refusing to launch and never falling back to 32323-32324"
        )


def _wait_for_ring_ports_to_clear(deadline: _Deadline) -> None:
    if all(_port_is_free(port) for port in RING_PORTS):
        return
    while deadline.remaining() > 0:
        if all(_port_is_free(port) for port in RING_PORTS):
            return
        time.sleep(min(0.05, deadline.remaining()))
    occupied = [port for port in RING_PORTS if not _port_is_free(port)]
    if occupied:
        raise RingLaunchError(f"Ring launch left configured port(s) occupied: {occupied}")


def _stop_process_group(
    process: subprocess.Popen[str], deadline: _Deadline
) -> tuple[str, str]:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    term_timeout = min(TERM_GRACE_SECONDS, deadline.remaining())
    try:
        return process.communicate(timeout=term_timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        kill_timeout = min(TERM_GRACE_SECONDS, deadline.remaining())
        return process.communicate(timeout=kill_timeout)


def _launcher_command(case: str, output_dir: Path) -> list[str]:
    return [
        str(LAUNCHER),
        "--backend",
        "ring",
        "--repeat-hosts",
        "2",
        "--connections-per-ip",
        "1",
        "--starting-port",
        "33323",
        "--",
        str(PYTHON),
        str(WORKER),
        "--case",
        case,
        "--output-dir",
        str(output_dir),
    ]


def _attach_timeout_output(
    error: BaseException,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
) -> None:
    for label, output in (("stdout", stdout), ("stderr", stderr)):
        if isinstance(output, bytes):
            text = output.decode(errors="replace")
        else:
            text = "" if output is None else str(output)
        if len(text) > MAX_DIAGNOSTIC_CHARS:
            text = text[:MAX_DIAGNOSTIC_CHARS] + "\n...[truncated]"
        error.add_note(f"launcher {label} captured during timeout cleanup:\n{text}")


ERROR_KEYS = {
    "case",
    "rank",
    "world_size",
    "status",
    "exit_code",
    "error",
    "events",
}
SUCCESS_COMMON_KEYS = ERROR_KEYS | {"local_layers", "partition"}
FORWARD_KEYS = SUCCESS_COMMON_KEYS | {
    "input_shape",
    "logits_shape",
    "reference_logits_shape",
    "logits",
    "reference_logits",
    "checksum",
}
CACHE_KEYS = SUCCESS_COMMON_KEYS | {
    "prefill_input_shapes",
    "decode_input_shape",
    "decode_logits_shape",
    "reference_decode_logits_shape",
    "decode_logits",
    "reference_decode_logits",
    "checksum",
    "prefill_caches",
    "caches",
}


def _require_exact_keys(value: dict[str, Any], expected: set[str], path: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise RingResultError(f"{path} keys differ: missing={missing}, extra={extra}")


def _require_int(value: Any, path: str) -> int:
    if type(value) is not int:
        raise RingResultError(f"{path} must be an exact integer")
    return value


def _require_number(value: Any, path: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise RingResultError(f"{path} must be a finite non-boolean number")
    return float(value)


def _require_shape(value: Any, path: str, dimensions: int) -> list[int]:
    if type(value) is not list or len(value) != dimensions:
        raise RingResultError(f"{path} must contain exactly {dimensions} dimensions")
    shape = [_require_int(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if any(dimension <= 0 for dimension in shape):
        raise RingResultError(f"{path} dimensions must be positive")
    return shape


def _numeric_tensor_shape(value: Any, path: str) -> list[int]:
    if type(value) is list:
        if not value:
            raise RingResultError(f"{path} cannot contain an empty dimension")
        child_shapes = [
            _numeric_tensor_shape(child, f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
        if any(shape != child_shapes[0] for shape in child_shapes[1:]):
            raise RingResultError(f"{path} must be rectangular")
        return [len(value), *child_shapes[0]]
    _require_number(value, path)
    return []


def _validate_events(
    events: Any,
    rank: int,
    calls: list[tuple[str, str]],
    path: str,
) -> None:
    if type(events) is not list:
        raise RingResultError(f"{path} must be a list")
    operation = "recv" if rank == 0 else "send"
    peer = 1 if rank == 0 else 0
    expected = []
    for call, target in calls:
        expected.extend(
            [
                {"call": call, "event": operation, "peer": peer},
                {"call": call, "event": "all_gather"},
                {"call": call, "event": "eval_complete", "target": target},
            ]
        )
    if events != expected:
        raise RingResultError(
            f"{path} distributed event count/order differs: "
            f"expected {expected!r}, got {events!r}"
        )


def _validate_partial_events(events: Any, path: str) -> None:
    if type(events) is not list:
        raise RingResultError(f"{path} must be a list")
    for index, event in enumerate(events):
        event_path = f"{path}[{index}]"
        if type(event) is not dict:
            raise RingResultError(f"{event_path} must be an object")
        event_name = event.get("event")
        expected_keys = (
            {"call", "event", "peer"}
            if event_name in ("recv", "send")
            else {"call", "event"}
            if event_name == "all_gather"
            else {"call", "event", "target"}
            if event_name == "eval_complete"
            else set()
        )
        if not expected_keys:
            raise RingResultError(f"{event_path} has an unknown event")
        _require_exact_keys(event, expected_keys, event_path)
        if type(event["call"]) is not str or not event["call"]:
            raise RingResultError(f"{event_path}.call must be a non-empty string")
        if "peer" in event:
            peer = _require_int(event["peer"], f"{event_path}.peer")
            if peer not in (0, 1):
                raise RingResultError(f"{event_path}.peer must be rank 0 or 1")
        if "target" in event and (
            type(event["target"]) is not str or not event["target"]
        ):
            raise RingResultError(f"{event_path}.target must be a non-empty string")


def _validate_cache_record(
    record: Any,
    expected_layer: int,
    expected_offset: int,
    path: str,
) -> None:
    if type(record) is not dict:
        raise RingResultError(f"{path} must be an object")
    expected_keys = {
        "layer",
        "offset",
        "reference_offset",
        "keys_hash",
        "reference_keys_hash",
        "values_hash",
        "reference_values_hash",
    }
    _require_exact_keys(record, expected_keys, path)
    if _require_int(record["layer"], f"{path}.layer") != expected_layer:
        raise RingResultError(f"{path}.layer reports wrong ownership")
    for field in ("offset", "reference_offset"):
        if _require_int(record[field], f"{path}.{field}") != expected_offset:
            raise RingResultError(f"{path}.{field} must equal {expected_offset}")
    for field in (
        "keys_hash",
        "reference_keys_hash",
        "values_hash",
        "reference_values_hash",
    ):
        value = record[field]
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RingResultError(f"{path}.{field} must be a SHA-256 hex digest")


def _validate_rank_result(
    result: Any,
    expected_rank: int,
    expected_case: str,
    path: str,
) -> None:
    if type(result) is not dict:
        raise RingResultError(f"{path} must contain an object")
    status = result.get("status")
    expected_keys = (
        ERROR_KEYS
        if status != "ok"
        else CACHE_KEYS
        if expected_case == "cache_dependency"
        else FORWARD_KEYS
    )
    _require_exact_keys(result, expected_keys, path)
    if type(result["case"]) is not str or result["case"] != expected_case:
        raise RingResultError(f"{path}.case must equal {expected_case!r}")
    if _require_int(result["rank"], f"{path}.rank") != expected_rank:
        raise RingResultError(f"{path}.rank must equal {expected_rank}")
    if _require_int(result["world_size"], f"{path}.world_size") != 2:
        raise RingResultError(f"{path}.world_size must equal 2")
    if type(status) is not str or status not in ("ok", "error"):
        raise RingResultError(f"{path}.status is invalid")
    exit_code = _require_int(result["exit_code"], f"{path}.exit_code")
    _validate_partial_events(result["events"], f"{path}.events")
    if status == "error":
        if exit_code == 0 or type(result["error"]) is not str or not result["error"]:
            raise RingResultError(f"{path} has an invalid error result")
        raise RingResultError(
            f"rank {expected_rank} reported failure: "
            f"exit_code={exit_code}, error={result['error']!r}"
        )
    if exit_code != 0 or result["error"] is not None:
        raise RingResultError(f"{path} successful result has failure metadata")

    local_layers = result["local_layers"]
    expected_layer = 1 if expected_rank == 0 else 0
    if (
        type(local_layers) is not list
        or len(local_layers) != 1
        or _require_int(local_layers[0], f"{path}.local_layers[0]") != expected_layer
    ):
        raise RingResultError(f"{path}.local_layers reports wrong ownership")
    partition = result["partition"]
    if type(partition) is not dict:
        raise RingResultError(f"{path}.partition must be an object")
    _require_exact_keys(partition, {"start", "end"}, f"{path}.partition")
    if (
        _require_int(partition["start"], f"{path}.partition.start") != expected_layer
        or _require_int(partition["end"], f"{path}.partition.end")
        != expected_layer + 1
    ):
        raise RingResultError(f"{path}.partition reports wrong bounds")

    if expected_case == "cache_dependency":
        prefill_shapes = result["prefill_input_shapes"]
        if type(prefill_shapes) is not list or len(prefill_shapes) != 2:
            raise RingResultError(f"{path}.prefill_input_shapes is invalid")
        for index, shape in enumerate(prefill_shapes):
            if (
                _require_shape(
                    shape, f"{path}.prefill_input_shapes[{index}]", 2
                )
                != [2, 2]
            ):
                raise RingResultError(f"{path}.prefill_input_shapes is invalid")
        decode_input_shape = _require_shape(
            result["decode_input_shape"], f"{path}.decode_input_shape", 2
        )
        if decode_input_shape != [2, 1]:
            raise RingResultError(f"{path}.decode_input_shape must equal [2, 1]")
        expected_logits_shape = [*decode_input_shape, 32]
        for field in ("decode_logits_shape", "reference_decode_logits_shape"):
            if _require_shape(result[field], f"{path}.{field}", 3) != expected_logits_shape:
                raise RingResultError(f"{path}.{field} is invalid")
        for field in ("decode_logits", "reference_decode_logits"):
            if _numeric_tensor_shape(result[field], f"{path}.{field}") != expected_logits_shape:
                raise RingResultError(f"{path}.{field} nested shape is invalid")
        _validate_events(
            result["events"],
            expected_rank,
            [
                ("prefill_0", "cache_state"),
                ("prefill_1", "cache_state"),
                ("decode", "logits_and_cache_state"),
            ],
            f"{path}.events",
        )
        for field, offset in (("prefill_caches", 4), ("caches", 5)):
            records = result[field]
            if type(records) is not list or len(records) != 1:
                raise RingResultError(f"{path}.{field} must contain one cache")
            _validate_cache_record(
                records[0], expected_layer, offset, f"{path}.{field}[0]"
            )
    else:
        input_shape = _require_shape(result["input_shape"], f"{path}.input_shape", 2)
        if input_shape != [2, 4]:
            raise RingResultError(f"{path}.input_shape must equal [2, 4]")
        expected_logits_shape = [*input_shape, 32]
        for field in ("logits_shape", "reference_logits_shape"):
            if _require_shape(result[field], f"{path}.{field}", 3) != expected_logits_shape:
                raise RingResultError(f"{path}.{field} is invalid")
        for field in ("logits", "reference_logits"):
            if _numeric_tensor_shape(result[field], f"{path}.{field}") != expected_logits_shape:
                raise RingResultError(f"{path}.{field} nested shape is invalid")
        _validate_events(
            result["events"],
            expected_rank,
            [("forward", "logits")],
            f"{path}.events",
        )
    _require_number(result["checksum"], f"{path}.checksum")


def _load_rank_results(
    output_dir: Path, expected_case: str
) -> list[dict[str, Any]]:
    results = []
    for expected_rank in range(2):
        result_path = output_dir / f"rank-{expected_rank}.json"
        if not result_path.is_file():
            raise RingResultError(f"missing rank {expected_rank} JSON: {result_path}")
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RingResultError(
                f"malformed rank {expected_rank} JSON: {result_path}: {exc}"
            ) from exc
        _validate_rank_result(
            result, expected_rank, expected_case, f"rank {expected_rank} JSON"
        )
        results.append(result)
    return results


def _launch_case(
    case: str,
    output_dir: Path,
    *,
    lock_path: Path = RING_LOCK_PATH,
) -> list[dict[str, Any]]:
    if not WORKER.is_file():
        raise RingLaunchError(f"two-rank worker is missing: {WORKER}")
    deadline = _module_deadline()
    with _ring_launch_lock(deadline, lock_path):
        deadline.require(
            "launcher start", reserve=PROCESS_CLEANUP_RESERVE_SECONDS
        )
        _assert_ring_ports_free()
        output_dir.mkdir(parents=True, exist_ok=False)
        environment = os.environ.copy()
        source_path = str(ROOT / "src")
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_path
            if not existing_pythonpath
            else os.pathsep.join((source_path, existing_pythonpath))
        )
        command = _launcher_command(case, output_dir)
        process: subprocess.Popen[str] | None = None
        primary_error: BaseException | None = None
        cleanup_errors: list[BaseException] = []
        results: list[dict[str, Any]] | None = None
        process_cleanup_attempted = False
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(
                    timeout=deadline.require(
                        "launcher completion",
                        reserve=PROCESS_CLEANUP_RESERVE_SECONDS,
                    )
                )
            except subprocess.TimeoutExpired as exc:
                timeout_error = RingLaunchError(
                    f"two-rank case {case!r} exceeded the single overall "
                    f"{LAUNCH_TIMEOUT_SECONDS}s budget; exact launcher process group "
                    f"{process.pid} will be stopped"
                )
                process_cleanup_attempted = True
                try:
                    captured_stdout, captured_stderr = _stop_process_group(
                        process, deadline
                    )
                    _attach_timeout_output(
                        timeout_error, captured_stdout, captured_stderr
                    )
                except BaseException as cleanup_error:
                    timeout_error.add_note(
                        f"secondary cleanup failure: {cleanup_error}"
                    )
                raise timeout_error from exc
            if process.returncode != 0:
                raise RingLaunchError(
                    f"launcher exited nonzero ({process.returncode}) for case {case!r}\n"
                    f"stdout:\n{stdout}\nstderr:\n{stderr}"
                )
            try:
                results = _load_rank_results(output_dir, case)
            except RingResultError as exc:
                raise RingResultError(
                    f"{exc}\nlauncher stdout:\n{stdout}\nlauncher stderr:\n{stderr}"
                ) from exc
        except BaseException as exc:
            primary_error = exc

        if (
            process is not None
            and not process_cleanup_attempted
            and process.poll() is None
        ):
            try:
                _stop_process_group(process, deadline)
            except BaseException as exc:
                cleanup_errors.append(exc)
        try:
            _wait_for_ring_ports_to_clear(deadline)
        except BaseException as exc:
            cleanup_errors.append(exc)

        if cleanup_errors:
            if primary_error is None:
                primary_error = cleanup_errors.pop(0)
            for cleanup_error in cleanup_errors:
                primary_error.add_note(f"secondary cleanup failure: {cleanup_error}")
        if primary_error is not None:
            raise primary_error
        if results is None:
            raise AssertionError("launcher completed without results or an error")
        return results


def _flatten(values: Any) -> list[float]:
    if isinstance(values, list):
        flattened: list[float] = []
        for value in values:
            flattened.extend(_flatten(value))
        return flattened
    return [float(values)]


def _assert_close(actual: Any, expected: Any) -> None:
    actual_values = _flatten(actual)
    expected_values = _flatten(expected)
    assert len(actual_values) == len(expected_values)
    assert actual_values == pytest.approx(expected_values, rel=1e-5, abs=1e-5)


@pytest.mark.local_integration
def test_occupied_configured_ring_port_is_rejected_without_fallback() -> None:
    with _ring_launch_lock(_module_deadline(), RING_LOCK_PATH):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", RING_PORTS[0]))
            listener.listen()
            with pytest.raises(
                RingLaunchError,
                match="33323.*refusing to launch.*never falling back to 32323-32324",
            ):
                _assert_ring_ports_free()


@pytest.mark.local_integration
def test_two_rank_forward_matches_unpartitioned_reference(tmp_path: Path) -> None:
    results = _launch_case("forward", tmp_path / "qwen32-forward")

    assert [result["rank"] for result in results] == [0, 1]
    assert [result["local_layers"] for result in results] == [[1], [0]]
    assert all(result["input_shape"][0] > 1 for result in results)
    for result in results:
        _assert_close(result["logits"], result["reference_logits"])
        assert math.isfinite(result["checksum"])
    _assert_close(results[0]["logits"], results[1]["logits"])
    assert results[0]["checksum"] == pytest.approx(
        results[1]["checksum"], rel=1e-7, abs=1e-7
    )


@pytest.mark.local_integration
def test_discarded_prefill_logits_still_materialize_all_pipeline_caches(
    tmp_path: Path,
) -> None:
    results = _launch_case("cache_dependency", tmp_path / "qwen32-cache")

    cache_records = sorted(
        (record for result in results for record in result["caches"]),
        key=lambda record: record["layer"],
    )
    prefill_cache_records = sorted(
        (record for result in results for record in result["prefill_caches"]),
        key=lambda record: record["layer"],
    )
    assert [record["layer"] for record in prefill_cache_records] == [0, 1]
    assert all(record["offset"] == 4 for record in prefill_cache_records)
    for record in prefill_cache_records:
        assert record["offset"] == record["reference_offset"]
        assert record["keys_hash"] == record["reference_keys_hash"]
        assert record["values_hash"] == record["reference_values_hash"]
    assert [record["layer"] for record in cache_records] == [0, 1]
    assert all(record["offset"] == 5 for record in cache_records)
    for record in cache_records:
        assert record["offset"] == record["reference_offset"]
        assert record["keys_hash"] == record["reference_keys_hash"]
        assert record["values_hash"] == record["reference_values_hash"]
    for result in results:
        _assert_close(result["decode_logits"], result["reference_decode_logits"])
        assert math.isfinite(result["checksum"])
    _assert_close(results[0]["decode_logits"], results[1]["decode_logits"])
    assert results[0]["checksum"] == pytest.approx(
        results[1]["checksum"], rel=1e-7, abs=1e-7
    )


@pytest.mark.local_integration
def test_cache_send_dependency_is_required_without_mutating_adapter(
    tmp_path: Path,
) -> None:
    adapter = ROOT / "src" / "qwen32_cluster" / "qwen3_pipeline.py"
    before = hashlib.sha256(adapter.read_bytes()).hexdigest()
    output_dir = tmp_path / "qwen32-cache-dependency-bypassed"

    with pytest.raises(
        (RingLaunchError, RingResultError),
        match="deadline|missing|failure|nonzero",
    ):
        _launch_case("cache_dependency_bypassed", output_dir)

    marker = output_dir / "rank-1-cache-evaluated-without-send.marker"
    assert marker.read_text(encoding="utf-8") == "cache-only evaluation completed"
    after = hashlib.sha256(adapter.read_bytes()).hexdigest()
    assert after == before
    assert "mx.depends(cache[-1].keys, sent_h)" in adapter.read_text(encoding="utf-8")


@pytest.mark.local_integration
def test_sequence_corruption_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RingResultError, match="event count/order differs"):
        _launch_case("sequence_corruption", tmp_path / "qwen32-corrupt-sequence")


def _valid_forward_result(rank: int) -> dict[str, Any]:
    operation = "recv" if rank == 0 else "send"
    peer = 1 if rank == 0 else 0
    logits = [[[0.0] * 32 for _ in range(4)] for _ in range(2)]
    return {
        "case": "forward",
        "rank": rank,
        "world_size": 2,
        "status": "ok",
        "exit_code": 0,
        "error": None,
        "events": [
            {"call": "forward", "event": operation, "peer": peer},
            {"call": "forward", "event": "all_gather"},
            {"call": "forward", "event": "eval_complete", "target": "logits"},
        ],
        "local_layers": [1] if rank == 0 else [0],
        "partition": {"start": 1 if rank == 0 else 0, "end": 2 if rank == 0 else 1},
        "input_shape": [2, 4],
        "logits_shape": [2, 4, 32],
        "reference_logits_shape": [2, 4, 32],
        "logits": logits,
        "reference_logits": deepcopy(logits),
        "checksum": 0.0,
    }


@pytest.mark.parametrize(
    "corruption",
    [
        "bool-rank",
        "extra-key",
        "missing-key",
        "wrong-case",
        "wrong-shape",
        "ragged-tensor",
        "non-finite",
        "fabricated-event",
    ],
)
def test_result_schema_rejects_altered_runtime_observations(
    tmp_path: Path,
    corruption: str,
) -> None:
    output_dir = tmp_path / corruption
    output_dir.mkdir()
    payloads = [_valid_forward_result(rank) for rank in range(2)]
    target = payloads[0]
    if corruption == "bool-rank":
        target["rank"] = False
    elif corruption == "extra-key":
        target["fabricated"] = True
    elif corruption == "missing-key":
        target.pop("input_shape")
    elif corruption == "wrong-case":
        target["case"] = "cache_dependency"
    elif corruption == "wrong-shape":
        target["logits_shape"] = [2, 3, 32]
    elif corruption == "ragged-tensor":
        target["logits"][0].pop()
    elif corruption == "non-finite":
        target["logits"][0][0][0] = float("nan")
    else:
        target["events"][0]["event"] = "send"
    for rank, payload in enumerate(payloads):
        (output_dir / f"rank-{rank}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    with pytest.raises(RingResultError):
        _load_rank_results(output_dir, "forward")


def test_primary_result_error_survives_secondary_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CompletedProcess:
        pid = 999_999
        returncode = 0

        def communicate(self, timeout: float):
            return "", ""

        def poll(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: CompletedProcess())
    monkeypatch.setattr("test_two_rank_pipeline._assert_ring_ports_free", lambda: None)
    monkeypatch.setattr(
        "test_two_rank_pipeline._load_rank_results",
        lambda *args, **kwargs: (_ for _ in ()).throw(RingResultError("primary result")),
    )
    monkeypatch.setattr(
        "test_two_rank_pipeline._wait_for_ring_ports_to_clear",
        lambda *args, **kwargs: (_ for _ in ()).throw(RingLaunchError("cleanup failed")),
    )

    with pytest.raises(RingResultError, match="primary result") as caught:
        _launch_case(
            "forward",
            tmp_path / "supervisor",
            lock_path=tmp_path / "supervisor.lock",
        )

    assert any("cleanup failed" in note for note in caught.value.__notes__)


def test_ring_launch_lock_has_a_bounded_contention_failure(tmp_path: Path) -> None:
    lock_path = tmp_path / "contention.lock"
    with _ring_launch_lock(_Deadline(1.0), lock_path):
        with pytest.raises(RingLaunchError, match="lock.*budget"):
            with _ring_launch_lock(_Deadline(0.05), lock_path):
                pytest.fail("contended lock unexpectedly acquired")


def test_pure_lock_is_isolated_while_global_ring_lock_is_held(
    tmp_path: Path,
) -> None:
    with _ring_launch_lock(_Deadline(1.0), RING_LOCK_PATH):
        with _ring_launch_lock(_Deadline(1.0), tmp_path / "isolated.lock"):
            assert True


def test_timeout_error_retains_cleanup_stdout_and_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TimedOutProcess:
        pid = 999_998
        returncode = None

        def communicate(self, timeout: float):
            raise subprocess.TimeoutExpired(["mlx.launch"], timeout)

        def poll(self):
            return None

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: TimedOutProcess())
    monkeypatch.setattr("test_two_rank_pipeline._assert_ring_ports_free", lambda: None)
    monkeypatch.setattr(
        "test_two_rank_pipeline._stop_process_group",
        lambda *args, **kwargs: ("captured stdout", "captured stderr"),
    )
    monkeypatch.setattr(
        "test_two_rank_pipeline._wait_for_ring_ports_to_clear",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(RingLaunchError, match="single overall") as caught:
        _launch_case(
            "forward",
            tmp_path / "timeout",
            lock_path=tmp_path / "timeout.lock",
        )

    rendered = "\n".join([str(caught.value), *caught.value.__notes__])
    assert "captured stdout" in rendered
    assert "captured stderr" in rendered
