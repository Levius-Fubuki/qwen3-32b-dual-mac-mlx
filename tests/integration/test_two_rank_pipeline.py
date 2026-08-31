from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "tests" / "integration" / "two_rank_pipeline_case.py"
LAUNCHER = Path("/Users/Shared/mlx-cluster/.venv/bin/mlx.launch")
PYTHON = Path("/Users/Shared/mlx-cluster/.venv/bin/python")
RING_PORTS = (33323, 33324)
LAUNCH_TIMEOUT_SECONDS = 90
TERM_GRACE_SECONDS = 5
EXPECTED_SEQUENCES = {
    0: ["group_ready", "model_partitioned", "receive", "all_gather", "result"],
    1: ["group_ready", "model_partitioned", "send", "all_gather", "result"],
}


class RingLaunchError(AssertionError):
    pass


class RingResultError(AssertionError):
    pass


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


def _wait_for_ring_ports_to_clear() -> None:
    deadline = time.monotonic() + TERM_GRACE_SECONDS
    while time.monotonic() < deadline:
        if all(_port_is_free(port) for port in RING_PORTS):
            return
        time.sleep(0.05)
    occupied = [port for port in RING_PORTS if not _port_is_free(port)]
    raise RingLaunchError(f"Ring launch left configured port(s) occupied: {occupied}")


def _stop_process_group(process: subprocess.Popen[str]) -> tuple[str, str]:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        return process.communicate(timeout=TERM_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.communicate(timeout=TERM_GRACE_SECONDS)


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


def _load_rank_results(output_dir: Path) -> list[dict[str, Any]]:
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
        if not isinstance(result, dict):
            raise RingResultError(f"rank {expected_rank} JSON must be an object")
        if result.get("rank") != expected_rank:
            raise RingResultError(
                f"wrong rank in {result_path}: expected {expected_rank}, "
                f"got {result.get('rank')!r}"
            )
        if result.get("status") != "ok" or result.get("exit_code") != 0:
            raise RingResultError(
                f"rank {expected_rank} reported failure: "
                f"status={result.get('status')!r}, "
                f"exit_code={result.get('exit_code')!r}, "
                f"error={result.get('error')!r}"
            )
        if result.get("world_size") != 2:
            raise RingResultError(
                f"rank {expected_rank} reported wrong world size: "
                f"{result.get('world_size')!r}"
            )
        expected_sequence = EXPECTED_SEQUENCES[expected_rank]
        if result.get("sequence") != expected_sequence:
            raise RingResultError(
                f"rank {expected_rank} sequence/order corruption: expected "
                f"{expected_sequence!r}, got {result.get('sequence')!r}"
            )
        results.append(result)
    return results


def _launch_case(case: str, output_dir: Path) -> list[dict[str, Any]]:
    if not WORKER.is_file():
        raise RingLaunchError(f"two-rank worker is missing: {WORKER}")
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
        try:
            stdout, stderr = process.communicate(timeout=LAUNCH_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            stdout, stderr = _stop_process_group(process)
            raise RingLaunchError(
                f"two-rank case {case!r} exceeded the {LAUNCH_TIMEOUT_SECONDS}s "
                f"hard deadline; exact launcher process group {process.pid} was stopped\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            ) from exc
        if process.returncode != 0:
            raise RingLaunchError(
                f"launcher exited nonzero ({process.returncode}) for case {case!r}\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        try:
            return _load_rank_results(output_dir)
        except RingResultError as exc:
            raise RingResultError(
                f"{exc}\nlauncher stdout:\n{stdout}\nlauncher stderr:\n{stderr}"
            ) from exc
    finally:
        if process.poll() is None:
            _stop_process_group(process)
        _wait_for_ring_ports_to_clear()


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
    assert all(result["batch_size"] > 1 for result in results)
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
    with pytest.raises(RingResultError, match="sequence/order corruption"):
        _launch_case("sequence_corruption", tmp_path / "qwen32-corrupt-sequence")
