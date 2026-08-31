from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import conftest


PROJECT_ROOT = Path(__file__).parents[2]
PROBE = PROJECT_ROOT / "tests" / "gate_cases" / "test_gate_probe.py"


def valid_hostfile_payload() -> dict:
    return {"backend": "ring", "hosts": conftest.EXPECTED_RING_HOSTS}


def valid_sidecar() -> dict:
    return {
        "name": "calibration-4bit-m4-40-m3-24-p512",
        "quantization_bits": 4,
        "stage_layers": [40, 24],
        "prefill_step_size": 512,
        "context_limit": 8192,
    }


def write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def run_probe(tmp_path: Path, nodeid: str, *options: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    sentinel = tmp_path / "executed.txt"
    sentinel.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["QWEN32_GATE_SENTINEL"] = str(sentinel)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", f"{PROBE}::{nodeid}", *options],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    executed = sentinel.read_text(encoding="utf-8").splitlines() if sentinel.exists() else []
    return result, executed


def assert_skipped(result: subprocess.CompletedProcess[str], executed: list[str]) -> None:
    assert "1 skipped" in result.stdout
    assert executed == []


def assert_executed(result: subprocess.CompletedProcess[str], executed: list[str], expected: str) -> None:
    assert "1 passed" in result.stdout
    assert executed == [expected]


@pytest.mark.parametrize(
    "payload",
    [
        {"backend": "jaccl", "hosts": conftest.EXPECTED_RING_HOSTS},
        {"backend": "jaccl-ring", "hosts": conftest.EXPECTED_RING_HOSTS},
        {
            "backend": "ring",
            "hosts": conftest.EXPECTED_RING_HOSTS,
            "envs": ["MLX_METAL_FAST_SYNCH=1"],
        },
        {"backend": "ring", "hosts": conftest.EXPECTED_RING_HOSTS, "unexpected": True},
        conftest.EXPECTED_RING_HOSTS,
    ],
    ids=["jaccl", "jaccl-ring", "environment-override", "unexpected-root-field", "bare-list-root"],
)
def test_hostfile_rejects_unsafe_policy_shapes(tmp_path: Path, payload: object) -> None:
    hostfile = write_json(tmp_path / "hosts.json", payload)
    assert not conftest._valid_hostfile(str(hostfile))


def test_hostfile_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    hosts = json.dumps(conftest.EXPECTED_RING_HOSTS)
    hostfile = tmp_path / "hosts.json"
    hostfile.write_text(f'{{"backend":"ring","backend":"ring","hosts":{hosts}}}', encoding="utf-8")
    assert not conftest._valid_hostfile(str(hostfile))


def test_hostfile_accepts_only_exact_safe_ring_policy(tmp_path: Path) -> None:
    hostfile = write_json(tmp_path / "hosts.json", valid_hostfile_payload())
    assert conftest._valid_hostfile(str(hostfile))


def test_profile_file_accepts_valid_profiles_collection(tmp_path: Path) -> None:
    source = PROJECT_ROOT / "config" / "profiles.json"
    profile_file = tmp_path / "profiles.json"
    profile_file.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    assert conftest._canonical_profile_file(str(profile_file))


def test_profile_file_accepts_valid_single_profile_sidecar(tmp_path: Path) -> None:
    profile_file = write_json(tmp_path / "selected-profile.json", valid_sidecar())
    assert conftest._canonical_profile_file(str(profile_file))


def test_profile_file_rejects_invalid_single_profile_sidecar(tmp_path: Path) -> None:
    profile = valid_sidecar()
    profile["stage_layers"] = [64, 0]
    profile_file = write_json(tmp_path / "selected-profile.json", profile)
    assert not conftest._canonical_profile_file(str(profile_file))


def test_real_inherited_cluster_gate_handles_omitted_malformed_and_valid_hostfile(tmp_path: Path) -> None:
    malformed = write_json(
        tmp_path / "malformed-hosts.json",
        {"backend": "jaccl", "hosts": conftest.EXPECTED_RING_HOSTS},
    )
    valid = write_json(tmp_path / "valid-hosts.json", valid_hostfile_payload())

    result, executed = run_probe(tmp_path, "TestInheritedCluster::test_body")
    assert_skipped(result, executed)
    result, executed = run_probe(tmp_path, "TestInheritedCluster::test_body", "--hostfile", str(malformed))
    assert_skipped(result, executed)
    result, executed = run_probe(tmp_path, "TestInheritedCluster::test_body", "--hostfile", str(valid))
    assert_executed(result, executed, "cluster")


def test_real_profile_required_cluster_gate_handles_omitted_malformed_and_valid_profile(tmp_path: Path) -> None:
    hostfile = write_json(tmp_path / "hosts.json", valid_hostfile_payload())
    malformed = valid_sidecar()
    malformed["stage_layers"] = [64, 0]
    malformed_profile = write_json(tmp_path / "malformed-profile.json", malformed)
    valid_profile = write_json(tmp_path / "valid-profile.json", valid_sidecar())
    common = ("--hostfile", str(hostfile))

    result, executed = run_probe(tmp_path, "test_profile_required_cluster", *common)
    assert_skipped(result, executed)
    result, executed = run_probe(
        tmp_path,
        "test_profile_required_cluster",
        *common,
        "--profile-file",
        str(malformed_profile),
    )
    assert_skipped(result, executed)
    result, executed = run_probe(
        tmp_path,
        "test_profile_required_cluster",
        *common,
        "--profile-file",
        str(valid_profile),
    )
    assert_executed(result, executed, "profile-cluster")


def test_real_live_api_gate_handles_omitted_malformed_and_valid_base_url(tmp_path: Path) -> None:
    result, executed = run_probe(tmp_path, "test_live_api")
    assert_skipped(result, executed)
    result, executed = run_probe(tmp_path, "test_live_api", "--base-url", "http://localhost:18080")
    assert_skipped(result, executed)
    result, executed = run_probe(tmp_path, "test_live_api", "--base-url", "http://127.0.0.1:18080")
    assert_executed(result, executed, "live-api")


def test_real_model_metadata_gate_handles_omitted_malformed_and_valid_profile(tmp_path: Path) -> None:
    malformed_profile = write_json(tmp_path / "malformed-profile.json", {"name": "incomplete"})
    valid_profile = write_json(tmp_path / "valid-profile.json", valid_sidecar())

    result, executed = run_probe(tmp_path, "test_model_metadata")
    assert_skipped(result, executed)
    result, executed = run_probe(
        tmp_path,
        "test_model_metadata",
        "--profile-file",
        str(malformed_profile),
    )
    assert_skipped(result, executed)
    result, executed = run_probe(tmp_path, "test_model_metadata", "--profile-file", str(valid_profile))
    assert_executed(result, executed, "model-metadata")


def test_real_combined_markers_require_both_cluster_and_profile_inputs(tmp_path: Path) -> None:
    hostfile = write_json(tmp_path / "hosts.json", valid_hostfile_payload())
    profile = write_json(tmp_path / "profile.json", valid_sidecar())

    result, executed = run_probe(tmp_path, "test_combined_markers", "--hostfile", str(hostfile))
    assert_skipped(result, executed)
    result, executed = run_probe(
        tmp_path,
        "test_combined_markers",
        "--hostfile",
        str(hostfile),
        "--profile-file",
        str(profile),
    )
    assert_executed(result, executed, "combined")
