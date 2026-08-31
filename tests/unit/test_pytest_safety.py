from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import conftest


class FakeConfig:
    def __init__(self, **options: str | None) -> None:
        self.options = options

    def getoption(self, name: str) -> str | None:
        return self.options[name.removeprefix("--").replace("-", "_")]


class FakeItem:
    def __init__(self, **markers: dict) -> None:
        self.markers = {name: SimpleNamespace(kwargs=kwargs) for name, kwargs in markers.items()}
        self.added_markers: list = []

    def get_closest_marker(self, name: str):
        return self.markers.get(name)

    def add_marker(self, marker) -> None:
        self.added_markers.append(marker)


def write_valid_hostfile(path: Path) -> None:
    path.write_text(json.dumps(conftest.EXPECTED_RING_HOSTS), encoding="utf-8")


def valid_sidecar() -> dict:
    return {
        "name": "calibration-4bit-m4-40-m3-24-p512",
        "quantization_bits": 4,
        "stage_layers": [40, 24],
        "prefill_step_size": 512,
        "context_limit": 8192,
    }


def test_cluster_stage_requiring_profile_skips_without_explicit_profile_file(tmp_path: Path) -> None:
    hostfile = tmp_path / "hosts.json"
    write_valid_hostfile(hostfile)
    item = FakeItem(cluster={"requires_profile": True})
    config = FakeConfig(hostfile=str(hostfile), profile_file=None, base_url=None)

    conftest.pytest_collection_modifyitems(config, [item])

    assert len(item.added_markers) == 1
    assert "--profile-file" in item.added_markers[0].mark.kwargs["reason"]


def test_ordinary_cluster_test_remains_hostfile_only(tmp_path: Path) -> None:
    hostfile = tmp_path / "hosts.json"
    write_valid_hostfile(hostfile)
    item = FakeItem(cluster={})
    config = FakeConfig(hostfile=str(hostfile), profile_file=None, base_url=None)

    conftest.pytest_collection_modifyitems(config, [item])

    assert item.added_markers == []


def test_profile_file_accepts_valid_profiles_collection(tmp_path: Path) -> None:
    source = Path(__file__).parents[2] / "config" / "profiles.json"
    profile_file = tmp_path / "profiles.json"
    profile_file.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    assert conftest._canonical_profile_file(str(profile_file))


def test_profile_file_accepts_valid_single_profile_sidecar(tmp_path: Path) -> None:
    profile_file = tmp_path / "selected-profile.json"
    profile_file.write_text(json.dumps(valid_sidecar()), encoding="utf-8")

    assert conftest._canonical_profile_file(str(profile_file))


def test_profile_file_rejects_invalid_single_profile_sidecar(tmp_path: Path) -> None:
    profile = valid_sidecar()
    profile["stage_layers"] = [64, 0]
    profile_file = tmp_path / "selected-profile.json"
    profile_file.write_text(json.dumps(profile), encoding="utf-8")

    assert not conftest._canonical_profile_file(str(profile_file))
