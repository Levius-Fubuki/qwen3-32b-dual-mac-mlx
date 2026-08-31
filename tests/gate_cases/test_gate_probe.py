from __future__ import annotations

import os
from pathlib import Path

import pytest


def record_execution(name: str) -> None:
    raw_path = os.environ.get("QWEN32_GATE_SENTINEL")
    if raw_path:
        with Path(raw_path).open("a", encoding="utf-8") as stream:
            stream.write(f"{name}\n")


@pytest.mark.cluster
class TestInheritedCluster:
    def test_body(self) -> None:
        record_execution("cluster")


@pytest.mark.cluster(requires_profile=True)
def test_profile_required_cluster() -> None:
    record_execution("profile-cluster")


@pytest.mark.live_api
def test_live_api() -> None:
    record_execution("live-api")


@pytest.mark.model_metadata
def test_model_metadata() -> None:
    record_execution("model-metadata")


@pytest.mark.cluster(requires_profile=True)
@pytest.mark.model_metadata
def test_combined_markers() -> None:
    record_execution("combined")
