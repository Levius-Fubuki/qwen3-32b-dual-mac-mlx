from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest


EXPECTED_RING_HOSTS = [
    {"ssh": "127.0.0.1", "ips": ["169.254.217.74"]},
    {"ssh": "kelly@169.254.82.82", "ips": ["169.254.82.82"]},
]
CANONICAL_PROFILE_FILE = (Path(__file__).parents[1] / "config" / "profiles.json").resolve()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--hostfile", help="Path to the validated two-host MLX Ring hostfile")
    parser.addoption("--profile-file", help="Path to the canonical model profile file")
    parser.addoption("--base-url", help="Explicit loopback URL for live API tests")


def _valid_hostfile(raw_path: str | None) -> bool:
    if not raw_path:
        return False
    try:
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    hosts = payload.get("hosts") if isinstance(payload, dict) else payload
    return hosts == EXPECTED_RING_HOSTS


def _canonical_profile_file(raw_path: str | None) -> bool:
    if not raw_path:
        return False
    try:
        return Path(raw_path).resolve(strict=True) == CANONICAL_PROFILE_FILE
    except OSError:
        return False


def _loopback_base_url(raw_url: str | None) -> bool:
    if not raw_url:
        return False
    try:
        parsed = urlsplit(raw_url)
        return (
            parsed.scheme == "http"
            and parsed.hostname == "127.0.0.1"
            and parsed.port is not None
            and 1 <= parsed.port <= 65535
            and parsed.path in ("", "/")
            and not parsed.query
            and not parsed.fragment
            and parsed.username is None
            and parsed.password is None
        )
    except ValueError:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    hostfile_ok = _valid_hostfile(config.getoption("--hostfile"))
    profile_ok = _canonical_profile_file(config.getoption("--profile-file"))
    base_url_ok = _loopback_base_url(config.getoption("--base-url"))

    skip_cluster = pytest.mark.skip(reason="requires explicit validated --hostfile")
    skip_metadata = pytest.mark.skip(reason="requires canonical explicit --profile-file")
    skip_live = pytest.mark.skip(reason="requires explicit http://127.0.0.1:<port> --base-url")

    for item in items:
        if item.get_closest_marker("cluster") and not hostfile_ok:
            item.add_marker(skip_cluster)
        if item.get_closest_marker("model_metadata") and not profile_ok:
            item.add_marker(skip_metadata)
        if item.get_closest_marker("live_api") and not base_url_ok:
            item.add_marker(skip_live)
