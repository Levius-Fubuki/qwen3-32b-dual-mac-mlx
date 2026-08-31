from __future__ import annotations

from urllib.parse import urlsplit

import pytest

from qwen32_cluster.contracts import _load_json
from qwen32_cluster.profiles import Profile, ProfilesConfig


EXPECTED_RING_HOSTS = [
    {"ssh": "127.0.0.1", "ips": ["169.254.217.74"]},
    {"ssh": "kelly@169.254.82.82", "ips": ["169.254.82.82"]},
]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--hostfile", help="Path to the validated two-host MLX Ring hostfile")
    parser.addoption("--profile-file", help="Path to the canonical model profile file")
    parser.addoption("--base-url", help="Explicit loopback URL for live API tests")


def _valid_hostfile(raw_path: str | None) -> bool:
    if not raw_path:
        return False
    try:
        payload = _load_json(raw_path)
    except (OSError, TypeError, ValueError):
        return False
    return payload == {"backend": "ring", "hosts": EXPECTED_RING_HOSTS}


def _canonical_profile_file(raw_path: str | None) -> bool:
    if not raw_path:
        return False
    try:
        payload = _load_json(raw_path)
        if set(payload) == {"profiles", "server"}:
            ProfilesConfig.from_dict(payload)
        else:
            Profile.from_dict(payload)
    except (OSError, TypeError, ValueError):
        return False
    return True


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
    skip_profile = pytest.mark.skip(reason="requires explicit validated --profile-file")
    skip_live = pytest.mark.skip(reason="requires explicit http://127.0.0.1:<port> --base-url")

    for item in items:
        cluster_marker = item.get_closest_marker("cluster")
        if cluster_marker and not hostfile_ok:
            item.add_marker(skip_cluster)
        requires_profile = bool(item.get_closest_marker("model_metadata")) or bool(
            cluster_marker and cluster_marker.kwargs.get("requires_profile", False)
        )
        if requires_profile and not profile_ok:
            item.add_marker(skip_profile)
        if item.get_closest_marker("live_api") and not base_url_ok:
            item.add_marker(skip_live)
