from __future__ import annotations

import dataclasses
import ipaddress
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


ALLOWED_SSH_ENDPOINTS = frozenset({"127.0.0.1", "kelly@169.254.82.82"})
EXPECTED_RING_HOSTS = (
    ("127.0.0.1", ("169.254.217.74",)),
    ("kelly@169.254.82.82", ("169.254.82.82",)),
)
EXPECTED_GUARDRAILS = ((0, 10844792422), (1, 12133282611))
CANONICAL_RING_HOSTFILE = "/Users/Shared/mlx-cluster/hosts.json"
CANONICAL_RING_STARTING_PORT = 33323
CANONICAL_DEPLOYMENT_PORTS = (18081, 18080, 8080)


class RunStatus(str, Enum):
    PASS = "PASS"
    OUTPUT_FAIL = "OUTPUT_FAIL"
    TIMEOUT = "TIMEOUT"
    PEER_LOST = "PEER_LOST"
    MEMORY_GUARD = "MEMORY_GUARD"
    SWAP_GUARD = "SWAP_GUARD"
    GPU_UNHEALTHY = "GPU_UNHEALTHY"


def _require_exact_keys(data: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(data, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    actual = set(data)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected {sorted(unexpected)}")
        raise ValueError(f"invalid {label}: {', '.join(details)}")


def _require_int(value: Any, label: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be at most {maximum}")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _validate_port(value: Any, label: str) -> int:
    return _require_int(value, label, minimum=1, maximum=65535)


def _load_json(path: str | Path) -> Mapping[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("configuration root must be a JSON object")
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Serialize a contract value as stable compact UTF-8 JSON with a final newline."""

    return json.dumps(
        _jsonable(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


@dataclass(frozen=True)
class ClusterHost:
    rank: int
    name: str
    ssh: str
    thunderbolt_ip: str
    mlx_peak_guardrail_bytes: int

    def __post_init__(self) -> None:
        _require_int(self.rank, "rank", minimum=0)
        _require_string(self.name, "name")
        if self.ssh not in ALLOWED_SSH_ENDPOINTS:
            raise ValueError(f"ssh must be one of {sorted(ALLOWED_SSH_ENDPOINTS)}")
        _require_string(self.thunderbolt_ip, "thunderbolt_ip")
        try:
            address = ipaddress.ip_address(self.thunderbolt_ip)
        except ValueError as exc:
            raise ValueError("thunderbolt_ip must be a valid IP address") from exc
        if not isinstance(address, ipaddress.IPv4Address) or not address.is_link_local:
            raise ValueError("thunderbolt_ip must be a link-local IPv4 address")
        _require_int(self.mlx_peak_guardrail_bytes, "mlx_peak_guardrail_bytes", minimum=1)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ClusterHost:
        keys = {"rank", "name", "ssh", "thunderbolt_ip", "mlx_peak_guardrail_bytes"}
        _require_exact_keys(data, keys, "cluster host")
        return cls(**{key: data[key] for key in keys})

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "name": self.name,
            "ssh": self.ssh,
            "thunderbolt_ip": self.thunderbolt_ip,
            "mlx_peak_guardrail_bytes": self.mlx_peak_guardrail_bytes,
        }


@dataclass(frozen=True)
class RingHost:
    ssh: str
    ips: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.ssh not in ALLOWED_SSH_ENDPOINTS:
            raise ValueError(f"ssh must be one of {sorted(ALLOWED_SSH_ENDPOINTS)}")
        if not isinstance(self.ips, tuple) or not self.ips:
            raise ValueError("ring host ips must be a non-empty tuple")
        for value in self.ips:
            if not isinstance(value, str) or not value:
                raise ValueError("ring host ips must contain non-empty strings")
            try:
                address = ipaddress.ip_address(value)
            except ValueError as exc:
                raise ValueError("ring host ips must contain valid IP addresses") from exc
            if not isinstance(address, ipaddress.IPv4Address) or not address.is_link_local:
                raise ValueError("ring host ips must contain link-local IPv4 addresses")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RingHost:
        _require_exact_keys(data, {"ssh", "ips"}, "ring host")
        ips = data["ips"]
        if not isinstance(ips, list):
            raise ValueError("ring host ips must be a JSON list")
        return cls(ssh=data["ssh"], ips=tuple(ips))

    def to_dict(self) -> dict[str, Any]:
        return {"ssh": self.ssh, "ips": list(self.ips)}


@dataclass(frozen=True)
class Ring:
    hostfile: str
    starting_port: int
    hosts: tuple[RingHost, ...]

    def __post_init__(self) -> None:
        _require_string(self.hostfile, "ring hostfile")
        if not Path(self.hostfile).is_absolute():
            raise ValueError("ring hostfile must be an absolute path")
        if self.hostfile != CANONICAL_RING_HOSTFILE:
            raise ValueError(f"ring hostfile must be {CANONICAL_RING_HOSTFILE}")
        _validate_port(self.starting_port, "ring starting_port")
        if self.starting_port != CANONICAL_RING_STARTING_PORT:
            raise ValueError(f"ring starting_port must be {CANONICAL_RING_STARTING_PORT}")
        actual = tuple((host.ssh, host.ips) for host in self.hosts)
        if actual != EXPECTED_RING_HOSTS:
            raise ValueError("ring hosts and ips must match the fixed M3/M4 order")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Ring:
        _require_exact_keys(data, {"hostfile", "starting_port", "hosts"}, "ring")
        hosts = data["hosts"]
        if not isinstance(hosts, list):
            raise ValueError("ring hosts must be a JSON list")
        return cls(
            hostfile=data["hostfile"],
            starting_port=data["starting_port"],
            hosts=tuple(RingHost.from_dict(host) for host in hosts),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hostfile": self.hostfile,
            "starting_port": self.starting_port,
            "hosts": [host.to_dict() for host in self.hosts],
        }


@dataclass(frozen=True)
class DeploymentPorts:
    internal: int
    canary: int
    public: int

    def __post_init__(self) -> None:
        values = (
            _validate_port(self.internal, "internal port"),
            _validate_port(self.canary, "canary port"),
            _validate_port(self.public, "public port"),
        )
        if len(set(values)) != len(values):
            raise ValueError("deployment ports must be distinct")
        if values != CANONICAL_DEPLOYMENT_PORTS:
            raise ValueError(
                "deployment ports must be internal=18081, canary=18080, public=8080"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DeploymentPorts:
        _require_exact_keys(data, {"internal", "canary", "public"}, "deployment ports")
        return cls(internal=data["internal"], canary=data["canary"], public=data["public"])

    def to_dict(self) -> dict[str, int]:
        return {"internal": self.internal, "canary": self.canary, "public": self.public}


@dataclass(frozen=True)
class ClusterConfig:
    hosts: tuple[ClusterHost, ...]
    ring: Ring
    deployment_ports: DeploymentPorts

    def __post_init__(self) -> None:
        if tuple(host.rank for host in self.hosts) != (0, 1):
            raise ValueError("cluster hosts must have unique ranks in order 0, 1")
        expected = (
            (0, "M3", "127.0.0.1", "169.254.217.74"),
            (1, "M4", "kelly@169.254.82.82", "169.254.82.82"),
        )
        actual = tuple((host.rank, host.name, host.ssh, host.thunderbolt_ip) for host in self.hosts)
        if actual != expected:
            raise ValueError("cluster hosts must match the fixed M3/M4 topology")
        guardrails = tuple((host.rank, host.mlx_peak_guardrail_bytes) for host in self.hosts)
        if guardrails != EXPECTED_GUARDRAILS:
            raise ValueError("cluster guardrails must match the fixed per-rank safety policy")
        ring_endpoints = tuple((host.ssh, host.ips[0]) for host in self.ring.hosts)
        control_endpoints = tuple((host.ssh, host.thunderbolt_ip) for host in self.hosts)
        if ring_endpoints != control_endpoints:
            raise ValueError("ring endpoints must correspond to cluster control endpoints")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ClusterConfig:
        _require_exact_keys(data, {"hosts", "ring", "deployment_ports"}, "cluster configuration")
        hosts = data["hosts"]
        if not isinstance(hosts, list):
            raise ValueError("cluster hosts must be a JSON list")
        return cls(
            hosts=tuple(ClusterHost.from_dict(host) for host in hosts),
            ring=Ring.from_dict(data["ring"]),
            deployment_ports=DeploymentPorts.from_dict(data["deployment_ports"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hosts": [host.to_dict() for host in self.hosts],
            "ring": self.ring.to_dict(),
            "deployment_ports": self.deployment_ports.to_dict(),
        }


def load_cluster(path: str | Path) -> ClusterConfig:
    return ClusterConfig.from_dict(_load_json(path))
