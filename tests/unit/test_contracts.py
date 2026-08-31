from __future__ import annotations

import dataclasses
import ipaddress
import json
from importlib import import_module
from pathlib import Path

import pytest


def contracts():
    try:
        return import_module("qwen32_cluster.contracts")
    except ModuleNotFoundError as exc:
        pytest.fail(f"contracts module is not implemented: {exc}")


def valid_cluster_payload() -> dict:
    return {
        "hosts": [
            {
                "rank": 0,
                "name": "M3",
                "ssh": "127.0.0.1",
                "thunderbolt_ip": "169.254.217.74",
                "mlx_peak_guardrail_bytes": 10844792422,
            },
            {
                "rank": 1,
                "name": "M4",
                "ssh": "kelly@169.254.82.82",
                "thunderbolt_ip": "169.254.82.82",
                "mlx_peak_guardrail_bytes": 12133282611,
            },
        ],
        "ring": {
            "hostfile": "/Users/Shared/mlx-cluster/hosts.json",
            "starting_port": 33323,
            "hosts": [
                {"ssh": "127.0.0.1", "ips": ["169.254.217.74"]},
                {"ssh": "kelly@169.254.82.82", "ips": ["169.254.82.82"]},
            ],
        },
        "deployment_ports": {"internal": 18081, "canary": 18080, "public": 8080},
    }


def test_run_status_values_are_strict_and_complete() -> None:
    module = contracts()
    assert {member.name: member.value for member in module.RunStatus} == {
        "PASS": "PASS",
        "OUTPUT_FAIL": "OUTPUT_FAIL",
        "TIMEOUT": "TIMEOUT",
        "PEER_LOST": "PEER_LOST",
        "MEMORY_GUARD": "MEMORY_GUARD",
        "SWAP_GUARD": "SWAP_GUARD",
        "GPU_UNHEALTHY": "GPU_UNHEALTHY",
    }
    with pytest.raises(ValueError):
        module.RunStatus("pass")


def test_cluster_host_is_frozen_and_serializes_deterministically() -> None:
    module = contracts()
    host = module.ClusterHost(
        rank=1,
        name="M4",
        ssh="kelly@169.254.82.82",
        thunderbolt_ip="169.254.82.82",
        mlx_peak_guardrail_bytes=12133282611,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        host.rank = 0
    assert module.canonical_json(host) == (
        '{"mlx_peak_guardrail_bytes":12133282611,"name":"M4","rank":1,'
        '"ssh":"kelly@169.254.82.82","thunderbolt_ip":"169.254.82.82"}\n'
    )


def test_cluster_host_rejects_integer_form_thunderbolt_ip() -> None:
    module = contracts()
    numeric_ip = int(ipaddress.IPv4Address("169.254.217.74"))
    with pytest.raises(ValueError, match="thunderbolt_ip must be a non-empty string"):
        module.ClusterHost(0, "M3", "127.0.0.1", numeric_ip, 10844792422)


def test_cluster_host_parser_rejects_integer_form_thunderbolt_ip() -> None:
    module = contracts()
    payload = valid_cluster_payload()["hosts"][0]
    payload["thunderbolt_ip"] = int(ipaddress.IPv4Address("169.254.217.74"))
    with pytest.raises(ValueError, match="thunderbolt_ip must be a non-empty string"):
        module.ClusterHost.from_dict(payload)


def test_ring_host_rejects_integer_form_ip() -> None:
    module = contracts()
    numeric_ip = int(ipaddress.IPv4Address("169.254.217.74"))
    with pytest.raises(ValueError, match="ring host ips must contain non-empty strings"):
        module.RingHost("127.0.0.1", (numeric_ip,))


def test_ring_host_parser_rejects_integer_form_ip() -> None:
    module = contracts()
    numeric_ip = int(ipaddress.IPv4Address("169.254.217.74"))
    with pytest.raises(ValueError, match="ring host ips must contain non-empty strings"):
        module.RingHost.from_dict({"ssh": "127.0.0.1", "ips": [numeric_ip]})


def test_cluster_config_round_trips_canonical_json(tmp_path: Path) -> None:
    module = contracts()
    path = tmp_path / "cluster.json"
    path.write_text(json.dumps(valid_cluster_payload()), encoding="utf-8")
    cluster = module.load_cluster(path)
    assert cluster.to_dict() == valid_cluster_payload()
    assert module.canonical_json(cluster) == module.canonical_json(valid_cluster_payload())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["hosts"][1].update(rank=0),
        lambda value: value["hosts"][0].update(thunderbolt_ip="not-an-ip"),
        lambda value: value["ring"]["hosts"][0].update(ips=["169.254.82.82"]),
        lambda value: value["ring"].update(hosts=list(reversed(value["ring"]["hosts"]))),
    ],
    ids=["duplicate-ranks", "invalid-thunderbolt-ip", "unexpected-ring-ip", "reordered-ring-hosts"],
)
def test_cluster_rejects_invalid_topology(tmp_path: Path, mutate) -> None:
    module = contracts()
    payload = valid_cluster_payload()
    mutate(payload)
    path = tmp_path / "cluster.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        module.load_cluster(path)


def test_cluster_rejects_unknown_fields(tmp_path: Path) -> None:
    module = contracts()
    payload = valid_cluster_payload()
    payload["surprise"] = True
    path = tmp_path / "cluster.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        module.load_cluster(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["hosts"][0].update(mlx_peak_guardrail_bytes=1),
        lambda value: value["hosts"][1].update(mlx_peak_guardrail_bytes=2**60),
        lambda value: value["ring"].update(hostfile="/tmp/alternate-hosts.json"),
        lambda value: value["ring"].update(starting_port=1024),
        lambda value: value["ring"].update(starting_port=33324),
        lambda value: value["deployment_ports"].update(internal=18082),
        lambda value: value["deployment_ports"].update(canary=18079),
        lambda value: value["deployment_ports"].update(public=8081),
        lambda value: value["deployment_ports"].update(public=18081),
    ],
    ids=[
        "arbitrary-rank0-guardrail",
        "high-rank1-guardrail",
        "alternate-hostfile",
        "low-ring-port",
        "wrong-ring-port",
        "altered-internal-port",
        "altered-canary-port",
        "altered-public-port",
        "colliding-deployment-port",
    ],
)
def test_cluster_rejects_noncanonical_safety_inputs(tmp_path: Path, mutate) -> None:
    module = contracts()
    payload = valid_cluster_payload()
    mutate(payload)
    path = tmp_path / "cluster.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        module.load_cluster(path)


def test_repository_cluster_config_has_fixed_endpoints() -> None:
    module = contracts()
    cluster = module.load_cluster(Path(__file__).parents[2] / "config" / "cluster.json")
    assert [(host.rank, host.name, host.ssh, host.thunderbolt_ip) for host in cluster.hosts] == [
        (0, "M3", "127.0.0.1", "169.254.217.74"),
        (1, "M4", "kelly@169.254.82.82", "169.254.82.82"),
    ]
    assert [host.mlx_peak_guardrail_bytes for host in cluster.hosts] == [10844792422, 12133282611]
    assert cluster.ring.hostfile == "/Users/Shared/mlx-cluster/hosts.json"
    assert cluster.ring.starting_port == 33323
    assert cluster.deployment_ports.to_dict() == {"internal": 18081, "canary": 18080, "public": 8080}
