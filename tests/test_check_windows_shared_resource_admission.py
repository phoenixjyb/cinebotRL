import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/check_windows_shared_resource_admission.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("resource_admission", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _module()


def _snapshot(
    *,
    windows_free_gib: float = 16.0,
    gpu_free_mib: int = 18_000,
    cad_processes: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    total_memory_kib = 32 * 1024 * 1024
    total_gpu_mib = 24_000
    return {
        "windows": {
            "total_physical_memory_kib": total_memory_kib,
            "free_physical_memory_kib": int(
                windows_free_gib * 1024 * 1024
            ),
            "cad_processes": (
                [] if cad_processes is None else cad_processes
            ),
        },
        "gpu": {
            "gpu_count": 1,
            "total_memory_mib": total_gpu_mib,
            "used_memory_mib": total_gpu_mib - gpu_free_mib,
            "free_memory_mib": gpu_free_mib,
        },
    }


def test_healthy_snapshot_passes_without_consuming_authorization() -> None:
    result = MODULE.evaluate_snapshot(_snapshot())

    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["runtime_started"] is False
    assert result["authorization_consumed"] is False


@pytest.mark.parametrize(
    ("snapshot", "failed_check"),
    [
        (
            _snapshot(
                cad_processes=[
                    {
                        "name": "ugraf.exe",
                        "pid": 27816,
                        "working_set_mib": 700.0,
                        "private_memory_mib": 20_000.0,
                    }
                ]
            ),
            "cad_processes_absent",
        ),
        (_snapshot(windows_free_gib=5.7), "windows_free_memory_sufficient"),
        (_snapshot(gpu_free_mib=10_268), "gpu_free_memory_sufficient"),
    ],
)
def test_busy_shared_host_fails_closed(
    snapshot: dict[str, object],
    failed_check: str,
) -> None:
    result = MODULE.evaluate_snapshot(snapshot)

    assert result["passed"] is False
    assert result["checks"][failed_check] is False
    assert result["runtime_started"] is False
    assert result["authorization_consumed"] is False


def test_malformed_probe_fails_closed() -> None:
    result = MODULE.evaluate_snapshot(
        {
            "windows": {
                "total_physical_memory_kib": True,
                "free_physical_memory_kib": "lots",
                "cad_processes": "none",
            },
            "gpu": {
                "gpu_count": 1,
                "total_memory_mib": 24_000,
                "used_memory_mib": 1_000,
                "free_memory_mib": 18_000,
            },
        }
    )

    assert result["passed"] is False
    assert result["checks"]["windows_memory_probe_valid"] is False
    assert result["checks"]["cad_process_probe_valid"] is False
    assert result["checks"]["gpu_memory_probe_valid"] is False


def test_cli_snapshot_writes_machine_readable_rejection(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.json"
    output = tmp_path / "result.json"
    snapshot.write_text(
        json.dumps(_snapshot(windows_free_gib=4.0)),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--snapshot",
            str(snapshot),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 5
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["runtime_started"] is False
    assert payload["authorization_consumed"] is False
