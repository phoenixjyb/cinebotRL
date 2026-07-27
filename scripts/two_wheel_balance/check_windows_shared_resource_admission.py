#!/usr/bin/env python3
"""Fail closed when shared Windows resources are unsafe for Isaac coexistence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any


SCHEMA = "cinebotrl_windows_shared_resource_admission_v2"
LAUNCH_MINIMUM_WINDOWS_FREE_MEMORY_GIB = 5.0
LAUNCH_MINIMUM_GPU_FREE_MEMORY_MIB = 9_216
RUNTIME_MINIMUM_WINDOWS_FREE_MEMORY_GIB = 1.5
RUNTIME_MINIMUM_GPU_FREE_MEMORY_MIB = 2_048
CAD_COEXISTENCE_ALLOWED = True
NVIDIA_SMI = Path("/usr/lib/wsl/lib/nvidia-smi")
POWERSHELL = Path(
    "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
)
CAD_PROCESS_NAMES = (
    "creoparametric.exe",
    "parametric.exe",
    "proe.exe",
    "sldworks.exe",
    "ugraf.exe",
    "xtop.exe",
)


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("resource snapshot must be a JSON object")
    return payload


def _windows_snapshot() -> dict[str, Any]:
    names = ",".join(f'"{name}"' for name in CAD_PROCESS_NAMES)
    script = f"""
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$cadNames = @({names})
$os = Get-CimInstance Win32_OperatingSystem
$cad = @(
  Get-CimInstance Win32_Process |
    Where-Object {{ $cadNames -contains $_.Name.ToLowerInvariant() }} |
    Sort-Object Name, ProcessId |
    ForEach-Object {{
      $process = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
      [pscustomobject]@{{
        name = $_.Name.ToLowerInvariant()
        pid = [int]$_.ProcessId
        working_set_mib = if ($process) {{
          [math]::Round($process.WorkingSet64 / 1MB, 3)
        }} else {{ $null }}
        private_memory_mib = if ($process) {{
          [math]::Round($process.PrivateMemorySize64 / 1MB, 3)
        }} else {{ $null }}
      }}
    }}
)
[pscustomobject]@{{
  total_physical_memory_kib = [int64]$os.TotalVisibleMemorySize
  free_physical_memory_kib = [int64]$os.FreePhysicalMemory
  cad_processes = $cad
}} | ConvertTo-Json -Compress -Depth 4
"""
    result = subprocess.run(
        [
            str(POWERSHELL),
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            script,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _gpu_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        [
            str(NVIDIA_SMI),
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [
        [int(value.strip()) for value in line.split(",")]
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    if len(rows) != 1 or len(rows[0]) != 3:
        raise ValueError("expected exactly one GPU memory row")
    total_mib, used_mib, free_mib = rows[0]
    return {
        "gpu_count": 1,
        "total_memory_mib": total_mib,
        "used_memory_mib": used_mib,
        "free_memory_mib": free_mib,
    }


def probe_live_snapshot() -> dict[str, Any]:
    if not POWERSHELL.is_file():
        raise FileNotFoundError(f"PowerShell probe is unavailable: {POWERSHELL}")
    if not NVIDIA_SMI.is_file():
        raise FileNotFoundError(f"NVIDIA probe is unavailable: {NVIDIA_SMI}")
    return {
        "windows": _windows_snapshot(),
        "gpu": _gpu_snapshot(),
    }


def evaluate_snapshot(
    snapshot: dict[str, Any],
    *,
    phase: str = "launch",
) -> dict[str, Any]:
    if phase not in {"launch", "runtime"}:
        raise ValueError(f"unsupported resource-admission phase: {phase}")
    minimum_windows_free_memory_gib = (
        LAUNCH_MINIMUM_WINDOWS_FREE_MEMORY_GIB
        if phase == "launch"
        else RUNTIME_MINIMUM_WINDOWS_FREE_MEMORY_GIB
    )
    minimum_gpu_free_memory_mib = (
        LAUNCH_MINIMUM_GPU_FREE_MEMORY_MIB
        if phase == "launch"
        else RUNTIME_MINIMUM_GPU_FREE_MEMORY_MIB
    )
    windows = snapshot.get("windows")
    gpu = snapshot.get("gpu")
    windows = windows if isinstance(windows, dict) else {}
    gpu = gpu if isinstance(gpu, dict) else {}

    free_memory_kib = windows.get("free_physical_memory_kib")
    total_memory_kib = windows.get("total_physical_memory_kib")
    cad_processes = windows.get("cad_processes")
    gpu_count = gpu.get("gpu_count")
    gpu_free_mib = gpu.get("free_memory_mib")
    gpu_total_mib = gpu.get("total_memory_mib")
    gpu_used_mib = gpu.get("used_memory_mib")
    gpu_unaccounted_mib = (
        gpu_total_mib - gpu_used_mib - gpu_free_mib
        if all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (gpu_total_mib, gpu_used_mib, gpu_free_mib)
        )
        else None
    )

    free_memory_gib = (
        float(free_memory_kib) / (1024.0 * 1024.0)
        if isinstance(free_memory_kib, (int, float))
        and not isinstance(free_memory_kib, bool)
        else None
    )
    cad_rows_valid = isinstance(cad_processes, list) and all(
        isinstance(row, dict)
        and isinstance(row.get("name"), str)
        and isinstance(row.get("pid"), int)
        and not isinstance(row.get("pid"), bool)
        for row in cad_processes
    )
    gpu_values_valid = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (gpu_count, gpu_free_mib, gpu_total_mib, gpu_used_mib)
    )
    checks = {
        "windows_memory_probe_valid": (
            isinstance(total_memory_kib, int)
            and not isinstance(total_memory_kib, bool)
            and total_memory_kib > 0
            and isinstance(free_memory_kib, int)
            and not isinstance(free_memory_kib, bool)
            and 0 <= free_memory_kib <= total_memory_kib
        ),
        "windows_free_memory_sufficient": (
            free_memory_gib is not None
            and free_memory_gib >= minimum_windows_free_memory_gib
        ),
        "cad_process_probe_valid": cad_rows_valid,
        "cad_coexistence_allowed": (
            cad_rows_valid and CAD_COEXISTENCE_ALLOWED
        ),
        "gpu_memory_probe_valid": (
            gpu_values_valid
            and gpu_count == 1
            and isinstance(gpu_unaccounted_mib, int)
            and 0 <= gpu_unaccounted_mib <= 1024
        ),
        "gpu_free_memory_sufficient": (
            isinstance(gpu_free_mib, int)
            and not isinstance(gpu_free_mib, bool)
            and gpu_free_mib >= minimum_gpu_free_memory_mib
        ),
    }
    return {
        "schema": SCHEMA,
        "phase": phase,
        "thresholds": {
            "minimum_windows_free_memory_gib": minimum_windows_free_memory_gib,
            "minimum_gpu_free_memory_mib": minimum_gpu_free_memory_mib,
            "launch_minimum_windows_free_memory_gib": (
                LAUNCH_MINIMUM_WINDOWS_FREE_MEMORY_GIB
            ),
            "launch_minimum_gpu_free_memory_mib": (
                LAUNCH_MINIMUM_GPU_FREE_MEMORY_MIB
            ),
            "runtime_minimum_windows_free_memory_gib": (
                RUNTIME_MINIMUM_WINDOWS_FREE_MEMORY_GIB
            ),
            "runtime_minimum_gpu_free_memory_mib": (
                RUNTIME_MINIMUM_GPU_FREE_MEMORY_MIB
            ),
            "cad_coexistence_allowed": CAD_COEXISTENCE_ALLOWED,
            "cad_process_names": list(CAD_PROCESS_NAMES),
        },
        "observed": {
            "windows_total_memory_gib": (
                float(total_memory_kib) / (1024.0 * 1024.0)
                if isinstance(total_memory_kib, int)
                and not isinstance(total_memory_kib, bool)
                else None
            ),
            "windows_free_memory_gib": free_memory_gib,
            "cad_processes": cad_processes if cad_rows_valid else None,
            "gpu_count": gpu_count,
            "gpu_total_memory_mib": gpu_total_mib,
            "gpu_used_memory_mib": gpu_used_mib,
            "gpu_free_memory_mib": gpu_free_mib,
            "gpu_unaccounted_memory_mib": gpu_unaccounted_mib,
        },
        "checks": checks,
        "runtime_started": phase == "runtime",
        "authorization_consumed": phase == "runtime",
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Evaluate a saved synthetic snapshot instead of probing the host.",
    )
    parser.add_argument(
        "--phase",
        choices=("launch", "runtime"),
        default="launch",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        snapshot = (
            _load_object(args.snapshot)
            if args.snapshot is not None
            else probe_live_snapshot()
        )
        result = evaluate_snapshot(snapshot, phase=args.phase)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        result = {
            "schema": SCHEMA,
            "phase": args.phase,
            "error": f"{type(error).__name__}: {error}",
            "runtime_started": args.phase == "runtime",
            "authorization_consumed": args.phase == "runtime",
            "passed": False,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
