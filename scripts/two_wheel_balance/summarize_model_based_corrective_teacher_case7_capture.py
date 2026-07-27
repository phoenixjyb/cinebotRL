#!/usr/bin/env python3
"""Seal one authorized case-7 corrective-label capture without training it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.two_wheel_balance.summarize_model_based_corrective_teacher_case30_capture import (  # noqa: E402
    summarize as summarize_capture,
)
from scripts.two_wheel_balance.check_windows_shared_resource_admission import (  # noqa: E402
    CAD_PROCESS_NAMES,
    MINIMUM_GPU_FREE_MEMORY_MIB,
    MINIMUM_WINDOWS_FREE_MEMORY_GIB,
    SCHEMA as RESOURCE_ADMISSION_SCHEMA,
)


NAMESPACE = "20260724_model_based_corrective_teacher_case7_capture_v1_exclusive"
CAPTURE_NAME = "case_0007_corrective_teacher_capture_v2.npz"
RESOURCE_ADMISSION_NAME = "resource_admission.json"


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _identity(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def validate_resource_admission(
    root: Path,
) -> tuple[dict[str, bool], dict[str, Any]]:
    payload = _load_object(root / RESOURCE_ADMISSION_NAME)
    thresholds = payload.get("thresholds")
    observed = payload.get("observed")
    probe_checks = payload.get("checks")
    thresholds = thresholds if isinstance(thresholds, dict) else {}
    observed = observed if isinstance(observed, dict) else {}
    probe_checks = probe_checks if isinstance(probe_checks, dict) else {}

    cad_processes = observed.get("cad_processes")
    windows_free_gib = observed.get("windows_free_memory_gib")
    gpu_free_mib = observed.get("gpu_free_memory_mib")
    required_probe_checks = {
        "windows_memory_probe_valid",
        "windows_free_memory_sufficient",
        "cad_process_probe_valid",
        "cad_processes_absent",
        "gpu_memory_probe_valid",
        "gpu_free_memory_sufficient",
    }
    checks = {
        "schema": payload.get("schema") == RESOURCE_ADMISSION_SCHEMA,
        "passed": payload.get("passed") is True,
        "generated_before_runtime": payload.get("runtime_started") is False
        and payload.get("authorization_consumed") is False,
        "minimum_windows_free_memory_gib": thresholds.get(
            "minimum_windows_free_memory_gib"
        )
        == MINIMUM_WINDOWS_FREE_MEMORY_GIB,
        "minimum_gpu_free_memory_mib": thresholds.get(
            "minimum_gpu_free_memory_mib"
        )
        == MINIMUM_GPU_FREE_MEMORY_MIB,
        "cad_process_names": thresholds.get("cad_process_names")
        == list(CAD_PROCESS_NAMES),
        "required_probe_checks": set(probe_checks) == required_probe_checks
        and all(probe_checks.get(name) is True for name in required_probe_checks),
        "cad_processes_absent": cad_processes == [],
        "windows_free_memory_sufficient": (
            isinstance(windows_free_gib, (int, float))
            and not isinstance(windows_free_gib, bool)
            and windows_free_gib >= MINIMUM_WINDOWS_FREE_MEMORY_GIB
        ),
        "gpu_free_memory_sufficient": (
            isinstance(gpu_free_mib, int)
            and not isinstance(gpu_free_mib, bool)
            and gpu_free_mib >= MINIMUM_GPU_FREE_MEMORY_MIB
        ),
    }
    return checks, payload


def summarize(
    root: Path,
    admission_path: Path,
    *,
    runtime_commit: str,
    playback_exit_code: int,
    gpu_release_passed: bool,
) -> dict[str, object]:
    result = summarize_capture(
        root,
        admission_path,
        runtime_commit=runtime_commit,
        playback_exit_code=playback_exit_code,
        gpu_release_passed=gpu_release_passed,
        expected_case=7,
        expected_namespace=NAMESPACE,
        capture_name=CAPTURE_NAME,
        plan_identity_name="case7_plan",
    )
    resource_checks, resource_payload = validate_resource_admission(root)
    resource_passed = all(resource_checks.values())
    base_passed = result.get("passed") is True
    base_conversion_admitted = (
        result.get("capture_admitted_for_dataset_conversion") is True
    )
    result["resource_admission_checks"] = resource_checks
    result["resource_admission"] = _identity(root / RESOURCE_ADMISSION_NAME)
    result["resource_admission_observed"] = resource_payload.get("observed")
    result["shared_windows_resource_admission_passed"] = resource_passed
    result["capture_admitted_for_dataset_conversion"] = (
        base_conversion_admitted and resource_passed
    )
    result["passed"] = base_passed and resource_passed
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--playback-exit-code", type=int, required=True)
    parser.add_argument("--gpu-release-passed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.root,
        args.admission,
        runtime_commit=args.runtime_commit,
        playback_exit_code=args.playback_exit_code,
        gpu_release_passed=bool(args.gpu_release_passed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
