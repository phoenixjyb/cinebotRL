#!/usr/bin/env python3
"""Seal one authorized case-8 validation corrective-label capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.two_wheel_balance.summarize_model_based_corrective_teacher_case30_capture import (  # noqa: E402
    summarize as summarize_capture,
)
from scripts.two_wheel_balance.summarize_model_based_corrective_teacher_case7_capture import (  # noqa: E402
    validate_resource_admission,
    validate_resource_monitor,
)


NAMESPACE = (
    "20260728_model_based_corrective_teacher_"
    "case8_validation_capture_v1_coexistence"
)
CAPTURE_NAME = "case_0008_corrective_teacher_capture_v2.npz"
RESOURCE_ADMISSION_NAME = "resource_admission.json"
RESOURCE_MONITOR_NAME = "resource_monitor.json"


def _identity(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _load(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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
        expected_case=8,
        expected_split="validation",
        expected_namespace=NAMESPACE,
        capture_name=CAPTURE_NAME,
        plan_identity_name="case8_plan",
    )
    resource_checks, resource_payload = validate_resource_admission(root)
    monitor_checks, monitor_payload = validate_resource_monitor(root)
    resource_passed = all(resource_checks.values())
    monitor_passed = all(monitor_checks.values())
    base_passed = result.get("passed") is True
    base_conversion_admitted = (
        result.get("capture_admitted_for_dataset_conversion") is True
    )
    admission = _load(admission_path)
    split_checks = {
        "admission_validation_split": admission.get("split") == "validation",
        "case8_only": admission.get("case") == 8,
        "validation_case_opened": admission.get(
            "corrective_target_admission_passed"
        )
        is True,
        "normalized_dataset_closed": admission.get(
            "dataset_creation_authorized"
        )
        is False,
        "training_closed": admission.get("bc_authorized") is False
        and admission.get("ppo_authorized") is False
        and admission.get("training_started") is False,
    }
    result["validation_capture_checks"] = split_checks
    result["resource_admission_checks"] = resource_checks
    result["resource_admission"] = _identity(root / RESOURCE_ADMISSION_NAME)
    result["resource_admission_observed"] = resource_payload.get("observed")
    result["shared_windows_resource_admission_passed"] = resource_passed
    result["resource_monitor_checks"] = monitor_checks
    result["resource_monitor"] = _identity(root / RESOURCE_MONITOR_NAME)
    result["resource_monitor_summary"] = {
        "sample_count": monitor_payload.get("sample_count"),
        "minimum_observed_windows_free_memory_gib": monitor_payload.get(
            "minimum_observed_windows_free_memory_gib"
        ),
        "minimum_observed_gpu_free_memory_mib": monitor_payload.get(
            "minimum_observed_gpu_free_memory_mib"
        ),
        "termination_requested": monitor_payload.get(
            "termination_requested"
        ),
        "process_exit_observed": monitor_payload.get(
            "process_exit_observed"
        ),
    }
    result["shared_windows_resource_monitor_passed"] = monitor_passed
    passed = (
        base_passed
        and resource_passed
        and monitor_passed
        and all(split_checks.values())
    )
    result["capture_admitted_for_dataset_conversion"] = (
        base_conversion_admitted and passed
    )
    result["conversion_authorized"] = False
    result["dataset_creation_authorized"] = False
    result["bc_authorized"] = False
    result["ppo_authorized"] = False
    result["training_started"] = False
    result["valid_for_training"] = False
    result["passed"] = passed
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
