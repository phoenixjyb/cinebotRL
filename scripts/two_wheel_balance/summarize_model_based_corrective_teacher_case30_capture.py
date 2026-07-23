#!/usr/bin/env python3
"""Seal one authorized case-30 corrective-label capture without training it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_corrective_capture import (  # noqa: E402
    CORRECTIVE_CAPTURE_ADMISSION_SCHEMA,
    load_corrective_capture,
)


NAMESPACE = "20260722_model_based_corrective_teacher_case30_capture_v2_exclusive"
CAPTURE_NAME = "case_0030_corrective_teacher_capture_v2.npz"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _identity(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _single_result(payload: dict[str, object]) -> dict[str, object]:
    results = payload.get("results", [])
    return results[0] if len(results) == 1 and isinstance(results[0], dict) else {}


def summarize(
    root: Path,
    admission_path: Path,
    *,
    runtime_commit: str,
    playback_exit_code: int,
    gpu_release_passed: bool,
    expected_case: int = 30,
    expected_namespace: str = NAMESPACE,
    capture_name: str = CAPTURE_NAME,
    plan_identity_name: str = "case30_plan",
) -> dict[str, object]:
    contract_path = root / "contract.json"
    gate_path = root / f"case_{expected_case:04d}.json"
    heartbeat_path = root / "runtime_heartbeat.json"
    capture_path = root / "capture" / capture_name
    contract = _load(contract_path)
    admission = _load(admission_path)
    gate = _load(gate_path)
    heartbeat = _load(heartbeat_path)
    result = _single_result(gate)
    capture_metadata: dict[str, object] = {}
    capture_payload: dict[str, np.ndarray] = {}
    capture_error = None
    try:
        capture_metadata, capture_payload = load_corrective_capture(
            capture_path,
            expected_case=expected_case,
            expected_split="train",
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        capture_error = str(exc)

    identities = contract.get("identities", {})
    plan_sha = identities.get(plan_identity_name, {}).get("sha256")
    profile_sha = identities.get("corrective_profile", {}).get("sha256")
    pair_sha = identities.get("paired_final_status", {}).get("sha256")
    sample_count = int(capture_metadata.get("sample_count", 0))
    perturbation_rows = int(
        np.count_nonzero(capture_payload.get("perturbation_active", []))
    )
    requested_normalized_max = (
        np.max(
            np.abs(capture_payload["requested_corrective_normalized_actions"]),
            axis=0,
        ).tolist()
        if capture_payload
        else None
    )
    effective_normalized_max = (
        np.max(
            np.abs(capture_payload["effective_corrective_normalized_actions"]),
            axis=0,
        ).tolist()
        if capture_payload
        else None
    )
    command_clipped_rows = (
        np.count_nonzero(capture_payload["command_clipped"], axis=0).tolist()
        if capture_payload
        else None
    )
    requested_vs_effective_delta_abs_max = (
        np.max(
            np.abs(capture_payload["requested_vs_effective_residual_delta"]),
            axis=0,
        ).tolist()
        if capture_payload
        else None
    )
    amplitude_limited_rows = (
        np.count_nonzero(capture_payload["amplitude_limited"], axis=0).tolist()
        if capture_payload
        else None
    )
    slew_limited_rows = (
        np.count_nonzero(capture_payload["slew_limited"], axis=0).tolist()
        if capture_payload
        else None
    )
    source_clock_end = (
        float(capture_payload["source_time_s"][-1]) if capture_payload else None
    )
    execution_clock_end = (
        float(capture_payload["execution_time_s"][-1]) if capture_payload else None
    )
    source_duration = result.get("source_duration_s")
    execution_duration = result.get("execution_duration_s")
    telemetry = result.get("corrective_teacher_telemetry")
    if not isinstance(telemetry, dict):
        telemetry = {}

    admission_checks = {
        "schema": admission.get("schema") == CORRECTIVE_CAPTURE_ADMISSION_SCHEMA,
        "passed": admission.get("passed") is True,
        "runtime_commit": admission.get("runtime_commit") == runtime_commit,
        "runtime_authorized": admission.get("runtime_authorized") is True,
        "capture_authorized": admission.get("label_capture_authorized") is True,
        "pair_admitted": admission.get("corrective_target_admission_passed") is True,
        "dataset_closed": admission.get("dataset_creation_authorized") is False,
        "training_closed": admission.get("bc_authorized") is False
        and admission.get("ppo_authorized") is False
        and admission.get("training_started") is False,
        "contract_identity": admission.get("contract_sha256")
        == (
            hashlib.sha256(contract_path.read_bytes()).hexdigest()
            if contract_path.is_file()
            else None
        ),
        "plan_identity": admission.get("plan_sha256") == plan_sha,
        "profile_identity": admission.get("corrective_profile_sha256") == profile_sha,
        "pair_identity": admission.get("paired_final_status_sha256") == pair_sha,
    }
    gate_checks = {
        "exit_zero": playback_exit_code == 0,
        "single_case": gate.get("cases") == [expected_case]
        and result.get("case") == expected_case,
        "passed": gate.get("passed") is True and result.get("passed") is True,
        "dynamic": result.get("dynamic_quality_passed") is True,
        "thermal": result.get("thermal_admission_passed") is True,
        "controller": result.get("controller_evidence_passed") is True,
        "perturbation": result.get("perturbation_contract_passed") is True,
        "source": gate.get("trajectory_command_source")
        == "model_based_planner_plus_corrective_teacher",
        "capture_requested": gate.get("corrective_teacher_capture_started") is True
        and gate.get("corrective_teacher_label_capture_authorized") is True,
        "capture_written": result.get("corrective_teacher_labels_captured") is True
        and result.get("executed_corrective_teacher_capture")
        == str(capture_path.resolve()),
        "legacy_capture_closed": gate.get("raw_teacher_capture_started") is False
        and gate.get("normalized_dataset_capture_started") is False
        and gate.get("policy_trace_started") is False
        and gate.get("shadow_teacher_trace_started") is False,
        "training_closed": gate.get("training_started") is False
        and gate.get("ppo_authorized") is False,
        "heartbeat": heartbeat.get("case") == expected_case
        and int(heartbeat.get("completed_steps", 0)) > 0
        and heartbeat.get("capture_outputs_enabled") is True,
        "gpu_released": gpu_release_passed,
    }
    archive_checks = {
        "loaded": capture_error is None and bool(capture_payload),
        "sample_count": sample_count >= 2
        and sample_count == result.get("completed_steps")
        and sample_count == telemetry.get("sample_count")
        and sample_count == heartbeat.get("completed_steps"),
        "plan_identity": capture_metadata.get("plan_sha256") == plan_sha,
        "runtime_identity": capture_metadata.get("runtime_commit") == runtime_commit,
        "profile_identity": capture_metadata.get("corrective_profile_sha256")
        == profile_sha,
        "pair_identity": capture_metadata.get("paired_final_status_sha256") == pair_sha,
        "source_clock": source_clock_end is not None
        and isinstance(source_duration, (int, float))
        and abs(source_clock_end - float(source_duration)) <= 1e-6,
        "execution_clock": execution_clock_end is not None
        and isinstance(execution_duration, (int, float))
        and abs(execution_clock_end - float(execution_duration)) <= 1e-6,
        "perturbation_rows": perturbation_rows == 20,
        "requested_reserved_margin": requested_normalized_max is not None
        and max(requested_normalized_max) < 0.95 - 1e-6,
        "effective_reserved_margin": effective_normalized_max is not None
        and max(effective_normalized_max) < 0.95 - 1e-6,
        "supervisor_contract": capture_metadata.get("safety_supervisor_contract")
        == "requested_teacher_intent_and_effective_applied_command_separate_v1"
        and capture_metadata.get("training_target_contract")
        == "effective_post_supervisor_residual_v1",
        "clipping_telemetry": command_clipped_rows is not None
        and requested_vs_effective_delta_abs_max is not None,
        "initialization_excluded": bool(capture_payload)
        and not bool(np.any(capture_payload["initialization_mask"])),
        "training_closed": capture_metadata.get("valid_for_training") is False
        and capture_metadata.get("normalized_training_dataset_created") is False
        and capture_metadata.get("bc_authorized") is False
        and capture_metadata.get("ppo_authorized") is False
        and capture_metadata.get("training_started") is False,
    }
    contract_checks = {
        "namespace": contract.get("namespace") == expected_namespace,
        "case_split": contract.get("case") == expected_case
        and contract.get("split") == "train",
        "runtime_commit_descends_reviewed_parent": admission.get(
            "reviewed_parent_commit"
        )
        == contract.get("reviewed_parent_commit"),
    }
    passed = (
        all(admission_checks.values())
        and all(gate_checks.values())
        and all(archive_checks.values())
        and all(contract_checks.values())
    )
    return {
        "schema": "cinebotrl_two_wheel_riser_corrective_teacher_capture_final_v2",
        "namespace": expected_namespace,
        "runtime_commit": runtime_commit,
        "case": expected_case,
        "split": "train",
        "playback_exit_code": playback_exit_code,
        "admission_checks": admission_checks,
        "gate_checks": gate_checks,
        "archive_checks": archive_checks,
        "contract_checks": contract_checks,
        "capture_error": capture_error,
        "capture_metrics": {
            "sample_count": sample_count,
            "source_clock_end_s": source_clock_end,
            "execution_clock_end_s": execution_clock_end,
            "perturbation_active_rows": perturbation_rows,
            "requested_normalized_action_abs_max": requested_normalized_max,
            "effective_normalized_action_abs_max": effective_normalized_max,
            "command_clipped_rows": command_clipped_rows,
            "requested_vs_effective_residual_delta_abs_max": (
                requested_vs_effective_delta_abs_max
            ),
            "amplitude_limited_rows": amplitude_limited_rows,
            "slew_limited_rows": slew_limited_rows,
        },
        "gate": _identity(gate_path),
        "heartbeat": _identity(heartbeat_path),
        "capture": _identity(capture_path),
        "admission": _identity(admission_path),
        "contract": _identity(contract_path),
        "dynamic_quality_passed": gate_checks["dynamic"],
        "capture_admitted_for_dataset_conversion": passed,
        "normalized_training_dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
        "passed": passed,
    }


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
