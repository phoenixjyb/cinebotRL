#!/usr/bin/env python3
"""Seal one paired case-8 validation baseline/corrective canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_corrective_validation import (  # noqa: E402
    assess_paired_corrective_validation,
)


NAMESPACE = (
    "20260728_model_based_corrective_teacher_case8_validation_pair_v2_coexistence"
)
CASE = 8
SPLIT = "validation"
EXPECTED_SCALES = [0.05, 0.05, 0.02]
EXPECTED_SOURCES = {
    "baseline": "model_based_planner_plus_zero_policy_residual",
    "candidate": "model_based_planner_plus_corrective_teacher",
}


def _identity(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _single_result(payload: dict[str, object]) -> dict[str, object]:
    results = payload.get("results", [])
    return results[0] if len(results) == 1 and isinstance(results[0], dict) else {}


def _capture_closed(payload: dict[str, object], result: dict[str, object]) -> bool:
    return bool(
        payload.get("raw_teacher_capture_started") is False
        and payload.get("normalized_dataset_capture_started") is False
        and payload.get("policy_trace_started") is False
        and payload.get("shadow_teacher_trace_started") is False
        and payload.get("corrective_teacher_capture_started") is False
        and payload.get("corrective_teacher_label_capture_authorized") is False
        and result.get("executed_residual_dataset") is None
        and result.get("executed_raw_teacher_capture") is None
        and result.get("executed_policy_trace") is None
        and result.get("executed_shadow_teacher_trace") is None
        and result.get("executed_corrective_teacher_capture") is None
    )


def _resource_admission_passed(payload: dict[str, object]) -> bool:
    thresholds = payload.get("thresholds", {})
    observed = payload.get("observed", {})
    checks = payload.get("checks", {})
    return bool(
        payload.get("schema") == "cinebotrl_windows_shared_resource_admission_v2"
        and payload.get("phase") == "launch"
        and payload.get("passed") is True
        and isinstance(checks, dict)
        and checks
        and all(checks.values())
        and thresholds.get("minimum_windows_free_memory_gib") == 5.0
        and thresholds.get("minimum_gpu_free_memory_mib") == 9216
        and thresholds.get("cad_coexistence_allowed") is True
        and observed.get("windows_free_memory_gib", 0.0) >= 5.0
        and observed.get("gpu_free_memory_mib", 0) >= 9216
    )


def _resource_monitor_passed(payload: dict[str, object]) -> bool:
    thresholds = payload.get("runtime_thresholds", {})
    minimum_windows_free = payload.get(
        "minimum_observed_windows_free_memory_gib"
    )
    minimum_gpu_free = payload.get("minimum_observed_gpu_free_memory_mib")
    return bool(
        payload.get("schema") == "cinebotrl_windows_shared_resource_monitor_v1"
        and payload.get("passed") is True
        and payload.get("sample_count", 0) > 0
        and payload.get("termination_requested") is False
        and payload.get("process_exit_observed") is True
        and thresholds.get("minimum_windows_free_memory_gib") == 1.5
        and thresholds.get("minimum_gpu_free_memory_mib") == 2048
        and isinstance(minimum_windows_free, (int, float))
        and not isinstance(minimum_windows_free, bool)
        and minimum_windows_free >= 1.5
        and isinstance(minimum_gpu_free, int)
        and not isinstance(minimum_gpu_free, bool)
        and minimum_gpu_free >= 2048
    )


def _metrics(
    payload: dict[str, object],
    result: dict[str, object],
    *,
    plan_sha256: str,
    physics_seed: int,
    candidate: bool,
) -> dict[str, object]:
    telemetry = result.get("corrective_teacher_telemetry", {})
    normalized_max = (
        telemetry.get("normalized_action_abs_max")
        if candidate
        else result.get("residual_action_abs_max")
    )
    return {
        "case": result.get("case"),
        "split": SPLIT,
        "plan_sha256": plan_sha256,
        "physics_seed": physics_seed,
        "source_duration_s": result.get("source_duration_s"),
        "execution_duration_s": result.get("execution_duration_s"),
        "dynamic_quality_passed": result.get("dynamic_quality_passed"),
        "position_error_p95_m": result.get("position_error_p95_m"),
        "position_error_max_m": result.get("position_error_max_m"),
        "attitude_error_max_deg": result.get("attitude_error_max_deg"),
        "pitch_max_deg": result.get("pitch_max_deg"),
        "riser_error_max_m": result.get("riser_servo_error_max_m"),
        "action_saturation_ratio": result.get("action_saturation_ratio"),
        "normalized_residual_action_abs_max": normalized_max,
        "dataset_created": False,
        "training_started": False,
        "ppo_started": False,
    }


def summarize(
    root: Path,
    admission_path: Path,
    *,
    runtime_commit: str,
    baseline_exit_code: int,
    candidate_exit_code: int,
    gpu_release_passed: bool,
) -> dict[str, object]:
    baseline_path = root / "baseline/case_0008.json"
    candidate_path = root / "candidate/case_0008.json"
    baseline_heartbeat_path = root / "baseline/runtime_heartbeat.json"
    candidate_heartbeat_path = root / "candidate/runtime_heartbeat.json"
    resource_admission_path = root / "resource_admission.json"
    baseline_resource_monitor_path = root / "baseline/resource_monitor.json"
    candidate_resource_monitor_path = root / "candidate/resource_monitor.json"
    contract_path = root / "contract.json"
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)
    baseline_result = _single_result(baseline)
    candidate_result = _single_result(candidate)
    baseline_heartbeat = _load(baseline_heartbeat_path)
    candidate_heartbeat = _load(candidate_heartbeat_path)
    resource_admission = _load(resource_admission_path)
    baseline_resource_monitor = _load(baseline_resource_monitor_path)
    candidate_resource_monitor = _load(candidate_resource_monitor_path)
    admission = _load(admission_path)
    contract = _load(contract_path)
    identities = contract.get("identities", {})
    plan_sha = identities.get("case8_plan", {}).get("sha256")
    corrective_sha = identities.get("corrective_profile", {}).get("sha256")
    perturbation_sha = identities.get("perturbation_profile", {}).get("sha256")
    physics_seed = contract.get("controller_arguments", {}).get("reset_seed")
    baseline_metrics = _metrics(
        baseline,
        baseline_result,
        plan_sha256=plan_sha,
        physics_seed=physics_seed,
        candidate=False,
    )
    candidate_metrics = _metrics(
        candidate,
        candidate_result,
        plan_sha256=plan_sha,
        physics_seed=physics_seed,
        candidate=True,
    )
    try:
        pair_report = assess_paired_corrective_validation(
            baseline_metrics, candidate_metrics
        )
    except (TypeError, ValueError) as exc:
        pair_report = {
            "validation_pair_passed": False,
            "error": str(exc),
        }
    baseline_perturbation = baseline_result.get(
        "deterministic_wrench_perturbation", {}
    )
    candidate_perturbation = candidate_result.get(
        "deterministic_wrench_perturbation", {}
    )
    rollout_checks = {
        "baseline_exit_zero": baseline_exit_code == 0,
        "candidate_exit_zero": candidate_exit_code == 0,
        "baseline_single_case": baseline.get("cases") == [CASE]
        and baseline_result.get("case") == CASE,
        "candidate_single_case": candidate.get("cases") == [CASE]
        and candidate_result.get("case") == CASE,
        "baseline_source": baseline.get("trajectory_command_source")
        == EXPECTED_SOURCES["baseline"],
        "candidate_source": candidate.get("trajectory_command_source")
        == EXPECTED_SOURCES["candidate"],
        "model_based_command_base": baseline.get("policy_command_base")
        == candidate.get("policy_command_base")
        == "model_based_planner",
        "residual_scales": baseline.get("residual_action_scales")
        == candidate.get("residual_action_scales")
        == EXPECTED_SCALES,
        "baseline_zero_action": baseline_result.get("residual_action_abs_max")
        == [0.0, 0.0, 0.0],
        "candidate_profile_bound": candidate.get(
            "corrective_teacher_enabled"
        )
        is True
        and candidate.get("corrective_teacher_profile", {}).get("sha256")
        == corrective_sha,
        "baseline_profile_absent": baseline.get("corrective_teacher_enabled")
        is False
        and baseline.get("corrective_teacher_profile") is None,
        "perturbation_profile_bound": baseline.get(
            "deterministic_wrench_profile", {}
        ).get("sha256")
        == candidate.get("deterministic_wrench_profile", {}).get("sha256")
        == perturbation_sha,
        "same_perturbation_profile": baseline_perturbation.get("profile")
        == candidate_perturbation.get("profile"),
        "both_perturbations_exact": all(
            telemetry.get("triggered") is True
            and telemetry.get("released_after_pulse") is True
            and telemetry.get("active_step_count")
            == telemetry.get("expected_active_step_count")
            == 20
            for telemetry in (baseline_perturbation, candidate_perturbation)
        ),
        "perturbation_passed": baseline_result.get(
            "perturbation_contract_passed"
        )
        is True
        and candidate_result.get("perturbation_contract_passed") is True,
        "baseline_capture_closed": _capture_closed(baseline, baseline_result),
        "candidate_capture_closed": _capture_closed(candidate, candidate_result),
        "candidate_labels_not_captured": candidate_result.get(
            "corrective_teacher_labels_captured"
        )
        is False,
        "baseline_heartbeat": baseline_heartbeat.get("case") == CASE
        and int(baseline_heartbeat.get("completed_steps", 0)) > 0,
        "candidate_heartbeat": candidate_heartbeat.get("case") == CASE
        and int(candidate_heartbeat.get("completed_steps", 0)) > 0,
        "gpu_released": gpu_release_passed,
        "shared_resource_admission": _resource_admission_passed(
            resource_admission
        ),
        "baseline_resource_monitor": _resource_monitor_passed(
            baseline_resource_monitor
        ),
        "candidate_resource_monitor": _resource_monitor_passed(
            candidate_resource_monitor
        ),
    }
    contract_checks = {
        "namespace": contract.get("namespace") == NAMESPACE,
        "case_split": contract.get("case") == CASE
        and contract.get("split") == SPLIT,
        "runtime_commit": admission.get("runtime_commit") == runtime_commit,
        "admission_passed": admission.get("passed") is True,
        "runtime_authorized": admission.get("runtime_authorized") is True,
        "gpu_launch_authorized": admission.get("gpu_launch_authorized") is True,
        "label_capture_closed": admission.get("label_capture_authorized") is False,
        "dataset_closed": admission.get("dataset_creation_authorized") is False,
        "training_closed": admission.get("bc_authorized") is False
        and admission.get("ppo_authorized") is False
        and admission.get("training_started") is False,
        "contract_identity": admission.get("contract_sha256")
        == hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    }
    pair_passed = pair_report.get("validation_pair_passed") is True
    passed = (
        all(rollout_checks.values())
        and all(contract_checks.values())
        and pair_passed
    )
    return {
        "schema": (
            "cinebotrl_two_wheel_riser_corrective_teacher_"
            "case8_validation_pair_final_v1"
        ),
        "runtime_commit": runtime_commit,
        "namespace": NAMESPACE,
        "case": CASE,
        "split": SPLIT,
        "baseline_exit_code": baseline_exit_code,
        "candidate_exit_code": candidate_exit_code,
        "rollout_checks": rollout_checks,
        "contract_checks": contract_checks,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "paired_admission": pair_report,
        "baseline": _identity(baseline_path),
        "candidate": _identity(candidate_path),
        "baseline_heartbeat": _identity(baseline_heartbeat_path),
        "candidate_heartbeat": _identity(candidate_heartbeat_path),
        "resource_admission": _identity(resource_admission_path),
        "baseline_resource_monitor": _identity(
            baseline_resource_monitor_path
        ),
        "candidate_resource_monitor": _identity(
            candidate_resource_monitor_path
        ),
        "admission": _identity(admission_path),
        "contract": _identity(contract_path),
        "dynamic_pair_completed": all(rollout_checks.values()),
        "validation_pair_passed": pair_passed,
        "teacher_admission_opened": False,
        "label_capture_authorized": False,
        "dataset_created": False,
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
    parser.add_argument("--baseline-exit-code", type=int, required=True)
    parser.add_argument("--candidate-exit-code", type=int, required=True)
    parser.add_argument(
        "--gpu-release-passed", type=int, choices=(0, 1), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.root,
        args.admission,
        runtime_commit=args.runtime_commit,
        baseline_exit_code=args.baseline_exit_code,
        candidate_exit_code=args.candidate_exit_code,
        gpu_release_passed=bool(args.gpu_release_passed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
