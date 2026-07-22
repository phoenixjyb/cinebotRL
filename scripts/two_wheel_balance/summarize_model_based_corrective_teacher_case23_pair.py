#!/usr/bin/env python3
"""Seal one paired case-23 baseline/corrective-teacher canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_corrective_teacher import (  # noqa: E402
    assess_paired_corrective_rollouts,
)


NAMESPACE = "20260723_model_based_corrective_teacher_case23_pair_v1_exclusive"
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
        and result.get("executed_residual_dataset") is None
        and result.get("executed_raw_teacher_capture") is None
        and result.get("executed_policy_trace") is None
        and result.get("executed_shadow_teacher_trace") is None
    )


def _rollout_metrics(
    payload: dict[str, object],
    result: dict[str, object],
    *,
    split: str,
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
        "split": split,
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
    baseline_path = root / "baseline/case_0023.json"
    candidate_path = root / "candidate/case_0023.json"
    baseline_heartbeat_path = root / "baseline/runtime_heartbeat.json"
    candidate_heartbeat_path = root / "candidate/runtime_heartbeat.json"
    contract_path = root / "contract.json"
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)
    baseline_heartbeat = _load(baseline_heartbeat_path)
    candidate_heartbeat = _load(candidate_heartbeat_path)
    admission = _load(admission_path)
    contract = _load(contract_path)
    baseline_result = _single_result(baseline)
    candidate_result = _single_result(candidate)
    identities = contract.get("identities", {})
    plan_sha = identities.get("case23_plan", {}).get("sha256")
    controller = contract.get("controller_arguments", {})
    split = contract.get("split")
    physics_seed = controller.get("reset_seed")
    baseline_metrics = _rollout_metrics(
        baseline,
        baseline_result,
        split=split,
        plan_sha256=plan_sha,
        physics_seed=physics_seed,
        candidate=False,
    )
    candidate_metrics = _rollout_metrics(
        candidate,
        candidate_result,
        split=split,
        plan_sha256=plan_sha,
        physics_seed=physics_seed,
        candidate=True,
    )
    try:
        pair_report = assess_paired_corrective_rollouts(
            baseline_metrics, candidate_metrics
        )
    except (TypeError, ValueError) as exc:
        pair_report = {
            "corrective_target_admission_passed": False,
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
        "baseline_single_case": baseline.get("cases") == [23]
        and baseline_result.get("case") == 23,
        "candidate_single_case": candidate.get("cases") == [23]
        and candidate_result.get("case") == 23,
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
        "candidate_profile_present": candidate.get("corrective_teacher_enabled")
        is True
        and candidate.get("corrective_teacher_profile") is not None,
        "baseline_profile_absent": baseline.get("corrective_teacher_enabled")
        is False
        and baseline.get("corrective_teacher_profile") is None,
        "same_perturbation_profile": baseline_perturbation.get("profile")
        == candidate_perturbation.get("profile"),
        "both_perturbations_exact": all(
            telemetry.get("triggered") is True
            and telemetry.get("released_after_pulse") is True
            and telemetry.get("active_step_count")
            == telemetry.get("expected_active_step_count")
            for telemetry in (baseline_perturbation, candidate_perturbation)
        ),
        "perturbation_passed": baseline_result.get("perturbation_contract_passed")
        is True
        and candidate_result.get("perturbation_contract_passed") is True,
        "baseline_capture_closed": _capture_closed(baseline, baseline_result),
        "candidate_capture_closed": _capture_closed(candidate, candidate_result),
        "candidate_labels_not_captured": candidate_result.get(
            "corrective_teacher_labels_captured"
        )
        is False,
        "baseline_heartbeat": baseline_heartbeat.get("case") == 23
        and int(baseline_heartbeat.get("completed_steps", 0)) > 0,
        "candidate_heartbeat": candidate_heartbeat.get("case") == 23
        and int(candidate_heartbeat.get("completed_steps", 0)) > 0,
        "gpu_released": gpu_release_passed,
    }
    contract_checks = {
        "namespace": contract.get("namespace") == NAMESPACE,
        "case_split": contract.get("case") == 23 and split == "train",
        "runtime_commit": admission.get("runtime_commit") == runtime_commit,
        "admission_passed": admission.get("passed") is True,
        "runtime_authorized": admission.get("runtime_authorized") is True,
        "label_capture_closed": admission.get("label_capture_authorized") is False,
        "dataset_closed": admission.get("dataset_creation_authorized") is False,
        "training_closed": admission.get("bc_authorized") is False
        and admission.get("ppo_authorized") is False
        and admission.get("training_started") is False,
        "contract_identity": admission.get("contract_sha256")
        == hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    }
    pair_passed = pair_report.get("corrective_target_admission_passed") is True
    passed = (
        all(rollout_checks.values())
        and all(contract_checks.values())
        and pair_passed
    )
    return {
        "schema": "cinebotrl_two_wheel_riser_corrective_teacher_case23_pair_final_v1",
        "runtime_commit": runtime_commit,
        "namespace": NAMESPACE,
        "case": 23,
        "split": "train",
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
        "admission": _identity(admission_path),
        "contract": _identity(contract_path),
        "dynamic_pair_completed": all(rollout_checks.values()),
        "corrective_target_admission_passed": pair_passed,
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
    parser.add_argument("--gpu-release-passed", type=int, choices=(0, 1), required=True)
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
