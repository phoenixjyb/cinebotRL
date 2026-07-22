#!/usr/bin/env python3
"""Finalize a bounded teacher-41 validation-case residual-policy canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ZERO_SOURCE = "zero_policy_action_baseline"
LEARNED_SOURCE = "torchscript_residual_policy"


def identity(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _single_result(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", [])
    return results[0] if isinstance(results, list) and len(results) == 1 else {}


def finalize(
    root: Path,
    *,
    case: int,
    source_duration_s: float,
    execution_duration_s: float,
    tracking_profile: str,
    runtime_commit: str,
    learned_exit_code: int,
    zero_exit_code: int,
    gate_exit_code: int,
) -> dict[str, object]:
    padded = f"{case:04d}"
    admission_path = root / "admission.json"
    learned_path = root / f"learned/case_{padded}.json"
    zero_path = root / f"zero/case_{padded}.json"
    summary_path = root / "summary.json"
    learned_heartbeat_path = root / "learned/runtime_heartbeat.json"
    zero_heartbeat_path = root / "zero/runtime_heartbeat.json"
    admission = load_json(admission_path)
    learned_payload = load_json(learned_path)
    zero_payload = load_json(zero_path)
    summary = load_json(summary_path)
    learned_heartbeat = load_json(learned_heartbeat_path)
    zero_heartbeat = load_json(zero_heartbeat_path)
    learned = _single_result(learned_payload)
    zero = _single_result(zero_payload)

    def rollout_contract(payload: dict[str, Any], result: dict[str, Any], source: str) -> bool:
        return (
            payload.get("cases") == [case]
            and payload.get("trajectory_command_source") == source
            and payload.get("tracking_profile") == tracking_profile
            and payload.get("phase_feedforward_contract")
            == "derivatives_scaled_by_progress_v1"
            and payload.get("raw_teacher_capture_started") is False
            and payload.get("normalized_dataset_capture_started") is False
            and result.get("case") == case
            and result.get("source_duration_s") == source_duration_s
            and result.get("execution_duration_s") == execution_duration_s
            and result.get("executed_residual_dataset") is None
        )

    checks = {
        "admission_present_and_passed": admission.get("passed") is True
        and admission.get("runtime_commit") == runtime_commit
        and admission.get("case") == case
        and admission.get("split") == "validation",
        "learned_process_exit_zero": learned_exit_code == 0,
        "learned_rollout_contract": rollout_contract(
            learned_payload, learned, LEARNED_SOURCE
        ),
        "learned_dynamic_quality_passed": learned_payload.get("passed") is True
        and learned_payload.get("dynamic_quality_passed") is True
        and learned.get("passed") is True
        and learned.get("dynamic_quality_passed") is True,
        "learned_residual_bounded": bool(learned)
        and max(learned.get("residual_action_abs_max", [float("inf")])) <= 1.0 + 1e-6,
        "zero_process_bounded": zero_exit_code in {0, 6},
        "zero_baseline_recorded": rollout_contract(zero_payload, zero, ZERO_SOURCE),
        "zero_action_is_null": zero.get("residual_action_abs_max") == [0.0, 0.0, 0.0],
        "comparison_gate_exit_zero": gate_exit_code == 0,
        "comparison_gate_passed": summary.get("passed") is True
        and summary.get("cases") == [case]
        and summary.get("case_count") == 1
        and summary.get("expected_tracking_profile") == tracking_profile
        and summary.get("maximum_regression_fraction") == 0.05
        and summary.get("minimum_zero_improvement_fraction") == 0.05,
        "runtime_heartbeats_present": learned_heartbeat.get("schema")
        == "cinebotrl_two_wheel_riser_runtime_heartbeat_v1"
        and zero_heartbeat.get("schema")
        == "cinebotrl_two_wheel_riser_runtime_heartbeat_v1"
        and learned_heartbeat.get("case") == case
        and zero_heartbeat.get("case") == case,
        "no_dataset_created": learned_heartbeat.get("dataset_created") is False
        and zero_heartbeat.get("dataset_created") is False
        and learned_heartbeat.get("valid_for_training") is False
        and zero_heartbeat.get("valid_for_training") is False,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    return {
        "schema": "cinebotrl_two_wheel_riser_initial_teacher41_validation_canary_final_v1",
        "runtime_commit": runtime_commit,
        "case": case,
        "split": "validation",
        "checks": checks,
        "process_exit_codes": {
            "learned": learned_exit_code,
            "zero": zero_exit_code,
            "comparison_gate": gate_exit_code,
        },
        "evidence": {
            "admission": identity(admission_path),
            "learned": identity(learned_path),
            "zero": identity(zero_path),
            "comparison_summary": identity(summary_path),
            "learned_heartbeat": identity(learned_heartbeat_path),
            "zero_heartbeat": identity(zero_heartbeat_path),
        },
        "dynamic_canary_passed": passed,
        "remaining_validation_cases_authorized": False,
        "broad_rollout_authorized": False,
        "dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "ppo_started": False,
        "holdout_opened": False,
        "valid_for_training": False,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--source-duration-s", type=float, required=True)
    parser.add_argument("--execution-duration-s", type=float, required=True)
    parser.add_argument("--tracking-profile", required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--learned-exit-code", type=int, required=True)
    parser.add_argument("--zero-exit-code", type=int, required=True)
    parser.add_argument("--gate-exit-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(
        args.root,
        case=args.case,
        source_duration_s=args.source_duration_s,
        execution_duration_s=args.execution_duration_s,
        tracking_profile=args.tracking_profile,
        runtime_commit=args.runtime_commit,
        learned_exit_code=args.learned_exit_code,
        zero_exit_code=args.zero_exit_code,
        gate_exit_code=args.gate_exit_code,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
