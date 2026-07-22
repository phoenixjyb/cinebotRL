#!/usr/bin/env python3
"""Finalize the bounded teacher-41 cases 16/22/32 validation tranche."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ZERO_SOURCE = "zero_policy_action_baseline"
LEARNED_SOURCE = "torchscript_residual_policy"
EXPECTED_CASES = [16, 22, 32]


def identity(path: Path) -> dict[str, str] | None:
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


def _rollout_contract(
    payload: dict[str, Any],
    result: dict[str, Any],
    *,
    case: int,
    source: str,
    source_duration_s: float,
    execution_duration_s: float,
    tracking_profile: str,
) -> bool:
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


def finalize(
    root: Path,
    *,
    contract: dict[str, Any],
    contract_sha256: str,
    runtime_commit: str,
    process_status: dict[str, Any],
    gate_exit_code: int,
) -> dict[str, Any]:
    cases = contract.get("cases", [])
    case_contracts = contract.get("case_contracts", {})
    tracking_profile = contract.get("controller_contract", {}).get("tracking_profile")
    admission_path = root / "admission.json"
    summary_path = root / "summary.json"
    admission = load_json(admission_path)
    summary = load_json(summary_path)
    learned_codes = process_status.get("learned", {})
    zero_codes = process_status.get("zero", {})
    checks: dict[str, bool] = {
        "contract_cases_exact": cases == EXPECTED_CASES,
        "contract_cpu_ready": contract.get("cpu_contract_ready") is True,
        "contract_runtime_was_closed": contract.get("runtime_authorized") is False
        and contract.get("gpu_launch_authorized") is False
        and contract.get("runtime_authorization_token_issued") is False,
        "contract_learning_closed": contract.get("dataset_creation_authorized") is False
        and contract.get("bc_authorized") is False
        and contract.get("ppo_authorized") is False
        and contract.get("holdout_opened") is False,
        "admission_present_and_passed": admission.get("passed") is True
        and admission.get("runtime_commit") == runtime_commit
        and admission.get("cases") == EXPECTED_CASES
        and admission.get("split") == "validation"
        and admission.get("cpu_contract", {}).get("sha256") == contract_sha256
        and admission.get("policy", {}).get("sha256")
        == contract.get("inputs", {}).get("policy_torchscript", {}).get("sha256"),
        "process_status_contract_exact": process_status.get("schema")
        == "cinebotrl_two_wheel_riser_initial_teacher41_validation_tranche_process_status_v1"
        and process_status.get("cases") == EXPECTED_CASES
        and set(process_status) == {"schema", "cases", "learned", "zero"},
    }
    evidence: dict[str, Any] = {
        "admission": identity(admission_path),
        "comparison_summary": identity(summary_path),
        "cases": {},
    }

    for case in EXPECTED_CASES:
        key = str(case)
        expected = case_contracts.get(key, {})
        learned_path = root / f"learned/case_{case:04d}.json"
        zero_path = root / f"zero/case_{case:04d}.json"
        learned_heartbeat_path = root / f"learned/case_{case:04d}_runtime_heartbeat.json"
        zero_heartbeat_path = root / f"zero/case_{case:04d}_runtime_heartbeat.json"
        learned_payload = load_json(learned_path)
        zero_payload = load_json(zero_path)
        learned = _single_result(learned_payload)
        zero = _single_result(zero_payload)
        learned_heartbeat = load_json(learned_heartbeat_path)
        zero_heartbeat = load_json(zero_heartbeat_path)
        checks.update(
            {
                f"case{case}_contract_present": expected.get("source_duration_s")
                is not None
                and expected.get("execution_duration_s") is not None
                and expected.get("camera_lever_arm_cap_m") == 0.05,
                f"case{case}_learned_process_exit_zero": learned_codes.get(key) == 0,
                f"case{case}_learned_rollout_contract": _rollout_contract(
                    learned_payload,
                    learned,
                    case=case,
                    source=LEARNED_SOURCE,
                    source_duration_s=expected.get("source_duration_s"),
                    execution_duration_s=expected.get("execution_duration_s"),
                    tracking_profile=tracking_profile,
                ),
                f"case{case}_learned_dynamic_quality_passed": learned_payload.get(
                    "passed"
                )
                is True
                and learned_payload.get("dynamic_quality_passed") is True
                and learned.get("passed") is True
                and learned.get("dynamic_quality_passed") is True,
                f"case{case}_learned_residual_bounded": bool(learned)
                and max(learned.get("residual_action_abs_max", [float("inf")]))
                <= 1.0 + 1e-6,
                f"case{case}_zero_process_bounded": zero_codes.get(key) in {0, 6},
                f"case{case}_zero_rollout_contract": _rollout_contract(
                    zero_payload,
                    zero,
                    case=case,
                    source=ZERO_SOURCE,
                    source_duration_s=expected.get("source_duration_s"),
                    execution_duration_s=expected.get("execution_duration_s"),
                    tracking_profile=tracking_profile,
                ),
                f"case{case}_zero_action_is_null": zero.get("residual_action_abs_max")
                == [0.0, 0.0, 0.0],
                f"case{case}_heartbeats_present": learned_heartbeat.get("schema")
                == "cinebotrl_two_wheel_riser_runtime_heartbeat_v1"
                and zero_heartbeat.get("schema")
                == "cinebotrl_two_wheel_riser_runtime_heartbeat_v1"
                and learned_heartbeat.get("case") == case
                and zero_heartbeat.get("case") == case,
                f"case{case}_no_dataset_created": learned_heartbeat.get(
                    "dataset_created"
                )
                is False
                and zero_heartbeat.get("dataset_created") is False
                and learned_heartbeat.get("valid_for_training") is False
                and zero_heartbeat.get("valid_for_training") is False,
            }
        )
        evidence["cases"][key] = {
            "learned": identity(learned_path),
            "zero": identity(zero_path),
            "learned_heartbeat": identity(learned_heartbeat_path),
            "zero_heartbeat": identity(zero_heartbeat_path),
        }

    checks.update(
        {
            "comparison_gate_exit_zero": gate_exit_code == 0,
            "comparison_gate_passed": summary.get("passed") is True
            and summary.get("cases") == EXPECTED_CASES
            and summary.get("case_count") == len(EXPECTED_CASES)
            and summary.get("expected_tracking_profile") == tracking_profile
            and summary.get("maximum_regression_fraction") == 0.05
            and summary.get("minimum_zero_improvement_fraction") == 0.05
            and summary.get("policy_sha256")
            == contract.get("inputs", {}).get("policy_torchscript", {}).get("sha256"),
            "comparison_rows_complete": [
                row.get("case") for row in summary.get("rows", [])
            ]
            == EXPECTED_CASES
            and all(
                bool(row.get("checks")) and all(row["checks"].values())
                for row in summary.get("rows", [])
            ),
        }
    )
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    return {
        "schema": "cinebotrl_two_wheel_riser_initial_teacher41_validation_tranche_final_v1",
        "runtime_commit": runtime_commit,
        "cpu_contract_sha256": contract_sha256,
        "cases": EXPECTED_CASES,
        "split": "validation",
        "checks": checks,
        "process_exit_codes": {
            "learned": learned_codes,
            "zero": zero_codes,
            "comparison_gate": gate_exit_code,
        },
        "evidence": evidence,
        "validation_tranche_passed": passed,
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
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--process-status", type=Path, required=True)
    parser.add_argument("--gate-exit-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(
        args.root,
        contract=load_json(args.contract),
        contract_sha256=hashlib.sha256(args.contract.read_bytes()).hexdigest(),
        runtime_commit=args.runtime_commit,
        process_status=load_json(args.process_status),
        gate_exit_code=args.gate_exit_code,
    )
    result["evidence"]["cpu_contract"] = identity(args.contract)
    result["evidence"]["process_status"] = identity(args.process_status)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
