#!/usr/bin/env python3
"""Validate one authorized case-32 natural-error validation capture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.two_wheel_balance import (
    validate_model_based_corrective_teacher_case8_validation_capture as shared,
)

EXPECTED_CAPTURE = shared.EXPECTED_CAPTURE
EXPECTED_HOLDOUT = shared.EXPECTED_HOLDOUT
EXPECTED_SCALES = shared.EXPECTED_SCALES
_authorization_checks = shared._authorization_checks
validate_shared = shared.validate


SCHEMA = "cinebotrl_two_wheel_riser_corrective_teacher_capture_contract_v2"
REVIEWED_PARENT = "93cbe5c9f17ec92d2871ec0a5ed15b45f7989cc0"
NAMESPACE = (
    "20260804_model_based_corrective_teacher_"
    "case32_validation_capture_v1_coexistence"
)
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case32_validation_capture_contract_v1.json"
)
PAIR_SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_teacher_"
    "case32_validation_natural_error_pair_final_v1"
)
EXPECTED_PROFILE_MAXIMUM_RESIDUALS = [
    0.014007529727093336,
    0.00630392556773864,
    0.0008681364516619957,
]
EXPECTED_EXECUTION = {
    "case": 32,
    "split": "validation",
    "rollout": "complete_model_based_planner_plus_corrective_teacher",
    "maximum_runtime_seconds": 600,
    "authorization_consumed_before_isaac": True,
    "fresh_namespace_required": True,
    "exclusive_isaac_runtime_required": True,
    "shared_windows_resource_admission_required": True,
    "resource_admission_before_token_consumption": True,
    "launch_minimum_windows_free_memory_gib": 5.0,
    "launch_minimum_gpu_free_memory_mib": 9_216,
    "cad_coexistence_allowed": True,
    "runtime_resource_monitor_required": True,
    "runtime_minimum_windows_free_memory_gib": 1.5,
    "runtime_minimum_gpu_free_memory_mib": 2_048,
    "dynamic_gate_required_before_save": True,
    "finalizer_reopens_archive": True,
    "external_wrench_forbidden": True,
    "capture_only": True,
}
REQUIRED_IDENTITIES = {
    "paired_final_status",
    "case32_plan",
    "corrective_profile",
    "drive_profile_selection",
    "lqr_gains",
    "robot_build_audit",
    "robot_urdf",
    "robot_usd",
    "playback",
    "corrective_teacher_runtime",
    "corrective_capture_runtime",
    "capture_validator_runtime",
    "capture_finalizer_runtime",
    "resource_finalizer_runtime",
    "contract_validator",
    "preflight_wrapper",
    "shared_windows_resource_guard",
    "shared_windows_resource_monitor",
    "capture_finalizer",
}


def validate(
    contract_path: Path,
    repo: Path,
    *,
    namespace: str,
    authorization_file: Path | None = None,
    authorization_sha256: str | None = None,
) -> dict[str, object]:
    return validate_shared(
        contract_path,
        repo,
        namespace=namespace,
        authorization_file=authorization_file,
        authorization_sha256=authorization_sha256,
        expected_case=32,
        expected_namespace=NAMESPACE,
        contract_relative_path=CONTRACT_RELATIVE_PATH,
        reviewed_parent=REVIEWED_PARENT,
        plan_identity_name="case32_plan",
        pair_schema=PAIR_SCHEMA,
        required_identities=REQUIRED_IDENTITIES,
        expected_profile_maximum_residuals=(
            EXPECTED_PROFILE_MAXIMUM_RESIDUALS
        ),
        expected_execution=EXPECTED_EXECUTION,
        validation_cases_opened=[32],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        args.contract,
        args.repo_root,
        namespace=args.namespace,
        authorization_file=args.authorization_file,
        authorization_sha256=args.authorization_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
