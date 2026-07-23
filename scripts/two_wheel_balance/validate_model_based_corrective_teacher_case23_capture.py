#!/usr/bin/env python3
"""Validate the no-token case-23 corrective-label capture preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.two_wheel_balance.validate_model_based_corrective_teacher_case30_capture import (  # noqa: E402
    EXPECTED_CAPTURE,
    EXPECTED_HOLDOUT,
    EXPECTED_SCALES,
    SCHEMA,
    validate as validate_capture,
)


ADMISSION_SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_teacher_capture_admission_v2"
)
REVIEWED_PARENT = "48214ba7f868d04f4e0c456f839cd7528cf5bd50"
NAMESPACE = "20260723_model_based_corrective_teacher_case23_capture_v1_exclusive"
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case23_capture_contract_v1.json"
)
PAIR_SCHEMA = "cinebotrl_two_wheel_riser_corrective_teacher_case23_pair_final_v1"
EXPECTED_EXECUTION = {
    "case": 23,
    "rollout": "complete_model_based_planner_plus_corrective_teacher",
    "maximum_runtime_seconds": 600,
    "authorization_consumed_before_isaac": True,
    "fresh_namespace_required": True,
    "exclusive_gpu_required": True,
    "dynamic_gate_required_before_save": True,
    "finalizer_reopens_archive": True,
    "capture_only": True,
}
REQUIRED_IDENTITIES = {
    "paired_final_status",
    "case23_plan",
    "perturbation_profile",
    "corrective_profile",
    "lqr_gains",
    "robot_build_audit",
    "robot_urdf",
    "robot_usd",
    "playback",
    "corrective_teacher_runtime",
    "corrective_capture_runtime",
    "capture_validator_runtime",
    "capture_finalizer_runtime",
    "contract_validator",
    "preflight_wrapper",
    "capture_finalizer",
}
TRACKED_IDENTITIES = REQUIRED_IDENTITIES - {"case23_plan"}


def validate(
    contract_path: Path,
    repo: Path,
    *,
    namespace: str,
    authorization_file: Path | None = None,
) -> dict[str, object]:
    result = validate_capture(
        contract_path,
        repo,
        namespace=namespace,
        authorization_file=authorization_file,
        expected_case=23,
        expected_namespace=NAMESPACE,
        contract_relative_path=CONTRACT_RELATIVE_PATH,
        reviewed_parent=REVIEWED_PARENT,
        plan_identity_name="case23_plan",
        pair_schema=PAIR_SCHEMA,
        required_identities=REQUIRED_IDENTITIES,
        tracked_identities=TRACKED_IDENTITIES,
        expected_execution=EXPECTED_EXECUTION,
    )
    result["schema"] = ADMISSION_SCHEMA
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        args.contract,
        args.repo_root,
        namespace=args.namespace,
        authorization_file=args.authorization_file,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
