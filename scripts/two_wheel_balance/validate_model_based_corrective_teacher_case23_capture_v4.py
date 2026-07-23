#!/usr/bin/env python3
"""Validate the fresh no-token case-23 corrective-label capture v4 preflight."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
import re
import stat
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.two_wheel_balance.validate_model_based_corrective_teacher_case23_capture import (  # noqa: E402
    ADMISSION_SCHEMA,
    PAIR_SCHEMA,
    drive_profile_checks,
)
from scripts.two_wheel_balance.validate_model_based_corrective_teacher_case30_capture import (  # noqa: E402
    validate as validate_capture,
)


REVIEWED_PARENT = "472130ef622ef90afd6f470783f834d014e41ac0"
NAMESPACE = "20260723_model_based_corrective_teacher_case23_capture_v4_exclusive"
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case23_capture_contract_v4.json"
)
EXPECTED_EXECUTION = {
    "case": 23,
    "split": "train",
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
    authorization_sha256: str | None = None,
) -> dict[str, object]:
    result = validate_capture(
        contract_path,
        repo,
        namespace=namespace,
        authorization_file=None,
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
    profile_checks = drive_profile_checks(result)
    profile_passed = all(profile_checks.values())
    result.setdefault("checks", {})["active_drive_profile"] = profile_passed
    result["drive_profile_checks"] = profile_checks
    result["cpu_contract_ready"] = bool(
        result.get("cpu_contract_ready") and profile_passed
    )
    cpu_passed = bool(result.get("passed") and profile_passed)
    token_present = authorization_file is not None and authorization_file.is_file()
    token_mode = (
        stat.S_IMODE(authorization_file.stat().st_mode)
        if token_present
        else None
    )
    token_hash = (
        hashlib.sha256(authorization_file.read_bytes()).hexdigest()
        if token_present
        else None
    )
    authorization_path_outside_repo = bool(
        token_present
        and not authorization_file.resolve().is_relative_to(repo.resolve())
    )
    authorization_checks = {
        "authorization_file_present": token_present,
        "authorization_mode_0600": token_mode == 0o600,
        "authorization_not_symlink": token_present
        and not authorization_file.is_symlink(),
        "authorization_file_outside_repository": authorization_path_outside_repo,
        "authorization_hash_is_out_of_band": (
            isinstance(authorization_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", authorization_sha256) is not None
            and authorization_sha256
            not in contract_path.read_text(encoding="utf-8")
        ),
        "authorization_hash_matches": (
            token_present
            and isinstance(authorization_sha256, str)
            and hmac.compare_digest(token_hash, authorization_sha256)
        ),
    }
    runtime_authorized = bool(
        cpu_passed
        and authorization_file is not None
        and all(authorization_checks.values())
    )
    result["authorization_checks"] = authorization_checks
    result["authorization_file"] = (
        None
        if authorization_file is None
        else str(authorization_file.resolve())
    )
    result["authorization_consumed_before_isaac"] = runtime_authorized
    result["runtime_authorized"] = runtime_authorized
    result["gpu_launch_authorized"] = runtime_authorized
    result["label_capture_authorized"] = runtime_authorized
    result["passed"] = cpu_passed and (
        authorization_file is None or runtime_authorized
    )
    result["schema"] = ADMISSION_SCHEMA
    return result


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
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
