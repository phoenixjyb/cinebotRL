#!/usr/bin/env python3
"""Validate one out-of-band-authorized case-7 corrective-label capture."""

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
    drive_profile_checks,
)
from scripts.two_wheel_balance.validate_model_based_corrective_teacher_case30_capture import (  # noqa: E402
    validate as validate_capture,
)


REVIEWED_PARENT = "b3cffb43d877e0d59eaaed818a9f88e6daa1f968"
NAMESPACE = "20260724_model_based_corrective_teacher_case7_capture_v1_exclusive"
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case7_capture_contract_v1.json"
)
PAIR_SCHEMA = "cinebotrl_two_wheel_riser_corrective_teacher_case7_pair_final_v1"
PROJECTION_SCHEMA = "cinebotrl_two_wheel_riser_case7_pair_projection_audit_v1"
EXPECTED_PROFILE_MAXIMUM_RESIDUALS = [
    0.019165321461451848,
    0.010077209250079967,
    0.0012628383956008627,
]
EXPECTED_EXECUTION = {
    "case": 7,
    "split": "train",
    "rollout": "complete_model_based_planner_plus_corrective_teacher",
    "maximum_runtime_seconds": 600,
    "authorization_consumed_before_isaac": True,
    "fresh_namespace_required": True,
    "exclusive_gpu_required": True,
    "shared_windows_resource_admission_required": True,
    "resource_admission_before_token_consumption": True,
    "minimum_windows_free_memory_gib": 12.0,
    "minimum_gpu_free_memory_mib": 16_384,
    "cad_processes_must_be_absent": True,
    "dynamic_gate_required_before_save": True,
    "finalizer_reopens_archive": True,
    "capture_only": True,
}
REQUIRED_IDENTITIES = {
    "paired_final_status",
    "paired_projection_audit",
    "case7_plan",
    "perturbation_profile",
    "perturbation_runtime",
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
    "shared_windows_resource_guard",
    "capture_finalizer",
}
TRACKED_IDENTITIES = REQUIRED_IDENTITIES


def _projection_checks(result: dict[str, object]) -> dict[str, bool]:
    identities = result.get("identities", {})
    row = identities.get("paired_projection_audit", {})
    payload: dict[str, object] = {}
    if isinstance(row, dict) and row.get("passed") is True:
        payload = json.loads(
            Path(str(row["path"])).read_text(encoding="utf-8")
        )
    baseline = payload.get("baseline", {})
    candidate = payload.get("candidate", {})
    candidate_sample_count = (
        candidate.get("sample_count") if isinstance(candidate, dict) else None
    )
    return {
        "schema_case": payload.get("schema") == PROJECTION_SCHEMA
        and payload.get("case") == 7,
        "pair_projection_passed": payload.get("passed") is True,
        "baseline_exact_zero_passed": isinstance(baseline, dict)
        and baseline.get("enabled") is False
        and baseline.get("passed") is True
        and baseline.get("projection_affected_sample_count") == 0,
        "candidate_projection_passed": isinstance(candidate, dict)
        and candidate.get("enabled") is True
        and candidate.get("passed") is True
        and isinstance(candidate_sample_count, int)
        and not isinstance(candidate_sample_count, bool)
        and candidate_sample_count > 0,
        "observer_did_not_modify_commands": isinstance(candidate, dict)
        and candidate.get("observer_modified_commands") is False
        and candidate.get("applied_to_commands") is False,
        "projection_audit_created_no_labels_or_training": (
            payload.get("labels_captured") is False
            and payload.get("dataset_created") is False
            and payload.get("training_started") is False
            and payload.get("valid_for_training") is False
        ),
    }


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
        expected_case=7,
        expected_namespace=NAMESPACE,
        contract_relative_path=CONTRACT_RELATIVE_PATH,
        reviewed_parent=REVIEWED_PARENT,
        plan_identity_name="case7_plan",
        pair_schema=PAIR_SCHEMA,
        required_identities=REQUIRED_IDENTITIES,
        tracked_identities=TRACKED_IDENTITIES,
        expected_execution=EXPECTED_EXECUTION,
        expected_profile_maximum_residuals=EXPECTED_PROFILE_MAXIMUM_RESIDUALS,
    )
    profile_checks = drive_profile_checks(result)
    projection_checks = _projection_checks(result)
    evidence_passed = all(profile_checks.values()) and all(
        projection_checks.values()
    )
    result.setdefault("checks", {})["active_drive_profile"] = all(
        profile_checks.values()
    )
    result["checks"]["paired_projection_evidence"] = all(
        projection_checks.values()
    )
    result["drive_profile_checks"] = profile_checks
    result["paired_projection_checks"] = projection_checks
    result["cpu_contract_ready"] = bool(
        result.get("cpu_contract_ready") and evidence_passed
    )
    cpu_passed = bool(result.get("passed") and evidence_passed)

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
    authorization_checks = {
        "authorization_file_present": token_present,
        "authorization_mode_0600": token_mode == 0o600,
        "authorization_not_symlink": token_present
        and not authorization_file.is_symlink(),
        "authorization_file_outside_repository": bool(
            token_present
            and not authorization_file.resolve().is_relative_to(repo.resolve())
        ),
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
    result["dataset_creation_authorized"] = False
    result["bc_authorized"] = False
    result["ppo_authorized"] = False
    result["training_started"] = False
    result["valid_for_training"] = False
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
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
