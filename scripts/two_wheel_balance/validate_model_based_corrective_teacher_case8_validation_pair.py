#!/usr/bin/env python3
"""Validate the fail-closed case-8 held-out validation pair route."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Mapping


SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_teacher_"
    "case8_validation_pair_contract_v1"
)
ADMISSION_SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_teacher_"
    "case8_validation_pair_admission_v1"
)
REVIEWED_PARENT = "2e83ff1b102f2860988ff6f3f4e6bfc7a399defa"
NAMESPACE = (
    "20260724_model_based_corrective_teacher_case8_validation_pair_v1_exclusive"
)
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case8_validation_pair_contract_v1.json"
)
EXPECTED_SELECTED = [8, 16]
EXPECTED_SOURCE_VALIDATION = [8, 16, 22, 32, 78]
EXPECTED_ELIGIBLE_VALIDATION = [8, 16, 22, 32]
EXPECTED_RESIDUAL_SCALES = [0.05, 0.05, 0.02]
EXPECTED_CONTROLLER_ARGUMENTS = {
    "case": 8,
    "configuration_seed": 20260716,
    "reset_seed": 20260724,
    "controller_wz_kp": 1.05,
    "maximum_duration_scale": 3.0,
    "camera_lever_arm_compensation_enabled": True,
    "camera_lever_arm_compensation_gain": 1.0,
    "maximum_camera_lever_arm_correction_m": 0.05,
    "policy_command_base": "model_based_planner",
    "zero_policy_action": True,
}
EXPECTED_DYNAMIC_THRESHOLDS = {
    "maximum_pitch_deg": 12.0,
    "maximum_position_p95_m": 0.15,
    "maximum_position_error_m": 0.25,
    "maximum_attitude_p95_deg": 5.0,
    "maximum_attitude_error_deg": 10.0,
    "maximum_riser_servo_error_m": 0.03,
    "maximum_proxy_servo_error_deg": 5.0,
    "maximum_internal_proxy_rate_deg_s": 360.0,
    "maximum_saturation_ratio": 0.2,
}
EXPECTED_PAIR_CONTRACT = {
    "baseline": "complete_model_based_planner_plus_exact_zero_residual",
    "candidate": (
        "complete_model_based_planner_plus_case8_validation_corrective"
    ),
    "rollout_order": ["baseline", "candidate"],
    "same_plan_seed_physics_and_perturbation_required": True,
    "candidate_requires_baseline_dynamic_pass": True,
    "teacher_admission_during_pair": False,
    "label_capture_during_pair": False,
    "dataset_creation_during_pair": False,
    "minimum_position_p95_improvement_m": 0.003,
    "minimum_position_p95_relative_improvement": 0.02,
    "maximum_position_error_regression_m": 0.005,
    "maximum_attitude_error_regression_deg": 0.1,
    "maximum_pitch_regression_deg": 0.5,
    "maximum_riser_error_regression_m": 0.002,
    "saturation_regression_allowed": False,
    "maximum_runtime_seconds_per_rollout": 600,
}
REQUIRED_IDENTITIES = {
    "selection",
    "readiness_audit",
    "readiness_auditor",
    "profile_proposal",
    "profile_builder",
    "case8_plan",
    "perturbation_profile",
    "corrective_profile",
    "lqr_gains",
    "robot_build_audit",
    "robot_urdf",
    "robot_usd",
    "playback",
    "corrective_teacher_runtime",
    "perturbation_runtime",
    "validation_assessment",
    "preflight_wrapper",
    "contract_validator",
    "paired_finalizer",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(
    repo: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=check,
    )


def _identity(repo: Path, payload: Mapping[str, object]) -> dict[str, object]:
    path = Path(str(payload.get("path", "")))
    if not path.is_absolute():
        path = repo / path
    exists = path.is_file()
    actual_sha = _sha256(path) if exists else None
    result = _git(repo, "hash-object", str(path), check=False) if exists else None
    actual_blob = (
        result.stdout.strip() if result is not None and result.returncode == 0 else None
    )
    checks = {
        "file_exists": exists,
        "sha256_matches": actual_sha == payload.get("sha256"),
        "git_blob_matches": actual_blob == payload.get("git_blob_sha1"),
    }
    return {
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "git_blob_sha1": actual_blob,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _load(rows: Mapping[str, Mapping[str, object]], name: str) -> dict[str, object]:
    row = rows.get(name, {})
    if row.get("passed") is not True:
        return {}
    return json.loads(Path(str(row["path"])).read_text(encoding="utf-8"))


def _all_closed(payload: Mapping[str, object]) -> bool:
    return all(
        payload.get(field) is False
        for field in (
            "runtime_authorized",
            "gpu_launch_authorized",
            "label_capture_authorized",
            "dataset_conversion_authorized",
            "dataset_merge_authorized",
            "bc_authorized",
            "ppo_authorized",
            "training_started",
            "valid_for_training",
        )
        if field in payload
    )


def validate(
    contract_path: Path,
    repo: Path,
    *,
    namespace: str,
    authorization_file: Path | None = None,
) -> dict[str, object]:
    repo = repo.resolve()
    contract_path = contract_path.resolve()
    canonical_path = (repo / CONTRACT_RELATIVE_PATH).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    identities = contract.get("identities", {})
    rows = {
        name: _identity(repo, value)
        for name, value in identities.items()
        if isinstance(value, Mapping)
    }
    selection = _load(rows, "selection")
    readiness = _load(rows, "readiness_audit")
    proposal = _load(rows, "profile_proposal")
    corrective = _load(rows, "corrective_profile")
    perturbation = _load(rows, "perturbation_profile")
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    upstream = _git(repo, "rev-parse", "@{u}").stdout.strip()
    tracked_clean = (
        _git(repo, "diff", "--quiet", check=False).returncode == 0
        and _git(repo, "diff", "--cached", "--quiet", check=False).returncode
        == 0
    )
    contract_blob = _git(
        repo, "hash-object", str(contract_path), check=False
    ).stdout.strip()
    committed_blob = _git(
        repo, "rev-parse", f"HEAD:{CONTRACT_RELATIVE_PATH}", check=False
    ).stdout.strip()
    case_row = next(
        (
            row
            for row in selection.get("selected_rows", [])
            if isinstance(row, Mapping) and row.get("case") == 8
        ),
        {},
    )
    plan_sha = identities.get("case8_plan", {}).get("sha256")
    proposal_ids = proposal.get("identities", {})
    document_checks = {
        "selection": selection.get("schema")
        == "cinebotrl_two_wheel_riser_model_based_corrective_validation_selection_v1"
        and selection.get("passed") is True
        and selection.get("selected_cases") == EXPECTED_SELECTED
        and selection.get("source_validation_cases") == EXPECTED_SOURCE_VALIDATION
        and selection.get("eligible_validation_cases")
        == EXPECTED_ELIGIBLE_VALIDATION
        and case_row.get("selection_role")
        == "same_seed_validation_paired_canary_required"
        and case_row.get("plan_sha256") == plan_sha
        and _all_closed(selection),
        "readiness": readiness.get("schema")
        == "cinebotrl_two_wheel_riser_case8_validation_pair_readiness_cpu_v1"
        and readiness.get("case") == 8
        and readiness.get("split") == "validation"
        and readiness.get("passed") is True
        and readiness.get("inputs", {}).get("plan", {}).get("sha256")
        == plan_sha
        and readiness.get("case_specific_profile_required") is True
        and readiness.get("case7_profile_reuse_authorized") is False
        and _all_closed(readiness),
        "proposal": proposal.get("schema")
        == (
            "cinebotrl_two_wheel_riser_case8_validation_pair_"
            "profile_proposal_cpu_v1"
        )
        and proposal.get("case") == 8
        and proposal.get("split") == "validation"
        and proposal.get("passed") is True
        and all(proposal.get("input_checks", {}).values())
        and all(proposal.get("shape_checks", {}).values())
        and all(proposal.get("formula_checks", {}).values())
        and all(proposal.get("validation_profile_checks", {}).values())
        and proposal.get("train_profile_reuse_authorized") is False
        and proposal.get("pair_profile_cpu_ready") is True
        and proposal_ids.get("plan", {}).get("sha256") == plan_sha
        and proposal_ids.get("corrective_profile", {}).get("sha256")
        == identities.get("corrective_profile", {}).get("sha256")
        and proposal_ids.get("wrench_profile", {}).get("sha256")
        == identities.get("perturbation_profile", {}).get("sha256")
        and _all_closed(proposal),
        "corrective": corrective.get("case") == 8
        and corrective.get("maximum_residuals")
        == [
            0.015366143432421054,
            0.008147585946902451,
            0.0010025138153900227,
        ],
        "perturbation": perturbation.get("case") == 8
        and perturbation.get("start_phase_time_s") == 2.8513062688185196
        and perturbation.get("duration_steps") == 20
        and perturbation.get("force_body_x_n") == 18.0
        and perturbation.get("application_height_m") == 0.5,
    }
    namespace_path = repo / "artifacts/two_wheel_riser" / namespace
    checks = {
        "schema_case_split": contract.get("schema") == SCHEMA
        and contract.get("case") == 8
        and contract.get("split") == "validation",
        "reviewed_parent_exact": contract.get("reviewed_parent_commit")
        == REVIEWED_PARENT,
        "reviewed_parent_is_ancestor": _git(
            repo,
            "merge-base",
            "--is-ancestor",
            REVIEWED_PARENT,
            head,
            check=False,
        ).returncode
        == 0,
        "head_matches_upstream": head == upstream,
        "tracked_worktree_clean": tracked_clean,
        "canonical_contract_path": contract_path == canonical_path,
        "contract_is_tracked": _git(
            repo,
            "ls-files",
            "--error-unmatch",
            CONTRACT_RELATIVE_PATH,
            check=False,
        ).returncode
        == 0,
        "contract_blob_matches_head": bool(contract_blob)
        and contract_blob == committed_blob,
        "namespace_exact_and_fresh": contract.get("namespace")
        == namespace
        == NAMESPACE
        and not namespace_path.exists(),
        "identity_set_and_hashes": set(identities) == REQUIRED_IDENTITIES
        and len(rows) == len(REQUIRED_IDENTITIES)
        and all(row["passed"] for row in rows.values()),
        "documents": all(document_checks.values()),
        "residual_scales": contract.get("residual_action_scales")
        == EXPECTED_RESIDUAL_SCALES,
        "controller_arguments": contract.get("controller_arguments")
        == EXPECTED_CONTROLLER_ARGUMENTS,
        "dynamic_thresholds": contract.get("unchanged_dynamic_gate_thresholds")
        == EXPECTED_DYNAMIC_THRESHOLDS,
        "pair_contract": contract.get("paired_experiment_contract")
        == EXPECTED_PAIR_CONTRACT,
        "route_complete": contract.get("cpu_preflight_ready") is True
        and contract.get("runtime_route_contract_ready") is True
        and contract.get("execution_route_complete") is True,
        "runtime_authorization_absent": contract.get("runtime_authorized") is False
        and contract.get("gpu_launch_authorized") is False
        and contract.get("authorization_token_issued") is False
        and contract.get("runtime_authorization_token_sha256") == "",
        "learning_closed": contract.get("teacher_admission_authorized") is False
        and contract.get("label_capture_authorized") is False
        and contract.get("dataset_creation_authorized") is False
        and contract.get("bc_authorized") is False
        and contract.get("ppo_authorized") is False
        and contract.get("training_started") is False
        and contract.get("valid_for_training") is False,
        "authorization_file_absent": authorization_file is None,
    }
    passed = all(checks.values())
    return {
        "schema": ADMISSION_SCHEMA,
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "contract_git_blob_sha1": contract_blob,
        "reviewed_parent_commit": REVIEWED_PARENT,
        "runtime_commit": head,
        "upstream_commit": upstream,
        "case": 8,
        "split": "validation",
        "namespace": namespace,
        "identities": rows,
        "document_checks": document_checks,
        "checks": checks,
        "authorization_token_issued": False,
        "cpu_contract_ready": passed,
        "runtime_route_contract_ready": passed,
        "execution_route_complete": passed,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "teacher_admission_authorized": False,
        "label_capture_authorized": False,
        "dataset_creation_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
        "passed": passed,
    }


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
