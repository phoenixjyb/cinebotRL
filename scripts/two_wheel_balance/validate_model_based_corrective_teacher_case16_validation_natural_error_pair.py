#!/usr/bin/env python3
"""Validate the fail-closed case-16 natural-error validation pair route."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from validate_model_based_corrective_teacher_case8_validation_pair import (  # noqa: E402
    _all_closed,
    _git,
    _identity,
    _load,
)


SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_teacher_case16_validation_"
    "natural_error_pair_contract_v1"
)
ADMISSION_SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_teacher_case16_validation_"
    "natural_error_pair_admission_v1"
)
REVIEWED_PARENT = "c92c428785be987ab13e558aa07abc2713a7a0c5"
NAMESPACE = (
    "20260728_model_based_corrective_teacher_case16_validation_"
    "natural_error_pair_v2_coexistence"
)
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/model_based_corrective_teacher_"
    "case16_validation_natural_error_pair_contract_v1.json"
)
EXPECTED_SELECTED = [8, 16]
EXPECTED_SOURCE_VALIDATION = [8, 16, 22, 32, 78]
EXPECTED_ELIGIBLE_VALIDATION = [8, 16, 22, 32]
EXPECTED_RESIDUAL_SCALES = [0.05, 0.05, 0.02]
EXPECTED_MAXIMUM_RESIDUALS = [
    0.004255377317959039,
    0.007046567106873897,
    0.0010219869160504214,
]
EXPECTED_CONTROLLER_ARGUMENTS = {
    "case": 16,
    "configuration_seed": 20260716,
    "reset_seed": 20260732,
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
        "complete_model_based_planner_plus_case16_validation_"
        "natural_error_corrective"
    ),
    "rollout_order": ["baseline", "candidate"],
    "same_plan_seed_and_physics_required": True,
    "external_wrench_forbidden": True,
    "safety_projection_required": True,
    "effective_projection_telemetry_required": True,
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
EXPECTED_RESOURCE_CONTRACT = {
    "shared_windows_resource_admission_required": True,
    "resource_admission_before_token_consumption": True,
    "launch_minimum_windows_free_memory_gib": 5.0,
    "launch_minimum_gpu_free_memory_mib": 9216,
    "cad_coexistence_allowed": True,
    "runtime_resource_monitor_required_per_rollout": True,
    "runtime_minimum_windows_free_memory_gib": 1.5,
    "runtime_minimum_gpu_free_memory_mib": 2048,
}
REQUIRED_IDENTITIES = {
    "selection",
    "readiness_audit",
    "readiness_auditor",
    "profile_proposal",
    "profile_builder",
    "case16_plan",
    "baseline_dynamic_gate",
    "safety_projection_source",
    "corrective_profile",
    "lqr_gains",
    "robot_build_audit",
    "robot_urdf",
    "robot_usd",
    "playback",
    "projection_telemetry_engine",
    "projection_evidence_engine",
    "case16_playback_adapter",
    "corrective_teacher_runtime",
    "validation_assessment",
    "natural_error_finalizer_engine",
    "preflight_wrapper",
    "route_validator_engine",
    "contract_builder",
    "contract_validator",
    "paired_finalizer",
    "shared_windows_resource_guard",
    "shared_windows_resource_monitor",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authorization_checks(
    authorization_file: Path | None,
    authorization_sha256: str | None,
    repo: Path,
    contract_text: str,
) -> dict[str, bool]:
    present = authorization_file is not None and authorization_file.is_file()
    mode = (
        stat.S_IMODE(authorization_file.stat().st_mode)
        if present and authorization_file is not None
        else None
    )
    is_symlink = (
        authorization_file.is_symlink()
        if present and authorization_file is not None
        else False
    )
    actual_hash = _sha256(authorization_file) if present else None
    return {
        "authorization_file_present": present,
        "authorization_mode_0600": mode == 0o600,
        "authorization_not_symlink": present and not is_symlink,
        "authorization_file_outside_repository": present
        and authorization_file is not None
        and not authorization_file.resolve().is_relative_to(repo),
        "authorization_hash_is_out_of_band": (
            isinstance(authorization_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", authorization_sha256) is not None
            and authorization_sha256 not in contract_text
        ),
        "authorization_hash_matches": (
            present
            and isinstance(authorization_sha256, str)
            and hmac.compare_digest(str(actual_hash), authorization_sha256)
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
    repo = repo.resolve()
    contract_path = contract_path.resolve()
    canonical_path = (repo / CONTRACT_RELATIVE_PATH).resolve()
    contract_text = contract_path.read_text(encoding="utf-8")
    contract = json.loads(contract_text)
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
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    upstream = _git(repo, "rev-parse", "@{u}").stdout.strip()
    tracked_clean = (
        _git(repo, "diff", "--quiet", check=False).returncode == 0
        and _git(
            repo, "diff", "--cached", "--quiet", check=False
        ).returncode
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
            if isinstance(row, Mapping) and row.get("case") == 16
        ),
        {},
    )
    plan_sha = identities.get("case16_plan", {}).get("sha256")
    gate_sha = identities.get("baseline_dynamic_gate", {}).get("sha256")
    proposal_ids = proposal.get("identities", {})
    held_out = proposal.get("held_out_validation_contract", {})
    document_checks = {
        "selection": selection.get("schema")
        == (
            "cinebotrl_two_wheel_riser_model_based_corrective_"
            "validation_selection_v1"
        )
        and selection.get("passed") is True
        and selection.get("selected_cases") == EXPECTED_SELECTED
        and selection.get("source_validation_cases")
        == EXPECTED_SOURCE_VALIDATION
        and selection.get("eligible_validation_cases")
        == EXPECTED_ELIGIBLE_VALIDATION
        and case_row.get("selection_role")
        == "same_seed_validation_paired_canary_required"
        and case_row.get("plan_sha256") == plan_sha
        and case_row.get("dynamic_gate_sha256") == gate_sha
        and _all_closed(selection),
        "readiness": readiness.get("schema")
        == (
            "cinebotrl_two_wheel_riser_case16_validation_"
            "pair_readiness_cpu_v1"
        )
        and readiness.get("case") == 16
        and readiness.get("split") == "validation"
        and readiness.get("passed") is True
        and readiness.get("inputs", {}).get("plan", {}).get("sha256")
        == plan_sha
        and readiness.get("inputs", {}).get("dynamic_gate", {}).get("sha256")
        == gate_sha
        and readiness.get("safe_window_absent_requires_structural_profile")
        is True
        and readiness.get("case2_profile_reuse_authorized") is False
        and readiness.get("case7_profile_reuse_authorized") is False
        and _all_closed(readiness),
        "proposal": proposal.get("schema")
        == (
            "cinebotrl_two_wheel_riser_case16_validation_natural_error_"
            "profile_proposal_cpu_v1"
        )
        and proposal.get("case") == 16
        and proposal.get("split") == "validation"
        and proposal.get("passed") is True
        and all(proposal.get("input_checks", {}).values())
        and all(proposal.get("shape_checks", {}).values())
        and all(proposal.get("gate_checks", {}).values())
        and all(proposal.get("formula_checks", {}).values())
        and all(proposal.get("validation_profile_checks", {}).values())
        and proposal.get("validation_pair_profile_cpu_ready") is True
        and proposal.get("train_profile_reuse_authorized") is False
        and proposal.get("natural_error_contract", {}).get(
            "external_wrench_required"
        )
        is False
        and proposal.get("natural_error_contract", {}).get(
            "external_perturbation_forbidden"
        )
        is True
        and held_out.get("teacher_admission_authorized") is False
        and held_out.get("label_capture_authorized") is False
        and held_out.get("dataset_creation_authorized") is False
        and held_out.get("dataset_merge_authorized") is False
        and proposal_ids.get("plan", {}).get("sha256") == plan_sha
        and proposal_ids.get("dynamic_gate", {}).get("sha256") == gate_sha
        and proposal_ids.get("safety_projection_source", {}).get("sha256")
        == identities.get("safety_projection_source", {}).get("sha256")
        and proposal_ids.get("corrective_profile", {}).get("sha256")
        == identities.get("corrective_profile", {}).get("sha256")
        and _all_closed(proposal),
        "corrective": corrective.get("case") == 16
        and corrective.get("maximum_residuals")
        == EXPECTED_MAXIMUM_RESIDUALS,
    }
    namespace_path = repo / "artifacts/two_wheel_riser" / namespace
    authorization_requested = (
        authorization_file is not None or authorization_sha256 is not None
    )
    authorization_checks = _authorization_checks(
        authorization_file,
        authorization_sha256,
        repo,
        contract_text,
    )
    checks = {
        "schema_case_split": contract.get("schema") == SCHEMA
        and contract.get("case") == 16
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
        "dynamic_thresholds": contract.get(
            "unchanged_dynamic_gate_thresholds"
        )
        == EXPECTED_DYNAMIC_THRESHOLDS,
        "pair_contract": contract.get("paired_experiment_contract")
        == EXPECTED_PAIR_CONTRACT,
        "resource_contract": contract.get("shared_resource_contract")
        == EXPECTED_RESOURCE_CONTRACT,
        "route_complete": contract.get("cpu_preflight_ready") is True
        and contract.get("runtime_route_contract_ready") is True
        and contract.get("execution_route_complete") is True,
        "runtime_authorization_absent": contract.get("runtime_authorized")
        is False
        and contract.get("gpu_launch_authorized") is False
        and contract.get("authorization_token_issued") is False
        and contract.get("runtime_authorization_token_sha256") == "",
        "validation_learning_closed": contract.get(
            "teacher_admission_authorized"
        )
        is False
        and contract.get("label_capture_authorized") is False
        and contract.get("dataset_creation_authorized") is False
        and contract.get("bc_authorized") is False
        and contract.get("ppo_authorized") is False
        and contract.get("training_started") is False
        and contract.get("valid_for_training") is False,
        "authorization_state": (
            all(authorization_checks.values())
            if authorization_requested
            else authorization_file is None and authorization_sha256 is None
        ),
    }
    passed = all(checks.values())
    runtime_authorized = authorization_requested and passed
    return {
        "schema": ADMISSION_SCHEMA,
        "contract": str(contract_path),
        "contract_sha256": hashlib.sha256(
            contract_path.read_bytes()
        ).hexdigest(),
        "contract_git_blob_sha1": contract_blob,
        "reviewed_parent_commit": REVIEWED_PARENT,
        "runtime_commit": head,
        "upstream_commit": upstream,
        "case": 16,
        "split": "validation",
        "namespace": namespace,
        "identities": rows,
        "document_checks": document_checks,
        "checks": checks,
        "authorization_checks": authorization_checks,
        "authorization_token_issued": runtime_authorized,
        "cpu_contract_ready": passed,
        "runtime_route_contract_ready": passed,
        "execution_route_complete": passed,
        "runtime_authorized": runtime_authorized,
        "gpu_launch_authorized": runtime_authorized,
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
