#!/usr/bin/env python3
"""Validate the CPU-only, fail-closed case-2 natural-error pair contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_teacher_"
    "case2_natural_error_pair_contract_v1"
)
ADMISSION_SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_teacher_"
    "case2_natural_error_pair_admission_v1"
)
REVIEWED_PARENT = "5f6725817ea80aaf2e6568e34ceedd3248cd1593"
NAMESPACE = (
    "20260724_model_based_corrective_teacher_case2_"
    "natural_error_pair_v1_exclusive"
)
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case2_natural_error_pair_contract_v1.json"
)
EXPECTED_HOLDOUT = [3, 5, 13, 19, 24]
EXPECTED_VALIDATION = [8, 16, 22, 32, 78]
EXPECTED_SELECTED = [30, 23, 6, 2, 7]
REQUIRED_IDENTITIES = {
    "selection",
    "readiness_audit",
    "readiness_auditor",
    "profile_proposal",
    "profile_builder",
    "case2_plan",
    "baseline_dynamic_gate",
    "safety_projection_source",
    "corrective_profile",
    "lqr_gains",
    "robot_build_audit",
    "robot_urdf",
    "robot_usd",
    "playback",
    "case2_playback_adapter",
    "corrective_teacher_runtime",
    "preflight_wrapper",
    "contract_validator",
    "paired_finalizer",
}
EXPECTED_RESIDUAL_SCALES = [0.05, 0.05, 0.02]
EXPECTED_CONTROLLER_ARGUMENTS = {
    "case": 2,
    "configuration_seed": 20260716,
    "reset_seed": 20260718,
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
        "complete_model_based_planner_plus_case2_natural_error_"
        "corrective_teacher"
    ),
    "rollout_order": ["baseline", "candidate"],
    "same_plan_seed_and_physics_required": True,
    "external_wrench_forbidden": True,
    "safety_projection_required": True,
    "effective_projection_telemetry_required": True,
    "candidate_requires_baseline_dynamic_pass": True,
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(
    repo: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=check,
    )


def identity_row(repo: Path, payload: dict[str, object]) -> dict[str, object]:
    path = Path(str(payload.get("path", "")))
    if not path.is_absolute():
        path = repo / path
    exists = path.is_file()
    actual_sha = sha256_file(path) if exists else None
    result = git(repo, "hash-object", str(path), check=False) if exists else None
    actual_blob = (
        result.stdout.strip()
        if result is not None and result.returncode == 0
        else None
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


def _load_json(
    rows: dict[str, dict[str, object]], name: str
) -> dict[str, object]:
    row = rows.get(name, {})
    if row.get("passed") is not True:
        return {}
    return json.loads(Path(str(row["path"])).read_text(encoding="utf-8"))


def _closed(payload: dict[str, object]) -> bool:
    return all(
        payload.get(field) is False
        for field in (
            "runtime_authorized",
            "gpu_launch_authorized",
            "label_capture_authorized",
            "bc_authorized",
            "ppo_authorized",
            "training_started",
            "valid_for_training",
        )
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
        name: identity_row(repo, value)
        for name, value in identities.items()
        if isinstance(value, dict)
    }
    selection = _load_json(rows, "selection")
    readiness = _load_json(rows, "readiness_audit")
    proposal = _load_json(rows, "profile_proposal")
    corrective = _load_json(rows, "corrective_profile")

    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    upstream = git(repo, "rev-parse", "@{u}").stdout.strip()
    tracked_clean = (
        git(repo, "diff", "--quiet", check=False).returncode == 0
        and git(repo, "diff", "--cached", "--quiet", check=False).returncode
        == 0
    )
    contract_tracked = (
        git(
            repo,
            "ls-files",
            "--error-unmatch",
            CONTRACT_RELATIVE_PATH,
            check=False,
        ).returncode
        == 0
    )
    contract_blob = git(
        repo, "hash-object", str(contract_path), check=False
    ).stdout.strip()
    committed_blob = git(
        repo, "rev-parse", f"HEAD:{CONTRACT_RELATIVE_PATH}", check=False
    ).stdout.strip()
    reviewed_parent_is_ancestor = (
        git(
            repo,
            "merge-base",
            "--is-ancestor",
            REVIEWED_PARENT,
            head,
            check=False,
        ).returncode
        == 0
    )
    namespace_path = repo / "artifacts/two_wheel_riser" / namespace
    selected_case2 = next(
        (
            row
            for row in selection.get("selected_rows", [])
            if isinstance(row, dict) and row.get("case") == 2
        ),
        {},
    )
    readiness_inputs = readiness.get("inputs", {})
    proposal_identities = proposal.get("identities", {})

    selection_checks = {
        "schema_passed": selection.get("schema")
        == "cinebotrl_two_wheel_riser_model_based_pair_tranche_selection_v1"
        and selection.get("passed") is True,
        "selected_cases": selection.get("selected_cases") == EXPECTED_SELECTED,
        "case2_requires_pair": selected_case2.get("selection_role")
        == "same_seed_paired_canary_required",
        "plan_bound": selected_case2.get("plan_sha256")
        == identities.get("case2_plan", {}).get("sha256"),
        "gate_bound": selected_case2.get("dynamic_gate_sha256")
        == identities.get("baseline_dynamic_gate", {}).get("sha256"),
        "splits_closed": selection.get("validation_cases")
        == EXPECTED_VALIDATION
        and selection.get("holdout_cases") == EXPECTED_HOLDOUT,
        "runtime_learning_closed": selection.get("runtime_authorized") is False
        and selection.get("gpu_launch_authorized") is False
        and selection.get("label_capture_authorized") is False
        and selection.get("dataset_merge_authorized") is False
        and selection.get("bc_authorized") is False
        and selection.get("ppo_authorized") is False
        and selection.get("training_started") is False,
    }
    readiness_checks = {
        "schema_case_passed": readiness.get("schema")
        == "cinebotrl_two_wheel_riser_case2_pair_readiness_cpu_v1"
        and readiness.get("case") == 2
        and readiness.get("passed") is True,
        "all_checks_passed": all(
            all(readiness.get(section, {}).values())
            for section in (
                "selection_checks",
                "plan_checks",
                "gate_checks",
                "metric_checks",
                "source_execution_clock_checks",
            )
        ),
        "plan_bound": readiness_inputs.get("plan", {}).get("sha256")
        == identities.get("case2_plan", {}).get("sha256"),
        "gate_bound": readiness_inputs.get("dynamic_gate", {}).get("sha256")
        == identities.get("baseline_dynamic_gate", {}).get("sha256"),
        "structural_profile_required": readiness.get(
            "safe_window_absent_requires_structural_profile"
        )
        is True
        and readiness.get("case23_profile_reuse_authorized") is False
        and readiness.get("case6_profile_reuse_authorized") is False,
        "runtime_learning_closed": _closed(readiness)
        and readiness.get("dataset_conversion_authorized") is False
        and readiness.get("dataset_merge_authorized") is False,
    }
    proposal_checks = {
        "schema_case_passed": proposal.get("schema")
        == (
            "cinebotrl_two_wheel_riser_case2_"
            "natural_error_profile_proposal_cpu_v1"
        )
        and proposal.get("case") == 2
        and proposal.get("split") == "train"
        and proposal.get("passed") is True,
        "all_checks_passed": all(proposal.get("input_checks", {}).values())
        and all(proposal.get("shape_checks", {}).values())
        and all(proposal.get("gate_checks", {}).values())
        and all(proposal.get("formula_checks", {}).values()),
        "plan_bound": proposal_identities.get("plan", {}).get("sha256")
        == identities.get("case2_plan", {}).get("sha256"),
        "gate_bound": proposal_identities.get("dynamic_gate", {}).get("sha256")
        == identities.get("baseline_dynamic_gate", {}).get("sha256"),
        "projection_bound": proposal_identities.get(
            "safety_projection_source", {}
        ).get("sha256")
        == identities.get("safety_projection_source", {}).get("sha256"),
        "profile_bound": proposal_identities.get(
            "corrective_profile", {}
        ).get("sha256")
        == identities.get("corrective_profile", {}).get("sha256"),
        "natural_error_no_wrench": proposal.get(
            "natural_error_pair_profile_cpu_ready"
        )
        is True
        and proposal.get("natural_error_contract", {}).get(
            "external_wrench_required"
        )
        is False
        and proposal.get("natural_error_contract", {}).get(
            "external_wrench_profile_created"
        )
        is False,
        "effective_projection_required": proposal.get(
            "natural_error_contract", {}
        ).get("capture_labels_must_use_effective_projected_residual")
        is True,
        "profile_ready_route_closed": proposal.get(
            "pair_profile_cpu_ready"
        )
        is True
        and proposal.get("runtime_route_implemented") is False
        and proposal.get("authorization_token_issued") is False,
        "runtime_learning_closed": _closed(proposal)
        and proposal.get("dataset_conversion_authorized") is False
        and proposal.get("dataset_merge_authorized") is False,
    }
    corrective_checks = {
        "schema": corrective.get("schema")
        == "cinebotrl_two_wheel_riser_corrective_teacher_profile_v1",
        "case": corrective.get("case") == 2,
        "maximum_residuals": corrective.get("maximum_residuals")
        == [
            0.010247377951381045,
            0.004541053111745168,
            0.000614431056640985,
        ],
        "maximum_slew_rates": corrective.get("maximum_slew_rates")
        == [
            0.025618444878452613,
            0.011352632779362921,
            0.0015360776416024624,
        ],
    }
    checks = {
        "schema_matches": contract.get("schema") == SCHEMA,
        "case_is_2_train": contract.get("case") == 2
        and contract.get("split") == "train",
        "reviewed_parent_exact": contract.get("reviewed_parent_commit")
        == REVIEWED_PARENT,
        "reviewed_parent_is_ancestor": reviewed_parent_is_ancestor,
        "head_matches_upstream": head == upstream,
        "tracked_worktree_clean": tracked_clean,
        "canonical_contract_path": contract_path == canonical_path,
        "contract_is_tracked": contract_tracked,
        "contract_blob_matches_head": bool(contract_blob)
        and contract_blob == committed_blob,
        "namespace_exact_and_fresh": contract.get("namespace")
        == namespace
        == NAMESPACE
        and not namespace_path.exists(),
        "all_identity_hashes_match": set(identities) == REQUIRED_IDENTITIES
        and len(rows) == len(REQUIRED_IDENTITIES)
        and all(row["passed"] for row in rows.values()),
        "all_identity_blobs_pinned": all(
            isinstance(identities.get(name), dict)
            and bool(identities[name].get("git_blob_sha1"))
            for name in REQUIRED_IDENTITIES
        ),
        "no_perturbation_identity": all(
            "wrench" not in name and "perturbation" not in name
            for name in identities
        ),
        "selection_contract": all(selection_checks.values()),
        "readiness_contract": all(readiness_checks.values()),
        "profile_proposal_contract": all(proposal_checks.values()),
        "corrective_profile_contract": all(corrective_checks.values()),
        "residual_scales_exact": contract.get("residual_action_scales")
        == EXPECTED_RESIDUAL_SCALES,
        "controller_arguments_exact": contract.get("controller_arguments")
        == EXPECTED_CONTROLLER_ARGUMENTS,
        "dynamic_thresholds_exact": contract.get(
            "unchanged_dynamic_gate_thresholds"
        )
        == EXPECTED_DYNAMIC_THRESHOLDS,
        "pair_contract_exact": contract.get("paired_experiment_contract")
        == EXPECTED_PAIR_CONTRACT,
        "cpu_preflight_ready": contract.get("cpu_preflight_ready") is True,
        "runtime_route_contract_ready": contract.get(
            "runtime_route_contract_ready"
        )
        is True,
        "execution_route_complete": contract.get(
            "execution_route_complete"
        )
        is True,
        "runtime_authorization_absent": contract.get(
            "runtime_authorized"
        )
        is False
        and contract.get("gpu_launch_authorized") is False
        and contract.get("authorization_token_issued") is False
        and contract.get("runtime_authorization_token_sha256") == "",
        "capture_and_training_closed": contract.get(
            "label_capture_authorized"
        )
        is False
        and contract.get("dataset_creation_authorized") is False
        and contract.get("bc_authorized") is False
        and contract.get("ppo_authorized") is False
        and contract.get("training_started") is False
        and contract.get("valid_for_training") is False,
        "validation_holdout_closed": contract.get("validation_cases")
        == EXPECTED_VALIDATION
        and contract.get("validation_opened") is False
        and contract.get("holdout_cases") == EXPECTED_HOLDOUT
        and contract.get("holdout_opened") is False,
        "authorization_file_absent": authorization_file is None,
    }
    cpu_passed = all(checks.values())
    return {
        "schema": ADMISSION_SCHEMA,
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "contract_git_blob_sha1": contract_blob,
        "reviewed_parent_commit": REVIEWED_PARENT,
        "runtime_commit": head,
        "upstream_commit": upstream,
        "case": 2,
        "split": "train",
        "namespace": namespace,
        "identities": rows,
        "selection_checks": selection_checks,
        "readiness_checks": readiness_checks,
        "profile_proposal_checks": proposal_checks,
        "corrective_profile_checks": corrective_checks,
        "checks": checks,
        "authorization_file": (
            None
            if authorization_file is None
            else str(authorization_file.resolve())
        ),
        "authorization_token_issued": False,
        "authorization_consumed_before_isaac": False,
        "cpu_contract_ready": cpu_passed,
        "runtime_route_contract_ready": cpu_passed,
        "execution_route_complete": cpu_passed,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "label_capture_authorized": False,
        "dataset_creation_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
        "passed": cpu_passed,
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
