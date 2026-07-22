#!/usr/bin/env python3
"""Validate the CPU-only teacher-41 case-8 learned-policy canary contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


SCHEMA = "cinebotrl_two_wheel_riser_initial_teacher41_case8_canary_cpu_contract_v1"
ADMISSION_SCHEMA = (
    "cinebotrl_two_wheel_riser_initial_teacher41_case8_canary_cpu_admission_v1"
)
REVIEWED_PARENT = "e8a7b7e5748586ecbf95aa1a50ca663c35de13a9"
NAMESPACE = "20260722_initial_teacher41_masked_bc_case8_canary_v1_exclusive"
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/initial_teacher41_case8_canary_cpu_contract_v1.json"
)
EXPECTED_IDENTITIES = {
    "balance_controller",
    "case8_plan",
    "case8_plan_admission",
    "case8_plan_report",
    "contract_validator",
    "lqr_gains",
    "plan_manifest",
    "plan_summary",
    "playback",
    "policy_admission",
    "policy_final",
    "policy_module",
    "policy_report",
    "policy_torchscript",
    "raw_teacher_gate",
    "recovery_evidence",
    "riser_control",
    "riser_loader",
    "robot_build_audit",
    "robot_urdf",
    "robot_usd",
    "rollout_gate",
    "rs4_attitude",
    "runtime_heartbeat",
    "teacher_gate",
    "tracking",
}
EXPECTED_PLAN = {
    "case": 8,
    "plan_sha256": (
        "f07ff020128dee70ea9c8c2d806dc75c8e0ef3964dccb4e0aabfd1b0048f3655"
    ),
    "source_pose_count": 663,
    "execution_state_count": 663,
    "source_duration_s": 12.940941,
    "execution_duration_s": 18.1173174,
}
EXPECTED_COMMON_CONTROLLER = {
    "controller_wz_kp": 1.05,
    "maximum_duration_scale": 3.0,
    "camera_lever_arm_compensation_enabled": True,
    "camera_lever_arm_compensation_gain": 1.0,
    "maximum_camera_lever_arm_correction_m": 0.05,
    "residual_action_scales": [0.35, 0.4, 0.1],
    "tracking_profile": "riser_recovery_direction_v4_camera_lever_arm_v1",
    "phase_feedforward_contract": "derivatives_scaled_by_progress_v1",
    "position_observation_link": "physical_cam_link_fk",
    "target_attitude_contract": "semantic_dfr_to_physical_cam_v1",
    "hardware_proxy_command_contract": "semantic_attitude_position_only",
}
EXPECTED_ROLLOUTS = {
    "zero": {
        "trajectory_command_source": "zero_policy_action_baseline",
        "residual_policy": None,
        "residual_policy_device": None,
        "zero_policy_action": True,
    },
    "learned": {
        "trajectory_command_source": "torchscript_residual_policy",
        "residual_policy_identity": "policy_torchscript",
        "residual_policy_device": "cuda",
        "zero_policy_action": False,
    },
}
EXPECTED_THRESHOLDS = {
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
EXPECTED_COMPARISON = {
    "mode": "validation_canary",
    "case": 8,
    "maximum_teacher_regression_fraction": 0.05,
    "minimum_zero_improvement_fraction": 0.05,
    "teacher_reference_identity": "teacher_gate",
    "zero_reference_created_in_namespace": True,
    "learned_reference_created_in_namespace": True,
}
EXPECTED_POLICY_SHA256 = (
    "0d796c600c6dca7dce176da555f4cd1f769163f41093d2b6313f4e6264888db7"
)
EXPECTED_DATASET_SHA256 = (
    "03e3f2b8b4a6b7626a9b43f1fb2a88cbbfdfceb4b6373a51abdb21590bf53497"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    raw_path = Path(str(payload.get("path", "")))
    path = raw_path if raw_path.is_absolute() else repo / raw_path
    exists = path.is_file()
    actual_sha = sha256_file(path) if exists else None
    expected_blob = payload.get("git_blob_sha1")
    actual_blob = None
    committed_blob = None
    if exists and expected_blob is not None:
        result = git(repo, "hash-object", str(path), check=False)
        actual_blob = result.stdout.strip() if result.returncode == 0 else None
        try:
            relative = path.resolve().relative_to(repo)
        except ValueError:
            relative = None
        if relative is not None:
            result = git(
                repo,
                "rev-parse",
                f"HEAD:{relative.as_posix()}",
                check=False,
            )
            committed_blob = result.stdout.strip() if result.returncode == 0 else None
    checks = {
        "file_exists": exists,
        "sha256_matches": actual_sha == payload.get("sha256"),
        "git_blob_matches": expected_blob is None or actual_blob == expected_blob,
        "committed_git_blob_matches": expected_blob is None
        or committed_blob == expected_blob,
    }
    return {
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "git_blob_sha1": actual_blob,
        "committed_git_blob_sha1": committed_blob,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _single_result(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", [])
    return results[0] if isinstance(results, list) and len(results) == 1 else {}


def semantic_checks(
    contract: dict[str, Any],
    *,
    plan_report: dict[str, Any],
    plan_admission: dict[str, Any],
    teacher_gate: dict[str, Any],
    raw_teacher_gate: dict[str, Any],
    policy_final: dict[str, Any],
    policy_report: dict[str, Any],
    policy_admission: dict[str, Any],
) -> dict[str, bool]:
    teacher = _single_result(teacher_gate)
    raw_teacher = _single_result(raw_teacher_gate)
    selected_plan = plan_admission.get("selected_plan", {})
    policy_identity = contract.get("identities", {}).get("policy_torchscript", {})
    teacher_metrics = (
        "position_error_p95_m",
        "position_error_max_m",
        "attitude_error_p95_deg",
        "attitude_error_max_deg",
        "pitch_p95_deg",
        "pitch_max_deg",
        "riser_servo_error_p95_m",
        "riser_servo_error_max_m",
        "proxy_servo_error_p95_deg",
        "proxy_servo_error_max_deg",
    )
    checks = {
        "schema_matches": contract.get("schema") == SCHEMA,
        "case_is_validation_case8": contract.get("case") == 8
        and contract.get("split") == "validation",
        "namespace_matches": contract.get("namespace") == NAMESPACE,
        "reviewed_parent_matches": contract.get("reviewed_policy_parent_commit")
        == REVIEWED_PARENT,
        "identity_set_exact": set(contract.get("identities", {}))
        == EXPECTED_IDENTITIES,
        "plan_contract_matches": contract.get("plan_contract") == EXPECTED_PLAN,
        "controller_contract_matches": contract.get("common_controller_arguments")
        == EXPECTED_COMMON_CONTROLLER,
        "rollout_contract_matches": contract.get("rollouts") == EXPECTED_ROLLOUTS,
        "dynamic_thresholds_unchanged": contract.get("dynamic_gate_thresholds")
        == EXPECTED_THRESHOLDS,
        "comparison_contract_matches": contract.get("comparison_gate")
        == EXPECTED_COMPARISON,
        "plan_report_identity_and_clocks": plan_report.get("case") == 8
        and plan_report.get("plan_sha256") == EXPECTED_PLAN["plan_sha256"]
        and all(
            plan_report.get(name) == expected
            for name, expected in EXPECTED_PLAN.items()
            if name != "plan_sha256"
        ),
        "plan_report_integrity_and_kinematics": plan_report.get("passed") is True
        and plan_report.get("timing_transition_kinematic_gate_passed") is True
        and all(plan_report.get("kinematic_checks", {}).values())
        and all(plan_report.get("derivation_checks", {}).values())
        and plan_report.get("valid_for_training") is False,
        "plan_admission_exact": plan_admission.get("passed") is True
        and plan_admission.get("valid_for_training") is False
        and selected_plan.get("case") == 8
        and selected_plan.get("plan_sha256") == EXPECTED_PLAN["plan_sha256"]
        and selected_plan.get("passed") is True,
        "teacher_reference_exact": teacher_gate.get("cases") == [8]
        and teacher_gate.get("trajectory_command_source") == "deterministic_teacher"
        and teacher_gate.get("tracking_profile")
        == EXPECTED_COMMON_CONTROLLER["tracking_profile"]
        and teacher_gate.get("phase_feedforward_contract")
        == EXPECTED_COMMON_CONTROLLER["phase_feedforward_contract"]
        and teacher_gate.get("position_observation_link")
        == EXPECTED_COMMON_CONTROLLER["position_observation_link"]
        and teacher_gate.get("target_attitude_contract")
        == EXPECTED_COMMON_CONTROLLER["target_attitude_contract"]
        and teacher_gate.get("hardware_proxy_command_contract")
        == EXPECTED_COMMON_CONTROLLER["hardware_proxy_command_contract"]
        and teacher_gate.get("camera_lever_arm_compensation_enabled") is True
        and teacher_gate.get("camera_lever_arm_compensation_gain") == 1.0
        and teacher_gate.get("maximum_camera_lever_arm_correction_m") == 0.05
        and teacher_gate.get("controller_overrides") == {"wz_kp": 1.05}
        and teacher_gate.get("maximum_duration_scale") == 3.0
        and teacher_gate.get("residual_policy") is None
        and teacher_gate.get("passed") is True
        and teacher_gate.get("dynamic_quality_passed") is True,
        "teacher_result_exact": teacher.get("case") == 8
        and teacher.get("source_duration_s") == EXPECTED_PLAN["source_duration_s"]
        and teacher.get("execution_duration_s")
        == EXPECTED_PLAN["execution_duration_s"]
        and teacher.get("dynamic_quality_passed") is True
        and teacher.get("passed") is True
        and teacher.get("residual_action_abs_max") == [0.0, 0.0, 0.0]
        and teacher.get("executed_residual_dataset") is None,
        "raw_teacher_matches_reference": raw_teacher_gate.get("cases") == [8]
        and raw_teacher_gate.get("raw_teacher_capture_started") is True
        and raw_teacher_gate.get("normalized_dataset_capture_started") is False
        and raw_teacher_gate.get("passed") is True
        and raw_teacher.get("case") == 8
        and raw_teacher.get("source_duration_s")
        == EXPECTED_PLAN["source_duration_s"]
        and raw_teacher.get("execution_duration_s")
        == EXPECTED_PLAN["execution_duration_s"]
        and raw_teacher.get("raw_residual_label_applied_to_commands") is False
        and raw_teacher.get("executed_residual_dataset") is None
        and bool(raw_teacher.get("executed_raw_teacher_capture"))
        and all(raw_teacher.get(metric) == teacher.get(metric) for metric in teacher_metrics),
        "policy_final_is_offline_only": policy_final.get("passed") is True
        and policy_final.get("offline_gate_passed") is True
        and policy_final.get("case8_canary_proposal_ready") is True
        and policy_final.get("learned_rollout_authorized") is False
        and policy_final.get("learned_rollout_started") is False
        and policy_final.get("holdout_opened") is False
        and policy_final.get("ppo_authorized") is False
        and policy_final.get("ppo_started") is False
        and policy_final.get("torchscript", {}).get("sha256")
        == EXPECTED_POLICY_SHA256,
        "policy_report_matches_architecture": policy_report.get("offline_gate_passed")
        is True
        and policy_report.get("policy_architecture")
        == "state_shared_lookahead_fusion_previous_action_masked_v1"
        and policy_report.get("masked_observation_indices") == [23, 24, 25]
        and policy_report.get("previous_action_observation_contract")
        == "masked_after_normalization_v1"
        and policy_report.get("offline_gate_splits") == ["validation"]
        and policy_report.get("dataset_sha256") == EXPECTED_DATASET_SHA256
        and policy_report.get("holdout_used_for_model_selection") is False
        and policy_report.get("holdout_metrics_computed") is False
        and policy_report.get("learned_rollout_started") is False
        and policy_report.get("ppo_started") is False
        and policy_report.get("torchscript_sha256") == EXPECTED_POLICY_SHA256,
        "policy_admission_kept_runtime_closed": policy_admission.get("passed") is True
        and policy_admission.get("validation_only_model_selection") is True
        and policy_admission.get("holdout_opened") is False
        and policy_admission.get("learned_rollout_authorized") is False
        and policy_admission.get("ppo_authorized") is False,
        "policy_identity_exact": policy_identity.get("sha256")
        == EXPECTED_POLICY_SHA256,
        "no_capture_or_training_side_effects": contract.get("one_case_only") is True
        and contract.get("dataset_creation_authorized") is False
        and contract.get("raw_teacher_capture_authorized") is False
        and contract.get("policy_trace_capture_authorized") is False
        and contract.get("shadow_teacher_capture_authorized") is False
        and contract.get("bc_authorized") is False
        and contract.get("ppo_authorized") is False
        and contract.get("holdout_opened") is False,
        "cpu_boundary_preserved": contract.get("cpu_preflight_ready") is True
        and contract.get("runtime_authorized") is False
        and contract.get("gpu_launch_authorized") is False
        and contract.get("dynamic_canary_authorized") is False
        and contract.get("case78_authorized") is False
        and contract.get("broad_rollout_authorized") is False
        and contract.get("valid_for_training") is False,
        "no_runtime_token": "runtime_authorization_token_sha256" not in contract
        and "authorization_sha256" not in contract,
    }
    return {name: bool(value) for name, value in checks.items()}


def validate(
    contract_path: Path,
    repo: Path,
    *,
    namespace: str,
) -> dict[str, object]:
    repo = repo.resolve()
    contract_path = contract_path.resolve()
    canonical_path = (repo / CONTRACT_RELATIVE_PATH).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    identities = contract.get("identities", {})
    rows = {
        name: identity_row(repo, payload)
        for name, payload in identities.items()
        if isinstance(payload, dict)
    }

    def read_json(name: str) -> dict[str, Any]:
        row = rows.get(name, {})
        if row.get("passed") is not True:
            return {}
        return json.loads(Path(str(row["path"])).read_text(encoding="utf-8"))

    checks = semantic_checks(
        contract,
        plan_report=read_json("case8_plan_report"),
        plan_admission=read_json("case8_plan_admission"),
        teacher_gate=read_json("teacher_gate"),
        raw_teacher_gate=read_json("raw_teacher_gate"),
        policy_final=read_json("policy_final"),
        policy_report=read_json("policy_report"),
        policy_admission=read_json("policy_admission"),
    )
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    upstream = git(repo, "rev-parse", "@{u}").stdout.strip()
    contract_blob = git(repo, "hash-object", str(contract_path), check=False).stdout.strip()
    committed_blob = git(
        repo, "rev-parse", f"HEAD:{CONTRACT_RELATIVE_PATH}", check=False
    ).stdout.strip()
    checks.update(
        {
            "head_matches_upstream": head == upstream,
            "tracked_worktree_clean": git(
                repo, "diff", "--quiet", check=False
            ).returncode
            == 0
            and git(repo, "diff", "--cached", "--quiet", check=False).returncode
            == 0,
            "reviewed_parent_is_ancestor": git(
                repo,
                "merge-base",
                "--is-ancestor",
                REVIEWED_PARENT,
                head,
                check=False,
            ).returncode
            == 0,
            "canonical_contract_path": contract_path == canonical_path,
            "contract_is_tracked": git(
                repo,
                "ls-files",
                "--error-unmatch",
                CONTRACT_RELATIVE_PATH,
                check=False,
            ).returncode
            == 0,
            "contract_blob_matches_head": bool(contract_blob)
            and contract_blob == committed_blob,
            "namespace_argument_matches": namespace == NAMESPACE,
            "namespace_is_fresh": not (
                repo / "artifacts/two_wheel_riser" / namespace
            ).exists(),
            "all_identity_hashes_match": set(rows) == EXPECTED_IDENTITIES
            and all(row["passed"] for row in rows.values()),
        }
    )
    passed = all(checks.values())
    return {
        "schema": ADMISSION_SCHEMA,
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "contract_git_blob_sha1": contract_blob,
        "runtime_commit": head,
        "upstream_commit": upstream,
        "reviewed_policy_parent_commit": REVIEWED_PARENT,
        "case": 8,
        "split": "validation",
        "namespace": namespace,
        "identities": rows,
        "checks": checks,
        "cpu_contract_ready": passed,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "dynamic_canary_authorized": False,
        "case78_authorized": False,
        "dataset_creation_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "holdout_opened": False,
        "valid_for_training": False,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite case-8 CPU admission: {args.output}")
    result = validate(args.contract, args.repo_root, namespace=args.namespace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
