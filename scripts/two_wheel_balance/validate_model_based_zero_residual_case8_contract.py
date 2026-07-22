#!/usr/bin/env python3
"""Validate the CPU-only model-based zero-residual case-8 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


SCHEMA = "cinebotrl_two_wheel_riser_model_based_zero_residual_case8_cpu_contract_v1"
ADMISSION_SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_zero_residual_case8_cpu_admission_v1"
)
REVIEWED_CONTROLLER_PARENT = "d46b70ccf3d45a991827404045946760dc108745"
ZERO_POLICY_SOURCE_COMMIT = "2d7ca8cc4676a5ff680049700562475940fec3b7"
NAMESPACE = "20260722_model_based_zero_residual_case8_canary_v1_exclusive"
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/model_based_zero_residual_case8_cpu_contract_v1.json"
)
ZERO_POLICY_CHECKPOINT_SHA256 = (
    "60377ad7b8b6618b614f9bd272a596574717ca436fcceaa1da827739d0f9e6d2"
)
ZERO_POLICY_TORCHSCRIPT_SHA256 = (
    "b1494f7af219d44cf966d7ba7781370afc1e8fe9575dd4e414d6ec0b7ea1ab19"
)
ZERO_POLICY_REPORT_SHA256 = (
    "55e3ab5cd1ad2c8ee3aac12b4f834b2db90c5a9704c2fe815dd477f95049ef7e"
)
EXPECTED_IDENTITIES = {
    "balance_controller",
    "case78_failure_audit",
    "case8_plan",
    "case8_plan_admission",
    "case8_plan_report",
    "checkpoint_builder",
    "contract_validator",
    "lqr_gains",
    "plan_manifest",
    "plan_summary",
    "playback",
    "policy_module",
    "recovery_evidence",
    "residual_dataset",
    "riser_control",
    "riser_loader",
    "robot_build_audit",
    "robot_urdf",
    "robot_usd",
    "rs4_attitude",
    "runtime_heartbeat",
    "teacher_gate",
    "tracking",
    "zero_policy_checkpoint",
    "zero_policy_report",
    "zero_policy_torchscript",
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
EXPECTED_CONTROLLER = {
    "policy_command_base": "model_based_planner",
    "policy_residual_contract": (
        "model_based_planner_plus_bounded_policy_residual_v1"
    ),
    "residual_action_scales": [0.05, 0.05, 0.02],
    "controller_wz_kp": 1.05,
    "maximum_duration_scale": 3.0,
    "camera_lever_arm_compensation_enabled": True,
    "camera_lever_arm_compensation_gain": 1.0,
    "maximum_camera_lever_arm_correction_m": 0.05,
    "tracking_profile": "riser_recovery_direction_v4_camera_lever_arm_v1",
    "phase_feedforward_contract": "derivatives_scaled_by_progress_v1",
    "position_observation_link": "physical_cam_link_fk",
    "target_attitude_contract": "semantic_dfr_to_physical_cam_v1",
    "hardware_proxy_command_contract": "semantic_attitude_position_only",
}
EXPECTED_ROLLOUTS = {
    "explicit_zero": {
        "trajectory_command_source": "model_based_planner_plus_zero_policy_residual",
        "residual_policy": None,
        "residual_policy_device": None,
        "zero_policy_action": True,
    },
    "zero_checkpoint": {
        "trajectory_command_source": (
            "model_based_planner_plus_torchscript_residual"
        ),
        "residual_policy_identity": "zero_policy_torchscript",
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
EXPECTED_PRESERVATION = {
    "case": 8,
    "both_dynamic_quality_must_pass": True,
    "both_must_complete_reference": True,
    "both_residual_action_abs_max_must_equal": [0.0, 0.0, 0.0],
    "maximum_position_metric_delta_m": 0.005,
    "maximum_attitude_metric_delta_deg": 0.05,
    "maximum_pitch_metric_delta_deg": 0.05,
    "maximum_riser_metric_delta_m": 0.001,
    "maximum_proxy_metric_delta_deg": 0.05,
    "dataset_must_remain_absent": True,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
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
                repo, "rev-parse", f"HEAD:{relative.as_posix()}", check=False
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
    zero_policy_report: dict[str, Any],
) -> dict[str, bool]:
    teacher = _single_result(teacher_gate)
    selected_plan = plan_admission.get("selected_plan", {})
    identities = contract.get("identities", {})
    checks = {
        "schema_matches": contract.get("schema") == SCHEMA,
        "case_is_validation_case8": contract.get("case") == 8
        and contract.get("split") == "validation",
        "namespace_matches": contract.get("namespace") == NAMESPACE,
        "reviewed_controller_parent_matches": contract.get(
            "reviewed_controller_parent_commit"
        )
        == REVIEWED_CONTROLLER_PARENT,
        "zero_policy_source_commit_matches": contract.get(
            "zero_policy_source_commit"
        )
        == ZERO_POLICY_SOURCE_COMMIT,
        "identity_set_exact": set(identities) == EXPECTED_IDENTITIES,
        "plan_contract_matches": contract.get("plan_contract") == EXPECTED_PLAN,
        "controller_contract_matches": contract.get("controller_contract")
        == EXPECTED_CONTROLLER,
        "rollout_contract_matches": contract.get("rollouts") == EXPECTED_ROLLOUTS,
        "dynamic_thresholds_unchanged": contract.get("dynamic_gate_thresholds")
        == EXPECTED_THRESHOLDS,
        "preservation_gate_exact": contract.get("preservation_gate")
        == EXPECTED_PRESERVATION,
        "plan_report_exact": plan_report.get("case") == 8
        and plan_report.get("plan_sha256") == EXPECTED_PLAN["plan_sha256"]
        and all(
            plan_report.get(name) == expected
            for name, expected in EXPECTED_PLAN.items()
            if name != "plan_sha256"
        )
        and plan_report.get("passed") is True
        and plan_report.get("timing_transition_kinematic_gate_passed") is True
        and all(plan_report.get("kinematic_checks", {}).values())
        and all(plan_report.get("derivation_checks", {}).values()),
        "plan_admission_exact": plan_admission.get("passed") is True
        and selected_plan.get("case") == 8
        and selected_plan.get("plan_sha256") == EXPECTED_PLAN["plan_sha256"]
        and selected_plan.get("passed") is True,
        "teacher_reference_exact": teacher_gate.get("cases") == [8]
        and teacher_gate.get("trajectory_command_source") == "deterministic_teacher"
        and teacher_gate.get("tracking_profile")
        == EXPECTED_CONTROLLER["tracking_profile"]
        and teacher_gate.get("phase_feedforward_contract")
        == EXPECTED_CONTROLLER["phase_feedforward_contract"]
        and teacher_gate.get("position_observation_link")
        == EXPECTED_CONTROLLER["position_observation_link"]
        and teacher_gate.get("target_attitude_contract")
        == EXPECTED_CONTROLLER["target_attitude_contract"]
        and teacher_gate.get("hardware_proxy_command_contract")
        == EXPECTED_CONTROLLER["hardware_proxy_command_contract"]
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
        "zero_policy_build_exact": zero_policy_report.get("passed") is True
        and zero_policy_report.get("source_commit") == ZERO_POLICY_SOURCE_COMMIT
        and zero_policy_report.get("policy_architecture")
        == "model_based_shared_encoder_zero_initialized_residual_v1"
        and zero_policy_report.get("command_contract")
        == EXPECTED_CONTROLLER["policy_residual_contract"]
        and zero_policy_report.get("residual_action_scales") == [0.05, 0.05, 0.02]
        and zero_policy_report.get("residual_head_exact_zero") is True
        and zero_policy_report.get("checkpoint", {}).get("sha256")
        == ZERO_POLICY_CHECKPOINT_SHA256
        and zero_policy_report.get("torchscript", {}).get("sha256")
        == ZERO_POLICY_TORCHSCRIPT_SHA256
        and zero_policy_report.get("runtime_authorized") is False
        and zero_policy_report.get("training_authorized") is False
        and zero_policy_report.get("training_started") is False
        and zero_policy_report.get("ppo_authorized") is False
        and zero_policy_report.get("holdout_opened") is False
        and zero_policy_report.get("valid_for_training") is False,
        "zero_policy_identities_exact": identities.get(
            "zero_policy_checkpoint", {}
        ).get("sha256")
        == ZERO_POLICY_CHECKPOINT_SHA256
        and identities.get("zero_policy_torchscript", {}).get("sha256")
        == ZERO_POLICY_TORCHSCRIPT_SHA256
        and identities.get("zero_policy_report", {}).get("sha256")
        == ZERO_POLICY_REPORT_SHA256,
        "no_runtime_or_learning_side_effects": contract.get("one_case_only") is True
        and contract.get("cpu_preflight_ready") is True
        and contract.get("runtime_authorized") is False
        and contract.get("gpu_launch_authorized") is False
        and contract.get("dynamic_canary_authorized") is False
        and contract.get("case78_authorized") is False
        and contract.get("broad_rollout_authorized") is False
        and contract.get("dataset_creation_authorized") is False
        and contract.get("raw_teacher_capture_authorized") is False
        and contract.get("policy_trace_capture_authorized") is False
        and contract.get("shadow_teacher_capture_authorized") is False
        and contract.get("bc_authorized") is False
        and contract.get("ppo_authorized") is False
        and contract.get("holdout_opened") is False
        and contract.get("valid_for_training") is False,
        "no_runtime_token": "runtime_authorization_token_sha256" not in contract
        and "authorization_sha256" not in contract,
    }
    return {name: bool(value) for name, value in checks.items()}


def validate(contract_path: Path, repo: Path, *, namespace: str) -> dict[str, Any]:
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
        zero_policy_report=read_json("zero_policy_report"),
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
            "tracked_worktree_clean": git(repo, "diff", "--quiet", check=False).returncode
            == 0
            and git(repo, "diff", "--cached", "--quiet", check=False).returncode
            == 0,
            "reviewed_parent_is_ancestor": git(
                repo,
                "merge-base",
                "--is-ancestor",
                REVIEWED_CONTROLLER_PARENT,
                head,
                check=False,
            ).returncode
            == 0,
            "zero_policy_source_is_ancestor": git(
                repo,
                "merge-base",
                "--is-ancestor",
                ZERO_POLICY_SOURCE_COMMIT,
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
        "reviewed_controller_parent_commit": REVIEWED_CONTROLLER_PARENT,
        "zero_policy_source_commit": ZERO_POLICY_SOURCE_COMMIT,
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
        raise ValueError(f"refusing to overwrite CPU admission: {args.output}")
    result = validate(args.contract, args.repo_root, namespace=args.namespace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
