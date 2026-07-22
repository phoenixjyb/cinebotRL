#!/usr/bin/env python3
"""Validate the CPU-only deterministic case-78 shadow-label contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


EXPECTED_REVIEWED_PARENT = "0ebc51d7e8eb5adb7136daa95dd301f976497214"
EXPECTED_IMPLEMENTATION = "95216534ead0d11cbf985deb57f6e115cc2c1e18"
EXPECTED_NAMESPACE = "20260722_case78_shadow_label_measurement_v1_exclusive"
EXPECTED_SCALES = [0.35, 0.4, 0.1]
EXPECTED_CONTROLLER = {
    "trajectory_command_source": "deterministic_teacher",
    "controller_wz_kp": 1.05,
    "maximum_duration_scale": 3.0,
    "camera_lever_arm_compensation_enabled": True,
    "camera_lever_arm_compensation_gain": 1.0,
    "maximum_camera_lever_arm_correction_m": 0.1,
    "residual_policy": None,
    "zero_policy_action": False,
    "shadow_teacher_trace_enabled": True,
    "residual_action_scales": EXPECTED_SCALES,
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def same_vector(left: Any, right: list[float], *, tolerance: float = 1e-12) -> bool:
    return bool(
        isinstance(left, list)
        and len(left) == len(right)
        and all(
            isinstance(value, (int, float))
            and abs(float(value) - expected) <= tolerance
            for value, expected in zip(left, right, strict=True)
        )
    )


def run_git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def semantic_checks(
    contract: dict[str, Any],
    audit: dict[str, Any],
    split: dict[str, Any],
    gate: dict[str, Any],
    final: dict[str, Any],
) -> dict[str, bool]:
    results = gate.get("results", [])
    case78 = results[0] if len(results) == 1 and isinstance(results[0], dict) else {}
    admitted = split.get("admitted_split_cases", {})
    return {
        "contract_schema": contract.get("schema")
        == "cinebotrl_two_wheel_riser_case78_shadow_label_cpu_contract_v1",
        "case_and_namespace": contract.get("case") == 78
        and contract.get("current_split") == "validation"
        and contract.get("namespace") == EXPECTED_NAMESPACE,
        "reviewed_lineage": contract.get("reviewed_parent_commit")
        == EXPECTED_REVIEWED_PARENT
        and contract.get("implementation_commit") == EXPECTED_IMPLEMENTATION,
        "exact_plan_contract": contract.get("plan_contract")
        == {
            "case": 78,
            "plan_sha256": (
                "28c69e20778e738d1ac4a0ae299160ed5764089094c2a0f9a018c49790860569"
            ),
            "source_pose_count": 6870,
            "execution_state_count": 6870,
            "source_duration_s": 135.487646,
            "execution_duration_s": 192.29956737098348,
        },
        "controller_is_exact": contract.get("controller_arguments")
        == EXPECTED_CONTROLLER,
        "dynamic_gates_unchanged": contract.get("dynamic_gate_thresholds")
        == EXPECTED_THRESHOLDS,
        "measurement_is_observational_only": contract.get(
            "measurement_contract"
        )
        == {
            "visited_state_source": "deterministic_controller",
            "raw_labels_applied_to_commands": False,
            "applied_residual_actions_must_be_zero": True,
            "trace_only": True,
            "dataset_present": False,
            "record_policy_rate_timestamps": True,
            "record_source_and_execution_clocks": True,
            "record_raw_and_normalized_labels": True,
            "record_command_reconstruction_fields": True,
        },
        "bounded_runtime_contract": contract.get("one_case_only") is True
        and contract.get("maximum_runtime_seconds") == 5400
        and contract.get("heartbeat_interval_policy_steps") == 2000,
        "cpu_only_and_no_learning": all(
            contract.get(name) is False
            for name in (
                "runtime_authorized",
                "gpu_launch_authorized",
                "shadow_measurement_authorized",
                "label_capture_authorized",
                "dataset_creation_authorized",
                "dagger_authorized",
                "bc_authorized",
                "ppo_authorized",
                "holdout_opened",
                "valid_for_training",
            )
        ),
        "residual_audit_retains_candidate_scale": audit.get("decision")
        == "retain_teacher40_scale_case78_series_measurement_required"
        and audit.get("teacher40_action_contract_retained") is True
        and same_vector(audit.get("teacher40_candidate_scale"), EXPECTED_SCALES)
        and audit.get("candidate_scale_maximum_compatibility_passed") is True
        and audit.get("case78_shadow_measurement_required_before_label_capture")
        is True
        and audit.get("bc_authorized") is False
        and audit.get("ppo_authorized") is False,
        "split_admits_validation_role_only": split.get("split_admitted") is True
        and 78 in admitted.get("validation", [])
        and 78 not in admitted.get("train", [])
        and split.get("case78_labels_available") is False
        and split.get("label_capture_authorized") is False,
        "passed_canary_is_bound": gate.get("cases") == [78]
        and gate.get("passed") is True
        and case78.get("case") == 78
        and case78.get("dynamic_quality_passed") is True
        and case78.get("thermal_admission_passed") is True
        and case78.get("controller_evidence_passed") is True
        and case78.get("maximum_camera_lever_arm_correction_m") == 0.1
        and case78.get("camera_recovery_governor_enabled") is False
        and case78.get("termination") is None
        and case78.get("executed_residual_dataset") is None,
        "passed_final_is_bound": final.get("case") == 78
        and final.get("passed") is True
        and final.get("dynamic_qualification_passed") is True
        and final.get("dataset_created") is False
        and final.get("bc_authorized") is False
        and final.get("ppo_authorized") is False
        and final.get("valid_for_training") is False,
    }


def verify_identity(root: Path, identity: dict[str, Any]) -> dict[str, Any]:
    relative = identity.get("path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("identity path must be repository-relative")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"identity path is missing or outside repository: {relative}")
    actual_sha = sha256_file(path)
    if actual_sha != identity.get("sha256"):
        raise ValueError(f"identity SHA-256 mismatch: {relative}")
    expected_blob = identity.get("git_blob_sha1")
    if expected_blob is not None:
        actual_blob = run_git(root, "hash-object", relative)
        committed_blob = run_git(root, "rev-parse", f"HEAD:{relative}")
        if actual_blob != expected_blob or committed_blob != expected_blob:
            raise ValueError(f"identity Git blob mismatch: {relative}")
    return {"path": str(path), "sha256": actual_sha, "git_blob_sha1": expected_blob}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    expected_contract = (
        root
        / "scripts/two_wheel_balance/case78_shadow_label_cpu_contract_v1.json"
    ).resolve()
    if args.contract.resolve() != expected_contract:
        raise ValueError("contract must use the canonical committed path")
    contract = load_json(expected_contract)
    if args.namespace != EXPECTED_NAMESPACE:
        raise ValueError("unexpected namespace")
    head = run_git(root, "rev-parse", "HEAD")
    upstream = run_git(root, "rev-parse", "@{upstream}")
    if head != upstream:
        raise ValueError("HEAD is not equal to configured upstream")
    if run_git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("tracked worktree is not clean")
    for commit in (EXPECTED_REVIEWED_PARENT, EXPECTED_IMPLEMENTATION):
        result = subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", commit, head],
            check=False,
        )
        if result.returncode != 0:
            raise ValueError(f"required lineage is not an ancestor: {commit}")
    identities = {
        name: verify_identity(root, identity)
        for name, identity in contract.get("identities", {}).items()
    }
    required = {
        "case78_plan",
        "lqr_gains",
        "robot_usd",
        "playback",
        "residual_dataset_contract",
        "tracking",
        "heartbeat_helper",
        "case78_gate",
        "case78_final_status",
        "residual_action_audit",
        "split_admission",
        "validator",
        "preflight_wrapper",
    }
    if set(identities) != required:
        raise ValueError("contract identity set is incomplete")
    audit = load_json(Path(identities["residual_action_audit"]["path"]))
    split = load_json(Path(identities["split_admission"]["path"]))
    gate = load_json(Path(identities["case78_gate"]["path"]))
    final = load_json(Path(identities["case78_final_status"]["path"]))
    checks = semantic_checks(contract, audit, split, gate, final)
    namespace_path = root / "artifacts/two_wheel_riser" / EXPECTED_NAMESPACE
    checks["fresh_namespace"] = not namespace_path.exists()
    checks["head_equals_upstream"] = head == upstream
    checks["tracked_worktree_clean"] = True
    checks["identity_set_complete"] = set(identities) == required
    if not all(checks.values()):
        raise ValueError(f"shadow-label CPU contract failed: {checks}")
    output = {
        "schema": "cinebotrl_two_wheel_riser_case78_shadow_label_preflight_v1",
        "case": 78,
        "namespace": EXPECTED_NAMESPACE,
        "runtime_commit": head,
        "reviewed_parent_commit": EXPECTED_REVIEWED_PARENT,
        "implementation_commit": EXPECTED_IMPLEMENTATION,
        "checks": checks,
        "identities": identities,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "shadow_measurement_authorized": False,
        "dataset_creation_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "runtime_started": False,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
