#!/usr/bin/env python3
"""Build the CPU-only model-based zero-residual case-78 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEWED_CASE8_RUNTIME_COMMIT = "cf0d4d0bb171063b50d5db58045d721158f8d566"
NAMESPACE = "20260722_model_based_zero_residual_case78_canary_v1_exclusive"
EXPECTED_PLAN = {
    "case": 78,
    "plan_sha256": (
        "28c69e20778e738d1ac4a0ae299160ed5764089094c2a0f9a018c49790860569"
    ),
    "source_pose_count": 6870,
    "execution_state_count": 6870,
    "source_duration_s": 135.487646,
    "execution_duration_s": 192.29956737098348,
}
EXPECTED_HASHES = {
    "case8_final": "24922a7a08e9262b6159732aac5dbee6689ffe13a46b0bb95d29182437c66d9d",
    "case78_plan": EXPECTED_PLAN["plan_sha256"],
    "plan_manifest": "8351514a361d3be4e5fbf57f2dbb019a7d8d2f5b86e89cea2553a1cfda3c64a1",
    "plan_summary": "72bdb9b64f71bde125507944a0f43f244bfb766e445e996693fba67270cb104d",
    "plan_case_report": "419c9568de492050bc6504900580e8759abe7d149a85a17eb083abfb9184fdf0",
    "camera_cap_cpu_contract": (
        "1c864557d6302613d111eccbafd914673d88316c2bcbc659e1afb028c7fec176"
    ),
    "teacher_gate": "ad0dc3ee618819ec808ac4d0318bda711dc2cba38dd041119a1f78584e97e459",
    "zero_policy_report": (
        "55e3ab5cd1ad2c8ee3aac12b4f834b2db90c5a9704c2fe815dd477f95049ef7e"
    ),
    "zero_policy_checkpoint": (
        "60377ad7b8b6618b614f9bd272a596574717ca436fcceaa1da827739d0f9e6d2"
    ),
    "zero_policy_torchscript": (
        "b1494f7af219d44cf966d7ba7781370afc1e8fe9575dd4e414d6ec0b7ea1ab19"
    ),
    "case78_failure_audit": (
        "97c90a0dc56450e4dc71654ac588eeffd09bd5d0db92bc3a4fbae265709241fd"
    ),
    "lqr_gains": "2d955a8878b1086836cfffdaf89e2cd2ecf7c2c4ab2467c24bbfa43cbbd4d5e6",
    "robot_usd": "89f8e38f9290c4a0fcf206dd6966f067f543888f5422f978e566dbb655efa9d0",
    "playback": "320019f164343d113bed74c4352686bcb12eb68404bfe911d23594f5f4fc81a3",
}
JSON_INPUTS = {
    "case8_final",
    "plan_case_report",
    "camera_cap_cpu_contract",
    "teacher_gate",
    "zero_policy_report",
    "case78_failure_audit",
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
    "maximum_camera_lever_arm_correction_m": 0.1,
    "tracking_profile": "riser_recovery_direction_v4_camera_lever_arm_v1",
    "position_observation_link": "physical_cam_link_fk",
    "target_attitude_contract": "semantic_dfr_to_physical_cam_v1",
    "hardware_proxy_command_contract": "semantic_attitude_position_only",
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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"missing case-78 contract input: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_contract(payloads: dict[str, dict[str, Any]], source_commit: str) -> dict[str, Any]:
    case8 = payloads["case8_final"]
    plan = payloads["plan_case_report"]
    cap = payloads["camera_cap_cpu_contract"]
    teacher_payload = payloads["teacher_gate"]
    results = teacher_payload.get("results", [])
    teacher = results[0] if isinstance(results, list) and len(results) == 1 else {}
    policy = payloads["zero_policy_report"]
    failure = payloads["case78_failure_audit"]
    checks = {
        "source_commit_present": len(source_commit) == 40,
        "case8_zero_residual_preservation_passed": case8.get("passed") is True
        and case8.get("schema")
        == "cinebotrl_two_wheel_riser_model_based_zero_residual_case8_final_v1"
        and case8.get("zero_residual_preservation_passed") is True
        and case8.get("case") == 8
        and all(float(value) == 0.0 for value in case8.get("metric_absolute_deltas", {}).values())
        and case8.get("zero_checkpoint_sha256")
        == EXPECTED_HASHES["zero_policy_torchscript"]
        and case8.get("dataset_creation_authorized") is False
        and case8.get("training_authorized") is False
        and case8.get("ppo_authorized") is False
        and case8.get("holdout_opened") is False,
        "case78_plan_exact": all(
            plan.get(key) == value for key, value in EXPECTED_PLAN.items()
        )
        and plan.get("passed") is True
        and plan.get("timing_transition_kinematic_gate_passed") is True
        and all(plan.get("kinematic_checks", {}).values())
        and plan.get("valid_for_training") is False,
        "case78_camera_cap_exact": cap.get("case") == 78
        and cap.get("plan_contract") == EXPECTED_PLAN
        and cap.get("controller_arguments", {}).get(
            "maximum_camera_lever_arm_correction_m"
        )
        == 0.1
        and cap.get("controller_arguments", {}).get("controller_wz_kp") == 1.05
        and cap.get("dynamic_gate_thresholds") == EXPECTED_THRESHOLDS
        and cap.get("runtime_authorized") is False
        and cap.get("gpu_launch_authorized") is False
        and cap.get("dataset_creation_authorized") is False,
        "teacher_reference_exact": teacher_payload.get("cases") == [78]
        and teacher_payload.get("trajectory_command_source") == "deterministic_teacher"
        and teacher_payload.get("tracking_profile")
        == EXPECTED_CONTROLLER["tracking_profile"]
        and teacher_payload.get("position_observation_link")
        == EXPECTED_CONTROLLER["position_observation_link"]
        and teacher_payload.get("target_attitude_contract")
        == EXPECTED_CONTROLLER["target_attitude_contract"]
        and teacher_payload.get("hardware_proxy_command_contract")
        == EXPECTED_CONTROLLER["hardware_proxy_command_contract"]
        and teacher_payload.get("maximum_camera_lever_arm_correction_m") == 0.1
        and teacher_payload.get("residual_policy") is None
        and teacher_payload.get("dynamic_quality_passed") is True
        and teacher_payload.get("passed") is True,
        "teacher_result_exact": teacher.get("case") == 78
        and teacher.get("source_duration_s") == EXPECTED_PLAN["source_duration_s"]
        and teacher.get("execution_duration_s")
        == EXPECTED_PLAN["execution_duration_s"]
        and teacher.get("completed_phase_time_s")
        == EXPECTED_PLAN["execution_duration_s"]
        and teacher.get("position_error_p95_m", 1.0) <= 0.15
        and teacher.get("position_error_max_m", 1.0) <= 0.25
        and teacher.get("residual_action_abs_max") == [0.0, 0.0, 0.0]
        and teacher.get("executed_residual_dataset") is None
        and teacher.get("passed") is True,
        "zero_policy_exact": policy.get("passed") is True
        and policy.get("policy_architecture")
        == "model_based_shared_encoder_zero_initialized_residual_v1"
        and policy.get("command_contract")
        == EXPECTED_CONTROLLER["policy_residual_contract"]
        and policy.get("residual_action_scales") == [0.05, 0.05, 0.02]
        and policy.get("residual_head_exact_zero") is True
        and policy.get("checkpoint", {}).get("sha256")
        == EXPECTED_HASHES["zero_policy_checkpoint"]
        and policy.get("torchscript", {}).get("sha256")
        == EXPECTED_HASHES["zero_policy_torchscript"]
        and policy.get("runtime_authorized") is False
        and policy.get("training_authorized") is False
        and policy.get("ppo_authorized") is False,
        "planner_imitation_failure_requires_pivot": failure.get("passed") is True
        and failure.get("failed_dynamic_gate") == "position_p95_bounded"
        and failure.get("architecture_audit", {}).get("required_contract_satisfied")
        is False
        and failure.get("architecture_audit", {}).get("checkpoint_classification")
        == "planner_imitation_bc_initialization_only"
        and failure.get("decision", {}).get("bc_retraining_authorized") is False
        and failure.get("decision", {}).get("ppo_authorized") is False,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"model-based zero-residual case-78 contract failed: {checks}")
    return {
        "schema": "cinebotrl_two_wheel_riser_model_based_zero_residual_case78_cpu_contract_v1",
        "source_commit": source_commit,
        "reviewed_case8_runtime_commit": REVIEWED_CASE8_RUNTIME_COMMIT,
        "case": 78,
        "split": "validation",
        "namespace": NAMESPACE,
        "plan_contract": EXPECTED_PLAN,
        "controller_contract": EXPECTED_CONTROLLER,
        "rollouts": {
            "explicit_zero": {
                "trajectory_command_source": (
                    "model_based_planner_plus_zero_policy_residual"
                ),
                "zero_policy_action": True,
                "residual_policy": None,
            },
            "zero_checkpoint": {
                "trajectory_command_source": (
                    "model_based_planner_plus_torchscript_residual"
                ),
                "zero_policy_action": False,
                "residual_policy_identity": "zero_policy_torchscript",
                "residual_policy_device": "cuda",
            },
        },
        "dynamic_gate_thresholds": EXPECTED_THRESHOLDS,
        "preservation_gate": {
            "both_dynamic_quality_must_pass": True,
            "both_must_complete_reference": True,
            "both_residual_action_abs_max_must_equal": [0.0, 0.0, 0.0],
            "maximum_position_metric_delta_m": 0.005,
            "maximum_attitude_metric_delta_deg": 0.05,
            "maximum_pitch_metric_delta_deg": 0.05,
            "maximum_riser_metric_delta_m": 0.001,
            "maximum_proxy_metric_delta_deg": 0.05,
            "dataset_must_remain_absent": True,
        },
        "runtime_bounds": {
            "explicit_zero_timeout_s": 5400,
            "zero_checkpoint_timeout_s": 5400,
            "maximum_combined_timeout_s": 10800,
            "heartbeat_interval_policy_steps": 2000,
        },
        "input_contract_checks": checks,
        "cpu_contract_ready": True,
        "runtime_authorization_token_issued": False,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "dynamic_canary_authorized": False,
        "dataset_creation_authorized": False,
        "case16_22_32_authorized": False,
        "broad_rollout_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "holdout_opened": False,
        "valid_for_training": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in EXPECTED_HASHES:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite case-78 CPU contract: {args.output}")
    actual_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    if args.source_commit != actual_head:
        raise ValueError("case-78 contract source commit does not match HEAD")
    if subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            REVIEWED_CASE8_RUNTIME_COMMIT,
            actual_head,
        ],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode != 0:
        raise ValueError("case-78 contract HEAD does not descend from case-8 runtime")
    paths = {name: getattr(args, name) for name in EXPECTED_HASHES}
    identities = {name: identity(path) for name, path in paths.items()}
    if any(
        identities[name]["sha256"] != expected
        for name, expected in EXPECTED_HASHES.items()
    ):
        raise ValueError("case-78 contract input hash mismatch")
    payloads = {name: load_json(paths[name]) for name in JSON_INPUTS}
    result = build_contract(payloads, args.source_commit)
    result["inputs"] = identities
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
