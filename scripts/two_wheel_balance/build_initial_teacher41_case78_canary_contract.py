#!/usr/bin/env python3
"""Build the CPU-only teacher-41 case-78 learned-policy canary contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEWED_CASE8_COMMIT = "02bcdfc49ef2c17c7784e4db7f70692572a23658"
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
    "case78_plan": EXPECTED_PLAN["plan_sha256"],
    "plan_manifest": "8351514a361d3be4e5fbf57f2dbb019a7d8d2f5b86e89cea2553a1cfda3c64a1",
    "plan_summary": "72bdb9b64f71bde125507944a0f43f244bfb766e445e996693fba67270cb104d",
    "plan_case_report": "419c9568de492050bc6504900580e8759abe7d149a85a17eb083abfb9184fdf0",
    "camera_cap_cpu_contract": "1c864557d6302613d111eccbafd914673d88316c2bcbc659e1afb028c7fec176",
    "teacher_gate": "ad0dc3ee618819ec808ac4d0318bda711dc2cba38dd041119a1f78584e97e459",
    "teacher_final": "63004e41d1185a8589c8715e620a5c976db44bfb6a130786214109a1ab2d5bd7",
    "label_admission": "cd752e402c912d7a83767544c8059d2068979d10868dff2408fd93836b71033d",
    "case8_final": "681bdaa0a2e20d4182d5de944f9b2d079b7b9e87a565070c0d976829a2c69446",
    "case8_summary": "bcf1d892216902590c259fbd325d35b9cb618152ea200bdee16ed31d91948c30",
    "policy_final": "d232ffac1f67d1a4510e2ad7e6670f82742b259433ed0ce6b15727c4e39db3d9",
    "policy_report": "b7915caddea9467847430a247924eae2e856ad486da06135e1b8f543c42b891a",
    "policy_torchscript": "0d796c600c6dca7dce176da555f4cd1f769163f41093d2b6313f4e6264888db7",
    "lqr_gains": "2d955a8878b1086836cfffdaf89e2cd2ecf7c2c4ab2467c24bbfa43cbbd4d5e6",
    "robot_usd": "89f8e38f9290c4a0fcf206dd6966f067f543888f5422f978e566dbb655efa9d0",
    "playback": "ffe45cd5747f6e628caebafbc405d589f34df764df944ddc2b36a2efd0926b1d",
    "rollout_gate": "e3e327f2b7bc7f3bdc5f7c27ba36edcf2f3660e5ed4c8b4e4324325764927f5b",
}
JSON_INPUTS = {
    "plan_case_report",
    "camera_cap_cpu_contract",
    "teacher_gate",
    "teacher_final",
    "label_admission",
    "case8_final",
    "case8_summary",
    "policy_final",
    "policy_report",
}
TRACKING_PROFILE = "riser_recovery_direction_v4_camera_lever_arm_v1"
ACTION_SCALES = [0.35, 0.4, 0.1]


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
    plan = payloads["plan_case_report"]
    cap = payloads["camera_cap_cpu_contract"]
    teacher_payload = payloads["teacher_gate"]
    teacher_results = teacher_payload.get("results", [])
    teacher = teacher_results[0] if len(teacher_results) == 1 else {}
    teacher_final = payloads["teacher_final"]
    label = payloads["label_admission"]
    case8_final = payloads["case8_final"]
    case8_summary = payloads["case8_summary"]
    policy_final = payloads["policy_final"]
    policy_report = payloads["policy_report"]
    checks = {
        "source_commit_present": len(source_commit) == 40,
        "case8_dynamic_canary_passed": case8_final.get("passed") is True
        and case8_final.get("dynamic_canary_passed") is True
        and case8_final.get("case") == 8
        and case8_final.get("case78_authorized") is False
        and case8_final.get("dataset_created") is False
        and case8_summary.get("passed") is True
        and case8_summary.get("cases") == [8]
        and case8_summary.get("policy_sha256")
        == EXPECTED_HASHES["policy_torchscript"],
        "case78_plan_exact": all(plan.get(key) == value for key, value in EXPECTED_PLAN.items())
        and plan.get("passed") is True
        and plan.get("timing_transition_kinematic_gate_passed") is True
        and all(plan.get("kinematic_checks", {}).values())
        and plan.get("valid_for_training") is False,
        "camera_cap_contract_exact": cap.get("case") == 78
        and cap.get("plan_contract") == EXPECTED_PLAN
        and cap.get("controller_arguments", {}).get(
            "maximum_camera_lever_arm_correction_m"
        )
        == 0.1
        and cap.get("controller_arguments", {}).get("controller_wz_kp") == 1.05
        and cap.get("controller_arguments", {}).get("maximum_duration_scale") == 3.0
        and cap.get("controller_arguments", {}).get("trajectory_command_source")
        == "deterministic_teacher"
        and cap.get("runtime_authorized") is False
        and cap.get("gpu_launch_authorized") is False,
        "teacher_reference_exact": teacher_payload.get("cases") == [78]
        and teacher_payload.get("trajectory_command_source") == "deterministic_teacher"
        and teacher_payload.get("tracking_profile") == TRACKING_PROFILE
        and teacher_payload.get("position_observation_link") == "physical_cam_link_fk"
        and teacher_payload.get("target_attitude_contract")
        == "semantic_dfr_to_physical_cam_v1"
        and teacher_payload.get("hardware_proxy_command_contract")
        == "semantic_attitude_position_only"
        and teacher_payload.get("controller_overrides") == {"wz_kp": 1.05}
        and teacher_payload.get("maximum_duration_scale") == 3.0
        and teacher_payload.get("camera_lever_arm_compensation_enabled") is True
        and teacher_payload.get("camera_lever_arm_compensation_gain") == 1.0
        and teacher_payload.get("maximum_camera_lever_arm_correction_m") == 0.1
        and teacher_payload.get("residual_policy") is None
        and teacher_payload.get("passed") is True
        and teacher_payload.get("dynamic_quality_passed") is True,
        "teacher_result_exact": teacher.get("case") == 78
        and teacher.get("source_duration_s") == EXPECTED_PLAN["source_duration_s"]
        and teacher.get("execution_duration_s") == EXPECTED_PLAN["execution_duration_s"]
        and teacher.get("completed_phase_time_s")
        == EXPECTED_PLAN["execution_duration_s"]
        and teacher.get("position_error_p95_m", 1.0) <= 0.15
        and teacher.get("position_error_max_m", 1.0) <= 0.25
        and teacher.get("residual_action_abs_max") == [0.0, 0.0, 0.0]
        and teacher.get("executed_residual_dataset") is None
        and teacher.get("passed") is True,
        "teacher_shadow_run_sealed_non_training": teacher_final.get("passed") is True
        and teacher_final.get("case") == 78
        and teacher_final.get("current_split") == "validation"
        and teacher_final.get("physical_quality_passed") is True
        and teacher_final.get("shadow_trace_passed") is True
        and teacher_final.get("labels_applied_to_commands") is False
        and teacher_final.get("dataset_created") is False
        and teacher_final.get("valid_for_training") is False,
        "case78_labels_admitted": label.get("case") == 78
        and label.get("split") == "validation"
        and label.get("action_scales") == ACTION_SCALES
        and label.get("label_admission_passed") is True
        and label.get("labels_applied_to_commands") is False
        and label.get("holdout_opened") is False
        and label.get("training_started") is False,
        "policy_is_same_offline_admitted_policy": policy_final.get("passed") is True
        and policy_final.get("offline_gate_passed") is True
        and policy_final.get("case8_canary_proposal_ready") is True
        and policy_final.get("learned_rollout_authorized") is False
        and policy_final.get("learned_rollout_started") is False
        and policy_final.get("holdout_opened") is False
        and policy_final.get("ppo_started") is False
        and policy_final.get("torchscript", {}).get("sha256")
        == EXPECTED_HASHES["policy_torchscript"],
        "policy_report_includes_case78_validation_only": policy_report.get(
            "offline_gate_passed"
        )
        is True
        and policy_report.get("policy_architecture")
        == "state_shared_lookahead_fusion_previous_action_masked_v1"
        and policy_report.get("masked_observation_indices") == [23, 24, 25]
        and policy_report.get("offline_gate_splits") == ["validation"]
        and policy_report.get("holdout_metrics_computed") is False
        and policy_report.get("holdout_used_for_model_selection") is False
        and policy_report.get("torchscript_sha256")
        == EXPECTED_HASHES["policy_torchscript"],
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"case-78 learned canary CPU contract failed: {checks}")
    return {
        "schema": "cinebotrl_two_wheel_riser_initial_teacher41_case78_canary_cpu_contract_v1",
        "source_commit": source_commit,
        "reviewed_case8_runtime_commit": REVIEWED_CASE8_COMMIT,
        "case": 78,
        "split": "validation",
        "namespace": "20260722_initial_teacher41_masked_bc_case78_canary_v1_exclusive",
        "plan_contract": EXPECTED_PLAN,
        "controller_contract": {
            "controller_wz_kp": 1.05,
            "maximum_duration_scale": 3.0,
            "camera_lever_arm_compensation_enabled": True,
            "camera_lever_arm_compensation_gain": 1.0,
            "maximum_camera_lever_arm_correction_m": 0.1,
            "residual_action_scales": ACTION_SCALES,
            "tracking_profile": TRACKING_PROFILE,
            "position_observation_link": "physical_cam_link_fk",
            "target_attitude_contract": "semantic_dfr_to_physical_cam_v1",
        },
        "comparison_contract": {
            "teacher_reference": "teacher_gate",
            "fresh_zero_required": True,
            "fresh_learned_required": True,
            "maximum_teacher_regression_fraction": 0.05,
            "minimum_zero_improvement_fraction": 0.05,
        },
        "runtime_bounds": {
            "zero_timeout_s": 5400,
            "learned_timeout_s": 5400,
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
        ["git", "merge-base", "--is-ancestor", REVIEWED_CASE8_COMMIT, actual_head],
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
