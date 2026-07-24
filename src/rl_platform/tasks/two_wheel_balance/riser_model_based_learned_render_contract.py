"""Admission contract for representative learned-policy RTX renders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .riser_model_based_learned_all79_contract import (
    ALL79_GATE_SCHEMA,
    CONTROL_OWNERSHIP,
    DEFAULT_EVALUATION_CONFIG,
    _exact_digest,
    _exact_source_manifest_valid,
    _mapping_matches_json_file,
    _plan_manifest_valid,
    _resolve_identity,
    balance_safety_snapshot_valid,
    sha256_file,
)
from .riser_model_based_policy_artifact import (
    model_based_residual_torchscript_valid,
)


SCHEMA = "cinebotrl_two_wheel_riser_model_based_learned_render_admission_v1"
REPRESENTATIVE_CASES = [1, 15, 31, 50, 73, 79]
CASE_ROLES = {
    "1": ["simple", "joint_adaptive"],
    "15": ["case15_yaw_limited"],
    "31": ["fixed_path", "low_height"],
    "50": ["low_height_shift"],
    "73": ["joint_adaptive", "high_riser_motion"],
    "79": ["long_duration"],
}
RENDER_CONFIG = {
    "tracking_profile": "riser_recovery_direction_v4_camera_lever_arm_v1",
    "policy_command_contract": "model_based_planner_plus_bounded_policy_residual_v1",
    "residual_action_scales": [0.05, 0.05, 0.02],
    "control_ownership": CONTROL_OWNERSHIP,
    "controller_wz_kp": 1.05,
    "maximum_duration_scale": 3.0,
    "camera_lever_arm_compensation_gain": 1.0,
    "maximum_camera_lever_arm_correction_m": 0.05,
    "video_frame_stride": 8,
    "video_fps": 25,
    "minimum_width": 1280,
    "minimum_height": 720,
    "minimum_duration_s": 3.0,
}
CODE_KEYS = {
    "playback",
    "admission_contract",
    "policy_artifact",
    "preflight_validator",
    "execution_wrapper",
    "media_auditor",
    "report_finalizer",
    "completion_auditor",
}
FIELDS = {
    "schema",
    "all79_report",
    "all79_admission",
    "all79_preflight",
    "policy",
    "plan_manifest",
    "source_manifest",
    "lqr_gains",
    "robot_build_audit",
    "robot_usd",
    "drive_profile_selection",
    "execution_commit",
    "code",
    "cases",
    "case_roles",
    "render_config",
    "all79_gate_passed",
    "render_evaluation_approved",
    "learned_render_authorized",
    "residual_capture_authorized",
    "bc_authorized",
    "ppo_authorized",
    "training_started",
}


def _load(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def validate_render_admission(
    admission: Mapping[str, Any],
    *,
    identity_root: Path,
    all79_report_path: Path,
    all79_report: Mapping[str, Any],
    all79_admission_path: Path,
    all79_preflight_path: Path,
    policy_path: Path,
    plan_manifest_path: Path,
    source_manifest_path: Path,
    lqr_gains_path: Path,
    robot_build_audit_path: Path,
    robot_usd_path: Path,
    drive_profile_selection_path: Path,
    code_paths: Mapping[str, Path],
    expected_execution_commit: str,
    require_authorized: bool,
) -> None:
    code = admission.get("code")
    code_valid = (
        isinstance(code, Mapping)
        and set(code) == CODE_KEYS
        and set(code_paths) == CODE_KEYS
        and all(
            _resolve_identity(
                code[name],
                directory=identity_root,
                expected_path=code_paths[name],
            )
            for name in CODE_KEYS
        )
    )
    policy_sha = sha256_file(policy_path) if policy_path.is_file() else None
    all79_rows = all79_report.get("rows")
    all_rows_balance_safety_valid = (
        isinstance(all79_rows, list)
        and len(all79_rows) == 79
        and all(
            isinstance(row, Mapping)
            and balance_safety_snapshot_valid(row.get("teacher_safety"))
            and balance_safety_snapshot_valid(row.get("learned_safety"))
            for row in all79_rows
        )
    )
    selected_rows_valid = (
        isinstance(all79_rows, list)
        and len(all79_rows) == 79
        and all(
            isinstance(all79_rows[case - 1], Mapping)
            and all79_rows[case - 1].get("case") == case
            and all79_rows[case - 1].get("checks")
            and all(
                value is True
                for value in all79_rows[case - 1]["checks"].values()
            )
            and balance_safety_snapshot_valid(
                all79_rows[case - 1].get("teacher_safety")
            )
            and balance_safety_snapshot_valid(
                all79_rows[case - 1].get("learned_safety")
            )
            for case in REPRESENTATIVE_CASES
        )
    )
    checks = {
        "fields": set(admission) == FIELDS,
        "schema": admission.get("schema") == SCHEMA,
        "all79_report": _resolve_identity(
            admission.get("all79_report"),
            directory=identity_root,
            expected_path=all79_report_path,
        )
        and _mapping_matches_json_file(all79_report_path, all79_report)
        and all79_report.get("schema")
        == ALL79_GATE_SCHEMA
        and all79_report.get("passed") is True
        and all79_report.get("cases") == list(range(1, 80))
        and all79_report.get("policy_sha256") == policy_sha
        and all79_report.get("execution_commit") == expected_execution_commit
        and all_rows_balance_safety_valid
        and selected_rows_valid,
        "all79_provenance": _resolve_identity(
            admission.get("all79_admission"),
            directory=identity_root,
            expected_path=all79_admission_path,
        )
        and _resolve_identity(
            admission.get("all79_preflight"),
            directory=identity_root,
            expected_path=all79_preflight_path,
        ),
        "policy": _resolve_identity(
            admission.get("policy"),
            directory=identity_root,
            expected_path=policy_path,
        )
        and model_based_residual_torchscript_valid(policy_path),
        "source": _resolve_identity(
            admission.get("source_manifest"),
            directory=identity_root,
            expected_path=source_manifest_path,
        )
        and _exact_source_manifest_valid(source_manifest_path),
        "plan": _resolve_identity(
            admission.get("plan_manifest"),
            directory=identity_root,
            expected_path=plan_manifest_path,
        )
        and _plan_manifest_valid(
            plan_manifest_path,
            source_manifest_path=source_manifest_path,
        ),
        "assets": all(
            _resolve_identity(
                admission.get(name),
                directory=identity_root,
                expected_path=path,
            )
            for name, path in {
                "lqr_gains": lqr_gains_path,
                "robot_build_audit": robot_build_audit_path,
                "robot_usd": robot_usd_path,
                "drive_profile_selection": drive_profile_selection_path,
            }.items()
        ),
        "commit": admission.get("execution_commit")
        == expected_execution_commit
        and _exact_digest(expected_execution_commit, 40),
        "code": code_valid,
        "selection": admission.get("cases") == REPRESENTATIVE_CASES
        and admission.get("case_roles") == CASE_ROLES,
        "config": admission.get("render_config") == RENDER_CONFIG
        and {
            key: admission["render_config"][key]
            for key in (
                "tracking_profile",
                "policy_command_contract",
                "residual_action_scales",
                "control_ownership",
                "controller_wz_kp",
                "maximum_duration_scale",
                "camera_lever_arm_compensation_gain",
                "maximum_camera_lever_arm_correction_m",
            )
        }
        == {
            key: DEFAULT_EVALUATION_CONFIG[key]
            for key in (
                "tracking_profile",
                "policy_command_contract",
                "residual_action_scales",
                "control_ownership",
                "controller_wz_kp",
                "maximum_duration_scale",
                "camera_lever_arm_compensation_gain",
                "maximum_camera_lever_arm_correction_m",
            )
        },
        "all79_passed": admission.get("all79_gate_passed") is True,
        "downstream_closed": admission.get("residual_capture_authorized") is False
        and admission.get("bc_authorized") is False
        and admission.get("ppo_authorized") is False
        and admission.get("training_started") is False,
        "authorized": not require_authorized
        or (
            admission.get("render_evaluation_approved") is True
            and admission.get("learned_render_authorized") is True
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"learned render admission failed: {checks}")
