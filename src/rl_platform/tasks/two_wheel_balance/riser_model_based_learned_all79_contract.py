"""Hash-bound admission contract for final learned all-79 evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .riser_model_based_corrective_bc_contract import (
    MODEL_BASED_CORRECTIVE_BC_EXECUTION_REPORT_SCHEMA,
    validate_bc_execution_report,
)
from .riser_model_based_corrective_corpus import DEFAULT_RESERVED_HOLDOUT_CASES
from .riser_model_based_policy_artifact import (
    model_based_residual_torchscript_valid,
)


MODEL_BASED_LEARNED_ALL79_ADMISSION_SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_learned_all79_admission_v1"
)
BC_REPORT_SCHEMA = MODEL_BASED_CORRECTIVE_BC_EXECUTION_REPORT_SCHEMA
VALIDATION_GATE_SCHEMA = (
    "cinebotrl_two_wheel_riser_residual_validation_canary_gate_v2"
)
HOLDOUT_GATE_SCHEMA = "cinebotrl_two_wheel_riser_residual_holdout_gate_v2"
ALL79_GATE_SCHEMA = "cinebotrl_two_wheel_riser_residual_all79_gate_v2"
BALANCE_SAFETY_CONTRACT = "balance_first_rollout_safety_v1"
BALANCE_SAFETY_SNAPSHOT_FIELDS = {
    "payload_dynamic_quality_passed",
    "payload_thermal_admission_passed",
    "result_dynamic_quality_passed",
    "result_thermal_admission_passed",
    "controller_evidence_passed",
    "completed_reference",
    "termination_absent",
    "no_termination_check",
    "pitch_bounded_check",
    "saturation_checks_passed",
    "thermal_checks_passed",
    "pitch_max_deg",
    "action_saturation_ratio",
    "riser_saturation_ratio",
    "proxy_saturation_ratio",
    "riser_thermal_load_max",
    "riser_peak_force_violation_count",
    "passed",
}
CODE_IDENTITY_KEYS = {
    "playback",
    "rollout_gate",
    "completion_auditor",
    "admission_contract",
    "policy_artifact",
    "preflight_validator",
    "execution_wrapper",
}
ALL79_CASES = list(range(1, 80))
DEFAULT_EVALUATION_CONFIG = {
    "tracking_profile": "riser_recovery_direction_v4_camera_lever_arm_v1",
    "policy_command_contract": "model_based_planner_plus_bounded_policy_residual_v1",
    "residual_action_scales": [0.05, 0.05, 0.02],
    "controller_wz_kp": 1.05,
    "maximum_duration_scale": 3.0,
    "camera_lever_arm_compensation_enabled": True,
    "camera_lever_arm_compensation_gain": 1.0,
    "maximum_camera_lever_arm_correction_m": 0.05,
    "maximum_regression_fraction": 0.05,
    "minimum_zero_improvement_fraction": 0.05,
    "position_error_p95_m_max": 0.15,
    "position_error_max_m_max": 0.25,
    "attitude_error_p95_deg_max": 5.0,
    "attitude_error_max_deg_max": 10.0,
    "maximum_pitch_deg": 12.0,
    "maximum_saturation_ratio": 0.20,
    "maximum_riser_thermal_load": 1.0,
    "maximum_riser_peak_force_violations": 0,
}
ADMISSION_FIELDS = {
    "schema",
    "bc_report",
    "policy",
    "plan_manifest",
    "source_manifest",
    "lqr_gains",
    "robot_build_audit",
    "robot_usd",
    "drive_profile_selection",
    "validation_gate_report",
    "holdout_gate_report",
    "execution_commit",
    "code",
    "evaluation_config",
    "validation_cases",
    "holdout_cases",
    "all79_cases",
    "model_selection_complete",
    "validation_gate_passed",
    "holdout_gate_passed",
    "holdout_opened_only_after_model_selection",
    "all79_evaluation_approved",
    "learned_rollout_authorized",
    "residual_capture_authorized",
    "bc_authorized",
    "ppo_authorized",
    "training_started",
}
PLAN_SOURCE_CHECKS = {
    "source_manifest_hash_bound",
    "source_json_hash_bound",
    "source_time_verbatim",
    "source_position_verbatim",
    "source_attitude_verbatim",
    "source_anchor_map_identity",
    "source_motion_direction_preserved",
    "source_start_preserved",
    "source_final_preserved",
    "path_length_within_5_percent",
    "source_polyline_p95_bounded",
    "source_polyline_max_bounded",
    "base_branch_step_bounded",
    "proxy_branch_step_bounded",
    "training_closed",
}
GATE_REPORT_FIELDS = {
    "schema",
    "policy_sha256",
    "cases",
    "case_count",
    "maximum_regression_fraction",
    "minimum_zero_improvement_fraction",
    "expected_tracking_profile",
    "policy_command_contract",
    "residual_action_scales",
    "balance_safety_contract",
    "maximum_pitch_deg",
    "maximum_saturation_ratio",
    "maximum_riser_thermal_load",
    "maximum_riser_peak_force_violations",
    "rollout_admission",
    "preflight_receipt",
    "plan_manifest",
    "execution_commit",
    "means",
    "aggregate_checks",
    "rows",
    "passed",
    "ppo_authorized",
}


def balance_safety_snapshot_valid(snapshot: object) -> bool:
    if not isinstance(snapshot, Mapping) or set(snapshot) != (
        BALANCE_SAFETY_SNAPSHOT_FIELDS
    ):
        return False
    boolean_fields = BALANCE_SAFETY_SNAPSHOT_FIELDS - {
        "pitch_max_deg",
        "action_saturation_ratio",
        "riser_saturation_ratio",
        "proxy_saturation_ratio",
        "riser_thermal_load_max",
        "riser_peak_force_violation_count",
    }
    numeric_fields = {
        "pitch_max_deg",
        "action_saturation_ratio",
        "riser_saturation_ratio",
        "proxy_saturation_ratio",
        "riser_thermal_load_max",
    }
    return (
        all(snapshot.get(name) is True for name in boolean_fields)
        and all(
            isinstance(snapshot.get(name), (int, float))
            and not isinstance(snapshot.get(name), bool)
            and math.isfinite(float(snapshot[name]))
            and float(snapshot[name]) >= 0.0
            for name in numeric_fields
        )
        and isinstance(snapshot.get("riser_peak_force_violation_count"), int)
        and not isinstance(snapshot.get("riser_peak_force_violation_count"), bool)
        and snapshot["riser_peak_force_violation_count"] >= 0
        and float(snapshot["pitch_max_deg"])
        <= DEFAULT_EVALUATION_CONFIG["maximum_pitch_deg"]
        and max(
            float(snapshot[name])
            for name in (
                "action_saturation_ratio",
                "riser_saturation_ratio",
                "proxy_saturation_ratio",
            )
        )
        <= DEFAULT_EVALUATION_CONFIG["maximum_saturation_ratio"]
        and float(snapshot["riser_thermal_load_max"])
        <= DEFAULT_EVALUATION_CONFIG["maximum_riser_thermal_load"]
        and snapshot["riser_peak_force_violation_count"]
        <= DEFAULT_EVALUATION_CONFIG["maximum_riser_peak_force_violations"]
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_digest(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_identity(
    identity: object,
    *,
    directory: Path,
    expected_path: Path,
) -> bool:
    if (
        not isinstance(identity, Mapping)
        or set(identity) != {"path", "sha256"}
        or not isinstance(identity.get("path"), str)
        or not identity["path"]
        or not _exact_digest(identity.get("sha256"), 64)
    ):
        return False
    path = Path(identity["path"])
    path = path if path.is_absolute() else directory / path
    path = path.resolve()
    return (
        path == expected_path.resolve()
        and path.is_file()
        and identity["sha256"] == sha256_file(path)
    )


def _artifact_identity_valid(identity: object, *, directory: Path) -> bool:
    if (
        not isinstance(identity, Mapping)
        or set(identity) != {"path", "sha256"}
        or not isinstance(identity.get("path"), str)
        or not identity["path"]
        or not _exact_digest(identity.get("sha256"), 64)
    ):
        return False
    path = Path(identity["path"])
    path = path if path.is_absolute() else directory / path
    return path.is_file() and identity["sha256"] == sha256_file(path)


def _identity_path(identity: object, *, directory: Path) -> Path | None:
    if (
        not isinstance(identity, Mapping)
        or set(identity) != {"path", "sha256"}
        or not isinstance(identity.get("path"), str)
        or not identity["path"]
    ):
        return None
    path = Path(identity["path"])
    path = path if path.is_absolute() else directory / path
    return path.resolve()


def _same_identity(left: object, right: object) -> bool:
    return (
        isinstance(left, Mapping)
        and isinstance(right, Mapping)
        and left.get("sha256") == right.get("sha256")
        and Path(str(left.get("path"))).name == Path(str(right.get("path"))).name
    )


def _bc_report_valid(
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> bool:
    admission_path = _identity_path(
        report.get("admission"),
        directory=report_path.parent,
    )
    if admission_path is None or not admission_path.is_file():
        return False
    admission = _load_json_object(admission_path)
    if admission is None:
        return False
    try:
        validate_bc_execution_report(
            report,
            admission_path=admission_path,
            admission=admission,
            report_directory=report_path.parent,
        )
    except ValueError:
        return False
    return (
        report.get("schema") == MODEL_BASED_CORRECTIVE_BC_EXECUTION_REPORT_SCHEMA
        and report.get("passed") is True
        and report.get("offline_gate_passed") is True
        and report.get("valid_for_dynamic_canary") is True
        and report.get("training_started") is True
        and report.get("ppo_authorized") is False
        and report.get("learned_rollout_authorized") is False
    )


def _split_gate_provenance_valid(
    report: Mapping[str, Any],
    *,
    mode: str,
    cases: list[int],
    policy_sha256: str,
    report_directory: Path,
) -> bool:
    admission_path = _identity_path(
        report.get("rollout_admission"),
        directory=report_directory,
    )
    preflight_path = _identity_path(
        report.get("preflight_receipt"),
        directory=report_directory,
    )
    if (
        admission_path is None
        or preflight_path is None
        or not _artifact_identity_valid(
            report.get("rollout_admission"),
            directory=report_directory,
        )
        or not _artifact_identity_valid(
            report.get("preflight_receipt"),
            directory=report_directory,
        )
        or not _artifact_identity_valid(
            report.get("plan_manifest"),
            directory=report_directory,
        )
    ):
        return False
    admission = _load_json_object(admission_path)
    preflight = _load_json_object(preflight_path)
    if admission is None or preflight is None:
        return False
    expected_stage = (
        admission.get("model_selection_complete") is False
        and admission.get("prior_validation_gate_passed") is False
        and admission.get("prior_validation_gate_report") is None
        if mode == "validation_canary"
        else (
            admission.get("model_selection_complete") is True
            and admission.get("prior_validation_gate_passed") is True
            and _artifact_identity_valid(
                admission.get("prior_validation_gate_report"),
                directory=admission_path.parent,
            )
        )
    )
    checks = preflight.get("checks")
    return (
        admission.get("schema")
        == "cinebotrl_two_wheel_riser_model_based_learned_split_admission_v1"
        and admission.get("mode") == mode
        and admission.get("cases") == cases
        and admission.get("execution_commit") == report.get("execution_commit")
        and admission.get("evaluation_config") == DEFAULT_EVALUATION_CONFIG
        and admission.get("policy", {}).get("sha256") == policy_sha256
        and _same_identity(admission.get("plan_manifest"), report.get("plan_manifest"))
        and admission.get("split_evaluation_approved") is True
        and admission.get("learned_rollout_authorized") is True
        and admission.get("residual_capture_authorized") is False
        and admission.get("bc_authorized") is False
        and admission.get("ppo_authorized") is False
        and admission.get("training_started") is False
        and expected_stage
        and preflight.get("schema")
        == "cinebotrl_two_wheel_riser_model_based_learned_split_preflight_v1"
        and preflight.get("mode") == mode
        and preflight.get("cases") == cases
        and preflight.get("execution_commit") == report.get("execution_commit")
        and _same_identity(
            preflight.get("admission"),
            report.get("rollout_admission"),
        )
        and preflight.get("policy", {}).get("sha256") == policy_sha256
        and _same_identity(
            preflight.get("plan_manifest"),
            report.get("plan_manifest"),
        )
        and isinstance(checks, Mapping)
        and bool(checks)
        and all(value is True for value in checks.values())
        and preflight.get("runtime_started") is False
        and preflight.get("dataset_written") is False
        and preflight.get("capture_started") is False
        and preflight.get("bc_started") is False
        and preflight.get("ppo_started") is False
        and preflight.get("passed") is True
    )


def _gate_report_valid(
    report: Mapping[str, Any],
    *,
    schema: str,
    cases: list[int],
    policy_sha256: str,
    report_directory: Path,
) -> bool:
    rows = report.get("rows")
    means = report.get("means")
    aggregates = report.get("aggregate_checks")
    return (
        set(report) == GATE_REPORT_FIELDS
        and report.get("schema") == schema
        and report.get("policy_sha256") == policy_sha256
        and report.get("cases") == cases
        and report.get("case_count") == len(cases)
        and report.get("maximum_regression_fraction")
        == DEFAULT_EVALUATION_CONFIG["maximum_regression_fraction"]
        and report.get("minimum_zero_improvement_fraction")
        == DEFAULT_EVALUATION_CONFIG["minimum_zero_improvement_fraction"]
        and report.get("expected_tracking_profile")
        == DEFAULT_EVALUATION_CONFIG["tracking_profile"]
        and report.get("policy_command_contract")
        == DEFAULT_EVALUATION_CONFIG["policy_command_contract"]
        and report.get("residual_action_scales")
        == DEFAULT_EVALUATION_CONFIG["residual_action_scales"]
        and report.get("balance_safety_contract") == BALANCE_SAFETY_CONTRACT
        and report.get("maximum_pitch_deg")
        == DEFAULT_EVALUATION_CONFIG["maximum_pitch_deg"]
        and report.get("maximum_saturation_ratio")
        == DEFAULT_EVALUATION_CONFIG["maximum_saturation_ratio"]
        and report.get("maximum_riser_thermal_load")
        == DEFAULT_EVALUATION_CONFIG["maximum_riser_thermal_load"]
        and report.get("maximum_riser_peak_force_violations")
        == DEFAULT_EVALUATION_CONFIG["maximum_riser_peak_force_violations"]
        and isinstance(rows, list)
        and len(rows) == len(cases)
        and all(
            isinstance(row, Mapping)
            and row.get("case") == case
            and isinstance(row.get("checks"), Mapping)
            and bool(row["checks"])
            and all(value is True for value in row["checks"].values())
            and balance_safety_snapshot_valid(row.get("teacher_safety"))
            and balance_safety_snapshot_valid(row.get("learned_safety"))
            and balance_safety_snapshot_valid(row.get("zero_safety"))
            and isinstance(row.get("learned_beats_zero_position_p95"), bool)
            and _artifact_identity_valid(
                row.get("teacher_rollout"),
                directory=report_directory,
            )
            and _artifact_identity_valid(
                row.get("learned_rollout"),
                directory=report_directory,
            )
            and _artifact_identity_valid(
                row.get("zero_rollout"),
                directory=report_directory,
            )
            for case, row in zip(cases, rows, strict=True)
        )
        and isinstance(means, Mapping)
        and set(means)
        == {
            "teacher_position_p95_m",
            "learned_position_p95_m",
            "zero_position_p95_m",
        }
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0.0
            for value in means.values()
        )
        and isinstance(aggregates, Mapping)
        and set(aggregates)
        == {
            "all_case_checks",
            "learned_position_mean_within_teacher_budget",
            "learned_beats_zero_by_required_mean",
            "learned_beats_zero_on_majority_of_cases",
        }
        and all(value is True for value in aggregates.values())
        and _split_gate_provenance_valid(
            report,
            mode=(
                "validation_canary"
                if schema == VALIDATION_GATE_SCHEMA
                else "holdout"
            ),
            cases=cases,
            policy_sha256=policy_sha256,
            report_directory=report_directory,
        )
        and report.get("passed") is True
        and report.get("ppo_authorized") is False
    )


def _mapping_matches_json_file(
    path: Path,
    value: Mapping[str, Any],
) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload == dict(value)


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _exact_source_manifest_valid(path: Path) -> bool:
    payload = _load_json_object(path)
    if payload is None:
        return False
    items = payload.get("items")
    return (
        payload.get("schema") == "gik_exact_source_reference_package_v1"
        and payload.get("trajectory_integrity_contract") == "exact_source_v1"
        and payload.get("episode_count") == len(ALL79_CASES)
        and payload.get("integrity_passed") is True
        and isinstance(items, list)
        and len(items) == len(ALL79_CASES)
        and all(
            isinstance(item, Mapping)
            and item.get("episode_index") == case
            and item.get("trajectory_integrity_contract") == "exact_source_v1"
            and item.get("source_time_strictly_increasing") is True
            and item.get("integrity_passed") is True
            and _exact_digest(item.get("source_json_sha256"), 64)
            and isinstance(item.get("source_pose_count"), int)
            and item["source_pose_count"] > 1
            for case, item in zip(ALL79_CASES, items, strict=True)
        )
    )


def _plan_manifest_valid(path: Path, *, source_manifest_path: Path) -> bool:
    payload = _load_json_object(path)
    source = _load_json_object(source_manifest_path)
    if payload is None or source is None:
        return False
    items = payload.get("items")
    source_items = source.get("items")
    if not isinstance(items, list) or not isinstance(source_items, list):
        return False
    if len(items) != len(ALL79_CASES) or len(source_items) != len(ALL79_CASES):
        return False
    if (
        payload.get("schema")
        != "cinebotrl_two_wheel_riser_smoothed_plan_export_v1"
        or payload.get("plan_schema")
        != "cinebotrl_two_wheel_riser_smoothed_plan_v1"
        or payload.get("source_manifest_sha256")
        != sha256_file(source_manifest_path)
        or payload.get("source_package_case_count") != len(ALL79_CASES)
        or payload.get("requested_cases") != ALL79_CASES
        or payload.get("attempted_cases") != ALL79_CASES
        or payload.get("portfolio_gate_passed") is not True
        or payload.get("isaac_started") is not False
        or payload.get("residual_capture_started") is not False
        or payload.get("bc_started") is not False
        or payload.get("ppo_started") is not False
    ):
        return False
    for case, item, source_item in zip(
        ALL79_CASES,
        items,
        source_items,
        strict=True,
    ):
        if not isinstance(item, Mapping) or not isinstance(source_item, Mapping):
            return False
        expected_file = f"case_{case:04d}_smoothed_riser_plan_v1.npz"
        plan_file = path.parent / expected_file
        checks = item.get("checks")
        if (
            item.get("case") != case
            or item.get("file") != expected_file
            or not _exact_digest(item.get("plan_sha256"), 64)
            or not plan_file.is_file()
            or item["plan_sha256"] != sha256_file(plan_file)
            or item.get("source_json_sha256")
            != source_item.get("source_json_sha256")
            or item.get("source_pose_count")
            != source_item.get("source_pose_count")
            or not isinstance(checks, Mapping)
            or not PLAN_SOURCE_CHECKS.issubset(checks)
            or any(checks[name] is not True for name in PLAN_SOURCE_CHECKS)
        ):
            return False
    return True


def validate_learned_all79_admission(
    admission: Mapping[str, Any],
    *,
    identity_root: Path,
    bc_report_path: Path,
    bc_report: Mapping[str, Any],
    policy_path: Path,
    plan_manifest_path: Path,
    source_manifest_path: Path,
    lqr_gains_path: Path,
    robot_build_audit_path: Path,
    robot_usd_path: Path,
    drive_profile_selection_path: Path,
    validation_report_path: Path,
    validation_report: Mapping[str, Any],
    holdout_report_path: Path,
    holdout_report: Mapping[str, Any],
    code_paths: Mapping[str, Path],
    expected_execution_commit: str,
    require_authorized: bool,
) -> None:
    validation_cases = bc_report.get("split_cases", {}).get("validation", [])
    policy_identity = bc_report.get("torchscript")
    policy_sha = (
        policy_identity.get("sha256")
        if isinstance(policy_identity, Mapping)
        else None
    )
    code = admission.get("code")
    code_valid = (
        isinstance(code, Mapping)
        and set(code) == CODE_IDENTITY_KEYS
        and set(code_paths) == CODE_IDENTITY_KEYS
        and all(
            _resolve_identity(
                code[name],
                directory=identity_root,
                expected_path=code_paths[name],
            )
            for name in CODE_IDENTITY_KEYS
        )
    )
    checks = {
        "fields": set(admission) == ADMISSION_FIELDS,
        "schema": admission.get("schema")
        == MODEL_BASED_LEARNED_ALL79_ADMISSION_SCHEMA,
        "bc_report": _resolve_identity(
            admission.get("bc_report"),
            directory=identity_root,
            expected_path=bc_report_path,
        )
        and _mapping_matches_json_file(bc_report_path, bc_report)
        and _bc_report_valid(bc_report, report_path=bc_report_path),
        "policy": _resolve_identity(
            admission.get("policy"),
            directory=identity_root,
            expected_path=policy_path,
        )
        and isinstance(policy_identity, Mapping)
        and admission.get("policy", {}).get("sha256")
        == policy_identity.get("sha256")
        and _exact_digest(policy_sha, 64)
        and model_based_residual_torchscript_valid(policy_path),
        "source_manifest": _resolve_identity(
            admission.get("source_manifest"),
            directory=identity_root,
            expected_path=source_manifest_path,
        )
        and _exact_source_manifest_valid(source_manifest_path),
        "plan_manifest": _resolve_identity(
            admission.get("plan_manifest"),
            directory=identity_root,
            expected_path=plan_manifest_path,
        )
        and _plan_manifest_valid(
            plan_manifest_path,
            source_manifest_path=source_manifest_path,
        ),
        "runtime_assets": all(
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
        "validation_report": _resolve_identity(
            admission.get("validation_gate_report"),
            directory=identity_root,
            expected_path=validation_report_path,
        )
        and _mapping_matches_json_file(validation_report_path, validation_report)
        and isinstance(validation_cases, list)
        and len(validation_cases) >= 2
        and _gate_report_valid(
            validation_report,
            schema=VALIDATION_GATE_SCHEMA,
            cases=validation_cases,
            policy_sha256=str(policy_sha),
            report_directory=validation_report_path.parent,
        ),
        "holdout_report": _resolve_identity(
            admission.get("holdout_gate_report"),
            directory=identity_root,
            expected_path=holdout_report_path,
        )
        and _mapping_matches_json_file(holdout_report_path, holdout_report)
        and _gate_report_valid(
            holdout_report,
            schema=HOLDOUT_GATE_SCHEMA,
            cases=DEFAULT_RESERVED_HOLDOUT_CASES,
            policy_sha256=str(policy_sha),
            report_directory=holdout_report_path.parent,
        ),
        "commit": admission.get("execution_commit")
        == expected_execution_commit
        == bc_report.get("execution_commit")
        and _exact_digest(expected_execution_commit, 40),
        "code": code_valid,
        "config": admission.get("evaluation_config") == DEFAULT_EVALUATION_CONFIG,
        "cases": admission.get("validation_cases") == validation_cases
        and admission.get("holdout_cases") == DEFAULT_RESERVED_HOLDOUT_CASES
        and admission.get("all79_cases") == ALL79_CASES,
        "prerequisites": admission.get("model_selection_complete") is True
        and admission.get("validation_gate_passed") is True
        and admission.get("holdout_gate_passed") is True
        and admission.get("holdout_opened_only_after_model_selection") is True,
        "downstream_closed": admission.get("residual_capture_authorized") is False
        and admission.get("bc_authorized") is False
        and admission.get("ppo_authorized") is False
        and admission.get("training_started") is False,
        "authorized": not require_authorized
        or (
            admission.get("all79_evaluation_approved") is True
            and admission.get("learned_rollout_authorized") is True
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"learned all-79 admission failed: {checks}")
