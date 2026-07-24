#!/usr/bin/env python3
"""Audit end-goal evidence for the arm-free two-wheel riser project."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_platform.tasks.two_wheel_balance import (  # noqa: E402
    riser_model_based_corrective_bc_contract as bc_contract,
)
from rl_platform.tasks.two_wheel_balance import (  # noqa: E402
    riser_model_based_corrective_training_dataset as training_dataset_contract,
)
from rl_platform.tasks.two_wheel_balance import (  # noqa: E402
    riser_model_based_learned_all79_contract as learned_all79_contract,
)


CODE_IDENTITY_KEYS = bc_contract.CODE_IDENTITY_KEYS
validate_bc_execution_admission = bc_contract.validate_bc_execution_admission
validate_bc_execution_report = bc_contract.validate_bc_execution_report
load_training_dataset = training_dataset_contract.load_training_dataset
validate_learned_all79_admission = (
    learned_all79_contract.validate_learned_all79_admission
)


DOC_ROOT = PROJECT_ROOT / "docs/03_training/two_wheel_balance"
DEFAULT_GOAL = DOC_ROOT / "riser_recursive_improvement_goal_v1.json"
DEFAULT_ASSET_AUDIT = DOC_ROOT / "evidence_20260715_riser/gate0_asset_audit.json"
DEFAULT_LQR_GATE = DOC_ROOT / "evidence_20260714_28kg/lqr_nominal_gate.json"
DEFAULT_BASELINE = (
    DOC_ROOT / "evidence_20260716_riser_gate0_gate3_online_comp/summary.json"
)
DEFAULT_EXACT_SOURCE = (
    DOC_ROOT / "evidence_20260717_riser_exact_source_gate_a_b/summary.json"
)
DEFAULT_HARDWARE = (
    DOC_ROOT / "evidence_20260723_hardware_production_candidate_v1/summary.json"
)
DEFAULT_BENCH = (
    DOC_ROOT / "evidence_20260723_riser_bench_750w_template_v1/summary.json"
)
DEFAULT_PENDING_ROUTE_QUEUE = (
    DOC_ROOT
    / "evidence_20260724_pending_corrective_route_queue_cpu_v4/summary.json"
)
EXPECTED_BRANCH = "codex/two-wheel-riser-rl"
EXPECTED_MOVABLE_JOINTS = {
    "joint1_gimbal_pitch",
    "joint2_gimbal_roll",
    "joint3_gimbal_yaw",
    "left_wheel_joint",
    "right_wheel_joint",
    "riser_joint",
}
EXPECTED_PENDING_ROUTE_ORDER = (
    "case23_conversion",
    "case6_pair",
    "case2_pair",
    "case7_pair",
    "case8_validation_pair",
    "case16_validation_pair",
)
PENDING_ROUTE_AUTHORIZATION_FIELDS = (
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
REQUIRED_COMPLETION_GATES = (
    "isolated_worktree_and_branch",
    "arm_free_robot_asset",
    "frozen_lqr_balance_baseline",
    "riser_height_and_speed_baseline",
    "exact_source_all79_reference",
    "riser_motor_and_mechanism_recommendation",
    "model_based_corrective_training_corpus",
    "projection_aware_bc_policy",
    "learned_policy_all79_dynamic_gate",
    "learned_policy_render_audit",
)
GOAL_COMPLETION_AUDIT_SCHEMA = (
    "cinebotrl_two_wheel_riser_goal_completion_audit_v5"
)
ROLLOUT_METRICS = (
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
ALL79_REPORT_FIELDS = {
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
    "control_ownership",
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
ALL79_ROW_FIELDS = {
    "case",
    "checks",
    "teacher",
    "learned",
    "teacher_safety",
    "learned_safety",
    "learned_residual_action_abs_max",
    "teacher_rollout",
    "learned_rollout",
}
LEARNED_RENDER_REPORT_FIELDS = {
    "schema",
    "policy",
    "source_all79_report",
    "render_admission",
    "render_preflight",
    "media_manifest",
    "visual_review",
    "cases",
    "rollout_gates",
    "videos",
    "visual_checks",
    "passed",
    "training_started",
    "ppo_authorized",
}
LEARNED_RENDER_VIDEO_FIELDS = {
    "case",
    "path",
    "sha256",
    "codec",
    "width",
    "height",
    "fps",
    "duration_s",
}
LEARNED_RENDER_VISUAL_CHECKS = {
    "robot_asset_intact",
    "riser_motion_visible",
    "camera_and_gimbal_visible",
    "wheel_ground_contact_plausible",
    "no_detached_links",
    "no_abnormal_oscillation",
}
LEARNED_RENDER_ROLLOUT_FIELDS = {"case", "path", "sha256"}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": _sha256(resolved)}


def _exact_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_identity(
    identity: object,
    *,
    directory: Path,
    expected_path: Path | None = None,
) -> Path:
    if (
        not isinstance(identity, Mapping)
        or set(identity) != {"path", "sha256"}
        or not isinstance(identity.get("path"), str)
        or not identity["path"]
        or not _exact_sha256(identity.get("sha256"))
    ):
        raise ValueError("artifact identity fields are invalid")
    path = Path(identity["path"])
    path = path if path.is_absolute() else directory / path
    path = path.resolve()
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError("artifact identity path mismatch")
    if not path.is_file() or _sha256(path) != identity["sha256"]:
        raise ValueError("artifact identity SHA-256 mismatch")
    return path


def _finite_nonnegative_metrics(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(ROLLOUT_METRICS)
        and all(
            isinstance(value[name], (int, float))
            and not isinstance(value[name], bool)
            and math.isfinite(float(value[name]))
            and float(value[name]) >= 0.0
            for name in ROLLOUT_METRICS
        )
    )


def _validate_all79_report(
    report: Mapping[str, Any],
    *,
    policy_sha256: str,
    report_directory: Path,
    admission_path: Path,
    preflight_path: Path,
    plan_manifest_path: Path,
    execution_commit: str,
) -> None:
    cases = list(range(1, 80))
    rows = report.get("rows")
    regression = report.get("maximum_regression_fraction")
    if (
        set(report) != ALL79_REPORT_FIELDS
        or report.get("schema")
        != learned_all79_contract.ALL79_GATE_SCHEMA
        or report.get("policy_sha256") != policy_sha256
        or report.get("cases") != cases
        or report.get("case_count") != len(cases)
        or not isinstance(regression, (int, float))
        or isinstance(regression, bool)
        or not 0.0 <= float(regression) <= 0.05
        or report.get("minimum_zero_improvement_fraction") is not None
        or report.get("expected_tracking_profile")
        != "riser_recovery_direction_v4_camera_lever_arm_v1"
        or report.get("policy_command_contract")
        != "model_based_planner_plus_bounded_policy_residual_v1"
        or report.get("residual_action_scales") != [0.05, 0.05, 0.02]
        or report.get("balance_safety_contract")
        != learned_all79_contract.BALANCE_SAFETY_CONTRACT
        or report.get("control_ownership")
        != learned_all79_contract.DEFAULT_EVALUATION_CONFIG[
            "control_ownership"
        ]
        or report.get("maximum_pitch_deg")
        != learned_all79_contract.DEFAULT_EVALUATION_CONFIG["maximum_pitch_deg"]
        or report.get("maximum_saturation_ratio")
        != learned_all79_contract.DEFAULT_EVALUATION_CONFIG[
            "maximum_saturation_ratio"
        ]
        or report.get("maximum_riser_thermal_load")
        != learned_all79_contract.DEFAULT_EVALUATION_CONFIG[
            "maximum_riser_thermal_load"
        ]
        or report.get("maximum_riser_peak_force_violations")
        != learned_all79_contract.DEFAULT_EVALUATION_CONFIG[
            "maximum_riser_peak_force_violations"
        ]
        or report.get("execution_commit") != execution_commit
        or not isinstance(rows, list)
        or len(rows) != len(cases)
        or report.get("passed") is not True
        or report.get("ppo_authorized") is not False
    ):
        raise ValueError("learned all-79 report contract mismatch")

    report_admission = _resolve_identity(
        report.get("rollout_admission"),
        directory=report_directory,
    )
    if _sha256(report_admission) != _sha256(admission_path):
        raise ValueError("learned all-79 admission identity mismatch")
    _resolve_identity(
        report.get("preflight_receipt"),
        directory=report_directory,
        expected_path=preflight_path,
    )
    _resolve_identity(
        report.get("plan_manifest"),
        directory=report_directory,
        expected_path=plan_manifest_path,
    )
    preflight = _load_json(preflight_path)
    if (
        preflight.get("schema")
        != "cinebotrl_two_wheel_riser_model_based_learned_all79_preflight_v1"
        or preflight.get("passed") is not True
        or preflight.get("execution_commit") != execution_commit
        or preflight.get("head") != execution_commit
        or preflight.get("runtime_started") is not False
        or preflight.get("dataset_created") is not False
        or preflight.get("residual_capture_started") is not False
        or preflight.get("bc_started") is not False
        or preflight.get("ppo_started") is not False
    ):
        raise ValueError("learned all-79 preflight receipt is invalid")

    learned_values: list[float] = []
    teacher_values: list[float] = []
    for expected_case, row in zip(cases, rows, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row) != ALL79_ROW_FIELDS
            or row.get("case") != expected_case
            or not isinstance(row.get("checks"), Mapping)
            or not row["checks"]
            or not all(value is True for value in row["checks"].values())
            or not _finite_nonnegative_metrics(row.get("teacher"))
            or not _finite_nonnegative_metrics(row.get("learned"))
            or not learned_all79_contract.balance_safety_snapshot_valid(
                row.get("teacher_safety")
            )
            or not learned_all79_contract.balance_safety_snapshot_valid(
                row.get("learned_safety")
            )
        ):
            raise ValueError(f"learned all-79 row {expected_case} is invalid")
        _resolve_identity(
            row.get("teacher_rollout"),
            directory=report_directory,
        )
        _resolve_identity(
            row.get("learned_rollout"),
            directory=report_directory,
        )
        residual = row.get("learned_residual_action_abs_max")
        if (
            not isinstance(residual, list)
            or len(residual) != 3
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0 + 1e-6
                for value in residual
            )
        ):
            raise ValueError(f"learned all-79 row {expected_case} residual is invalid")
        for metric in ROLLOUT_METRICS:
            if float(row["learned"][metric]) > (
                (1.0 + float(regression)) * float(row["teacher"][metric]) + 1e-9
            ):
                raise ValueError(
                    f"learned all-79 row {expected_case} regresses {metric}"
                )
        teacher_values.append(float(row["teacher"]["position_error_p95_m"]))
        learned_values.append(float(row["learned"]["position_error_p95_m"]))

    expected_means = {
        "teacher_position_p95_m": statistics.fmean(teacher_values),
        "learned_position_p95_m": statistics.fmean(learned_values),
    }
    means = report.get("means")
    aggregates = report.get("aggregate_checks")
    if (
        not isinstance(means, Mapping)
        or set(means) != set(expected_means)
        or any(
            not math.isclose(
                float(means[name]),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for name, expected in expected_means.items()
        )
        or not isinstance(aggregates, Mapping)
        or set(aggregates)
        != {"all_case_checks", "learned_position_mean_within_teacher_budget"}
        or not all(value is True for value in aggregates.values())
        or expected_means["learned_position_p95_m"]
        > (1.0 + float(regression)) * expected_means["teacher_position_p95_m"]
        + 1e-9
    ):
        raise ValueError("learned all-79 aggregate metrics mismatch")


def _validate_learned_render_report(
    report: Mapping[str, Any],
    *,
    report_directory: Path,
    policy_path: Path,
    policy_sha256: str,
    all79_report_path: Path,
) -> None:
    cases = report.get("cases")
    videos = report.get("videos")
    visual_checks = report.get("visual_checks")
    if (
        set(report) != LEARNED_RENDER_REPORT_FIELDS
        or report.get("schema")
        != "cinebotrl_two_wheel_riser_learned_render_audit_v2"
        or not isinstance(cases, list)
        or cases != [1, 15, 31, 50, 73, 79]
        or not isinstance(videos, list)
        or len(videos) != len(cases)
        or not isinstance(report.get("rollout_gates"), list)
        or len(report["rollout_gates"]) != len(cases)
        or not isinstance(visual_checks, Mapping)
        or set(visual_checks) != LEARNED_RENDER_VISUAL_CHECKS
        or not all(value is True for value in visual_checks.values())
        or report.get("passed") is not True
        or report.get("training_started") is not False
        or report.get("ppo_authorized") is not False
    ):
        raise ValueError("learned render report contract mismatch")
    _resolve_identity(
        report.get("policy"),
        directory=report_directory,
        expected_path=policy_path,
    )
    if report["policy"]["sha256"] != policy_sha256:
        raise ValueError("learned render policy SHA-256 mismatch")
    _resolve_identity(
        report.get("source_all79_report"),
        directory=report_directory,
        expected_path=all79_report_path,
    )
    admission_path = _resolve_identity(
        report.get("render_admission"),
        directory=report_directory,
    )
    preflight_path = _resolve_identity(
        report.get("render_preflight"),
        directory=report_directory,
    )
    media_path = _resolve_identity(
        report.get("media_manifest"),
        directory=report_directory,
    )
    review_path = _resolve_identity(
        report.get("visual_review"),
        directory=report_directory,
    )
    media = _load_json(media_path)
    review = _load_json(review_path)
    render_admission = _load_json(admission_path)
    render_preflight = _load_json(preflight_path)
    media_checks = media.get("media_checks")
    if (
        render_admission.get("schema")
        != "cinebotrl_two_wheel_riser_model_based_learned_render_admission_v1"
        or render_admission.get("cases") != cases
        or render_admission.get("all79_gate_passed") is not True
        or render_admission.get("render_evaluation_approved") is not True
        or render_admission.get("learned_render_authorized") is not True
        or render_admission.get("residual_capture_authorized") is not False
        or render_admission.get("bc_authorized") is not False
        or render_admission.get("ppo_authorized") is not False
        or render_admission.get("training_started") is not False
        or render_preflight.get("schema")
        != "cinebotrl_two_wheel_riser_model_based_learned_render_preflight_v1"
        or render_preflight.get("cases") != cases
        or render_preflight.get("passed") is not True
        or render_preflight.get("runtime_started") is not False
        or render_preflight.get("recording_started") is not False
        or _sha256(
            _resolve_identity(
                render_preflight.get("admission"),
                directory=preflight_path.parent,
            )
        )
        != _sha256(admission_path)
        or media.get("schema")
        != "cinebotrl_two_wheel_riser_learned_render_media_manifest_v1"
        or media.get("passed") is not True
        or media.get("manual_visual_review_required") is not True
        or media.get("cases") != cases
        or media.get("videos") != videos
        or media.get("rollout_gates") != report["rollout_gates"]
        or not isinstance(media_checks, Mapping)
        or set(media_checks) != {str(case) for case in cases}
        or any(
            not isinstance(value, Mapping)
            or not value
            or not all(check is True for check in value.values())
            for value in media_checks.values()
        )
        or media.get("policy") != report["policy"]
        or media.get("source_all79_report") != report["source_all79_report"]
        or _sha256(
            _resolve_identity(media.get("admission"), directory=media_path.parent)
        )
        != _sha256(admission_path)
        or _sha256(
            _resolve_identity(media.get("preflight"), directory=media_path.parent)
        )
        != _sha256(preflight_path)
        or media.get("runtime_started") is not True
        or media.get("recording_started") is not True
        or media.get("training_started") is not False
        or media.get("ppo_authorized") is not False
        or review.get("schema")
        != "cinebotrl_two_wheel_riser_learned_render_visual_review_v1"
        or review.get("cases") != cases
        or review.get("videos") != videos
        or review.get("visual_checks") != visual_checks
        or not isinstance(review.get("reviewer"), str)
        or not review["reviewer"].strip()
        or not isinstance(review.get("reviewed_at_utc"), str)
        or not review["reviewed_at_utc"].endswith("Z")
        or review.get("passed") is not True
    ):
        raise ValueError("learned render provenance or visual review is invalid")
    for expected_case, rollout in zip(
        cases,
        report["rollout_gates"],
        strict=True,
    ):
        if (
            not isinstance(rollout, Mapping)
            or set(rollout) != LEARNED_RENDER_ROLLOUT_FIELDS
            or rollout.get("case") != expected_case
        ):
            raise ValueError(f"learned render rollout {expected_case} is invalid")
        _resolve_identity(
            {"path": rollout["path"], "sha256": rollout["sha256"]},
            directory=report_directory,
        )
        rollout_path = _resolve_identity(
            {"path": rollout["path"], "sha256": rollout["sha256"]},
            directory=report_directory,
        )
        rollout_payload = _load_json(rollout_path)
        if (
            rollout_payload.get("cases") != [expected_case]
            or rollout_payload.get("passed") is not True
            or rollout_payload.get("trajectory_command_source")
            != "model_based_planner_plus_torchscript_residual"
            or rollout_payload.get("tracking_profile")
            != "riser_recovery_direction_v4_camera_lever_arm_v1"
            or rollout_payload.get("policy_command_base")
            != "model_based_planner"
            or rollout_payload.get("residual_action_scales")
            != [0.05, 0.05, 0.02]
            or any(
                rollout_payload.get(name) != value
                for name, value in learned_all79_contract.CONTROL_OWNERSHIP.items()
            )
        ):
            raise ValueError(
                f"learned render rollout {expected_case} contract is invalid"
            )
    for expected_case, video in zip(cases, videos, strict=True):
        if (
            not isinstance(video, Mapping)
            or set(video) != LEARNED_RENDER_VIDEO_FIELDS
            or video.get("case") != expected_case
            or not isinstance(video.get("codec"), str)
            or not video["codec"]
            or not isinstance(video.get("width"), int)
            or video["width"] < 640
            or not isinstance(video.get("height"), int)
            or video["height"] < 360
            or not isinstance(video.get("fps"), (int, float))
            or isinstance(video.get("fps"), bool)
            or float(video["fps"]) <= 0.0
            or not isinstance(video.get("duration_s"), (int, float))
            or isinstance(video.get("duration_s"), bool)
            or float(video["duration_s"]) <= 0.0
        ):
            raise ValueError(f"learned render video {expected_case} metadata is invalid")
        _resolve_identity(
            {"path": video["path"], "sha256": video["sha256"]},
            directory=report_directory,
        )


def _windows_to_wsl_path(value: str) -> str:
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", value)
    if match is None:
        raise ValueError(f"cannot translate Windows path to WSL: {value}")
    drive, suffix = match.groups()
    return f"/mnt/{drive.lower()}/{suffix.replace(chr(92), '/')}"


def _git_value(*args: str) -> str:
    command = ["git", "-C", str(PROJECT_ROOT), *args]
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        if os.name != "nt":
            raise
        wsl_root = _windows_to_wsl_path(str(PROJECT_ROOT))
        return subprocess.run(
            ["wsl.exe", "--exec", "git", "-C", wsl_root, *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()


def _git_state() -> dict[str, Any]:
    root = _git_value("rev-parse", "--show-toplevel")
    branch = _git_value("branch", "--show-current")
    head = _git_value("rev-parse", "HEAD")
    upstream = _git_value("rev-parse", "@{upstream}")
    tracked_dirty = bool(_git_value("status", "--porcelain", "--untracked-files=no"))
    return {
        "root": root,
        "branch": branch,
        "head": head,
        "upstream": upstream,
        "tracked_dirty": tracked_dirty,
    }


def _gate(
    *,
    required: bool,
    passed: bool,
    evidence: list[str],
    detail: str,
) -> dict[str, Any]:
    return {
        "required_for_goal": required,
        "passed": bool(passed),
        "evidence": evidence,
        "detail": detail,
    }


def _optional_learning_evidence(
    *,
    training_dataset: Path | None,
    bc_admission: Path | None,
    bc_report: Path | None,
    learned_all79_admission: Path | None,
    learned_all79_preflight_receipt: Path | None,
    learned_plan_manifest: Path | None,
    learned_source_manifest: Path | None,
    learned_lqr_gains: Path | None,
    learned_robot_build_audit: Path | None,
    learned_robot_usd: Path | None,
    learned_drive_profile_selection: Path | None,
    validation_gate_report: Path | None,
    holdout_gate_report: Path | None,
    all79_report: Path | None,
    learned_render_report: Path | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    provided = {
        "training_dataset": training_dataset,
        "bc_admission": bc_admission,
        "bc_report": bc_report,
        "learned_all79_admission": learned_all79_admission,
        "learned_all79_preflight_receipt": learned_all79_preflight_receipt,
        "learned_plan_manifest": learned_plan_manifest,
        "learned_source_manifest": learned_source_manifest,
        "learned_lqr_gains": learned_lqr_gains,
        "learned_robot_build_audit": learned_robot_build_audit,
        "learned_robot_usd": learned_robot_usd,
        "learned_drive_profile_selection": learned_drive_profile_selection,
        "validation_gate_report": validation_gate_report,
        "holdout_gate_report": holdout_gate_report,
        "all79_report": all79_report,
        "learned_render_report": learned_render_report,
    }
    identities = {
        name: _identity(path)
        for name, path in provided.items()
        if path is not None and path.is_file()
    }
    training_metadata: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    if any(path is not None for path in (training_dataset, bc_admission, bc_report)):
        if None in (training_dataset, bc_admission, bc_report):
            raise ValueError(
                "training dataset, BC admission, and BC report must be supplied together"
            )
        assert training_dataset is not None
        assert bc_admission is not None
        assert bc_report is not None
        training_metadata, _ = load_training_dataset(training_dataset)
        admission = _load_json(bc_admission)
        report = _load_json(bc_report)
        code_paths = {
            "trainer": PROJECT_ROOT
            / "scripts/two_wheel_balance/train_riser_residual_bc.py",
            "adapter": SRC_ROOT
            / "rl_platform/tasks/two_wheel_balance/"
            "riser_model_based_corrective_bc_adapter.py",
            "loss_module": SRC_ROOT
            / "rl_platform/tasks/two_wheel_balance/riser_model_based_bc_loss.py",
            "policy_module": SRC_ROOT
            / "rl_platform/tasks/two_wheel_balance/riser_residual_policy.py",
            "training_dataset_module": SRC_ROOT
            / "rl_platform/tasks/two_wheel_balance/"
            "riser_model_based_corrective_training_dataset.py",
            "admission_module": SRC_ROOT
            / "rl_platform/tasks/two_wheel_balance/"
            "riser_model_based_corrective_bc_contract.py",
        }
        if set(code_paths) != CODE_IDENTITY_KEYS:
            raise RuntimeError("internal BC code identity set drifted")
        execution_commit = report.get("execution_commit")
        if not isinstance(execution_commit, str):
            raise ValueError("BC report has no execution commit")
        validate_bc_execution_admission(
            admission,
            dataset_path=training_dataset,
            dataset_metadata=training_metadata,
            code_paths=code_paths,
            expected_execution_commit=execution_commit,
            require_authorized=True,
        )
        validate_bc_execution_report(
            report,
            admission_path=bc_admission,
            admission=admission,
            report_directory=bc_report.parent,
        )

    all79: dict[str, Any] | None = None
    if all79_report is not None:
        if (
            report is None
            or bc_report is None
            or learned_all79_admission is None
            or learned_all79_preflight_receipt is None
            or learned_plan_manifest is None
            or learned_source_manifest is None
            or learned_lqr_gains is None
            or learned_robot_build_audit is None
            or learned_robot_usd is None
            or learned_drive_profile_selection is None
            or validation_gate_report is None
            or holdout_gate_report is None
        ):
            raise ValueError(
                "learned all-79 report requires BC, admission, validation, and "
                "holdout evidence"
            )
        rollout_admission = _load_json(learned_all79_admission)
        validation_gate = _load_json(validation_gate_report)
        holdout_gate = _load_json(holdout_gate_report)
        policy_path = _resolve_identity(
            report["torchscript"],
            directory=bc_report.parent,
        )
        rollout_code_paths = {
            "playback": PROJECT_ROOT
            / "scripts/two_wheel_balance/smoke_riser_reference_playback.py",
            "rollout_gate": PROJECT_ROOT
            / "scripts/two_wheel_balance/gate_riser_residual_rollouts.py",
            "completion_auditor": Path(__file__).resolve(),
            "admission_contract": SRC_ROOT
            / "rl_platform/tasks/two_wheel_balance/"
            "riser_model_based_learned_all79_contract.py",
            "preflight_validator": PROJECT_ROOT
            / "scripts/two_wheel_balance/"
            "validate_model_based_learned_all79_admission.py",
            "execution_wrapper": PROJECT_ROOT
            / "scripts/two_wheel_balance/"
            "run_model_based_learned_all79_policy_gate.sh",
        }
        validate_learned_all79_admission(
            rollout_admission,
            identity_root=PROJECT_ROOT,
            bc_report_path=bc_report,
            bc_report=report,
            policy_path=policy_path,
            plan_manifest_path=learned_plan_manifest,
            source_manifest_path=learned_source_manifest,
            lqr_gains_path=learned_lqr_gains,
            robot_build_audit_path=learned_robot_build_audit,
            robot_usd_path=learned_robot_usd,
            drive_profile_selection_path=learned_drive_profile_selection,
            validation_report_path=validation_gate_report,
            validation_report=validation_gate,
            holdout_report_path=holdout_gate_report,
            holdout_report=holdout_gate,
            code_paths=rollout_code_paths,
            expected_execution_commit=str(report["execution_commit"]),
            require_authorized=True,
        )
        all79 = _load_json(all79_report)
        policy_sha = report["torchscript"]["sha256"]
        _validate_all79_report(
            all79,
            policy_sha256=policy_sha,
            report_directory=all79_report.parent,
            admission_path=learned_all79_admission,
            preflight_path=learned_all79_preflight_receipt,
            plan_manifest_path=learned_plan_manifest,
            execution_commit=str(report["execution_commit"]),
        )

    learned_render: dict[str, Any] | None = None
    if learned_render_report is not None:
        if report is None or all79 is None or all79_report is None:
            raise ValueError(
                "learned render report requires validated BC and all-79 reports"
            )
        learned_render = _load_json(learned_render_report)
        policy_path = _resolve_identity(
            report["torchscript"],
            directory=bc_report.parent,
        )
        _validate_learned_render_report(
            learned_render,
            report_directory=learned_render_report.parent,
            policy_path=policy_path,
            policy_sha256=report["torchscript"]["sha256"],
            all79_report_path=all79_report,
        )

    return {
        "training_metadata": training_metadata,
        "bc_report": report,
        "all79_report": all79,
        "learned_render_report": learned_render,
    }, identities


def build_report(
    *,
    goal: Mapping[str, Any],
    asset: Mapping[str, Any],
    lqr: Mapping[str, Any],
    baseline: Mapping[str, Any],
    exact_source: Mapping[str, Any],
    hardware: Mapping[str, Any],
    bench: Mapping[str, Any],
    pending_route_queue: Mapping[str, Any],
    git_state: Mapping[str, Any],
    learning: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    robot_contract = goal.get("robot_contract", {})
    current_stage = goal.get("current_stage", {})
    status_refresh = current_stage.get("status_refresh_20260723", {})
    residual_contract = status_refresh.get("residual_dnn_admission_contract", {})
    corrective = status_refresh.get("model_based_corrective_teacher", {})
    asset_checks = asset.get("checks", {})
    static = baseline.get("static_height_gate", {})
    dynamic = baseline.get("dynamic_riser_gate", {})
    source = exact_source.get("source_package", {})
    portfolio = exact_source.get("gate_b_portfolio_v4", {})
    recommendation = hardware.get("recommendation", {})
    training_metadata = learning.get("training_metadata")
    bc_report = learning.get("bc_report")
    all79 = learning.get("all79_report")
    learned_render = learning.get("learned_render_report")

    training_cases = (
        training_metadata.get("split_cases", {})
        if isinstance(training_metadata, Mapping)
        else {}
    )
    training_ready = (
        isinstance(training_metadata, Mapping)
        and training_metadata.get("valid_for_projection_aware_bc_input") is True
        and training_metadata.get("valid_for_training") is True
        and len(training_cases.get("train", [])) >= 4
        and len(training_cases.get("validation", [])) >= 2
        and not set(training_cases.get("train", [])).intersection(
            training_cases.get("validation", [])
        )
    )
    bc_ready = (
        isinstance(bc_report, Mapping)
        and bc_report.get("offline_gate_passed") is True
        and bc_report.get("passed") is True
        and bc_report.get("valid_for_dynamic_canary") is True
        and bc_report.get("training_started") is True
        and bc_report.get("ppo_authorized") is False
    )
    all79_ready = isinstance(all79, Mapping)
    render_ready = isinstance(learned_render, Mapping)
    admitted_architecture = bc_contract.DEFAULT_BC_TRAINING_CONFIG
    architecture_contract_ready = (
        isinstance(residual_contract, Mapping)
        and residual_contract.get("architecture")
        == admitted_architecture["policy_architecture"]
        and residual_contract.get("observation_dimension")
        == admitted_architecture["observation_dimension"]
        and residual_contract.get("base_observation_dimension")
        == admitted_architecture["base_observation_dimension"]
        and residual_contract.get("lookahead_horizon_count")
        == admitted_architecture["lookahead_horizon_count"]
        and residual_contract.get("lookahead_channel_count_per_horizon")
        == admitted_architecture["lookahead_channel_count"]
        and residual_contract.get("action_dimension")
        == admitted_architecture["action_dimension"]
        and residual_contract.get("zero_initialize_action_head")
        is admitted_architecture["zero_initialize_action_head"]
        and residual_contract.get("checkpoint_trained") is False
    )
    corrective_case_count = current_stage.get(
        "model_based_corrective_case_datasets_available"
    )
    corpus_case_count = current_stage.get(
        "model_based_corrective_training_corpus_cases_available"
    )
    next_case = corrective.get("next_case")
    pending_routes = pending_route_queue.get("routes", [])
    pending_route_keys = (
        [route.get("key") for route in pending_routes]
        if isinstance(pending_routes, list)
        and all(isinstance(route, Mapping) for route in pending_routes)
        else []
    )
    pending_identity_count = (
        sum(
            int(route.get("identity_count", -1))
            for route in pending_routes
        )
        if pending_route_keys
        else -1
    )
    pending_checks = pending_route_queue.get("checks", {})
    pending_route_queue_bound = (
        inputs.get("pending_route_queue", {}).get("sha256")
        == corrective.get("pending_corrective_route_queue_v4_summary_sha256")
    )
    pending_authorization_closed = all(
        pending_route_queue.get(field) is False
        for field in PENDING_ROUTE_AUTHORIZATION_FIELDS
    )
    pending_route_queue_ready = (
        pending_route_queue.get("schema")
        == (
            "cinebotrl_two_wheel_riser_model_based_corrective_"
            "pending_route_queue_v1"
        )
        and pending_route_queue.get("passed") is True
        and pending_route_queue_bound
        and pending_route_queue.get("execution_order")
        == list(EXPECTED_PENDING_ROUTE_ORDER)
        and pending_route_keys == list(EXPECTED_PENDING_ROUTE_ORDER)
        and pending_route_queue.get("ready_route_count")
        == len(EXPECTED_PENDING_ROUTE_ORDER)
        and pending_identity_count == 107
        and all(route.get("passed") is True for route in pending_routes)
        and isinstance(pending_checks, Mapping)
        and bool(pending_checks)
        and all(value is True for value in pending_checks.values())
        and pending_route_queue.get("next_bounded_action")
        == "authorize_exactly_one_case23_v4_cpu_conversion"
        and pending_authorization_closed
    )
    pre_training_readiness = {
        "architecture_contract_passed": architecture_contract_ready,
        "policy_architecture": admitted_architecture["policy_architecture"],
        "observation_dimension": admitted_architecture["observation_dimension"],
        "base_observation_dimension": admitted_architecture[
            "base_observation_dimension"
        ],
        "lookahead_horizon_count": admitted_architecture[
            "lookahead_horizon_count"
        ],
        "lookahead_channel_count_per_horizon": admitted_architecture[
            "lookahead_channel_count"
        ],
        "action_dimension": admitted_architecture["action_dimension"],
        "zero_initialize_action_head": admitted_architecture[
            "zero_initialize_action_head"
        ],
        "corrective_case_datasets_available": corrective_case_count,
        "corrective_training_corpus_cases_available": corpus_case_count,
        "minimum_train_cases": 4,
        "minimum_validation_cases": 2,
        "next_case": next_case,
        "pending_route_queue_passed": pending_route_queue_ready,
        "pending_route_queue_bound_to_goal": pending_route_queue_bound,
        "pending_route_queue_ready_count": (
            pending_route_queue.get("ready_route_count")
        ),
        "pending_route_queue_identity_count": pending_identity_count,
        "pending_route_queue_execution_order": pending_route_keys,
        "pending_route_queue_all_authorization_closed": (
            pending_authorization_closed
        ),
        "next_operation": pending_route_queue.get("next_bounded_action"),
        "next_operation_authorized": (
            pending_route_queue.get("dataset_conversion_authorized") is True
        ),
        "bc_authorized": current_stage.get("bc_authorized") is True,
        "training_authorized": current_stage.get("training_authorized") is True,
        "ppo_authorized": current_stage.get("ppo_authorized") is True,
        "runtime_authorized": current_stage.get("runtime_authorized") is True,
        "ready_for_bc_execution": training_ready and architecture_contract_ready,
    }

    gates = {
        "isolated_worktree_and_branch": _gate(
            required=True,
            passed=git_state.get("branch") == EXPECTED_BRANCH
            and git_state.get("head") == git_state.get("upstream")
            and git_state.get("tracked_dirty") is False,
            evidence=[
                f"git:{git_state.get('branch', '')}@{git_state.get('head', '')}"
            ],
            detail="dedicated riser branch is clean and synchronized",
        ),
        "arm_free_robot_asset": _gate(
            required=True,
            passed=asset.get("passed") is True
            and asset_checks.get("arm_joints_absent") is True
            and set(asset.get("movable_joint_names", [])) == EXPECTED_MOVABLE_JOINTS
            and asset_checks.get("wheel_track_620mm") is True
            and asset_checks.get("wheel_diameter_8in") is True
            and robot_contract.get("arm_joint_count") == 0,
            evidence=[inputs["asset"]["path"]],
            detail="two wheels, riser, physical gimbal, and no movable arm joints",
        ),
        "frozen_lqr_balance_baseline": _gate(
            required=True,
            passed=lqr.get("schema") == "recomo_two_wheel_lqr_nominal_gate_v1"
            and lqr.get("passed") is True
            and lqr.get("training_started") is False
            and lqr.get("selected", {}).get("success_rate") == 1.0,
            evidence=[inputs["lqr"]["path"]],
            detail="provisional 28 kg simulation LQR baseline passed its nominal gate",
        ),
        "riser_height_and_speed_baseline": _gate(
            required=True,
            passed=baseline.get("passed") is True
            and static.get("passed") is True
            and static.get("camera_height_targets_m") == [0.6, 0.9, 1.8]
            and dynamic.get("passed") is True
            and dynamic.get("requested_speeds_mps") == [0.1, 0.25, 0.5, 1.0]
            and float(dynamic.get("measured_speed_at_1mps_mps", 0.0)) >= 0.95
            and robot_contract.get("camera_height_m") == [0.6, 1.8]
            and robot_contract.get("riser_speed_mps") == 1.0,
            evidence=[inputs["baseline"]["path"]],
            detail="camera range is capped at 1.8 m and the simulated 1 m/s gate passed",
        ),
        "exact_source_all79_reference": _gate(
            required=True,
            passed=source.get("case_count") == 79
            and portfolio.get("exact_source_pass_count") == 79
            and portfolio.get("kinematic_pass_count", 0) >= 40
            and portfolio.get("valid_for_training") is False,
            evidence=[inputs["exact_source"]["path"]],
            detail="79/79 immutable source references exist; planning is not training",
        ),
        "riser_motor_and_mechanism_recommendation": _gate(
            required=True,
            passed=hardware.get("passed") is True
            and hardware.get("candidate_ready_for_supplier_and_bench_review") is True
            and hardware.get("checks", {}).get("camera_height_ceiling_is_1p8m")
            is True
            and hardware.get("checks", {}).get("motor_speed_covers_1mps") is True
            and hardware.get("checks", {}).get(
                "motor_is_pinned_48v_750w_brake_absolute"
            )
            is True
            and float(
                hardware.get("calculated", {}).get(
                    "motor_mechanical_power_from_rating_w", 0.0
                )
            )
            >= 740.0
            and bool(recommendation.get("production_design_review_candidate")),
            evidence=[inputs["hardware"]["path"]],
            detail="750 W servo plus guided belt/telescoping mechanism is design-review ready",
        ),
        "model_based_corrective_training_corpus": _gate(
            required=True,
            passed=training_ready,
            evidence=(
                [inputs["training_dataset"]["path"]]
                if "training_dataset" in inputs
                else []
            ),
            detail="requires at least four train and two disjoint validation cases",
        ),
        "projection_aware_bc_policy": _gate(
            required=True,
            passed=bc_ready,
            evidence=[inputs["bc_report"]["path"]] if "bc_report" in inputs else [],
            detail="requires an authorized real BC run and passing offline report",
        ),
        "learned_policy_all79_dynamic_gate": _gate(
            required=True,
            passed=all79_ready,
            evidence=(
                [inputs["all79_report"]["path"]]
                if "all79_report" in inputs
                else []
            ),
            detail="requires one hash-bound learned-policy result for every case 1-79",
        ),
        "learned_policy_render_audit": _gate(
            required=True,
            passed=render_ready,
            evidence=(
                [inputs["learned_render_report"]["path"]]
                if "learned_render_report" in inputs
                else []
            ),
            detail="requires at least three audited learned-policy rollout videos",
        ),
        "physical_riser_bench_qualification": _gate(
            required=False,
            passed=bench.get("passed") is True
            and bench.get("ready_for_production_design_review") is True,
            evidence=[inputs["bench"]["path"]],
            detail="deployment qualification remains separate from the requested recommendation",
        ),
    }
    missing = [
        name
        for name in REQUIRED_COMPLETION_GATES
        if not gates[name]["passed"]
    ]
    return {
        "schema": GOAL_COMPLETION_AUDIT_SCHEMA,
        "objective": goal.get("objective"),
        "required_completion_gates": list(REQUIRED_COMPLETION_GATES),
        "gates": gates,
        "required_gate_pass_count": len(REQUIRED_COMPLETION_GATES) - len(missing),
        "required_gate_count": len(REQUIRED_COMPLETION_GATES),
        "completion_blockers": missing,
        "pre_training_readiness": pre_training_readiness,
        "goal_achieved": not missing,
        "obstacle_avoidance_in_scope": False,
        "runtime_started": False,
        "bc_started_by_audit": False,
        "ppo_started_by_audit": False,
        "git": {
            "branch": git_state.get("branch"),
            "head": git_state.get("head"),
            "upstream": git_state.get("upstream"),
            "tracked_dirty": git_state.get("tracked_dirty"),
        },
        "inputs": dict(inputs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", type=Path, default=DEFAULT_GOAL)
    parser.add_argument("--asset-audit", type=Path, default=DEFAULT_ASSET_AUDIT)
    parser.add_argument("--lqr-gate", type=Path, default=DEFAULT_LQR_GATE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--exact-source", type=Path, default=DEFAULT_EXACT_SOURCE)
    parser.add_argument("--hardware", type=Path, default=DEFAULT_HARDWARE)
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument(
        "--pending-route-queue",
        type=Path,
        default=DEFAULT_PENDING_ROUTE_QUEUE,
    )
    parser.add_argument("--training-dataset", type=Path)
    parser.add_argument("--bc-admission", type=Path)
    parser.add_argument("--bc-report", type=Path)
    parser.add_argument("--learned-all79-admission", type=Path)
    parser.add_argument("--learned-all79-preflight-receipt", type=Path)
    parser.add_argument("--learned-plan-manifest", type=Path)
    parser.add_argument("--learned-source-manifest", type=Path)
    parser.add_argument("--learned-lqr-gains", type=Path)
    parser.add_argument("--learned-robot-build-audit", type=Path)
    parser.add_argument("--learned-robot-usd", type=Path)
    parser.add_argument("--learned-drive-profile-selection", type=Path)
    parser.add_argument("--validation-gate-report", type=Path)
    parser.add_argument("--holdout-gate-report", type=Path)
    parser.add_argument("--all79-report", type=Path)
    parser.add_argument("--learned-render-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    fixed_paths = {
        "auditor": Path(__file__),
        "goal": args.goal,
        "asset": args.asset_audit,
        "lqr": args.lqr_gate,
        "baseline": args.baseline,
        "exact_source": args.exact_source,
        "hardware": args.hardware,
        "bench": args.bench,
        "pending_route_queue": args.pending_route_queue,
    }
    inputs = {name: _identity(path) for name, path in fixed_paths.items()}
    learning, optional_identities = _optional_learning_evidence(
        training_dataset=args.training_dataset,
        bc_admission=args.bc_admission,
        bc_report=args.bc_report,
        learned_all79_admission=args.learned_all79_admission,
        learned_all79_preflight_receipt=args.learned_all79_preflight_receipt,
        learned_plan_manifest=args.learned_plan_manifest,
        learned_source_manifest=args.learned_source_manifest,
        learned_lqr_gains=args.learned_lqr_gains,
        learned_robot_build_audit=args.learned_robot_build_audit,
        learned_robot_usd=args.learned_robot_usd,
        learned_drive_profile_selection=args.learned_drive_profile_selection,
        validation_gate_report=args.validation_gate_report,
        holdout_gate_report=args.holdout_gate_report,
        all79_report=args.all79_report,
        learned_render_report=args.learned_render_report,
    )
    inputs.update(optional_identities)
    report = build_report(
        goal=_load_json(args.goal),
        asset=_load_json(args.asset_audit),
        lqr=_load_json(args.lqr_gate),
        baseline=_load_json(args.baseline),
        exact_source=_load_json(args.exact_source),
        hardware=_load_json(args.hardware),
        bench=_load_json(args.bench),
        pending_route_queue=_load_json(args.pending_route_queue),
        git_state=_git_state(),
        learning=learning,
        inputs=inputs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(report, indent=2))
    if report["goal_achieved"] or args.allow_incomplete:
        return 0
    return 8


if __name__ == "__main__":
    raise SystemExit(main())
