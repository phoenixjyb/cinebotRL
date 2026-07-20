#!/usr/bin/env python3
"""Seal a pass or fail Gate C canary summary from per-case runtime JSONs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_CONTROLLER_PROFILE = "structural_robust_v1"
EXPECTED_TRACKING_PROFILE = "riser_recovery_direction_v4"
CAMERA_LEVER_ARM_TRACKING_PROFILE = (
    "riser_recovery_direction_v4_camera_lever_arm_v1"
)
CAMERA_ERROR_GOVERNOR_TRACKING_PROFILE = (
    "riser_recovery_direction_v4_camera_lever_arm_error_governor_v1"
)
ZERO_PROGRESS_HOLD_TRACKING_PROFILE = (
    "riser_recovery_direction_v4_camera_lever_arm_zero_progress_hold_v1"
)
ZERO_PROGRESS_HOLD_VELOCITY_CAP_TRACKING_PROFILE = (
    "riser_recovery_direction_v4_camera_lever_arm_zero_progress_hold_velocity_cap_v1"
)
ZERO_PROGRESS_HOLD_VELOCITY_CAP_TOTAL_PITCH_TRACKING_PROFILE = (
    "riser_recovery_direction_v4_camera_lever_arm_"
    "zero_progress_hold_velocity_cap_total_pitch_limit_v1"
)
EXPECTED_TOTAL_PITCH_REFERENCE_LIMIT_RAD = math.radians(6.0)
PHASE_GOVERNOR_CONTRACT = "position_error_continuous_phase_scale_v1"
COMMANDED_BASE_PHASE_GOVERNOR_CONTRACT = (
    "commanded_base_and_camera_error_continuous_phase_scale_v1"
)
COMMANDED_BASE_PROGRESS_ERROR_SOURCE = (
    "lever_compensated_commanded_base_target"
)
CAMERA_LEVER_ARM_COMPENSATION_CONTRACT = (
    "measured_camera_to_base_xy_offset_v1"
)
CAMERA_ERROR_GOVERNOR_CONTRACT = (
    "saturated_camera_error_continuous_phase_cap_v1"
)
EXPECTED_RECOVERY_ERROR_RANGE_M = [0.2, 0.4]
EXPECTED_RISER_THERMAL_FORCE_CONTRACT = "leadshine_400w_first_order_monitor_v1"
EXPECTED_RECOVERY_TELEMETRY_SCHEMA = "riser_recovery_direction_policy_rate_v1"
EXPECTED_LONGITUDINAL_AUTHORITY_TELEMETRY_SCHEMA = (
    "riser_longitudinal_authority_policy_rate_v1"
)
REVIEWED_OPPOSING_VX_INTEGRAL_RESET_PARENT_COMMIT = (
    "35f775c39ed2d0c22b52be5dd8f9641354ee0b8f"
)
REQUIRED_CONTRACT_IDENTITIES = {
    "source_manifest",
    "portfolio_manifest",
    "case74_plan",
    "lqr_gains",
    "robot_usd",
    "tracking_controller",
    "riser_control",
    "recovery_evidence",
    "playback",
    "case74_wrapper",
    "shared_runner",
    "summarizer",
    "contract_validator",
    "portfolio_validator",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recovery_telemetry_passed(result: dict[str, object]) -> bool:
    telemetry = result.get("recovery_telemetry")
    if not isinstance(telemetry, dict):
        return False
    integer_fields = (
        "policy_rate_sample_count",
        "activation_step_count",
        "full_authority_step_count",
        "activation_segment_count",
        "motion_direction_sign_change_count",
        "feedback_direction_sign_change_count",
        "consecutive_active_motion_direction_chatter_count",
        "candidate_yaw_saturation_step_count",
        "legacy_yaw_saturation_step_count",
        "candidate_vs_legacy_delta_nonzero_step_count",
    )
    values = [telemetry.get(name) for name in integer_fields]
    if not all(isinstance(value, int) and value >= 0 for value in values):
        return False
    samples = telemetry["policy_rate_sample_count"]
    active = telemetry["activation_step_count"]
    return (
        telemetry.get("schema") == EXPECTED_RECOVERY_TELEMETRY_SCHEMA
        and samples == result.get("completed_steps")
        and 0 <= active <= samples
        and telemetry["full_authority_step_count"] <= active
        and telemetry["activation_segment_count"] <= active
        and telemetry["motion_direction_sign_change_count"] <= active
        and telemetry["feedback_direction_sign_change_count"] <= active
        and telemetry["consecutive_active_motion_direction_chatter_count"]
        <= active
        and telemetry["candidate_yaw_saturation_step_count"] <= active
        and telemetry["legacy_yaw_saturation_step_count"] <= active
        and telemetry["candidate_vs_legacy_delta_nonzero_step_count"] <= active
        and isinstance(
            telemetry.get("candidate_vs_legacy_yaw_delta_abs_max_rad_s"),
            (int, float),
        )
        and math.isfinite(
            telemetry["candidate_vs_legacy_yaw_delta_abs_max_rad_s"]
        )
        and telemetry["candidate_vs_legacy_yaw_delta_abs_max_rad_s"] >= 0.0
        and isinstance(telemetry.get("recovery_blend_max"), (int, float))
        and math.isfinite(telemetry["recovery_blend_max"])
        and 0.0 <= telemetry["recovery_blend_max"] <= 1.0
    )


def camera_lever_arm_telemetry_passed(
    payload: dict[str, object],
    result: dict[str, object],
    *,
    expected_gain: float,
    expected_maximum_correction_m: float,
) -> bool:
    numeric_fields = (
        "camera_lever_arm_correction_max_m",
        "camera_lever_arm_raw_correction_max_m",
        "camera_lever_arm_correction_saturation_ratio",
    )
    values = [result.get(name) for name in numeric_fields]
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in values
    ):
        return False
    correction_max, raw_max, saturation_ratio = values
    return (
        payload.get("camera_lever_arm_compensation_contract")
        == CAMERA_LEVER_ARM_COMPENSATION_CONTRACT
        and payload.get("camera_lever_arm_compensation_enabled") is True
        and payload.get("camera_lever_arm_compensation_gain") == expected_gain
        and payload.get("maximum_camera_lever_arm_correction_m")
        == expected_maximum_correction_m
        and result.get("camera_lever_arm_compensation_enabled") is True
        and result.get("camera_lever_arm_compensation_gain") == expected_gain
        and result.get("maximum_camera_lever_arm_correction_m")
        == expected_maximum_correction_m
        and payload.get("controller_evidence_passed") is True
        and result.get("controller_evidence_passed") is True
        and result.get("camera_lever_arm_telemetry_observed") is True
        and result.get("camera_lever_arm_telemetry_sample_count")
        == result.get("completed_steps")
        and 0.0 <= correction_max <= expected_maximum_correction_m + 1e-9
        and raw_max + 1e-12 >= correction_max
        and 0.0 <= saturation_ratio <= 1.0
    )


def commanded_base_progress_error_passed(
    payload: dict[str, object],
    result: dict[str, object],
    *,
    expected_maximum_correction_m: float,
) -> bool:
    numeric_fields = (
        "nominal_base_progress_error_p95_m",
        "nominal_base_progress_error_max_m",
        "commanded_base_progress_error_p95_m",
        "commanded_base_progress_error_max_m",
        "selected_base_progress_error_p95_m",
        "selected_base_progress_error_max_m",
        "selected_vs_nominal_base_progress_error_mean_delta_m",
        "selected_vs_nominal_base_progress_error_abs_max_delta_m",
        "maximum_commanded_base_progress_error_delta_m",
    )
    values = [result.get(name) for name in numeric_fields]
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value)
        for value in values
    ):
        return False
    nonnegative = values[:6]
    abs_delta = values[7]
    configured_delta = values[8]
    return bool(
        payload.get("phase_governor_enabled") is True
        and payload.get("phase_governor_contract")
        == COMMANDED_BASE_PHASE_GOVERNOR_CONTRACT
        and payload.get("commanded_base_progress_error_enabled") is True
        and payload.get("progress_base_error_source")
        == COMMANDED_BASE_PROGRESS_ERROR_SOURCE
        and result.get("phase_governor_contract")
        == COMMANDED_BASE_PHASE_GOVERNOR_CONTRACT
        and result.get("commanded_base_progress_error_enabled") is True
        and result.get("progress_base_error_source")
        == COMMANDED_BASE_PROGRESS_ERROR_SOURCE
        and result.get("progress_base_error_telemetry_observed") is True
        and result.get("progress_base_error_telemetry_sample_count")
        == result.get("completed_steps")
        and result.get("progress_base_error_selected_source_matches") is True
        and result.get("progress_base_error_command_delta_bounded") is True
        and result.get("checks", {}).get(
            "progress_base_error_telemetry_observed"
        )
        is True
        and result.get("checks", {}).get(
            "progress_base_error_selected_source_matches"
        )
        is True
        and result.get("checks", {}).get(
            "progress_base_error_command_delta_bounded"
        )
        is True
        and all(value >= 0.0 for value in nonnegative)
        and configured_delta == expected_maximum_correction_m
        and 0.0 <= abs_delta <= expected_maximum_correction_m + 1e-9
    )


def longitudinal_authority_telemetry_passed(
    payload: dict[str, object],
    result: dict[str, object],
    *,
    expected_deadband_mps: float,
) -> bool:
    telemetry = result.get("longitudinal_authority_telemetry")
    if not isinstance(telemetry, dict):
        return False
    integer_fields = (
        "policy_rate_sample_count",
        "controller_update_count",
        "held_controller_command_step_count",
        "reference_sign_change_count",
        "opposing_integral_sign_change_count",
        "integral_reset_count",
        "velocity_deficit_step_count",
        "total_pitch_limit_step_count",
    )
    numeric_fields = (
        "velocity_deficit_ratio",
        "velocity_deficit_mean_mps",
        "velocity_deficit_abs_max_mps",
        "deficit_pitch_contribution_mean",
        "deficit_pitch_rate_contribution_mean",
        "deficit_wheel_velocity_contribution_mean",
        "vx_integral_before_abs_max",
        "vx_integral_after_abs_max",
        "pitch_abs_max_rad",
        "pitch_rate_abs_max_rad_s",
        "total_pitch_reference_abs_max_rad",
        "common_action_abs_max",
        "reference_deadband_mps",
        "deficit_tolerance_mps",
    )
    if not all(
        isinstance(telemetry.get(name), int) and telemetry[name] >= 0
        for name in integer_fields
    ):
        return False
    if not all(
        isinstance(telemetry.get(name), (int, float))
        and math.isfinite(telemetry[name])
        for name in numeric_fields
    ):
        return False
    samples = telemetry["policy_rate_sample_count"]
    updates = telemetry["controller_update_count"]
    return bool(
        payload.get("opposing_vx_integral_deficit_reset_enabled") is True
        and payload.get("vx_integral_reset_reference_deadband_mps")
        == expected_deadband_mps
        and result.get("opposing_vx_integral_deficit_reset_enabled") is True
        and result.get("vx_integral_reset_reference_deadband_mps")
        == expected_deadband_mps
        and result.get("longitudinal_authority_telemetry_observed") is True
        and result.get("checks", {}).get(
            "longitudinal_authority_telemetry_observed"
        )
        is True
        and telemetry.get("schema")
        == EXPECTED_LONGITUDINAL_AUTHORITY_TELEMETRY_SCHEMA
        and samples == result.get("completed_steps")
        and 0 < updates <= samples
        and telemetry["held_controller_command_step_count"] == samples - updates
        and telemetry["reference_sign_change_count"] <= updates
        and telemetry["opposing_integral_sign_change_count"]
        <= telemetry["reference_sign_change_count"]
        and 0 < telemetry["integral_reset_count"] <= updates
        and telemetry["velocity_deficit_step_count"] <= samples
        and telemetry["total_pitch_limit_step_count"] <= samples
        and telemetry["reference_deadband_mps"] == expected_deadband_mps
        and telemetry["deficit_tolerance_mps"] > 0.0
        and 0.0 <= telemetry["velocity_deficit_ratio"] <= 1.0
        and telemetry["velocity_deficit_mean_mps"] >= 0.0
        and telemetry["velocity_deficit_abs_max_mps"] >= 0.0
        and telemetry["vx_integral_before_abs_max"] >= 0.0
        and telemetry["vx_integral_after_abs_max"] >= 0.0
        and telemetry["pitch_abs_max_rad"] >= 0.0
        and telemetry["pitch_rate_abs_max_rad_s"] >= 0.0
        and telemetry["total_pitch_reference_abs_max_rad"] >= 0.0
        and telemetry["common_action_abs_max"] >= 0.0
    )


def camera_error_governor_telemetry_passed(
    payload: dict[str, object],
    result: dict[str, object],
    *,
    expected_error_range_m: list[float],
    expected_minimum_scale: float,
) -> bool:
    sample_count = result.get("camera_recovery_telemetry_sample_count")
    activation_ratio = result.get("camera_recovery_activation_ratio")
    scale_min = result.get("camera_recovery_progress_scale_min")
    scale_mean = result.get("camera_recovery_progress_scale_mean")
    trace = result.get("trace")
    return (
        payload.get("camera_recovery_governor_enabled") is True
        and payload.get("camera_recovery_governor_contract")
        == CAMERA_ERROR_GOVERNOR_CONTRACT
        and payload.get("camera_recovery_error_range_m")
        == expected_error_range_m
        and payload.get("minimum_camera_recovery_scale")
        == expected_minimum_scale
        and result.get("camera_recovery_governor_enabled") is True
        and result.get("camera_recovery_governor_contract")
        == CAMERA_ERROR_GOVERNOR_CONTRACT
        and result.get("camera_recovery_error_range_m")
        == expected_error_range_m
        and result.get("minimum_camera_recovery_scale")
        == expected_minimum_scale
        and result.get("camera_recovery_telemetry_observed") is True
        and result.get("checks", {}).get("camera_recovery_telemetry_observed")
        is True
        and isinstance(sample_count, int)
        and sample_count == result.get("completed_steps")
        and all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in (activation_ratio, scale_min, scale_mean)
        )
        and 0.0 < activation_ratio <= 1.0
        and expected_minimum_scale - 1e-12 <= scale_min <= 1.0
        and scale_min <= scale_mean <= 1.0
        and isinstance(trace, list)
        and len(trace) > 0
        and all(
            isinstance(row.get("camera_recovery_progress_scale"), (int, float))
            and math.isfinite(row["camera_recovery_progress_scale"])
            and expected_minimum_scale - 1e-12
            <= row["camera_recovery_progress_scale"]
            <= 1.0
            and isinstance(row.get("camera_recovery_active"), bool)
            for row in trace
        )
    )


def zero_progress_hold_telemetry_passed(
    payload: dict[str, object],
    result: dict[str, object],
    *,
    expected_maximum_linear_velocity_mps: float | None = None,
    expected_phase_governor_contract: str = PHASE_GOVERNOR_CONTRACT,
) -> bool:
    completed_steps = result.get("completed_steps")
    hold_steps = result.get("progress_hold_step_count")
    hold_ratio = result.get("progress_hold_ratio")
    hold_segments = result.get("progress_hold_segment_count")
    progress_min = result.get("progress_scale_min")
    expected_tracking_overrides = {"minimum_progress_scale": 0.0}
    if expected_maximum_linear_velocity_mps is not None:
        expected_tracking_overrides["maximum_linear_velocity_mps"] = (
            expected_maximum_linear_velocity_mps
        )
    velocity_feedback = result.get("velocity_feedback_telemetry")
    velocity_cap_ok = expected_maximum_linear_velocity_mps is None or (
        payload.get("tracking_recovery_velocity_cap_enabled") is True
        and payload.get("maximum_linear_velocity_mps")
        == expected_maximum_linear_velocity_mps
        and result.get("maximum_linear_velocity_mps")
        == expected_maximum_linear_velocity_mps
        and isinstance(velocity_feedback, dict)
        and isinstance(
            velocity_feedback.get("effective_reference_abs_max_mps"),
            (int, float),
        )
        and math.isfinite(velocity_feedback["effective_reference_abs_max_mps"])
        and velocity_feedback["effective_reference_abs_max_mps"]
        <= expected_maximum_linear_velocity_mps + 1e-9
    )
    return (
        payload.get("phase_governor_enabled") is True
        and payload.get("phase_governor_contract")
        == expected_phase_governor_contract
        and payload.get("minimum_progress_scale") == 0.0
        and payload.get("tracking_overrides") == expected_tracking_overrides
        and result.get("minimum_progress_scale") == 0.0
        and result.get("outer_velocity_feedback_source") == "wheel_derived_vx"
        and isinstance(completed_steps, int)
        and completed_steps > 0
        and isinstance(hold_steps, int)
        and 0 < hold_steps <= completed_steps
        and isinstance(hold_segments, int)
        and 0 < hold_segments <= hold_steps
        and isinstance(hold_ratio, (int, float))
        and math.isfinite(hold_ratio)
        and math.isclose(
            hold_ratio, hold_steps / completed_steps, rel_tol=0.0, abs_tol=1e-12
        )
        and isinstance(progress_min, (int, float))
        and math.isfinite(progress_min)
        and progress_min == 0.0
        and velocity_cap_ok
    )


def contract_identity_rows_passed(admission: dict[str, object]) -> bool:
    identities = admission.get("identities")
    checks = admission.get("checks")
    if not isinstance(identities, dict) or set(identities) != REQUIRED_CONTRACT_IDENTITIES:
        return False
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        return False
    for name, row in identities.items():
        if not isinstance(row, dict) or row.get("passed") is not True:
            return False
        sha = row.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            return False
        if name != "source_manifest":
            blob = row.get("git_blob_sha1")
            if not isinstance(blob, str) or len(blob) != 40:
                return False
    contract_blob = admission.get("contract_git_blob_sha1")
    return isinstance(contract_blob, str) and len(contract_blob) == 40


def total_pitch_reference_limit_passed(
    payload: dict[str, object],
    result: dict[str, object],
    *,
    required: bool,
    require_opposing_vx_integral_deficit_reset: bool = False,
    expected_vx_integral_reset_deadband_mps: float = 0.05,
) -> bool:
    if not required:
        return True
    telemetry = result.get("velocity_feedback_telemetry")
    if not isinstance(telemetry, dict):
        return False
    total_max = telemetry.get("total_pitch_reference_abs_max_rad")
    velocity_offset_max = telemetry.get("pitch_reference_abs_max_rad")
    expected_controller_overrides: dict[str, object] = {
        "wz_kp": 1.05,
        "limit_total_pitch_reference": True,
    }
    if require_opposing_vx_integral_deficit_reset:
        expected_controller_overrides.update(
            {
                "reset_opposing_vx_integral_on_directional_deficit": True,
                "vx_integral_reset_reference_deadband_mps": (
                    expected_vx_integral_reset_deadband_mps
                ),
            }
        )
    return bool(
        payload.get("total_pitch_reference_limit_enabled") is True
        and payload.get("total_pitch_reference_limit_rad")
        == EXPECTED_TOTAL_PITCH_REFERENCE_LIMIT_RAD
        and result.get("total_pitch_reference_limit_enabled") is True
        and result.get("total_pitch_reference_limit_rad")
        == EXPECTED_TOTAL_PITCH_REFERENCE_LIMIT_RAD
        and payload.get("controller_overrides") == expected_controller_overrides
        and isinstance(total_max, (int, float))
        and math.isfinite(total_max)
        and EXPECTED_TOTAL_PITCH_REFERENCE_LIMIT_RAD - 1e-9
        <= total_max
        <= EXPECTED_TOTAL_PITCH_REFERENCE_LIMIT_RAD + 1e-9
        and isinstance(velocity_offset_max, (int, float))
        and math.isfinite(velocity_offset_max)
        and velocity_offset_max > EXPECTED_TOTAL_PITCH_REFERENCE_LIMIT_RAD + 1e-6
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-case74-contract", action="store_true")
    parser.add_argument("--expected-case74-contract-sha256")
    parser.add_argument(
        "--expected-tracking-profile",
        choices=(
            EXPECTED_TRACKING_PROFILE,
            CAMERA_LEVER_ARM_TRACKING_PROFILE,
            CAMERA_ERROR_GOVERNOR_TRACKING_PROFILE,
            ZERO_PROGRESS_HOLD_TRACKING_PROFILE,
            ZERO_PROGRESS_HOLD_VELOCITY_CAP_TRACKING_PROFILE,
            ZERO_PROGRESS_HOLD_VELOCITY_CAP_TOTAL_PITCH_TRACKING_PROFILE,
        ),
        default=EXPECTED_TRACKING_PROFILE,
    )
    parser.add_argument(
        "--require-camera-lever-arm-compensation", action="store_true"
    )
    parser.add_argument("--require-camera-error-recovery-governor", action="store_true")
    parser.add_argument("--require-zero-progress-hold", action="store_true")
    parser.add_argument("--require-recovery-velocity-cap", action="store_true")
    parser.add_argument("--require-total-pitch-reference-limit", action="store_true")
    parser.add_argument(
        "--require-commanded-base-progress-error", action="store_true"
    )
    parser.add_argument(
        "--require-opposing-vx-integral-deficit-reset", action="store_true"
    )
    parser.add_argument(
        "--expected-vx-integral-reset-reference-deadband-mps",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--expected-maximum-linear-velocity-mps", type=float, default=0.2
    )
    parser.add_argument("--expected-camera-lever-arm-gain", type=float, default=1.0)
    parser.add_argument(
        "--expected-maximum-camera-lever-arm-correction-m",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--expected-camera-recovery-error-start-m", type=float, default=0.13
    )
    parser.add_argument(
        "--expected-camera-recovery-error-full-m", type=float, default=0.155
    )
    parser.add_argument(
        "--expected-minimum-camera-recovery-scale", type=float, default=0.20
    )
    args = parser.parse_args()
    if args.require_camera_lever_arm_compensation and (
        args.expected_tracking_profile
        not in {
            CAMERA_LEVER_ARM_TRACKING_PROFILE,
            CAMERA_ERROR_GOVERNOR_TRACKING_PROFILE,
            ZERO_PROGRESS_HOLD_TRACKING_PROFILE,
            ZERO_PROGRESS_HOLD_VELOCITY_CAP_TRACKING_PROFILE,
            ZERO_PROGRESS_HOLD_VELOCITY_CAP_TOTAL_PITCH_TRACKING_PROFILE,
        }
    ):
        parser.error("camera lever-arm compensation requires its tracking profile")
    if args.require_camera_error_recovery_governor and (
        not args.require_camera_lever_arm_compensation
        or args.expected_tracking_profile != CAMERA_ERROR_GOVERNOR_TRACKING_PROFILE
    ):
        parser.error(
            "camera error recovery requires lever-arm compensation and its tracking profile"
        )
    if args.require_zero_progress_hold and (
        not args.require_camera_lever_arm_compensation
        or args.require_camera_error_recovery_governor
        or args.expected_tracking_profile
        not in {
            ZERO_PROGRESS_HOLD_TRACKING_PROFILE,
            ZERO_PROGRESS_HOLD_VELOCITY_CAP_TRACKING_PROFILE,
            ZERO_PROGRESS_HOLD_VELOCITY_CAP_TOTAL_PITCH_TRACKING_PROFILE,
        }
    ):
        parser.error(
            "zero-progress hold requires lever-arm compensation and its tracking profile"
        )
    if args.require_recovery_velocity_cap and (
        not args.require_zero_progress_hold
        or args.expected_tracking_profile
        not in {
            ZERO_PROGRESS_HOLD_VELOCITY_CAP_TRACKING_PROFILE,
            ZERO_PROGRESS_HOLD_VELOCITY_CAP_TOTAL_PITCH_TRACKING_PROFILE,
        }
        or not 0.0 < args.expected_maximum_linear_velocity_mps <= 0.4
    ):
        parser.error(
            "recovery velocity cap requires the capped hold profile and a limit in (0, 0.4]"
        )
    if args.require_total_pitch_reference_limit and (
        not args.require_recovery_velocity_cap
        or args.expected_tracking_profile
        != ZERO_PROGRESS_HOLD_VELOCITY_CAP_TOTAL_PITCH_TRACKING_PROFILE
    ):
        parser.error(
            "total pitch-reference limiting requires the capped hold total-pitch profile"
        )
    if args.require_commanded_base_progress_error and (
        not args.require_camera_lever_arm_compensation
        or not args.require_zero_progress_hold
    ):
        parser.error(
            "commanded-base progress error requires lever compensation and zero-progress hold"
        )
    if args.require_opposing_vx_integral_deficit_reset and (
        not args.require_commanded_base_progress_error
        or not args.require_total_pitch_reference_limit
        or not math.isfinite(
            args.expected_vx_integral_reset_reference_deadband_mps
        )
        or args.expected_vx_integral_reset_reference_deadband_mps <= 0.0
    ):
        parser.error(
            "opposing PI-memory reset requires commanded-base progress, the "
            "total-pitch limit, and a positive finite reference deadband"
        )
    expected_phase_governor_contract = (
        COMMANDED_BASE_PHASE_GOVERNOR_CONTRACT
        if args.require_commanded_base_progress_error
        else PHASE_GOVERNOR_CONTRACT
    )
    requested = [int(value) for value in args.cases.split(",")]
    admission = args.root / "admission.json"
    admission_payload = json.loads(admission.read_text(encoding="utf-8"))
    contract_admission_passed = not args.require_case74_contract or (
        requested == [74]
        and args.expected_case74_contract_sha256 is not None
        and admission_payload.get("schema")
        == "cinebotrl_case74_recovery_v4_runtime_contract_admission_v2"
        and admission_payload.get("contract_sha256")
        == args.expected_case74_contract_sha256
        and admission_payload.get("reviewed_controller_parent_commit")
        == "ba8f4e0b44dc15a60d61b8353a208032727ad0ae"
        and admission_payload.get("runtime_commit") == args.git_commit
        and admission_payload.get("upstream_commit") == args.git_commit
        and admission_payload.get("case") == 74
        and admission_payload.get("namespace") == args.root.name
        and admission_payload.get("tracking_profile") == EXPECTED_TRACKING_PROFILE
        and admission_payload.get("recovery_error_range_m")
        == EXPECTED_RECOVERY_ERROR_RANGE_M
        and admission_payload.get("identity_passed") is True
        and contract_identity_rows_passed(admission_payload)
        and admission_payload.get("runtime_authorized") is True
        and admission_payload.get("gate_c_execution_authorized") is True
        and admission_payload.get("residual_capture_authorized") is False
        and admission_payload.get("bc_authorized") is False
        and admission_payload.get("ppo_authorized") is False
        and admission_payload.get("valid_for_training") is False
    )
    zero_progress_hold_admission_passed = not args.require_zero_progress_hold or (
        requested == [42]
        and admission_payload.get("requested_cases") == [42]
        and admission_payload.get("namespace") == args.root.name
        and admission_payload.get("runtime_commit") == args.git_commit
        and admission_payload.get("upstream_commit") == args.git_commit
        and admission_payload.get("tracking_profile")
        == args.expected_tracking_profile
        and admission_payload.get("zero_progress_hold_required") is True
        and admission_payload.get("phase_governor_contract")
        == expected_phase_governor_contract
        and (
            not args.require_commanded_base_progress_error
            or (
                admission_payload.get(
                    "commanded_base_progress_error_required"
                )
                is True
                and admission_payload.get("progress_base_error_source")
                == COMMANDED_BASE_PROGRESS_ERROR_SOURCE
                and admission_payload.get(
                    "maximum_commanded_base_progress_error_delta_m"
                )
                == args.expected_maximum_camera_lever_arm_correction_m
            )
        )
        and (
            not args.require_opposing_vx_integral_deficit_reset
            or (
                admission_payload.get(
                    "opposing_vx_integral_deficit_reset_required"
                )
                is True
                and admission_payload.get(
                    "vx_integral_reset_reference_deadband_mps"
                )
                == args.expected_vx_integral_reset_reference_deadband_mps
                and admission_payload.get(
                    "longitudinal_authority_telemetry_schema"
                )
                == EXPECTED_LONGITUDINAL_AUTHORITY_TELEMETRY_SCHEMA
                and admission_payload.get("reviewed_controller_parent_commit")
                == REVIEWED_OPPOSING_VX_INTEGRAL_RESET_PARENT_COMMIT
                and admission_payload.get(
                    "reviewed_controller_parent_is_ancestor"
                )
                is True
            )
        )
        and admission_payload.get("minimum_progress_scale") == 0.0
        and (
            not args.require_recovery_velocity_cap
            or (
                admission_payload.get("recovery_velocity_cap_required") is True
                and admission_payload.get("maximum_linear_velocity_mps")
                == args.expected_maximum_linear_velocity_mps
                and (
                    not args.require_total_pitch_reference_limit
                    or (
                        admission_payload.get(
                            "total_pitch_reference_limit_required"
                        )
                        is True
                        and admission_payload.get(
                            "total_pitch_reference_limit_rad"
                        )
                        == EXPECTED_TOTAL_PITCH_REFERENCE_LIMIT_RAD
                    )
                )
            )
        )
        and admission_payload.get("root_velocity_outer_feedback_enabled") is False
        and admission_payload.get("runtime_authorized") is True
        and admission_payload.get("residual_capture_authorized") is False
        and admission_payload.get("bc_authorized") is False
        and admission_payload.get("ppo_authorized") is False
        and admission_payload.get("valid_for_training") is False
    )
    passed_cases = []
    gate_rows = []
    first_reject = None
    for case in requested:
        gate = args.root / "gates" / f"case_{case:04d}.json"
        log = args.root / "logs" / f"case_{case:04d}.log"
        if not gate.is_file():
            first_reject = {
                "case": case,
                "classification": "missing_runtime_json",
                "log": str(log.resolve()) if log.is_file() else None,
                "log_sha256": sha256_file(log) if log.is_file() else None,
            }
            break
        payload = json.loads(gate.read_text(encoding="utf-8"))
        result = payload.get("results", [{}])[0]
        physical_dynamic_passed = (
            payload.get("dynamic_quality_passed") is True
            and result.get("dynamic_quality_passed") is True
        )
        thermal_admission_passed = (
            payload.get("thermal_admission_passed") is True
            and result.get("thermal_admission_passed") is True
            and result.get("checks", {}).get("riser_thermal_force_observed") is True
            and result.get("checks", {}).get("riser_thermal_load_bounded") is True
            and result.get("checks", {}).get("riser_peak_force_bounded") is True
        )
        camera_lever_arm_evidence_passed = (
            not args.require_camera_lever_arm_compensation
            or camera_lever_arm_telemetry_passed(
                payload,
                result,
                expected_gain=args.expected_camera_lever_arm_gain,
                expected_maximum_correction_m=(
                    args.expected_maximum_camera_lever_arm_correction_m
                ),
            )
        )
        camera_error_governor_evidence_passed = (
            not args.require_camera_error_recovery_governor
            or camera_error_governor_telemetry_passed(
                payload,
                result,
                expected_error_range_m=[
                    args.expected_camera_recovery_error_start_m,
                    args.expected_camera_recovery_error_full_m,
                ],
                expected_minimum_scale=(
                    args.expected_minimum_camera_recovery_scale
                ),
            )
        )
        commanded_base_progress_error_evidence_passed = (
            not args.require_commanded_base_progress_error
            or commanded_base_progress_error_passed(
                payload,
                result,
                expected_maximum_correction_m=(
                    args.expected_maximum_camera_lever_arm_correction_m
                ),
            )
        )
        longitudinal_authority_evidence_passed = (
            not args.require_opposing_vx_integral_deficit_reset
            or longitudinal_authority_telemetry_passed(
                payload,
                result,
                expected_deadband_mps=(
                    args.expected_vx_integral_reset_reference_deadband_mps
                ),
            )
        )
        initialization_duration_s = float(
            result.get("initialization_duration_s", 0.0)
        )
        initialization_evidence_passed = (
            initialization_duration_s <= 0.0
            or (
                result.get("initialization_completed") is True
                and result.get("initialization_scored_as_source_tracking") is False
                and result.get("initialization_source_metric_samples") == 0
                and result.get("initialization_residual_label_samples") == 0
                and result.get("initialization_steps", 0) > 0
                and result.get("initialization_riser_thermal_sample_count")
                == result.get("initialization_steps")
                and result.get("checks", {}).get(
                    "initialization_action_saturation_bounded"
                )
                is True
                and result.get("checks", {}).get(
                    "initialization_riser_thermal_force_observed"
                )
                is True
                and result.get("checks", {}).get(
                    "initialization_riser_thermal_load_bounded"
                )
                is True
                and result.get("checks", {}).get(
                    "initialization_riser_peak_force_bounded"
                )
                is True
                and result.get("checks", {}).get(
                    "initialization_source_metrics_clean"
                )
                is True
            )
        )
        velocity_feedback = result.get("velocity_feedback_telemetry")
        velocity_feedback_evidence_passed = (
            velocity_feedback is None
            and result.get("velocity_feedback_telemetry_observed") is None
        ) or (
            isinstance(velocity_feedback, dict)
            and velocity_feedback.get("schema")
            == "riser_root_vs_wheel_velocity_policy_rate_v1"
            and velocity_feedback.get("policy_rate_sample_count")
            == result.get("completed_steps")
            and result.get("velocity_feedback_telemetry_observed") is True
            and result.get("checks", {}).get(
                "velocity_feedback_telemetry_observed"
            )
            is True
        )
        zero_progress_hold_evidence_passed = (
            not args.require_zero_progress_hold
            or zero_progress_hold_telemetry_passed(
                payload,
                result,
                expected_maximum_linear_velocity_mps=(
                    args.expected_maximum_linear_velocity_mps
                    if args.require_recovery_velocity_cap
                    else None
                ),
                expected_phase_governor_contract=(
                    expected_phase_governor_contract
                ),
            )
        )
        total_pitch_reference_limit_evidence_passed = (
            total_pitch_reference_limit_passed(
                payload,
                result,
                required=args.require_total_pitch_reference_limit,
                require_opposing_vx_integral_deficit_reset=(
                    args.require_opposing_vx_integral_deficit_reset
                ),
                expected_vx_integral_reset_deadband_mps=(
                    args.expected_vx_integral_reset_reference_deadband_mps
                ),
            )
        )
        runtime_contract_passed = (
            payload.get("training_started") is False
            and payload.get("ppo_authorized") is False
            and payload.get("trajectory_command_source") == "deterministic_teacher"
            and payload.get("residual_policy") is None
            and payload.get("controller_profile") == EXPECTED_CONTROLLER_PROFILE
            and payload.get("tracking_profile") == args.expected_tracking_profile
            and payload.get("tracking_direction_recovery_error_range_m")
            == EXPECTED_RECOVERY_ERROR_RANGE_M
            and payload.get("riser_thermal_force_contract")
            == EXPECTED_RISER_THERMAL_FORCE_CONTRACT
            and result.get("recovery_telemetry_observed") is True
            and recovery_telemetry_passed(result)
            and payload.get("cases") == [case]
            and len(payload.get("results", [])) == 1
            and result.get("executed_residual_dataset") is None
            and result.get("raw_residual_label_applied_to_commands") is False
            and camera_lever_arm_evidence_passed
            and camera_error_governor_evidence_passed
            and commanded_base_progress_error_evidence_passed
            and longitudinal_authority_evidence_passed
            and initialization_evidence_passed
            and velocity_feedback_evidence_passed
            and zero_progress_hold_evidence_passed
            and total_pitch_reference_limit_evidence_passed
            and zero_progress_hold_admission_passed
        )
        row = {
            "case": case,
            "gate": str(gate.resolve()),
            "gate_sha256": sha256_file(gate),
            "passed": physical_dynamic_passed
            and thermal_admission_passed
            and runtime_contract_passed,
            "physical_dynamic_quality_passed": physical_dynamic_passed,
            "thermal_admission_passed": thermal_admission_passed,
            "runtime_contract_passed": runtime_contract_passed,
            "controller_evidence_passed": (
                camera_lever_arm_evidence_passed
                and camera_error_governor_evidence_passed
                and initialization_evidence_passed
                and velocity_feedback_evidence_passed
                and zero_progress_hold_evidence_passed
                and commanded_base_progress_error_evidence_passed
                and longitudinal_authority_evidence_passed
                if args.require_camera_lever_arm_compensation
                else result.get("controller_evidence_passed")
            ),
            "camera_error_governor_evidence_passed": (
                camera_error_governor_evidence_passed
                if args.require_camera_error_recovery_governor
                else None
            ),
            "commanded_base_progress_error_evidence_passed": (
                commanded_base_progress_error_evidence_passed
                if args.require_commanded_base_progress_error
                else None
            ),
            "longitudinal_authority_evidence_passed": (
                longitudinal_authority_evidence_passed
                if args.require_opposing_vx_integral_deficit_reset
                else None
            ),
            "initialization_evidence_passed": initialization_evidence_passed,
            "velocity_feedback_evidence_passed": (
                velocity_feedback_evidence_passed
            ),
            "zero_progress_hold_evidence_passed": (
                zero_progress_hold_evidence_passed
            ),
            "recovery_velocity_cap_evidence_passed": (
                zero_progress_hold_evidence_passed
                if args.require_recovery_velocity_cap
                else None
            ),
            "total_pitch_reference_limit_evidence_passed": (
                total_pitch_reference_limit_evidence_passed
                if args.require_total_pitch_reference_limit
                else None
            ),
            "initialization_duration_s": initialization_duration_s,
            "initialization_steps": result.get("initialization_steps", 0),
            "initialization_completed": result.get(
                "initialization_completed", initialization_duration_s <= 0.0
            ),
            "initialization_scored_as_source_tracking": result.get(
                "initialization_scored_as_source_tracking", False
            ),
            "initialization_source_metric_samples": result.get(
                "initialization_source_metric_samples", 0
            ),
            "initialization_residual_label_samples": result.get(
                "initialization_residual_label_samples", 0
            ),
            "initialization_terminal_base_error_m": result.get(
                "initialization_terminal_base_error_m"
            ),
            "initialization_terminal_base_yaw_error_deg": result.get(
                "initialization_terminal_base_yaw_error_deg"
            ),
            "initialization_terminal_riser_error_m": result.get(
                "initialization_terminal_riser_error_m"
            ),
            "initialization_terminal_proxy_error_deg": result.get(
                "initialization_terminal_proxy_error_deg"
            ),
            "initialization_action_saturation_ratio": result.get(
                "initialization_action_saturation_ratio"
            ),
            "initialization_riser_thermal_load_max": result.get(
                "initialization_riser_thermal_load_max"
            ),
            "initialization_riser_effort_max_n": result.get(
                "initialization_riser_effort_max_n"
            ),
            "camera_recovery_activation_ratio": result.get(
                "camera_recovery_activation_ratio"
            ),
            "camera_recovery_progress_scale_min": result.get(
                "camera_recovery_progress_scale_min"
            ),
            "camera_lever_arm_compensation_enabled": result.get(
                "camera_lever_arm_compensation_enabled"
            ),
            "camera_lever_arm_correction_max_m": result.get(
                "camera_lever_arm_correction_max_m"
            ),
            "camera_lever_arm_raw_correction_max_m": result.get(
                "camera_lever_arm_raw_correction_max_m"
            ),
            "camera_lever_arm_correction_saturation_ratio": result.get(
                "camera_lever_arm_correction_saturation_ratio"
            ),
            "controller_profile": payload.get("controller_profile"),
            "tracking_profile": payload.get("tracking_profile"),
            "tracking_direction_recovery_error_range_m": payload.get(
                "tracking_direction_recovery_error_range_m"
            ),
            "riser_thermal_force_contract": payload.get(
                "riser_thermal_force_contract"
            ),
            "riser_thermal_load_max": result.get("riser_thermal_load_max"),
            "riser_effort_max_n": result.get("riser_effort_max_n"),
            "recovery_telemetry": result.get("recovery_telemetry"),
            "velocity_feedback_telemetry": result.get(
                "velocity_feedback_telemetry"
            ),
            "velocity_feedback_telemetry_observed": result.get(
                "velocity_feedback_telemetry_observed"
            ),
            "longitudinal_authority_telemetry": result.get(
                "longitudinal_authority_telemetry"
            ),
            "longitudinal_authority_telemetry_observed": result.get(
                "longitudinal_authority_telemetry_observed"
            ),
            "opposing_vx_integral_deficit_reset_enabled": result.get(
                "opposing_vx_integral_deficit_reset_enabled"
            ),
            "vx_integral_reset_reference_deadband_mps": result.get(
                "vx_integral_reset_reference_deadband_mps"
            ),
            "outer_velocity_feedback_source": result.get(
                "outer_velocity_feedback_source"
            ),
            "minimum_progress_scale": result.get("minimum_progress_scale"),
            "maximum_linear_velocity_mps": result.get(
                "maximum_linear_velocity_mps"
            ),
            "total_pitch_reference_limit_enabled": result.get(
                "total_pitch_reference_limit_enabled"
            ),
            "total_pitch_reference_limit_rad": result.get(
                "total_pitch_reference_limit_rad"
            ),
            "progress_scale_min": result.get("progress_scale_min"),
            "progress_hold_step_count": result.get("progress_hold_step_count"),
            "progress_hold_ratio": result.get("progress_hold_ratio"),
            "progress_hold_segment_count": result.get(
                "progress_hold_segment_count"
            ),
            "commanded_base_progress_error_enabled": result.get(
                "commanded_base_progress_error_enabled"
            ),
            "progress_base_error_source": result.get(
                "progress_base_error_source"
            ),
            "progress_base_error_telemetry_sample_count": result.get(
                "progress_base_error_telemetry_sample_count"
            ),
            "progress_base_error_selected_source_matches": result.get(
                "progress_base_error_selected_source_matches"
            ),
            "progress_base_error_command_delta_bounded": result.get(
                "progress_base_error_command_delta_bounded"
            ),
            "selected_vs_nominal_base_progress_error_mean_delta_m": result.get(
                "selected_vs_nominal_base_progress_error_mean_delta_m"
            ),
            "selected_vs_nominal_base_progress_error_abs_max_delta_m": result.get(
                "selected_vs_nominal_base_progress_error_abs_max_delta_m"
            ),
            "source_duration_s": result.get("source_duration_s"),
            "execution_duration_s": result.get("execution_duration_s"),
            "completed_steps": result.get("completed_steps"),
            "dynamic_quality_passed": result.get("dynamic_quality_passed"),
            "residual_label_envelope_passed": result.get(
                "residual_label_envelope_passed"
            ),
            "residual_label_admission_passed": result.get(
                "residual_label_admission_passed"
            ),
        }
        gate_rows.append(row)
        if row["passed"]:
            passed_cases.append(case)
            continue
        first_reject = {
            "case": case,
            "classification": (
                "thermal_admission_rejection"
                if physical_dynamic_passed and not thermal_admission_passed
                else (
                    "runtime_contract_rejection"
                    if physical_dynamic_passed and not runtime_contract_passed
                    else result.get("classification", "dynamic_gate_rejection")
                )
            ),
            "stage": result.get("stage", "dynamic_gate"),
            "physical_dynamic_quality_passed": physical_dynamic_passed,
            "thermal_admission_passed": thermal_admission_passed,
            "runtime_contract_passed": runtime_contract_passed,
            "exception_type": result.get("exception_type"),
            "exception_message": result.get("exception_message"),
            "normalized_action": result.get("normalized_action"),
            "gate_sha256": row["gate_sha256"],
            "log_sha256": sha256_file(log) if log.is_file() else None,
        }
        break

    not_started = requested[len(passed_cases) + (1 if first_reject else 0) :]
    passed = (
        first_reject is None
        and passed_cases == requested
        and contract_admission_passed
    )
    summary = {
        "schema": "cinebotrl_two_wheel_riser_gate_c_canary_v2",
        "git_commit": args.git_commit,
        "admission_sha256": sha256_file(admission),
        "case74_contract_required": args.require_case74_contract,
        "case74_contract_admission_passed": contract_admission_passed,
        "case74_contract_sha256": admission_payload.get("contract_sha256"),
        "requested_cases": requested,
        "dynamically_passed_cases": passed_cases,
        "first_dynamic_reject": first_reject,
        "not_started_cases": not_started,
        "gate_rows": gate_rows,
        "source_execution_timing_separated": all(
            row["source_duration_s"] is not None
            and row["execution_duration_s"] is not None
            for row in gate_rows
        ),
        "dynamic_quality_passed": bool(gate_rows)
        and len(gate_rows) == len(requested)
        and all(row["physical_dynamic_quality_passed"] for row in gate_rows),
        "thermal_admission_passed": bool(gate_rows)
        and len(gate_rows) == len(requested)
        and all(row["thermal_admission_passed"] for row in gate_rows),
        "runtime_contract_passed": bool(gate_rows)
        and len(gate_rows) == len(requested)
        and all(row["runtime_contract_passed"] for row in gate_rows),
        "expected_controller_profile": EXPECTED_CONTROLLER_PROFILE,
        "expected_tracking_profile": args.expected_tracking_profile,
        "camera_lever_arm_compensation_required": (
            args.require_camera_lever_arm_compensation
        ),
        "expected_camera_lever_arm_compensation_contract": (
            CAMERA_LEVER_ARM_COMPENSATION_CONTRACT
            if args.require_camera_lever_arm_compensation
            else None
        ),
        "expected_camera_lever_arm_gain": (
            args.expected_camera_lever_arm_gain
            if args.require_camera_lever_arm_compensation
            else None
        ),
        "expected_maximum_camera_lever_arm_correction_m": (
            args.expected_maximum_camera_lever_arm_correction_m
            if args.require_camera_lever_arm_compensation
            else None
        ),
        "camera_error_recovery_governor_required": (
            args.require_camera_error_recovery_governor
        ),
        "zero_progress_hold_required": args.require_zero_progress_hold,
        "recovery_velocity_cap_required": args.require_recovery_velocity_cap,
        "total_pitch_reference_limit_required": (
            args.require_total_pitch_reference_limit
        ),
        "commanded_base_progress_error_required": (
            args.require_commanded_base_progress_error
        ),
        "opposing_vx_integral_deficit_reset_required": (
            args.require_opposing_vx_integral_deficit_reset
        ),
        "expected_vx_integral_reset_reference_deadband_mps": (
            args.expected_vx_integral_reset_reference_deadband_mps
            if args.require_opposing_vx_integral_deficit_reset
            else None
        ),
        "expected_longitudinal_authority_telemetry_schema": (
            EXPECTED_LONGITUDINAL_AUTHORITY_TELEMETRY_SCHEMA
            if args.require_opposing_vx_integral_deficit_reset
            else None
        ),
        "zero_progress_hold_admission_passed": (
            zero_progress_hold_admission_passed
        ),
        "expected_phase_governor_contract": (
            expected_phase_governor_contract
            if args.require_zero_progress_hold
            else None
        ),
        "expected_progress_base_error_source": (
            COMMANDED_BASE_PROGRESS_ERROR_SOURCE
            if args.require_commanded_base_progress_error
            else None
        ),
        "commanded_base_progress_error_evidence_passed": (
            bool(gate_rows)
            and len(gate_rows) == len(requested)
            and all(
                row["commanded_base_progress_error_evidence_passed"] is True
                for row in gate_rows
            )
            if args.require_commanded_base_progress_error
            else None
        ),
        "longitudinal_authority_evidence_passed": (
            bool(gate_rows)
            and len(gate_rows) == len(requested)
            and all(
                row["longitudinal_authority_evidence_passed"] is True
                for row in gate_rows
            )
            if args.require_opposing_vx_integral_deficit_reset
            else None
        ),
        "expected_minimum_progress_scale": (
            0.0 if args.require_zero_progress_hold else None
        ),
        "expected_maximum_linear_velocity_mps": (
            args.expected_maximum_linear_velocity_mps
            if args.require_recovery_velocity_cap
            else None
        ),
        "expected_total_pitch_reference_limit_rad": (
            EXPECTED_TOTAL_PITCH_REFERENCE_LIMIT_RAD
            if args.require_total_pitch_reference_limit
            else None
        ),
        "total_pitch_reference_limit_evidence_passed": (
            bool(gate_rows)
            and len(gate_rows) == len(requested)
            and all(
                row["total_pitch_reference_limit_evidence_passed"] is True
                for row in gate_rows
            )
            if args.require_total_pitch_reference_limit
            else None
        ),
        "expected_camera_error_recovery_governor_contract": (
            CAMERA_ERROR_GOVERNOR_CONTRACT
            if args.require_camera_error_recovery_governor
            else None
        ),
        "expected_camera_recovery_error_range_m": (
            [
                args.expected_camera_recovery_error_start_m,
                args.expected_camera_recovery_error_full_m,
            ]
            if args.require_camera_error_recovery_governor
            else None
        ),
        "expected_minimum_camera_recovery_scale": (
            args.expected_minimum_camera_recovery_scale
            if args.require_camera_error_recovery_governor
            else None
        ),
        "controller_evidence_passed": bool(gate_rows)
        and len(gate_rows) == len(requested)
        and all(
            row["controller_evidence_passed"] is True for row in gate_rows
        )
        if args.require_camera_lever_arm_compensation
        else None,
        "expected_tracking_direction_recovery_error_range_m": (
            EXPECTED_RECOVERY_ERROR_RANGE_M
        ),
        "expected_riser_thermal_force_contract": (
            EXPECTED_RISER_THERMAL_FORCE_CONTRACT
        ),
        "expected_recovery_telemetry_schema": EXPECTED_RECOVERY_TELEMETRY_SCHEMA,
        "residual_label_envelope_passed": bool(gate_rows)
        and all(row["residual_label_envelope_passed"] is True for row in gate_rows),
        "residual_label_admission_passed": bool(gate_rows)
        and all(row["residual_label_admission_passed"] is True for row in gate_rows),
        "thresholds_relaxed": False,
        "actions_clipped": False,
        "residual_capture_started": False,
        "bc_started": False,
        "ppo_started": False,
        "passed": passed,
        "valid_for_final_gate_c": passed,
        "valid_for_training": False,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
