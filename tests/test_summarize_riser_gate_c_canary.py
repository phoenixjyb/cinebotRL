import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/summarize_riser_gate_c_canary.py"
CONTRACT_SHA = "c" * 64


def _admission(path: Path, commit: str, namespace: str) -> None:
    identity_names = {
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
    identities = {
        name: {
            "sha256": "a" * 64,
            "git_blob_sha1": None if name == "source_manifest" else "b" * 40,
            "passed": True,
        }
        for name in identity_names
    }
    path.write_text(
        json.dumps(
            {
                "schema": "cinebotrl_case74_recovery_v4_runtime_contract_admission_v2",
                "contract_sha256": CONTRACT_SHA,
                "reviewed_controller_parent_commit": (
                    "ba8f4e0b44dc15a60d61b8353a208032727ad0ae"
                ),
                "runtime_commit": commit,
                "upstream_commit": commit,
                "case": 74,
                "namespace": namespace,
                "tracking_profile": "riser_recovery_direction_v4",
                "recovery_error_range_m": [0.2, 0.4],
                "identity_passed": True,
                "runtime_authorized": True,
                "gate_c_execution_authorized": True,
                "contract_git_blob_sha1": "d" * 40,
                "identities": identities,
                "checks": {"all_required_checks": True},
                "residual_capture_authorized": False,
                "bc_authorized": False,
                "ppo_authorized": False,
                "valid_for_training": False,
            }
        )
    )


def _gate(
    path: Path, case: int, passed: bool, *, label_envelope_passed: bool = True
) -> None:
    path.write_text(
        json.dumps(
            {
                "passed": passed,
                "dynamic_quality_passed": passed,
                "thermal_admission_passed": passed,
                "training_started": False,
                "ppo_authorized": False,
                "trajectory_command_source": "deterministic_teacher",
                "residual_policy": None,
                "controller_profile": "structural_robust_v1",
                "tracking_profile": "riser_recovery_direction_v4",
                "tracking_direction_recovery_error_range_m": [0.2, 0.4],
                "riser_thermal_force_contract": (
                    "leadshine_400w_first_order_monitor_v1"
                ),
                "cases": [case],
                "passed_case_count": 1 if passed else 0,
                "results": [
                    {
                        "case": case,
                        "passed": passed,
                        "source_duration_s": 1.0,
                        "execution_duration_s": 3.0,
                        "completed_steps": 10,
                        "dynamic_quality_passed": passed,
                        "thermal_admission_passed": passed,
                        "residual_label_envelope_passed": label_envelope_passed,
                        "residual_label_admission_passed": (
                            passed and label_envelope_passed
                        ),
                        "riser_thermal_load_max": 0.4,
                        "riser_effort_max_n": 100.0,
                        "recovery_telemetry": {
                            "schema": "riser_recovery_direction_policy_rate_v1",
                            "policy_rate_sample_count": 10,
                            "activation_step_count": 4,
                            "full_authority_step_count": 2,
                            "activation_segment_count": 1,
                            "motion_direction_sign_change_count": 1,
                            "feedback_direction_sign_change_count": 1,
                            "consecutive_active_motion_direction_chatter_count": 1,
                            "candidate_yaw_saturation_step_count": 2,
                            "legacy_yaw_saturation_step_count": 0,
                            "candidate_vs_legacy_delta_nonzero_step_count": 4,
                            "candidate_vs_legacy_yaw_delta_abs_max_rad_s": 0.5,
                            "recovery_blend_max": 1.0,
                        },
                        "checks": {
                            "riser_thermal_force_observed": True,
                            "riser_thermal_load_bounded": True,
                            "riser_peak_force_bounded": True,
                        },
                        "executed_residual_dataset": None,
                        "raw_residual_label_applied_to_commands": False,
                        "recovery_telemetry_observed": True,
                        "classification": (
                            None
                            if passed
                            else "action_envelope_zero_clipping_rejection"
                        ),
                        "stage": "dynamic_gate",
                    }
                ],
            }
        )
    )


def _enable_camera_lever_arm_contract(path: Path) -> None:
    payload = json.loads(path.read_text())
    payload.update(
        {
            "tracking_profile": (
                "riser_recovery_direction_v4_camera_lever_arm_v1"
            ),
            "camera_lever_arm_compensation_contract": (
                "measured_camera_to_base_xy_offset_v1"
            ),
            "camera_lever_arm_compensation_enabled": True,
            "camera_lever_arm_compensation_gain": 1.0,
            "maximum_camera_lever_arm_correction_m": 0.05,
            "controller_evidence_passed": True,
        }
    )
    result = payload["results"][0]
    result.update(
        {
            "controller_evidence_passed": True,
            "camera_lever_arm_compensation_enabled": True,
            "camera_lever_arm_compensation_gain": 1.0,
            "maximum_camera_lever_arm_correction_m": 0.05,
            "camera_lever_arm_telemetry_observed": True,
            "camera_lever_arm_telemetry_sample_count": result["completed_steps"],
            "camera_lever_arm_correction_max_m": 0.05,
            "camera_lever_arm_raw_correction_max_m": 0.18,
            "camera_lever_arm_correction_saturation_ratio": 0.9,
        }
    )
    path.write_text(json.dumps(payload))


def _enable_camera_error_governor_contract(path: Path) -> None:
    _enable_camera_lever_arm_contract(path)
    payload = json.loads(path.read_text())
    payload.update(
        {
            "tracking_profile": (
                "riser_recovery_direction_v4_camera_lever_arm_error_governor_v1"
            ),
            "camera_recovery_governor_enabled": True,
            "camera_recovery_governor_contract": (
                "saturated_camera_error_continuous_phase_cap_v1"
            ),
            "camera_recovery_error_range_m": [0.13, 0.155],
            "minimum_camera_recovery_scale": 0.2,
        }
    )
    result = payload["results"][0]
    result.update(
        {
            "camera_recovery_governor_enabled": True,
            "camera_recovery_governor_contract": (
                "saturated_camera_error_continuous_phase_cap_v1"
            ),
            "camera_recovery_error_range_m": [0.13, 0.155],
            "minimum_camera_recovery_scale": 0.2,
            "camera_recovery_telemetry_observed": True,
            "camera_recovery_telemetry_sample_count": result["completed_steps"],
            "camera_recovery_progress_scale_min": 0.2,
            "camera_recovery_progress_scale_mean": 0.8,
            "camera_recovery_activation_ratio": 0.1,
            "trace": [
                {
                    "camera_recovery_progress_scale": 0.2,
                    "camera_recovery_active": True,
                },
                {
                    "camera_recovery_progress_scale": 1.0,
                    "camera_recovery_active": False,
                },
            ],
        }
    )
    result["checks"]["camera_recovery_telemetry_observed"] = True
    path.write_text(json.dumps(payload))


def _enable_zero_progress_hold_contract(path: Path) -> None:
    _enable_camera_lever_arm_contract(path)
    payload = json.loads(path.read_text())
    payload.update(
        {
            "tracking_profile": (
                "riser_recovery_direction_v4_camera_lever_arm_zero_progress_hold_v1"
            ),
            "phase_governor_enabled": True,
            "phase_governor_contract": "position_error_continuous_phase_scale_v1",
            "minimum_progress_scale": 0.0,
            "tracking_overrides": {"minimum_progress_scale": 0.0},
        }
    )
    result = payload["results"][0]
    result.update(
        {
            "minimum_progress_scale": 0.0,
            "progress_scale_min": 0.0,
            "progress_hold_step_count": 2,
            "progress_hold_ratio": 0.2,
            "progress_hold_segment_count": 1,
            "outer_velocity_feedback_source": "wheel_derived_vx",
        }
    )
    path.write_text(json.dumps(payload))


def _enable_zero_progress_hold_admission(path: Path, commit: str) -> None:
    payload = json.loads(path.read_text())
    payload.update(
        {
            "requested_cases": [42],
            "tracking_profile": (
                "riser_recovery_direction_v4_camera_lever_arm_zero_progress_hold_v1"
            ),
            "zero_progress_hold_required": True,
            "phase_governor_contract": "position_error_continuous_phase_scale_v1",
            "minimum_progress_scale": 0.0,
            "root_velocity_outer_feedback_enabled": False,
            "runtime_commit": commit,
            "upstream_commit": commit,
        }
    )
    path.write_text(json.dumps(payload))


def _enable_recovery_velocity_cap_contract(path: Path) -> None:
    _enable_zero_progress_hold_contract(path)
    payload = json.loads(path.read_text())
    payload.update(
        {
            "tracking_profile": (
                "riser_recovery_direction_v4_camera_lever_arm_"
                "zero_progress_hold_velocity_cap_v1"
            ),
            "tracking_recovery_velocity_cap_enabled": True,
            "maximum_linear_velocity_mps": 0.2,
            "tracking_overrides": {
                "minimum_progress_scale": 0.0,
                "maximum_linear_velocity_mps": 0.2,
            },
        }
    )
    result = payload["results"][0]
    result.update(
        {
            "maximum_linear_velocity_mps": 0.2,
            "velocity_feedback_telemetry": {
                "schema": "riser_root_vs_wheel_velocity_policy_rate_v1",
                "policy_rate_sample_count": result["completed_steps"],
                "effective_reference_abs_max_mps": 0.2,
            },
            "velocity_feedback_telemetry_observed": True,
        }
    )
    result["checks"]["velocity_feedback_telemetry_observed"] = True
    path.write_text(json.dumps(payload))


def _enable_recovery_velocity_cap_admission(path: Path, commit: str) -> None:
    _enable_zero_progress_hold_admission(path, commit)
    payload = json.loads(path.read_text())
    payload.update(
        {
            "tracking_profile": (
                "riser_recovery_direction_v4_camera_lever_arm_"
                "zero_progress_hold_velocity_cap_v1"
            ),
            "recovery_velocity_cap_required": True,
            "maximum_linear_velocity_mps": 0.2,
        }
    )
    path.write_text(json.dumps(payload))


def _enable_total_pitch_reference_limit_contract(path: Path) -> None:
    _enable_recovery_velocity_cap_contract(path)
    payload = json.loads(path.read_text())
    profile = (
        "riser_recovery_direction_v4_camera_lever_arm_"
        "zero_progress_hold_velocity_cap_total_pitch_limit_v1"
    )
    limit = math.radians(6.0)
    payload.update(
        {
            "tracking_profile": profile,
            "total_pitch_reference_limit_enabled": True,
            "total_pitch_reference_limit_rad": limit,
            "controller_overrides": {
                "wz_kp": 1.05,
                "limit_total_pitch_reference": True,
            },
        }
    )
    result = payload["results"][0]
    result.update(
        {
            "total_pitch_reference_limit_enabled": True,
            "total_pitch_reference_limit_rad": limit,
        }
    )
    result["velocity_feedback_telemetry"].update(
        {
            "total_pitch_reference_abs_max_rad": limit,
            "pitch_reference_abs_max_rad": math.radians(7.65),
        }
    )
    path.write_text(json.dumps(payload))


def _enable_total_pitch_reference_limit_admission(path: Path, commit: str) -> None:
    _enable_recovery_velocity_cap_admission(path, commit)
    payload = json.loads(path.read_text())
    payload.update(
        {
            "tracking_profile": (
                "riser_recovery_direction_v4_camera_lever_arm_"
                "zero_progress_hold_velocity_cap_total_pitch_limit_v1"
            ),
            "total_pitch_reference_limit_required": True,
            "total_pitch_reference_limit_rad": math.radians(6.0),
        }
    )
    path.write_text(json.dumps(payload))


def _enable_commanded_base_progress_error_contract(path: Path) -> None:
    _enable_total_pitch_reference_limit_contract(path)
    payload = json.loads(path.read_text())
    contract = "commanded_base_and_camera_error_continuous_phase_scale_v1"
    source = "lever_compensated_commanded_base_target"
    payload.update(
        {
            "phase_governor_contract": contract,
            "commanded_base_progress_error_enabled": True,
            "progress_base_error_source": source,
        }
    )
    result = payload["results"][0]
    result.update(
        {
            "phase_governor_contract": contract,
            "commanded_base_progress_error_enabled": True,
            "progress_base_error_source": source,
            "progress_base_error_telemetry_sample_count": result[
                "completed_steps"
            ],
            "progress_base_error_telemetry_observed": True,
            "progress_base_error_selected_source_matches": True,
            "progress_base_error_command_delta_bounded": True,
            "nominal_base_progress_error_p95_m": 0.20,
            "nominal_base_progress_error_max_m": 0.22,
            "commanded_base_progress_error_p95_m": 0.15,
            "commanded_base_progress_error_max_m": 0.17,
            "selected_base_progress_error_p95_m": 0.15,
            "selected_base_progress_error_max_m": 0.17,
            "selected_vs_nominal_base_progress_error_mean_delta_m": -0.04,
            "selected_vs_nominal_base_progress_error_abs_max_delta_m": 0.05,
            "maximum_commanded_base_progress_error_delta_m": 0.05,
        }
    )
    result["checks"].update(
        {
            "progress_base_error_telemetry_observed": True,
            "progress_base_error_selected_source_matches": True,
            "progress_base_error_command_delta_bounded": True,
        }
    )
    path.write_text(json.dumps(payload))


def _enable_commanded_base_progress_error_admission(
    path: Path,
    commit: str,
) -> None:
    _enable_total_pitch_reference_limit_admission(path, commit)
    payload = json.loads(path.read_text())
    payload.update(
        {
            "phase_governor_contract": (
                "commanded_base_and_camera_error_continuous_phase_scale_v1"
            ),
            "commanded_base_progress_error_required": True,
            "progress_base_error_source": (
                "lever_compensated_commanded_base_target"
            ),
            "maximum_commanded_base_progress_error_delta_m": 0.05,
        }
    )
    path.write_text(json.dumps(payload))


def _enable_opposing_vx_integral_reset_contract(path: Path) -> None:
    _enable_commanded_base_progress_error_contract(path)
    payload = json.loads(path.read_text())
    payload.update(
        {
            "opposing_vx_integral_deficit_reset_enabled": True,
            "vx_integral_reset_reference_deadband_mps": 0.05,
            "controller_overrides": {
                "wz_kp": 1.05,
                "limit_total_pitch_reference": True,
                "reset_opposing_vx_integral_on_directional_deficit": True,
                "vx_integral_reset_reference_deadband_mps": 0.05,
            },
        }
    )
    result = payload["results"][0]
    result.update(
        {
            "opposing_vx_integral_deficit_reset_enabled": True,
            "vx_integral_reset_reference_deadband_mps": 0.05,
            "longitudinal_authority_telemetry_observed": True,
            "longitudinal_authority_telemetry": {
                "schema": "riser_longitudinal_authority_policy_rate_v1",
                "policy_rate_sample_count": 10,
                "controller_update_count": 3,
                "held_controller_command_step_count": 7,
                "reference_sign_change_count": 2,
                "opposing_integral_sign_change_count": 1,
                "integral_reset_count": 2,
                "velocity_deficit_step_count": 4,
                "total_pitch_limit_step_count": 1,
                "velocity_deficit_ratio": 0.4,
                "velocity_deficit_mean_mps": 0.05,
                "velocity_deficit_abs_max_mps": 0.08,
                "deficit_pitch_contribution_mean": 0.1,
                "deficit_pitch_rate_contribution_mean": -0.2,
                "deficit_wheel_velocity_contribution_mean": -0.001,
                "vx_integral_before_abs_max": 0.3,
                "vx_integral_after_abs_max": 0.2,
                "pitch_abs_max_rad": 0.08,
                "pitch_rate_abs_max_rad_s": 0.3,
                "total_pitch_reference_abs_max_rad": math.radians(6.0),
                "common_action_abs_max": 0.5,
                "reference_deadband_mps": 0.05,
                "deficit_tolerance_mps": 0.03,
            },
        }
    )
    result["checks"]["longitudinal_authority_telemetry_observed"] = True
    path.write_text(json.dumps(payload))


def _enable_opposing_vx_integral_reset_admission(
    path: Path,
    commit: str,
) -> None:
    _enable_commanded_base_progress_error_admission(path, commit)
    payload = json.loads(path.read_text())
    payload.update(
        {
            "opposing_vx_integral_deficit_reset_required": True,
            "vx_integral_reset_reference_deadband_mps": 0.05,
            "longitudinal_authority_telemetry_schema": (
                "riser_longitudinal_authority_policy_rate_v1"
            ),
            "reviewed_controller_parent_commit": (
                "35f775c39ed2d0c22b52be5dd8f9641354ee0b8f"
            ),
            "reviewed_controller_parent_is_ancestor": True,
        }
    )
    path.write_text(json.dumps(payload))


def _enable_controller_vx_kp_contract(path: Path) -> None:
    _enable_opposing_vx_integral_reset_contract(path)
    payload = json.loads(path.read_text())
    payload["controller_overrides"]["vx_kp"] = 0.72
    payload["results"][0]["controller_vx_kp"] = 0.72
    path.write_text(json.dumps(payload))


def _enable_controller_vx_kp_admission(path: Path, commit: str) -> None:
    _enable_opposing_vx_integral_reset_admission(path, commit)
    payload = json.loads(path.read_text())
    payload.update(
        {
            "controller_vx_kp_required": True,
            "expected_controller_vx_kp": 0.72,
            "reviewed_controller_parent_commit": (
                "1e7ebbde4dcb241fde63275e5434dfa2fc4d1cb8"
            ),
        }
    )
    path.write_text(json.dumps(payload))


def _enable_initialization_preroll_evidence(path: Path) -> None:
    payload = json.loads(path.read_text())
    result = payload["results"][0]
    result.update(
        {
            "initialization_duration_s": 2.0,
            "initialization_steps": 400,
            "initialization_completed": True,
            "initialization_scored_as_source_tracking": False,
            "initialization_source_metric_samples": 0,
            "initialization_residual_label_samples": 0,
            "initialization_terminal_base_error_m": 0.03,
            "initialization_terminal_base_yaw_error_deg": 0.2,
            "initialization_terminal_riser_error_m": 0.002,
            "initialization_terminal_proxy_error_deg": 0.05,
            "initialization_action_saturation_ratio": 0.0,
            "initialization_riser_thermal_sample_count": 400,
            "initialization_riser_thermal_load_max": 0.001,
            "initialization_riser_effort_max_n": 20.0,
        }
    )
    result["checks"].update(
        {
            "initialization_action_saturation_bounded": True,
            "initialization_riser_thermal_force_observed": True,
            "initialization_riser_thermal_load_bounded": True,
            "initialization_riser_peak_force_bounded": True,
            "initialization_source_metrics_clean": True,
        }
    )
    path.write_text(json.dumps(payload))


def test_initialization_evidence_is_separate_and_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    commit = "a" * 40
    _admission(tmp_path / "admission.json", commit, tmp_path.name)
    gate = tmp_path / "gates/case_0042.json"
    _gate(gate, 42, True)
    _enable_initialization_preroll_evidence(gate)
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--git-commit",
            commit,
            "--cases",
            "42",
            "--output",
            str(output),
        ],
        check=True,
    )
    summary = json.loads(output.read_text())
    row = summary["gate_rows"][0]
    assert summary["passed"]
    assert row["initialization_evidence_passed"] is True
    assert row["initialization_steps"] == 400
    assert row["initialization_source_metric_samples"] == 0

    payload = json.loads(gate.read_text())
    payload["results"][0]["initialization_source_metric_samples"] = 1
    gate.write_text(json.dumps(payload))
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--git-commit",
            commit,
            "--cases",
            "42",
            "--output",
            str(output),
        ],
        check=True,
    )
    summary = json.loads(output.read_text())
    assert summary["dynamic_quality_passed"]
    assert not summary["runtime_contract_passed"]
    assert summary["first_dynamic_reject"]["classification"] == (
        "runtime_contract_rejection"
    )


def test_velocity_feedback_evidence_is_backward_compatible_and_fail_closed(
    tmp_path: Path,
) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    commit = "a" * 40
    _admission(tmp_path / "admission.json", commit, tmp_path.name)
    gate = tmp_path / "gates/case_0042.json"
    _gate(gate, 42, True)
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--git-commit",
            commit,
            "--cases",
            "42",
            "--output",
            str(output),
        ],
        check=True,
    )
    summary = json.loads(output.read_text())
    assert summary["gate_rows"][0]["velocity_feedback_evidence_passed"] is True

    payload = json.loads(gate.read_text())
    result = payload["results"][0]
    result["velocity_feedback_telemetry"] = {
        "schema": "riser_root_vs_wheel_velocity_policy_rate_v1",
        "policy_rate_sample_count": result["completed_steps"] - 1,
    }
    result["velocity_feedback_telemetry_observed"] = True
    result["checks"]["velocity_feedback_telemetry_observed"] = True
    gate.write_text(json.dumps(payload))
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--git-commit",
            commit,
            "--cases",
            "42",
            "--output",
            str(output),
        ],
        check=True,
    )
    summary = json.loads(output.read_text())
    assert summary["dynamic_quality_passed"]
    assert not summary["runtime_contract_passed"]
    assert summary["gate_rows"][0]["velocity_feedback_evidence_passed"] is False


def test_zero_progress_hold_evidence_is_explicit_and_fail_closed(
    tmp_path: Path,
) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    commit = "a" * 40
    _admission(tmp_path / "admission.json", commit, tmp_path.name)
    _enable_zero_progress_hold_admission(tmp_path / "admission.json", commit)
    gate = tmp_path / "gates/case_0042.json"
    _gate(gate, 42, True)
    _enable_zero_progress_hold_contract(gate)
    output = tmp_path / "summary.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--root",
        str(tmp_path),
        "--git-commit",
        commit,
        "--cases",
        "42",
        "--expected-tracking-profile",
        "riser_recovery_direction_v4_camera_lever_arm_zero_progress_hold_v1",
        "--require-camera-lever-arm-compensation",
        "--require-zero-progress-hold",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    summary = json.loads(output.read_text())
    row = summary["gate_rows"][0]
    assert summary["passed"]
    assert summary["zero_progress_hold_required"] is True
    assert summary["zero_progress_hold_admission_passed"] is True
    assert row["zero_progress_hold_evidence_passed"] is True
    assert row["progress_hold_step_count"] == 2

    payload = json.loads(gate.read_text())
    payload["results"][0].update(
        {
            "progress_scale_min": 0.01,
            "progress_hold_step_count": 0,
            "progress_hold_ratio": 0.0,
            "progress_hold_segment_count": 0,
        }
    )
    gate.write_text(json.dumps(payload))
    subprocess.run(command, check=True)
    summary = json.loads(output.read_text())
    assert summary["passed"]
    assert summary["gate_rows"][0]["zero_progress_hold_evidence_passed"] is True

    _enable_zero_progress_hold_contract(gate)
    payload = json.loads(gate.read_text())
    payload["results"][0]["progress_hold_ratio"] = 0.3
    gate.write_text(json.dumps(payload))
    subprocess.run(command, check=True)
    summary = json.loads(output.read_text())
    assert summary["dynamic_quality_passed"]
    assert not summary["runtime_contract_passed"]
    assert summary["gate_rows"][0]["zero_progress_hold_evidence_passed"] is False

    payload["results"][0]["progress_hold_ratio"] = 0.2
    payload["results"][0]["outer_velocity_feedback_source"] = "root_link_vx"
    gate.write_text(json.dumps(payload))
    subprocess.run(command, check=True)
    summary = json.loads(output.read_text())
    assert summary["dynamic_quality_passed"]
    assert not summary["runtime_contract_passed"]
    assert summary["gate_rows"][0]["zero_progress_hold_evidence_passed"] is False


def test_recovery_velocity_cap_is_independent_and_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    commit = "a" * 40
    _admission(tmp_path / "admission.json", commit, tmp_path.name)
    _enable_recovery_velocity_cap_admission(tmp_path / "admission.json", commit)
    gate = tmp_path / "gates/case_0042.json"
    _gate(gate, 42, True)
    _enable_recovery_velocity_cap_contract(gate)
    output = tmp_path / "summary.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--root",
        str(tmp_path),
        "--git-commit",
        commit,
        "--cases",
        "42",
        "--expected-tracking-profile",
        (
            "riser_recovery_direction_v4_camera_lever_arm_"
            "zero_progress_hold_velocity_cap_v1"
        ),
        "--require-camera-lever-arm-compensation",
        "--require-zero-progress-hold",
        "--require-recovery-velocity-cap",
        "--expected-maximum-linear-velocity-mps",
        "0.2",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    summary = json.loads(output.read_text())
    row = summary["gate_rows"][0]
    assert summary["passed"]
    assert summary["recovery_velocity_cap_required"] is True
    assert summary["expected_maximum_linear_velocity_mps"] == 0.2
    assert row["recovery_velocity_cap_evidence_passed"] is True
    assert row["maximum_linear_velocity_mps"] == 0.2

    payload = json.loads(gate.read_text())
    payload["results"][0]["velocity_feedback_telemetry"][
        "effective_reference_abs_max_mps"
    ] = 0.2001
    gate.write_text(json.dumps(payload))
    subprocess.run(command, check=True)
    summary = json.loads(output.read_text())
    assert summary["dynamic_quality_passed"]
    assert not summary["runtime_contract_passed"]
    assert summary["gate_rows"][0]["recovery_velocity_cap_evidence_passed"] is False

    payload["results"][0]["velocity_feedback_telemetry"][
        "effective_reference_abs_max_mps"
    ] = 0.2
    payload["tracking_overrides"]["maximum_linear_velocity_mps"] = 0.21
    gate.write_text(json.dumps(payload))
    subprocess.run(command, check=True)
    summary = json.loads(output.read_text())
    assert not summary["runtime_contract_passed"]
    assert summary["gate_rows"][0]["recovery_velocity_cap_evidence_passed"] is False


def test_summary_requires_total_pitch_reference_limit_evidence(tmp_path: Path) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    commit = "a" * 40
    _admission(tmp_path / "admission.json", commit, tmp_path.name)
    _enable_total_pitch_reference_limit_admission(
        tmp_path / "admission.json", commit
    )
    gate = tmp_path / "gates/case_0042.json"
    _gate(gate, 42, True)
    _enable_total_pitch_reference_limit_contract(gate)
    output = tmp_path / "summary.json"
    profile = (
        "riser_recovery_direction_v4_camera_lever_arm_"
        "zero_progress_hold_velocity_cap_total_pitch_limit_v1"
    )
    command = [
        sys.executable,
        str(SCRIPT),
        "--root",
        str(tmp_path),
        "--git-commit",
        commit,
        "--cases",
        "42",
        "--expected-tracking-profile",
        profile,
        "--require-camera-lever-arm-compensation",
        "--require-zero-progress-hold",
        "--require-recovery-velocity-cap",
        "--expected-maximum-linear-velocity-mps",
        "0.2",
        "--require-total-pitch-reference-limit",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    summary = json.loads(output.read_text())
    row = summary["gate_rows"][0]
    assert summary["passed"]
    assert summary["total_pitch_reference_limit_required"] is True
    assert summary["expected_total_pitch_reference_limit_rad"] == pytest.approx(
        math.radians(6.0)
    )
    assert row["total_pitch_reference_limit_evidence_passed"] is True

    payload = json.loads(gate.read_text())
    payload["results"][0]["velocity_feedback_telemetry"][
        "total_pitch_reference_abs_max_rad"
    ] = math.radians(6.01)
    gate.write_text(json.dumps(payload))
    subprocess.run(command, check=True)
    summary = json.loads(output.read_text())
    assert summary["dynamic_quality_passed"]
    assert not summary["runtime_contract_passed"]
    assert (
        summary["gate_rows"][0]["total_pitch_reference_limit_evidence_passed"]
        is False
    )

    payload["results"][0]["velocity_feedback_telemetry"][
        "total_pitch_reference_abs_max_rad"
    ] = math.radians(6.0)
    payload["controller_overrides"] = {"wz_kp": 1.05}
    gate.write_text(json.dumps(payload))
    subprocess.run(command, check=True)
    summary = json.loads(output.read_text())
    assert not summary["runtime_contract_passed"]
    assert (
        summary["gate_rows"][0]["total_pitch_reference_limit_evidence_passed"]
        is False
    )


def test_summary_requires_commanded_base_progress_error_evidence(
    tmp_path: Path,
) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    commit = "a" * 40
    _admission(tmp_path / "admission.json", commit, tmp_path.name)
    _enable_commanded_base_progress_error_admission(
        tmp_path / "admission.json", commit
    )
    gate = tmp_path / "gates/case_0042.json"
    _gate(gate, 42, True)
    _enable_commanded_base_progress_error_contract(gate)
    output = tmp_path / "summary.json"
    profile = (
        "riser_recovery_direction_v4_camera_lever_arm_"
        "zero_progress_hold_velocity_cap_total_pitch_limit_v1"
    )
    command = [
        sys.executable,
        str(SCRIPT),
        "--root",
        str(tmp_path),
        "--git-commit",
        commit,
        "--cases",
        "42",
        "--expected-tracking-profile",
        profile,
        "--require-camera-lever-arm-compensation",
        "--require-zero-progress-hold",
        "--require-recovery-velocity-cap",
        "--expected-maximum-linear-velocity-mps",
        "0.2",
        "--require-total-pitch-reference-limit",
        "--require-commanded-base-progress-error",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    summary = json.loads(output.read_text())
    row = summary["gate_rows"][0]
    assert summary["passed"]
    assert summary["commanded_base_progress_error_required"] is True
    assert summary["commanded_base_progress_error_evidence_passed"] is True
    assert row["commanded_base_progress_error_evidence_passed"] is True
    assert row["progress_base_error_selected_source_matches"] is True

    payload = json.loads(gate.read_text())
    payload["results"][0][
        "progress_base_error_selected_source_matches"
    ] = False
    payload["results"][0]["checks"][
        "progress_base_error_selected_source_matches"
    ] = False
    gate.write_text(json.dumps(payload))
    subprocess.run(command, check=True)
    summary = json.loads(output.read_text())
    assert summary["dynamic_quality_passed"]
    assert not summary["runtime_contract_passed"]
    assert summary["commanded_base_progress_error_evidence_passed"] is False

    payload["results"][0][
        "progress_base_error_selected_source_matches"
    ] = True
    payload["results"][0]["checks"][
        "progress_base_error_selected_source_matches"
    ] = True
    payload["results"][0][
        "selected_vs_nominal_base_progress_error_abs_max_delta_m"
    ] = 0.0501
    gate.write_text(json.dumps(payload))
    subprocess.run(command, check=True)
    summary = json.loads(output.read_text())
    assert not summary["runtime_contract_passed"]
    assert summary["commanded_base_progress_error_evidence_passed"] is False


def test_summary_requires_opposing_vx_integral_reset_evidence(
    tmp_path: Path,
) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    commit = "a" * 40
    admission = tmp_path / "admission.json"
    _admission(admission, commit, tmp_path.name)
    _enable_opposing_vx_integral_reset_admission(admission, commit)
    gate = tmp_path / "gates/case_0042.json"
    _gate(gate, 42, True)
    _enable_opposing_vx_integral_reset_contract(gate)
    output = tmp_path / "summary.json"
    profile = (
        "riser_recovery_direction_v4_camera_lever_arm_"
        "zero_progress_hold_velocity_cap_total_pitch_limit_v1"
    )
    command = [
        sys.executable,
        str(SCRIPT),
        "--root",
        str(tmp_path),
        "--git-commit",
        commit,
        "--cases",
        "42",
        "--expected-tracking-profile",
        profile,
        "--require-camera-lever-arm-compensation",
        "--require-zero-progress-hold",
        "--require-recovery-velocity-cap",
        "--expected-maximum-linear-velocity-mps",
        "0.2",
        "--require-total-pitch-reference-limit",
        "--require-commanded-base-progress-error",
        "--require-opposing-vx-integral-deficit-reset",
        "--expected-vx-integral-reset-reference-deadband-mps",
        "0.05",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    summary = json.loads(output.read_text())
    row = summary["gate_rows"][0]
    assert summary["passed"]
    assert summary["opposing_vx_integral_deficit_reset_required"] is True
    assert summary["longitudinal_authority_evidence_passed"] is True
    assert row["longitudinal_authority_evidence_passed"] is True
    assert row["longitudinal_authority_telemetry"]["integral_reset_count"] == 2

    payload = json.loads(gate.read_text())
    payload["results"][0]["longitudinal_authority_telemetry"][
        "integral_reset_count"
    ] = 0
    gate.write_text(json.dumps(payload))
    subprocess.run(command, check=True)
    summary = json.loads(output.read_text())
    assert summary["dynamic_quality_passed"]
    assert not summary["runtime_contract_passed"]
    assert summary["longitudinal_authority_evidence_passed"] is False

    _enable_opposing_vx_integral_reset_contract(gate)
    payload = json.loads(gate.read_text())
    payload["results"][0]["longitudinal_authority_telemetry"][
        "held_controller_command_step_count"
    ] = 6
    gate.write_text(json.dumps(payload))
    subprocess.run(command, check=True)
    summary = json.loads(output.read_text())
    assert summary["longitudinal_authority_evidence_passed"] is False

    _enable_opposing_vx_integral_reset_contract(gate)
    payload = json.loads(gate.read_text())
    del payload["results"][0]["longitudinal_authority_telemetry"]
    gate.write_text(json.dumps(payload))
    subprocess.run(command, check=True)
    summary = json.loads(output.read_text())
    assert summary["longitudinal_authority_evidence_passed"] is False

    _enable_opposing_vx_integral_reset_contract(gate)
    admission_payload = json.loads(admission.read_text())
    admission_payload["reviewed_controller_parent_commit"] = "f" * 40
    admission.write_text(json.dumps(admission_payload))
    subprocess.run(command, check=True)
    summary = json.loads(output.read_text())
    assert not summary["zero_progress_hold_admission_passed"]
    assert not summary["runtime_contract_passed"]


def test_summary_requires_exact_controller_vx_kp_evidence(tmp_path: Path) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    commit = "a" * 40
    admission = tmp_path / "admission.json"
    _admission(admission, commit, tmp_path.name)
    _enable_controller_vx_kp_admission(admission, commit)
    gate = tmp_path / "gates/case_0042.json"
    _gate(gate, 42, True)
    _enable_controller_vx_kp_contract(gate)
    output = tmp_path / "summary.json"
    profile = (
        "riser_recovery_direction_v4_camera_lever_arm_"
        "zero_progress_hold_velocity_cap_total_pitch_limit_v1"
    )
    command = [
        sys.executable,
        str(SCRIPT),
        "--root",
        str(tmp_path),
        "--git-commit",
        commit,
        "--cases",
        "42",
        "--expected-tracking-profile",
        profile,
        "--require-camera-lever-arm-compensation",
        "--require-zero-progress-hold",
        "--require-recovery-velocity-cap",
        "--expected-maximum-linear-velocity-mps",
        "0.2",
        "--require-total-pitch-reference-limit",
        "--require-commanded-base-progress-error",
        "--require-opposing-vx-integral-deficit-reset",
        "--expected-vx-integral-reset-reference-deadband-mps",
        "0.05",
        "--require-controller-vx-kp",
        "--expected-controller-vx-kp",
        "0.72",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    summary = json.loads(output.read_text())
    assert summary["passed"]
    assert summary["controller_vx_kp_required"] is True
    assert summary["expected_controller_vx_kp"] == 0.72
    assert summary["controller_vx_kp_evidence_passed"] is True
    assert summary["gate_rows"][0]["controller_vx_kp"] == 0.72

    payload = json.loads(gate.read_text())
    payload["results"][0]["controller_vx_kp"] = 0.71
    gate.write_text(json.dumps(payload))
    subprocess.run(command, check=True)
    summary = json.loads(output.read_text())
    assert summary["dynamic_quality_passed"]
    assert summary["controller_vx_kp_evidence_passed"] is False
    assert not summary["runtime_contract_passed"]
    assert not summary["passed"]

    _enable_controller_vx_kp_contract(gate)
    admission_payload = json.loads(admission.read_text())
    admission_payload["reviewed_controller_parent_commit"] = "f" * 40
    admission.write_text(json.dumps(admission_payload))
    subprocess.run(command, check=True)
    summary = json.loads(output.read_text())
    assert not summary["zero_progress_hold_admission_passed"]
    assert not summary["runtime_contract_passed"]
    assert not summary["passed"]


def test_summary_stops_at_first_reject_and_keeps_training_closed(tmp_path: Path) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    commit = "a" * 40
    _admission(tmp_path / "admission.json", commit, tmp_path.name)
    _gate(tmp_path / "gates/case_0001.json", 1, True)
    _gate(tmp_path / "gates/case_0002.json", 2, False)
    output = tmp_path / "summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--git-commit",
            commit,
            "--cases",
            "1,2,3",
            "--output",
            str(output),
        ],
        check=False,
    )
    summary = json.loads(output.read_text())
    assert result.returncode == 0
    assert summary["dynamically_passed_cases"] == [1]
    assert summary["first_dynamic_reject"]["case"] == 2
    assert summary["not_started_cases"] == [3]
    assert summary["source_execution_timing_separated"]
    assert not summary["residual_capture_started"]
    assert not summary["bc_started"]
    assert not summary["ppo_started"]
    assert not summary["passed"]
    assert summary["gate_rows"][0]["riser_thermal_load_max"] == 0.4


def test_dynamic_pass_is_independent_of_label_envelope(tmp_path: Path) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    commit = "a" * 40
    _admission(tmp_path / "admission.json", commit, tmp_path.name)
    _gate(
        tmp_path / "gates/case_0074.json",
        74,
        True,
        label_envelope_passed=False,
    )
    gate = tmp_path / "gates/case_0074.json"
    payload = json.loads(gate.read_text())
    payload["passed"] = False
    payload["passed_case_count"] = 0
    payload["results"][0]["passed"] = False
    gate.write_text(json.dumps(payload))
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--git-commit",
            commit,
            "--cases",
            "74",
            "--output",
            str(output),
            "--require-case74-contract",
            "--expected-case74-contract-sha256",
            CONTRACT_SHA,
        ],
        check=True,
    )
    summary = json.loads(output.read_text())
    assert summary["passed"]
    assert summary["dynamic_quality_passed"]
    assert not summary["residual_label_envelope_passed"]
    assert not summary["residual_label_admission_passed"]
    assert not summary["valid_for_training"]


def test_runtime_contract_rejects_wrong_tracking_profile(tmp_path: Path) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    commit = "a" * 40
    _admission(tmp_path / "admission.json", commit, tmp_path.name)
    gate = tmp_path / "gates/case_0074.json"
    _gate(gate, 74, True)
    payload = json.loads(gate.read_text())
    payload["tracking_profile"] = "riser_motion_direction_v3"
    gate.write_text(json.dumps(payload))
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--git-commit",
            commit,
            "--cases",
            "74",
            "--output",
            str(output),
        ],
        check=True,
    )
    summary = json.loads(output.read_text())
    assert summary["dynamic_quality_passed"]
    assert summary["thermal_admission_passed"]
    assert not summary["runtime_contract_passed"]
    assert summary["first_dynamic_reject"]["classification"] == (
        "runtime_contract_rejection"
    )
    assert not summary["passed"]


def test_camera_lever_arm_runtime_contract_passes_with_bounded_telemetry(
    tmp_path: Path,
) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    commit = "a" * 40
    _admission(tmp_path / "admission.json", commit, tmp_path.name)
    gate = tmp_path / "gates/case_0068.json"
    _gate(gate, 68, True)
    _enable_camera_lever_arm_contract(gate)
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--git-commit",
            commit,
            "--cases",
            "68",
            "--output",
            str(output),
            "--expected-tracking-profile",
            "riser_recovery_direction_v4_camera_lever_arm_v1",
            "--require-camera-lever-arm-compensation",
        ],
        check=True,
    )
    summary = json.loads(output.read_text())
    assert summary["passed"]
    assert summary["dynamic_quality_passed"]
    assert summary["thermal_admission_passed"]
    assert summary["controller_evidence_passed"]
    assert summary["runtime_contract_passed"]
    assert not summary["valid_for_training"]


def test_camera_lever_arm_runtime_contract_rejects_missing_policy_rate_sample(
    tmp_path: Path,
) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    commit = "a" * 40
    _admission(tmp_path / "admission.json", commit, tmp_path.name)
    gate = tmp_path / "gates/case_0068.json"
    _gate(gate, 68, True)
    _enable_camera_lever_arm_contract(gate)
    payload = json.loads(gate.read_text())
    payload["results"][0]["camera_lever_arm_telemetry_sample_count"] = 9
    gate.write_text(json.dumps(payload))
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--git-commit",
            commit,
            "--cases",
            "68",
            "--output",
            str(output),
            "--expected-tracking-profile",
            "riser_recovery_direction_v4_camera_lever_arm_v1",
            "--require-camera-lever-arm-compensation",
        ],
        check=True,
    )
    summary = json.loads(output.read_text())
    assert summary["dynamic_quality_passed"]
    assert summary["thermal_admission_passed"]
    assert not summary["controller_evidence_passed"]
    assert not summary["runtime_contract_passed"]
    assert not summary["passed"]


def test_camera_error_governor_runtime_contract_passes_with_bounded_telemetry(
    tmp_path: Path,
) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    commit = "a" * 40
    _admission(tmp_path / "admission.json", commit, tmp_path.name)
    gate = tmp_path / "gates/case_0020.json"
    _gate(gate, 20, True)
    _enable_camera_error_governor_contract(gate)
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--git-commit",
            commit,
            "--cases",
            "20",
            "--output",
            str(output),
            "--expected-tracking-profile",
            "riser_recovery_direction_v4_camera_lever_arm_error_governor_v1",
            "--require-camera-lever-arm-compensation",
            "--require-camera-error-recovery-governor",
        ],
        check=True,
    )
    summary = json.loads(output.read_text())
    assert summary["passed"]
    assert summary["runtime_contract_passed"]
    assert summary["controller_evidence_passed"]
    assert summary["gate_rows"][0]["camera_recovery_activation_ratio"] == 0.1


def test_camera_error_governor_runtime_contract_rejects_noop_activation(
    tmp_path: Path,
) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    commit = "a" * 40
    _admission(tmp_path / "admission.json", commit, tmp_path.name)
    gate = tmp_path / "gates/case_0020.json"
    _gate(gate, 20, True)
    _enable_camera_error_governor_contract(gate)
    payload = json.loads(gate.read_text())
    payload["results"][0]["camera_recovery_activation_ratio"] = 0.0
    gate.write_text(json.dumps(payload))
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--git-commit",
            commit,
            "--cases",
            "20",
            "--output",
            str(output),
            "--expected-tracking-profile",
            "riser_recovery_direction_v4_camera_lever_arm_error_governor_v1",
            "--require-camera-lever-arm-compensation",
            "--require-camera-error-recovery-governor",
        ],
        check=True,
    )
    summary = json.loads(output.read_text())
    assert summary["dynamic_quality_passed"]
    assert not summary["runtime_contract_passed"]
    assert not summary["passed"]


def test_runtime_contract_rejects_missing_thermal_force_gate(tmp_path: Path) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    commit = "a" * 40
    _admission(tmp_path / "admission.json", commit, tmp_path.name)
    gate = tmp_path / "gates/case_0074.json"
    _gate(gate, 74, True)
    payload = json.loads(gate.read_text())
    payload["results"][0]["checks"]["riser_thermal_load_bounded"] = False
    gate.write_text(json.dumps(payload))
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--git-commit",
            commit,
            "--cases",
            "74",
            "--output",
            str(output),
        ],
        check=True,
    )
    summary = json.loads(output.read_text())
    assert summary["dynamic_quality_passed"]
    assert not summary["thermal_admission_passed"]
    assert summary["runtime_contract_passed"]
    assert summary["first_dynamic_reject"]["classification"] == (
        "thermal_admission_rejection"
    )


def test_runtime_contract_rejects_missing_policy_rate_recovery_telemetry(
    tmp_path: Path,
) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    commit = "a" * 40
    _admission(tmp_path / "admission.json", commit, tmp_path.name)
    gate = tmp_path / "gates/case_0074.json"
    _gate(gate, 74, True)
    payload = json.loads(gate.read_text())
    payload["results"][0].pop("recovery_telemetry")
    gate.write_text(json.dumps(payload))
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--git-commit",
            commit,
            "--cases",
            "74",
            "--output",
            str(output),
        ],
        check=True,
    )
    summary = json.loads(output.read_text())
    assert summary["dynamic_quality_passed"]
    assert not summary["runtime_contract_passed"]
    assert not summary["passed"]


def test_case74_summary_rejects_missing_or_mismatched_contract(tmp_path: Path) -> None:
    (tmp_path / "gates").mkdir()
    (tmp_path / "logs").mkdir()
    commit = "a" * 40
    _admission(tmp_path / "admission.json", commit, tmp_path.name)
    _gate(tmp_path / "gates/case_0074.json", 74, True)
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--git-commit",
            commit,
            "--cases",
            "74",
            "--output",
            str(output),
            "--require-case74-contract",
            "--expected-case74-contract-sha256",
            "0" * 64,
        ],
        check=True,
    )
    summary = json.loads(output.read_text())
    assert summary["dynamic_quality_passed"]
    assert not summary["case74_contract_admission_passed"]
    assert not summary["passed"]
