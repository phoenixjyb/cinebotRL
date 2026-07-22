import copy

import pytest

from scripts.two_wheel_balance.build_initial_teacher41_validation_tranche_contract import (
    EXPECTED_CASES,
    POLICY_SHA256,
    REVIEWED_CASE78_RUNTIME_COMMIT,
    TRACKING_PROFILE,
    build_contract,
)


def _case78():
    summary_sha = "b" * 64
    final = {
        "schema": "cinebotrl_two_wheel_riser_initial_teacher41_validation_canary_final_v1",
        "runtime_commit": REVIEWED_CASE78_RUNTIME_COMMIT,
        "case": 78,
        "split": "validation",
        "checks": {"all": True},
        "process_exit_codes": {"learned": 0, "zero": 0, "comparison_gate": 0},
        "evidence": {"comparison_summary": {"sha256": summary_sha}},
        "dynamic_canary_passed": True,
        "remaining_validation_cases_authorized": False,
        "broad_rollout_authorized": False,
        "dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "ppo_started": False,
        "holdout_opened": False,
        "valid_for_training": False,
        "passed": True,
    }
    summary = {
        "schema": "cinebotrl_two_wheel_riser_residual_validation_canary_gate_v1",
        "policy_sha256": POLICY_SHA256,
        "cases": [78],
        "case_count": 1,
        "maximum_regression_fraction": 0.05,
        "minimum_zero_improvement_fraction": 0.05,
        "expected_tracking_profile": TRACKING_PROFILE,
        "passed": True,
    }
    return final, summary, summary_sha


def _inputs():
    reports = {}
    teachers = {}
    audits = {}
    for case, expected in EXPECTED_CASES.items():
        reports[case] = {
            "case": case,
            "plan_sha256": expected["plan_sha256"],
            "source_pose_count": expected["source_pose_count"],
            "execution_state_count": expected["execution_state_count"],
            "source_duration_s": expected["source_duration_s"],
            "execution_duration_s": expected["execution_duration_s"],
            "passed": True,
            "timing_transition_kinematic_gate_passed": True,
            "kinematic_checks": {"rates": True},
            "valid_for_training": False,
        }
        teachers[case] = {
            "cases": [case],
            "trajectory_command_source": "deterministic_teacher",
            "tracking_profile": TRACKING_PROFILE,
            "position_observation_link": "physical_cam_link_fk",
            "target_attitude_contract": "semantic_dfr_to_physical_cam_v1",
            "hardware_proxy_command_contract": "semantic_attitude_position_only",
            "controller_overrides": {"wz_kp": 1.05},
            "maximum_duration_scale": 3.0,
            "camera_lever_arm_compensation_enabled": True,
            "camera_lever_arm_compensation_gain": 1.0,
            "maximum_camera_lever_arm_correction_m": 0.05,
            "residual_policy": None,
            "raw_teacher_capture_started": True,
            "normalized_dataset_capture_started": False,
            "passed": True,
            "dynamic_quality_passed": True,
            "results": [
                {
                    "case": case,
                    "source_duration_s": expected["source_duration_s"],
                    "execution_duration_s": expected["execution_duration_s"],
                    "completed_phase_time_s": expected["execution_duration_s"],
                    "position_error_p95_m": 0.1,
                    "position_error_max_m": 0.14,
                    "residual_action_abs_max": [0.0, 0.0, 0.0],
                    "executed_residual_dataset": None,
                    "termination": None,
                    "passed": True,
                    "dynamic_quality_passed": True,
                }
            ],
        }
        audits[case] = {
            "schema": "cinebotrl_two_wheel_riser_raw_teacher_capture_audit_v1",
            "case": case,
            "checks": {"all": True},
            "capture_admission_passed": True,
            "gate_sha256": expected["teacher_gate_sha256"],
            "admission_sha256": (
                "95de47d95c825bb6a65cddddf4866525c05e70049ae932ab29d1c24568d15df8"
            ),
            "selection_sha256": (
                "e0f1d2b44061aabfe64ad2ffa3d23f57bf9b3e51015b2e3fa0703ba24316bb06"
            ),
            "source_duration_s": expected["source_duration_s"],
            "execution_duration_s": expected["execution_duration_s"],
            "raw_reconstruction_max_error": 1e-8,
            "action_scale_frozen": False,
            "valid_for_training": False,
            "bc_authorized": False,
            "ppo_authorized": False,
            "training_started": False,
            "passed": True,
        }
    return reports, teachers, audits


def _build():
    final, summary, summary_sha = _case78()
    reports, teachers, audits = _inputs()
    return build_contract(
        case78_final=final,
        case78_summary=summary,
        case78_summary_sha256=summary_sha,
        reports=reports,
        teachers=teachers,
        teacher_audits=audits,
        source_commit="a" * 40,
    )


def test_builds_cpu_only_validation_tranche_contract() -> None:
    result = _build()
    assert result["cases"] == [16, 22, 32]
    assert result["cpu_contract_ready"]
    assert result["controller_contract"]["maximum_camera_lever_arm_correction_m"] == 0.05
    assert not result["runtime_authorization_token_issued"]
    assert not result["runtime_namespace_created"]
    assert not result["gpu_launch_authorized"]
    assert not result["holdout_opened"]
    assert not result["ppo_authorized"]


def test_rejects_case78_failure_or_unsealed_summary() -> None:
    final, summary, summary_sha = _case78()
    reports, teachers, audits = _inputs()
    final["passed"] = False
    final["evidence"]["comparison_summary"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="validation tranche CPU contract"):
        build_contract(
            case78_final=final,
            case78_summary=summary,
            case78_summary_sha256=summary_sha,
            reports=reports,
            teachers=teachers,
            teacher_audits=audits,
            source_commit="a" * 40,
        )


def test_rejects_plan_clock_or_teacher_camera_contract_drift() -> None:
    final, summary, summary_sha = _case78()
    reports, teachers, audits = _inputs()
    reports = copy.deepcopy(reports)
    teachers = copy.deepcopy(teachers)
    reports[22]["execution_duration_s"] += 1.0
    teachers[32]["maximum_camera_lever_arm_correction_m"] = 0.1
    with pytest.raises(ValueError, match="validation tranche CPU contract"):
        build_contract(
            case78_final=final,
            case78_summary=summary,
            case78_summary_sha256=summary_sha,
            reports=reports,
            teachers=teachers,
            teacher_audits=audits,
            source_commit="a" * 40,
        )


def test_rejects_holdout_or_learning_side_effect() -> None:
    final, summary, summary_sha = _case78()
    reports, teachers, audits = _inputs()
    final["holdout_opened"] = True
    final["ppo_started"] = True
    with pytest.raises(ValueError, match="validation tranche CPU contract"):
        build_contract(
            case78_final=final,
            case78_summary=summary,
            case78_summary_sha256=summary_sha,
            reports=reports,
            teachers=teachers,
            teacher_audits=audits,
            source_commit="a" * 40,
        )


def test_rejects_teacher_audit_provenance_drift() -> None:
    final, summary, summary_sha = _case78()
    reports, teachers, audits = _inputs()
    audits[16]["checks"]["admission_plan_hash"] = False
    audits[22]["gate_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="validation tranche CPU contract"):
        build_contract(
            case78_final=final,
            case78_summary=summary,
            case78_summary_sha256=summary_sha,
            reports=reports,
            teachers=teachers,
            teacher_audits=audits,
            source_commit="a" * 40,
        )
