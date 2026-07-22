import json
from pathlib import Path
import numpy as np
import pytest

from rl_platform.tasks.two_wheel_balance.riser_corrective_teacher import (
    CorrectiveTeacherConfig,
    CorrectiveTeacherTelemetry,
    assess_paired_corrective_rollouts,
    build_corrective_teacher_action,
    load_corrective_teacher_profile,
)


def _paired_metrics() -> tuple[dict[str, object], dict[str, object]]:
    common = {
        "case": 30,
        "split": "train",
        "plan_sha256": "a" * 64,
        "physics_seed": 7,
        "source_duration_s": 18.0,
        "execution_duration_s": 29.0,
        "dynamic_quality_passed": True,
        "attitude_error_max_deg": 0.22,
        "pitch_max_deg": 7.0,
        "riser_error_max_m": 0.014,
        "action_saturation_ratio": 0.0,
        "dataset_created": False,
        "training_started": False,
        "ppo_started": False,
    }
    baseline = common | {
        "position_error_p95_m": 0.142,
        "position_error_max_m": 0.170,
        "normalized_residual_action_abs_max": [0.0, 0.0, 0.0],
    }
    candidate = common | {
        "position_error_p95_m": 0.136,
        "position_error_max_m": 0.168,
        "normalized_residual_action_abs_max": [0.5, 0.4, 0.3],
    }
    return baseline, candidate


def test_corrective_teacher_is_causal_bounded_and_slew_limited() -> None:
    output = build_corrective_teacher_action(
        np.array([0.30, -0.30, 0.10]),
        np.zeros(3),
        dt_s=0.005,
    )
    np.testing.assert_allclose(output.unbounded_residual, [0.058, -0.087, 0.0285])
    np.testing.assert_allclose(output.bounded_residual, [0.045, -0.045, 0.018])
    np.testing.assert_allclose(output.applied_residual, [0.0005, -0.0005, 0.0002])
    np.testing.assert_allclose(output.normalized_action, [0.01, -0.01, 0.01])
    assert output.amplitude_limited.tolist() == [True, True, True]
    assert output.slew_limited.tolist() == [True, True, True]


def test_corrective_teacher_deadband_emits_exact_zero() -> None:
    output = build_corrective_teacher_action(
        np.array([0.009, -0.009, 0.004]),
        np.zeros(3),
        dt_s=0.005,
    )
    np.testing.assert_array_equal(output.applied_residual, np.zeros(3))
    np.testing.assert_array_equal(output.normalized_action, np.zeros(3))


def test_corrective_teacher_profile_is_exact_and_hash_bound(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "schema": "cinebotrl_two_wheel_riser_corrective_teacher_profile_v1",
                "case": 30,
                "longitudinal_gain_s_inv": 0.2,
                "lateral_to_yaw_gain_rad_s_m": 0.3,
                "vertical_gain": 0.3,
                "deadbands_m": [0.01, 0.01, 0.005],
                "maximum_residuals": [0.045, 0.045, 0.018],
                "maximum_slew_rates": [0.1, 0.1, 0.04],
            }
        ),
        encoding="utf-8",
    )
    case, config, identity = load_corrective_teacher_profile(path)
    assert case == 30
    assert config == CorrectiveTeacherConfig()
    assert len(identity["sha256"]) == 64


def test_committed_case30_profile_matches_reviewed_default() -> None:
    path = (
        Path(__file__).parents[1]
        / "scripts/two_wheel_balance/model_based_corrective_teacher_case30_profile_v1.json"
    )
    case, config, _ = load_corrective_teacher_profile(path)
    assert case == 30
    assert config == CorrectiveTeacherConfig()


def test_committed_case23_profile_requires_explicit_case_identity() -> None:
    path = (
        Path(__file__).parents[1]
        / "scripts/two_wheel_balance/model_based_corrective_teacher_case23_profile_v1.json"
    )
    with pytest.raises(ValueError, match="not the reviewed case"):
        load_corrective_teacher_profile(path)
    case, config, _ = load_corrective_teacher_profile(path, expected_case=23)
    assert case == 23
    assert config == CorrectiveTeacherConfig()


def test_corrective_teacher_profile_rejects_extra_fields(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "schema": "cinebotrl_two_wheel_riser_corrective_teacher_profile_v1",
                "case": 30,
                "longitudinal_gain_s_inv": 0.2,
                "lateral_to_yaw_gain_rad_s_m": 0.3,
                "vertical_gain": 0.3,
                "deadbands_m": [0.01, 0.01, 0.005],
                "maximum_residuals": [0.045, 0.045, 0.018],
                "maximum_slew_rates": [0.1, 0.1, 0.04],
                "unreviewed": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected.*fields"):
        load_corrective_teacher_profile(path)


def test_corrective_teacher_telemetry_is_non_training() -> None:
    output = build_corrective_teacher_action(
        np.array([0.30, -0.30, 0.10]), np.zeros(3), dt_s=0.005
    )
    telemetry = CorrectiveTeacherTelemetry()
    telemetry.step(output)
    summary = telemetry.summary(enabled=True)
    assert summary["sample_count"] == 1
    assert summary["labels_captured"] is False
    assert summary["dataset_created"] is False
    assert max(summary["normalized_action_abs_max"]) < 0.95


def test_corrective_teacher_rejects_configuration_without_policy_margin() -> None:
    config = CorrectiveTeacherConfig(maximum_residuals=(0.05, 0.04, 0.01))
    with pytest.raises(ValueError, match="configuration"):
        config.validate()


def test_paired_admission_accepts_only_measurable_safe_improvement() -> None:
    baseline, candidate = _paired_metrics()
    report = assess_paired_corrective_rollouts(baseline, candidate)
    assert report["corrective_target_admission_passed"] is True
    assert report["label_capture_authorized"] is False
    assert report["valid_for_training"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        {"position_error_p95_m": 0.141},
        {"position_error_max_m": 0.176},
        {"normalized_residual_action_abs_max": [0.95, 0.1, 0.1]},
        {"dynamic_quality_passed": False},
        {"split": "validation"},
        {"dataset_created": True},
    ],
)
def test_paired_admission_rejects_weak_unsafe_or_leaking_candidate(mutation) -> None:
    baseline, candidate = _paired_metrics()
    candidate.update(mutation)
    report = assess_paired_corrective_rollouts(baseline, candidate)
    assert report["corrective_target_admission_passed"] is False


def test_paired_admission_rejects_identity_mismatch() -> None:
    baseline, candidate = _paired_metrics()
    candidate["physics_seed"] = 8
    report = assess_paired_corrective_rollouts(baseline, candidate)
    assert report["checks"]["same_case_plan_seed_and_clocks"] is False
    assert report["corrective_target_admission_passed"] is False
