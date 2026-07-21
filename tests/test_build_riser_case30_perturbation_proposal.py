import numpy as np
import pytest

from scripts.two_wheel_balance.build_riser_case30_perturbation_proposal import (
    select_localized_phase,
    validate_lqr_envelope,
)


def test_localized_phase_honors_margins_and_local_support() -> None:
    target = np.array([[0.0], [0.1], [5.0]])
    candidate = np.array([[0.0], [5.0], [0.05], [5.0]])
    times = np.array([0.0, 2.0, 4.0, 9.0])
    selected = select_localized_phase(
        target,
        candidate,
        times,
        execution_duration_s=10.0,
        local_support_count=2,
        start_margin_s=2.0,
        terminal_recovery_margin_s=2.0,
    )
    assert selected["candidate_index"] == 2
    assert selected["start_phase_time_s"] == 4.0
    assert selected["nearest_target_indices"] == [0, 1]


def test_localized_phase_rejects_missing_recovery_window() -> None:
    with pytest.raises(ValueError, match="recovery margins"):
        select_localized_phase(
            np.zeros((2, 1)),
            np.zeros((2, 1)),
            np.array([0.0, 1.0]),
            execution_duration_s=2.0,
            local_support_count=2,
        )


def _lqr_result(height: float) -> dict[str, object]:
    return {
        "passed": True,
        "push": {
            "forces_x_n": [-20.0, 20.0],
            "duration_steps": 20,
            "application_height_above_base_com_m": 0.5,
            "application": "global_x_force_plus_equivalent_global_y_pitch_torque",
        },
        "summary": {
            "scenarios": 56,
            "success_rate": 1.0,
            "peak_pitch_deg_max": 9.0,
            "action_saturation_ratio": 0.0,
            "riser_plant": {"riser_position_target_m": height},
        },
    }


def test_lqr_envelope_requires_three_complete_height_gates() -> None:
    summary = validate_lqr_envelope(
        [_lqr_result(0.0), _lqr_result(0.6), _lqr_result(1.2)]
    )
    assert all(summary["checks"].values())
    assert summary["scenario_count_total"] == 168
    assert summary["validated_force_frame"] == "global_x"
    assert summary["proposed_force_frame"] == "body_x"
    assert summary["frame_transfer_dynamically_validated"] is False


def test_lqr_envelope_rejects_saturation_or_force_drift() -> None:
    results = [_lqr_result(0.0), _lqr_result(0.6), _lqr_result(1.2)]
    results[1]["summary"]["action_saturation_ratio"] = 0.01
    with pytest.raises(ValueError, match="envelope contract failed"):
        validate_lqr_envelope(results)

    results = [_lqr_result(0.0), _lqr_result(0.6), _lqr_result(1.2)]
    results[2]["push"]["duration_steps"] = 21
    with pytest.raises(ValueError, match="envelope contract failed"):
        validate_lqr_envelope(results)
