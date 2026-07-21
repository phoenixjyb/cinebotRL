import numpy as np

from scripts.two_wheel_balance.diagnose_riser_policy_rate_shift import build_report


def test_policy_rate_diagnosis_keeps_training_closed() -> None:
    count = 20
    phase = np.arange(count, dtype=np.float64) * 0.005
    observations = np.zeros((count, 65), dtype=np.float32)
    teacher = {
        "phase_time_s": phase,
        "observations": observations,
        "post_step_position_error_m": np.linspace(0.01, 0.02, count),
    }
    learned_observations = observations.copy()
    learned_observations[:, 9] = np.linspace(0.0, 0.3, count)
    learned = {
        "phase_time_s": phase,
        "observations": learned_observations,
        "post_step_position_error_m": np.linspace(0.01, 0.05, count),
        "applied_residual_actions": np.zeros((count, 3), dtype=np.float32),
    }
    dataset = {
        "case_ids": np.full(count, 4, dtype=np.int16),
        "phase_time_s": phase,
        "actions": np.zeros((count, 3), dtype=np.float32),
    }
    report = build_report(
        teacher,
        learned,
        dataset,
        np.ones(65, dtype=np.float32),
        np.concatenate((np.ones(23), np.zeros(3), np.ones(39))),
        [f"channel_{index}" for index in range(65)],
        case=4,
    )
    assert report["classification"] == "no_single_non_output_policy_input_precursor_proven"
    assert report["strongest_outcome_coupled_association"] == "camera"
    assert report["group_metrics"]["previous_action"][
        "effective_standardized_delta_max"
    ] == 0.0
    assert report["teacher_relabel_capture_started"] is False
    assert report["dagger_authorized"] is False
    assert report["bc_authorized"] is False
    assert report["ppo_authorized"] is False
    assert report["valid_for_training"] is False


def test_policy_rate_diagnosis_rejects_teacher_trace_row_mismatch() -> None:
    count = 5
    phase = np.arange(count, dtype=np.float64)
    teacher = {
        "phase_time_s": phase,
        "observations": np.zeros((count, 65)),
        "post_step_position_error_m": np.zeros(count),
    }
    learned = {
        "phase_time_s": phase,
        "observations": np.zeros((count, 65)),
        "post_step_position_error_m": np.zeros(count),
        "applied_residual_actions": np.zeros((count, 3)),
    }
    dataset = {
        "case_ids": np.full(count - 1, 4),
        "phase_time_s": phase[:-1],
        "actions": np.zeros((count - 1, 3)),
    }
    try:
        build_report(
            teacher,
            learned,
            dataset,
            np.ones(65),
            np.ones(65),
            [f"channel_{index}" for index in range(65)],
            case=4,
        )
    except ValueError as error:
        assert "row counts differ" in str(error)
    else:
        raise AssertionError("mismatched teacher case was accepted")
