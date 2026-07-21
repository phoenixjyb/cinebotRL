import numpy as np

from scripts.two_wheel_balance.audit_riser_case30_perturbation_coverage import (
    audit_coverage,
)


def _inputs(perturbed_value: float) -> tuple:
    observation_count = 6
    phase = np.linspace(0.0, 1.0, 6)
    case4_observations = np.zeros((6, observation_count))
    case4_observations[:, 0] = 1.0
    case4_shadow = {
        "phase_time_s": phase,
        "observations": case4_observations,
        "shadow_teacher_normalized_residual_actions": np.column_stack((
            np.full(6, 0.1),
            np.zeros(6),
            np.zeros(6),
        )),
    }
    perturbed_observations = np.full(
        (6, observation_count), perturbed_value
    )
    if perturbed_value == 1.0:
        perturbed_observations[:] = 0.0
        perturbed_observations[:, 0] = 1.0
    case30_perturbed = {
        "phase_time_s": phase,
        "observations": perturbed_observations,
    }
    teacher = {
        "case_ids": np.array([4] * 6 + [30] * 6),
        "phase_time_s": np.concatenate((phase, phase)),
        "actions": np.zeros((12, 3)),
        "observations": np.concatenate((
            case4_observations,
            np.full((6, observation_count), 5.0),
        )),
    }
    metadata = {
        "split_cases": {
            "train": [30],
            "validation": [4],
            "holdout": [3],
        }
    }
    return (
        case4_shadow,
        case30_perturbed,
        teacher,
        metadata,
        np.zeros(observation_count),
        np.ones(observation_count),
        np.ones(observation_count),
    )


def test_near_perturbed_trace_materially_improves_coverage() -> None:
    report = audit_coverage(*_inputs(1.0))
    assert report["perturbed_to_nominal_score_ratio"] < 0.1
    assert report["state_coverage_materially_improved"]
    assert report["reference_calibrated_coverage_passed"]
    assert report["coverage_admission_passed"]
    assert not report["causal_attribution_to_perturbation_proven"]
    assert not report["dataset_created"]


def test_far_perturbed_trace_fails_coverage_without_admitting_data() -> None:
    report = audit_coverage(*_inputs(6.0))
    assert report["perturbed_to_nominal_score_ratio"] > 1.0
    assert not report["state_coverage_materially_improved"]
    assert not report["coverage_admission_passed"]
    assert not report["dagger_authorized"]
    assert not report["valid_for_training"]
