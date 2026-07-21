import numpy as np

from scripts.two_wheel_balance.rank_riser_dagger_localized_training_cases import (
    rank_localized_cases,
)


def _fixtures() -> tuple[dict, dict, dict, np.ndarray, np.ndarray, np.ndarray]:
    phase = np.arange(6, dtype=np.float64)
    case4_observations = np.zeros((6, 26), dtype=np.float64)
    case4_actions = np.zeros((6, 3), dtype=np.float64)
    shadow_observations = case4_observations.copy()
    shadow_observations[2:5, 0] = [1.0, 1.1, 1.2]
    shadow_actions = case4_actions.copy()
    shadow_actions[2:5, 0] = 0.2
    near_observations = shadow_observations.copy()
    far_observations = np.full((6, 26), 8.0, dtype=np.float64)
    observations = np.concatenate(
        [case4_observations, near_observations, far_observations]
    )
    actions = np.concatenate([case4_actions, case4_actions, case4_actions])
    dataset = {
        "observations": observations,
        "actions": actions,
        "case_ids": np.repeat([4, 18, 21], 6),
        "phase_time_s": np.tile(phase, 3),
    }
    shadow = {
        "observations": shadow_observations,
        "phase_time_s": phase,
        "shadow_teacher_normalized_residual_actions": shadow_actions,
    }
    metadata = {
        "split_cases": {
            "train": [18, 21],
            "validation": [4],
            "holdout": [3],
        }
    }
    observation_mean = np.zeros(26)
    observation_std = np.ones(26)
    observation_mask = np.ones(26)
    observation_mask[23:26] = 0.0
    return (
        shadow,
        dataset,
        metadata,
        observation_mean,
        observation_std,
        observation_mask,
    )


def test_localized_ranker_prefers_training_case_covering_hotspot() -> None:
    report = rank_localized_cases(*_fixtures())
    assert report["hotspot_row_count"] == 3
    assert report["top_training_cases"] == [18, 21]
    assert report["ranked_training_cases"][0]["case"] == 18
    assert report["previous_action_channels_effective"] is False
    assert report["holdout_opened"] is False
    assert report["runtime_authorized"] is False
    assert report["dataset_created"] is False
    assert report["dagger_authorized"] is False
    assert report["bc_authorized"] is False
    assert report["ppo_authorized"] is False


def test_localized_ranker_rejects_previous_action_enabled_policy() -> None:
    fixtures = list(_fixtures())
    fixtures[-1][23] = 1.0
    try:
        rank_localized_cases(*fixtures)
    except ValueError as error:
        assert "previous-action" in str(error)
    else:
        raise AssertionError("previous-action-enabled policy was accepted")


def test_localized_ranker_never_proposes_validation_or_holdout() -> None:
    report = rank_localized_cases(*_fixtures())
    proposed = set(report["proposed_runtime_cases"])
    assert not proposed & {3, 4}


def test_localized_ranker_fails_closed_when_training_coverage_is_distant() -> None:
    fixtures = list(_fixtures())
    fixtures[1]["observations"][6:] = 20.0
    report = rank_localized_cases(*fixtures)
    assert report["coverage_admission_passed"] is False
    assert report["proposed_runtime_cases"] == []
    assert report["classification"] == (
        "no_training_case_covers_case4_shadow_shift_region"
    )
    assert report["runtime_authorized"] is False
