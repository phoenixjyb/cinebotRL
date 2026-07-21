import numpy as np

from scripts.two_wheel_balance.diagnose_riser_shadow_teacher_gap import (
    build_report,
)


def _payload(shadow_offset: np.ndarray) -> tuple[dict, dict]:
    count = 20
    phase = np.arange(count, dtype=np.float64) * 0.005
    phase_actions = np.tile([0.1, -0.2, 0.05], (count, 1))
    applied = phase_actions.copy()
    shadow = {
        "phase_time_s": phase,
        "applied_residual_actions": applied,
        "shadow_teacher_normalized_residual_actions": (
            phase_actions + shadow_offset
        ),
    }
    dataset = {
        "case_ids": np.full(count, 4, dtype=np.int16),
        "phase_time_s": phase,
        "actions": phase_actions,
    }
    return shadow, dataset


def test_material_on_policy_gap_supports_proposal_but_not_training() -> None:
    shadow, dataset = _payload(np.array([0.08, 0.0, 0.0]))
    report = build_report(shadow, dataset, case=4)
    assert report["dagger_dataset_proposal_supported"] is True
    assert report["classification"] == (
        "on_policy_teacher_gap_supports_bounded_dagger_proposal"
    )
    assert report["material_shadow_shift_by_channel"] == [True, False, False]
    assert report["shadow_teacher_applied_to_commands"] is False
    assert report["dataset_created"] is False
    assert report["dagger_authorized"] is False
    assert report["bc_authorized"] is False
    assert report["ppo_authorized"] is False
    assert report["valid_for_training"] is False


def test_small_on_policy_gap_does_not_support_proposal() -> None:
    shadow, dataset = _payload(np.array([0.01, 0.01, 0.005]))
    report = build_report(shadow, dataset, case=4)
    assert report["dagger_dataset_proposal_supported"] is False
    assert report["material_shadow_shift_by_channel"] == [False, False, False]


def test_shadow_gap_rejects_missing_teacher_case() -> None:
    shadow, dataset = _payload(np.zeros(3))
    dataset["case_ids"][:] = 5
    try:
        build_report(shadow, dataset, case=4)
    except ValueError as error:
        assert "no usable case 4" in str(error)
    else:
        raise AssertionError("missing teacher case was accepted")
