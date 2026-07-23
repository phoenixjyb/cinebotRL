import pytest

from rl_platform.tasks.two_wheel_balance.riser_corrective_validation import (
    assess_paired_corrective_validation,
)


def _metrics(*, candidate: bool, split: str = "validation"):
    return {
        "case": 8,
        "split": split,
        "plan_sha256": "a" * 64,
        "physics_seed": 20260724,
        "source_duration_s": 12.940941,
        "execution_duration_s": 18.1173174,
        "dynamic_quality_passed": True,
        "position_error_p95_m": 0.120 if candidate else 0.131254,
        "position_error_max_m": 0.140 if candidate else 0.143331,
        "attitude_error_max_deg": 0.45,
        "pitch_max_deg": 6.2,
        "riser_error_max_m": 0.012,
        "action_saturation_ratio": 0.0,
        "normalized_residual_action_abs_max": (
            [0.30, 0.16, 0.05] if candidate else [0.0, 0.0, 0.0]
        ),
        "dataset_created": False,
        "training_started": False,
        "ppo_started": False,
    }


def test_validation_pair_passes_without_opening_teacher_admission() -> None:
    result = assess_paired_corrective_validation(
        _metrics(candidate=False), _metrics(candidate=True)
    )
    assert result["validation_pair_passed"] is True
    assert result["checks"]["validation_split_only"] is True
    assert result["teacher_admission_opened"] is False
    assert result["label_capture_authorized"] is False
    assert result["valid_for_training"] is False


def test_validation_pair_rejects_train_split() -> None:
    result = assess_paired_corrective_validation(
        _metrics(candidate=False, split="train"),
        _metrics(candidate=True, split="train"),
    )
    assert result["checks"]["validation_split_only"] is False
    assert result["validation_pair_passed"] is False


def test_validation_pair_rejects_identity_or_dataset_drift() -> None:
    baseline = _metrics(candidate=False)
    candidate = _metrics(candidate=True)
    candidate["physics_seed"] = 99
    candidate["dataset_created"] = True
    result = assess_paired_corrective_validation(baseline, candidate)
    assert result["checks"]["same_case_plan_seed_and_clocks"] is False
    assert result["checks"]["no_dataset_or_training"] is False
    assert result["validation_pair_passed"] is False


def test_validation_pair_rejects_missing_or_invalid_action() -> None:
    candidate = _metrics(candidate=True)
    candidate["normalized_residual_action_abs_max"] = [0.2, 0.1]
    with pytest.raises(ValueError, match="residual-action evidence"):
        assess_paired_corrective_validation(
            _metrics(candidate=False), candidate
        )
