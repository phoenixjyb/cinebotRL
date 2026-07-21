import pytest

from scripts.two_wheel_balance.build_riser_split_reset_fallback_proposal import (
    EXPECTED_HOLDOUT,
    EXPECTED_TRAIN,
    EXPECTED_VALIDATION,
    build_proposal,
)


def _inputs() -> tuple[dict, dict, dict, dict]:
    architecture = {
        "decision": "controlled_perturbation_contract_first",
        "current_split_cases": {
            "train": EXPECTED_TRAIN,
            "validation": EXPECTED_VALIDATION,
            "holdout": EXPECTED_HOLDOUT,
        },
        "options": {
            "transparent_split_reset": {
                "replacement_validation_candidate": 78,
            }
        },
    }
    coverage = {
        "coverage_admission_passed": False,
        "state_coverage_materially_improved": False,
        "dataset_created": False,
        "dagger_authorized": False,
        "valid_for_training": False,
    }
    unused = {
        "best_unused_admitted": {
            "case": 78,
            "plan": {"sha256": "plan78"},
        }
    }
    plan_summary = {
        "items": [{
            "case": 78,
            "plan_sha256": "plan78",
            "source_pose_count": 100,
            "execution_state_count": 100,
            "source_duration_s": 10.0,
            "execution_duration_s": 20.0,
            "path_metrics": {"source_path_length_m": 2.0},
            "kinematic_metrics": {
                "position_error_p95_m": 0.1,
                "position_error_max_m": 0.2,
            },
            "checks": {"source_time_verbatim": True},
            "kinematic_checks": {"position_p95_bounded": True},
            "timing_transition_kinematic_gate_passed": True,
            "valid_for_training": False,
        }]
    }
    return architecture, coverage, unused, plan_summary


def test_proposal_keeps_split_unapplied_and_holdout_closed() -> None:
    report = build_proposal(*_inputs())
    proposed = report["proposed_split_cases_not_applied"]
    assert 4 in proposed["train"]
    assert 4 not in proposed["validation"]
    assert 78 in proposed["validation"]
    assert proposed["holdout"] == EXPECTED_HOLDOUT
    assert not report["split_changed"]
    assert not report["runtime_authorized"]
    assert not report["dataset_created"]


def test_proposal_rejects_split_reset_when_perturbation_coverage_passes() -> None:
    inputs = list(_inputs())
    inputs[1]["coverage_admission_passed"] = True
    with pytest.raises(ValueError, match="input contract failed"):
        build_proposal(*inputs)
