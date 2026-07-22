import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/two_wheel_balance/build_model_based_corrective_teacher_proposal.py"
)
SPEC = importlib.util.spec_from_file_location("corrective_teacher_proposal", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _inputs():
    case8 = {
        "case": 8,
        "zero_residual_preservation_passed": True,
        "training_started": False,
    }
    case78 = {
        "case": 78,
        "zero_residual_preservation_passed": True,
        "training_started": False,
    }
    case30 = {
        "case": 30,
        "physical_quality_passed": True,
        "dagger_dataset_proposal_supported": False,
        "dataset_created": False,
        "valid_for_training": False,
    }
    summary = {
        "split_cases": {
            "train": [2, 30],
            "validation": [8, 78],
            "holdout": [3, 5, 13, 19, 24],
        },
        "base_dataset_rewrite_performed": False,
    }
    return case8, case78, case30, summary


def test_proposal_selects_training_case_and_keeps_runtime_closed() -> None:
    result = MODULE.build_proposal(*_inputs())
    assert result["candidate_case"] == 30
    assert result["candidate_split"] == "train"
    assert result["runtime_authorized"] is False
    assert result["label_capture_authorized"] is False
    assert result["bc_authorized"] is False
    assert result["ppo_authorized"] is False
    assert result["holdout_cases"] == [3, 5, 13, 19, 24]


@pytest.mark.parametrize("failure", ["case8", "case78", "case30", "split", "holdout"])
def test_proposal_rejects_missing_prerequisite(failure) -> None:
    case8, case78, case30, summary = _inputs()
    if failure == "case8":
        case8["zero_residual_preservation_passed"] = False
    elif failure == "case78":
        case78["zero_residual_preservation_passed"] = False
    elif failure == "case30":
        case30["physical_quality_passed"] = False
    elif failure == "split":
        summary["split_cases"]["train"].remove(30)
    else:
        summary["split_cases"]["holdout"] = [3, 5]
    with pytest.raises(ValueError, match="proposal inputs failed"):
        MODULE.build_proposal(case8, case78, case30, summary)
