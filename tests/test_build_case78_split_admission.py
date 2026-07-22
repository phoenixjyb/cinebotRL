import pytest

from scripts.two_wheel_balance.build_case78_split_admission import build_admission
from scripts.two_wheel_balance.build_riser_split_reset_fallback_proposal import (
    EXPECTED_HOLDOUT,
    EXPECTED_TRAIN,
    EXPECTED_VALIDATION,
)


def inputs() -> tuple[dict, dict, dict, dict]:
    proposed_train = sorted(EXPECTED_TRAIN + [4])
    proposed_validation = sorted(
        [case for case in EXPECTED_VALIDATION if case != 4] + [78]
    )
    proposal = {
        "decision": "transparent_split_reset_pending_case78_dynamic_qualification",
        "split_changed": False,
        "proposed_split_cases_not_applied": {
            "train": proposed_train,
            "validation": proposed_validation,
            "holdout": EXPECTED_HOLDOUT,
        },
    }
    dataset = {
        "split_cases": {
            "train": EXPECTED_TRAIN,
            "validation": EXPECTED_VALIDATION,
            "holdout": EXPECTED_HOLDOUT,
        },
        "trajectory_leakage": False,
        "case_count": 40,
        "captured_case_count": 41,
        "row_count": 403569,
    }
    final = {
        "case": 78,
        "passed": True,
        "dynamic_qualification_passed": True,
        "physical_quality_passed": True,
        "dataset_created": False,
        "split_changed": False,
    }
    result = {
        "case": 78,
        "completed_phase_time_s": 20.0,
        "execution_duration_s": 20.0,
        "source_duration_s": 10.0,
        "completed_steps": 100,
        "dynamic_quality_passed": True,
        "thermal_admission_passed": True,
        "controller_evidence_passed": True,
        "termination": None,
        "maximum_camera_lever_arm_correction_m": 0.10,
        "camera_recovery_governor_enabled": False,
        "position_error_p95_m": 0.12,
        "position_error_max_m": 0.18,
        "pitch_max_deg": 7.0,
        "residual_label_envelope_passed": False,
    }
    gate = {
        "cases": [78],
        "passed": True,
        "results": [result],
        "normalized_dataset_capture_started": False,
        "training_started": False,
    }
    return proposal, dataset, final, gate


def test_split_admission_swaps_only_case4_and_case78() -> None:
    report = build_admission(*inputs())
    admitted = report["admitted_split_cases"]
    assert 4 in admitted["train"]
    assert 4 not in admitted["validation"]
    assert 78 in admitted["validation"]
    assert admitted["holdout"] == EXPECTED_HOLDOUT
    assert report["split_admitted"] is True
    assert report["historical_dataset"]["rewrite_performed"] is False
    assert report["dataset_creation_authorized"] is False
    assert report["bc_authorized"] is False


def test_split_admission_rejects_failed_dynamic_or_admitted_labels() -> None:
    values = list(inputs())
    values[2]["dynamic_qualification_passed"] = False
    values[3]["results"][0]["residual_label_envelope_passed"] = True
    with pytest.raises(ValueError, match="contract failed"):
        build_admission(*values)
