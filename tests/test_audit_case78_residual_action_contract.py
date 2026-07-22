import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.two_wheel_balance.audit_case78_residual_action_contract import (
    EXPECTED_HOLDOUT,
    build_report,
    compute_development_distribution,
    overflow_statistics,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs() -> tuple[dict, dict, dict, dict, dict]:
    case78 = {
        "case": 78,
        "source_duration_s": 10.0,
        "execution_duration_s": 20.0,
        "completed_phase_time_s": 20.0,
        "dynamic_quality_passed": True,
        "thermal_admission_passed": True,
        "controller_evidence_passed": True,
        "termination": None,
        "raw_residual_command_abs_max": [0.3026041341602768, 0.18, 0.01],
        "normalized_residual_label_abs_max": [
            1.0086804472009228,
            0.45,
            0.1,
        ],
        "raw_residual_label_applied_to_commands": False,
        "residual_action_abs_max": [0.0, 0.0, 0.0],
        "executed_residual_dataset": None,
        "residual_label_envelope_passed": False,
        "residual_label_admission_passed": False,
    }
    gate = {
        "cases": [78],
        "passed": True,
        "dynamic_quality_passed": True,
        "residual_action_scales": [0.3, 0.4, 0.1],
        "residual_label_envelope_passed": False,
        "residual_label_admission_passed": False,
        "training_started": False,
        "ppo_authorized": False,
        "results": [case78],
    }
    summary = {
        "passed": True,
        "valid_for_bc_initialization": True,
        "dataset_admission_passed": True,
        "case_count": 40,
        "captured_case_count": 41,
        "row_count": 403569,
        "trajectory_leakage": False,
        "action_scales": [0.35, 0.4, 0.1],
        "action_abs_max": [0.8, 0.7, 0.15],
        "action_clip_ratio": [0.0, 0.0, 0.0],
        "physical_gimbal_labels_used_as_actions": False,
        "source_action_labels_used": False,
        "training_started": False,
        "bc_authorized": False,
        "ppo_authorized": False,
    }
    corpus = {
        "passed": True,
        "valid_for_bc_initialization": True,
        "action_scale_frozen": True,
        "case_count": 41,
        "row_count": 406837,
        "frozen_action_scales": [0.35, 0.4, 0.1],
        "raw_residual_abs_max": [0.28, 0.28, 0.015],
    }
    split = {
        "split_admitted": True,
        "admitted_split_cases": {"holdout": EXPECTED_HOLDOUT},
        "case78_labels_available": False,
        "dataset_creation_authorized": False,
        "label_capture_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
    }
    distribution = {
        "case_count": 35,
        "skipped_holdout_cases": EXPECTED_HOLDOUT,
        "holdout_files_opened": False,
        "raw_residual_abs_percentiles": {"95.0": [0.1, 0.1, 0.01]},
    }
    return gate, summary, corpus, split, distribution


def test_retains_teacher_scale_but_requires_case78_series() -> None:
    report = build_report(*_inputs())
    json.dumps(report)
    assert report["teacher40_action_contract_retained"] is True
    assert report["candidate_scale_normalized_abs_max"][0] < 1.0
    assert report["case78_runtime_scale_overflow_channels"] == [True, False, False]
    assert report["case78_runtime_scale_overflow_duration_s"] is None
    assert report["case78_shadow_measurement_required_before_label_capture"] is True
    assert report["holdout_opened"] is False
    assert report["bc_authorized"] is False
    assert report["ppo_authorized"] is False


def test_rejects_applied_residual_or_opened_holdout() -> None:
    values = list(_inputs())
    values[0]["results"][0]["raw_residual_label_applied_to_commands"] = True
    values[4]["holdout_files_opened"] = True
    with pytest.raises(ValueError, match="contract audit failed"):
        build_report(*values)


def test_development_distribution_skips_holdout_and_verifies_hashes(
    tmp_path: Path,
) -> None:
    source_rows = []
    for case, split in ((2, "train"), (4, "validation"), (3, "holdout")):
        path = tmp_path / f"case_{case:04d}.npz"
        raw = np.asarray([[0.1, -0.2, 0.01], [0.2, -0.1, 0.02]])
        np.savez_compressed(
            path,
            raw_residual_commands=raw,
            elapsed_time_s=np.asarray([0.0, 0.1]),
        )
        source_rows.append(
            {
                "case": case,
                "split": split,
                "row_count": 2,
                "raw_case": f"G:\\evidence\\{path.name}",
                "raw_case_sha256": _sha256(path),
            }
        )
    distribution = compute_development_distribution(
        {"source_rows": source_rows}, tmp_path
    )
    assert distribution["cases"] == [2, 4]
    assert distribution["skipped_holdout_cases"] == [3]
    assert distribution["holdout_files_opened"] is False
    assert distribution["row_count"] == 4
    overflow = overflow_statistics(
        distribution["_raw_chunks"],
        distribution["_elapsed_chunks"],
        np.asarray([0.15, 0.25, 0.03]),
    )
    assert overflow["overflow_sample_count"] == [2, 0, 0]
    assert overflow["overflow_duration_s"] == [0.0, 0.0, 0.0]


def test_development_distribution_rejects_tampered_raw_file(tmp_path: Path) -> None:
    path = tmp_path / "case_0002.npz"
    np.savez_compressed(
        path,
        raw_residual_commands=np.zeros((2, 3)),
        elapsed_time_s=np.asarray([0.0, 0.1]),
    )
    summary = {
        "source_rows": [
            {
                "case": 2,
                "split": "train",
                "row_count": 2,
                "raw_case": path.name,
                "raw_case_sha256": "0" * 64,
            }
        ]
    }
    with pytest.raises(ValueError, match="identity mismatch"):
        compute_development_distribution(summary, tmp_path)
