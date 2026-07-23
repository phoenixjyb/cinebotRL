import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/audit_model_based_corrective_temporal_projection.py"
)
DATASET = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_case30_effective_label_conversion_v1/"
    "case_0030_model_based_corrective_case_dataset_v1.npz"
)
PROFILE = (
    ROOT
    / "scripts/two_wheel_balance/model_based_corrective_teacher_case30_profile_v1.json"
)
TRACKED_SUMMARY = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_model_based_corrective_temporal_projection_v1/summary.json"
)

SPEC = importlib.util.spec_from_file_location("temporal_projection_audit", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_case30_temporal_projection_audit_passes() -> None:
    result = MODULE.audit(DATASET, PROFILE)
    assert result == json.loads(TRACKED_SUMMARY.read_text(encoding="utf-8"))
    assert result["passed"]
    assert result["requested_teacher_intent"]["slew_limit_passed"]
    assert not result["effective_post_supervisor_labels"]["slew_limit_passed"]
    assert result["effective_post_supervisor_labels"][
        "violation_count_per_channel"
    ] == [30, 49, 8]
    assert result["effective_post_supervisor_labels"][
        "all_violations_associated_with_command_clipping"
    ]
    assert result["safety_projection"][
        "final_command_reconstruction_max_error"
    ] <= MODULE.PROJECTION_COMMAND_TOLERANCE
    assert result["safety_projection"][
        "effective_action_reconstruction_max_error"
    ] <= MODULE.PROJECTION_ACTION_TOLERANCE
    assert not result["valid_for_training"]
    assert not result["runtime_authorized"]
    assert not result["bc_authorized"]
    assert not result["ppo_authorized"]


def test_temporal_rate_summary_rejects_unclipped_chatter() -> None:
    actions = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float64
    )
    result = MODULE._rate_summary(
        actions,
        np.array([0.0, 0.005]),
        np.array([0.05, 0.05, 0.02]),
        np.array([0.10, 0.10, 0.04]),
        np.zeros((1, 3), dtype=bool),
    )
    assert not result["slew_limit_passed"]
    assert result["violation_count_per_channel"] == [1, 0, 0]
    assert result["unclipped_violation_count_per_channel"] == [1, 0, 0]
    assert not result["all_violations_associated_with_command_clipping"]
