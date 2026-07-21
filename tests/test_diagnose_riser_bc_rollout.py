from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "two_wheel_balance"
    / "diagnose_riser_bc_rollout.py"
)
SPEC = importlib.util.spec_from_file_location("diagnose_riser_bc_rollout", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _metrics(aggregate_mse: float, correlations: list[float]) -> dict:
    return {
        "aggregate_mse": aggregate_mse,
        "correlation_per_action": correlations,
    }


def test_classification_localizes_previous_action_exposure_bias() -> None:
    result = MODULE.classify_diagnosis(
        _metrics(0.03, [0.96, 0.98, 0.99]),
        _metrics(0.55, [0.7, 0.7, 0.7]),
        _metrics(1.0, [0.0, 0.0, 0.0]),
        learned_position_p95_m=0.18,
        teacher_position_p95_m=0.128,
    )
    assert result["classification"] == "autoregressive_previous_action_exposure_bias"
    assert result["teacher_state_fit_passed"]
    assert not result["recursive_previous_action_stability_passed"]
    assert not result["comparison_tracking_passed"]


def test_classification_keeps_underfit_policy_separate() -> None:
    result = MODULE.classify_diagnosis(
        _metrics(0.30, [0.80, 0.90, 0.95]),
        _metrics(0.40, [0.7, 0.7, 0.7]),
        _metrics(1.0, [0.0, 0.0, 0.0]),
        learned_position_p95_m=0.18,
        teacher_position_p95_m=0.128,
    )
    assert result["classification"] == "teacher_state_policy_fit_failure"


def test_phase_rows_preserve_all_bins_and_action_channels() -> None:
    phase = np.asarray([0.0, 0.2, 0.6, 1.0])
    target = np.arange(12, dtype=np.float32).reshape(4, 3)
    rows = MODULE.phase_rows(phase, target, target + 1, target + 2, 2)
    assert len(rows) == 2
    assert sum(row["row_count"] for row in rows) == 4
    assert "residual_vx_teacher_mean" in rows[0]
    assert "residual_wz_recursive_previous_mae" in rows[0]
    assert "residual_riser_target_teacher_previous_mae" in rows[1]
