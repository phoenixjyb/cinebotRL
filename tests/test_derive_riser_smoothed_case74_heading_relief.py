import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/two_wheel_balance/derive_riser_smoothed_case74_heading_relief.py"
)
SPEC = importlib.util.spec_from_file_location("derive_case74_heading_relief", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _gate_payload() -> tuple[dict[str, object], dict[str, object]]:
    checks = {
        "completed_reference": True,
        "position_p95_bounded": False,
        "position_max_bounded": True,
        "thermal_bounded": True,
    }
    row = {
        "case": 74,
        "checks": checks,
        "completed_phase_time_s": 22.0,
        "execution_duration_s": 22.0,
        "thermal_admission_passed": True,
        "termination": None,
        "executed_residual_dataset": None,
        "raw_residual_label_applied_to_commands": False,
        "residual_action_abs_max": [0.0, 0.0, 0.0],
    }
    gate = {"cases": [74], "results": [row]}
    summary = {
        "dynamic_quality_passed": False,
        "residual_capture_started": False,
        "bc_started": False,
        "ppo_started": False,
        "valid_for_training": False,
    }
    return gate, summary


def test_gate_reject_accepts_only_completed_position_p95_failure() -> None:
    gate, summary = _gate_payload()
    assert MODULE._validate_gate_reject(gate, summary)["case"] == 74

    gate["results"][0]["checks"]["position_max_bounded"] = False
    with pytest.raises(ValueError, match="p95-only"):
        MODULE._validate_gate_reject(gate, summary)


def _write_arrays(
    path: Path,
    *,
    localized_delta: float = 0.0,
    outside_delta: float = 0.0,
) -> None:
    count = 590
    target = np.zeros((count, 3), dtype=np.float64)
    target[:, 2] = 1.0
    target[400, 0] = localized_delta
    target[100, 0] = outside_delta
    np.savez_compressed(
        path,
        target_semantic_dfr_quat_wxyz=np.tile(
            [1.0, 0.0, 0.0, 0.0], (count, 1)
        ),
        source_time_s=np.linspace(0.0, 10.0, count),
        source_target_position_world_m=np.zeros((count, 3)),
        source_target_semantic_dfr_quat_xyzw=np.tile(
            [0.0, 0.0, 0.0, 1.0], (count, 1)
        ),
        source_anchor_execution_index=np.arange(count),
        initialization_time_s=np.empty(0),
        initialization_state=np.empty((0, 7)),
        smoothed_target_position_source_frame_m=target,
        target_position_world_m=target,
        execution_time_s=np.linspace(0.0, 20.0, count),
        time_s=np.linspace(0.0, 20.0, count),
    )


def test_array_checks_reject_outside_window_change(tmp_path: Path) -> None:
    parent = tmp_path / "parent.npz"
    output = tmp_path / "output.npz"
    _write_arrays(parent)
    _write_arrays(output, localized_delta=0.01)
    checks = MODULE._array_checks(parent, output)
    assert all(checks.values())

    _write_arrays(output, localized_delta=0.01, outside_delta=0.001)
    checks = MODULE._array_checks(parent, output)
    assert not checks["smoothed_geometry_before_window_unchanged"]
    assert not checks["target_geometry_outside_window_unchanged"]
