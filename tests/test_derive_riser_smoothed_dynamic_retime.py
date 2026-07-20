import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/two_wheel_balance/derive_riser_smoothed_dynamic_retime.py"
)
SPEC = importlib.util.spec_from_file_location("derive_dynamic_retime", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_evidence(tmp_path: Path) -> tuple[Path, Path]:
    row = {
        "case": 7,
        "checks": {
            "completed_reference": True,
            "no_termination": True,
            "position_p95_bounded": False,
            "position_max_bounded": True,
            "thermal_bounded": True,
        },
        "completed_phase_time_s": 13.5,
        "execution_duration_s": 13.5,
        "dynamic_quality_passed": False,
        "thermal_admission_passed": True,
        "controller_evidence_passed": True,
        "termination": None,
        "executed_residual_dataset": None,
        "raw_residual_label_applied_to_commands": False,
        "residual_action_abs_max": [0.0, 0.0, 0.0],
    }
    gate = {"cases": [7], "results": [row]}
    summary = {
        "first_dynamic_reject": {
            "case": 7,
            "classification": "dynamic_gate_rejection",
            "stage": "dynamic_gate",
            "physical_dynamic_quality_passed": False,
            "thermal_admission_passed": True,
            "runtime_contract_passed": True,
        },
        "requested_cases": [7],
        "dynamically_passed_cases": [],
        "not_started_cases": [],
        "thermal_admission_passed": True,
        "runtime_contract_passed": True,
        "controller_evidence_passed": True,
        "residual_capture_started": False,
        "bc_started": False,
        "ppo_started": False,
        "valid_for_training": False,
    }
    gate_path = tmp_path / "gate.json"
    summary_path = tmp_path / "summary.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return gate_path, summary_path


def test_completed_position_p95_only_evidence_is_admitted(tmp_path: Path) -> None:
    gate, summary = _write_evidence(tmp_path)
    row = MODULE._gate_c_rejection(
        gate,
        summary,
        case=7,
        reject_mode="completed_position_p95_only",
    )
    assert row["case"] == 7


def test_completed_position_p95_mode_rejects_incomplete_or_mixed_failure(
    tmp_path: Path,
) -> None:
    gate, summary = _write_evidence(tmp_path)
    payload = json.loads(gate.read_text(encoding="utf-8"))
    payload["results"][0]["checks"]["completed_reference"] = False
    gate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="completed_position_p95_only"):
        MODULE._gate_c_rejection(
            gate,
            summary,
            case=7,
            reject_mode="completed_position_p95_only",
        )


def test_completed_position_p95_mode_requires_healthy_runtime_summary(
    tmp_path: Path,
) -> None:
    gate, summary = _write_evidence(tmp_path)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["first_dynamic_reject"]["runtime_contract_passed"] = False
    summary.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="healthy runtime evidence"):
        MODULE._gate_c_rejection(
            gate,
            summary,
            case=7,
            reject_mode="completed_position_p95_only",
        )


def test_completed_position_p95_mode_accepts_trailing_unstarted_cases(
    tmp_path: Path,
) -> None:
    gate, summary = _write_evidence(tmp_path)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload.update(
        {
            "requested_cases": [7, 13],
            "dynamically_passed_cases": [],
            "not_started_cases": [13],
            "thermal_admission_passed": False,
            "runtime_contract_passed": False,
            "controller_evidence_passed": False,
        }
    )
    summary.write_text(json.dumps(payload), encoding="utf-8")
    row = MODULE._gate_c_rejection(
        gate,
        summary,
        case=7,
        reject_mode="completed_position_p95_only",
    )
    assert row["case"] == 7


def test_retime_preserves_separate_initialization_arrays(tmp_path: Path) -> None:
    parent = tmp_path / "parent.npz"
    output = tmp_path / "output.npz"
    initialization_time = np.array([0.0, 0.5, 1.0])
    initialization_state = np.arange(21, dtype=np.float64).reshape(3, 7)
    np.savez_compressed(
        parent,
        initialization_time_s=initialization_time,
        initialization_state=initialization_state,
        execution_time_s=np.array([0.0, 1.0]),
    )
    np.savez_compressed(
        output,
        initialization_time_s=np.empty(0),
        initialization_state=np.empty((0, 7)),
        execution_time_s=np.array([0.0, 2.0]),
    )

    MODULE._restore_initialization_arrays(parent, output)

    with np.load(output, allow_pickle=False) as candidate:
        assert np.array_equal(candidate["initialization_time_s"], initialization_time)
        assert np.array_equal(candidate["initialization_state"], initialization_state)
        assert np.array_equal(candidate["execution_time_s"], np.array([0.0, 2.0]))
