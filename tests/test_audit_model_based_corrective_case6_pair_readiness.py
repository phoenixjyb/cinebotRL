import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "audit_model_based_corrective_case6_pair_readiness.py"
)
SPEC = importlib.util.spec_from_file_location("case6_pair_readiness", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _audit(**overrides):
    values = {
        "selection_path": MODULE.DEFAULT_SELECTION,
        "plan_path": MODULE.DEFAULT_PLAN,
        "gate_path": MODULE.DEFAULT_GATE,
    }
    values.update(overrides)
    return MODULE.audit_readiness(**values)


def test_current_case6_is_evidence_ready_but_not_pair_profile_ready() -> None:
    result = _audit()
    assert result["passed"] is True
    assert result["case"] == 6
    assert all(result["selection_checks"].values())
    assert all(result["plan_checks"].values())
    assert all(result["gate_checks"].values())
    assert all(result["metric_checks"].values())
    assert result["plan"]["sample_count"] == 807
    assert result["plan"]["transition_count"] == 806
    assert result["plan"]["minimum_target_camera_height_m"] >= 0.6
    assert result["plan"]["maximum_target_camera_height_m"] <= 1.8
    assert result["case_specific_profile_required"] is True
    assert result["case23_profile_reuse_authorized"] is False
    assert result["pair_profile_cpu_ready"] is False
    assert result["next_bounded_action"] == (
        "design_case6_specific_corrective_and_perturbation_profiles_cpu_only"
    )
    for field in (
        "runtime_authorized",
        "gpu_launch_authorized",
        "label_capture_authorized",
        "dataset_conversion_authorized",
        "dataset_merge_authorized",
        "bc_authorized",
        "ppo_authorized",
        "training_started",
        "valid_for_training",
    ):
        assert result[field] is False


def test_case6_risk_and_low_motion_window_are_explicit() -> None:
    result = _audit()
    plan = result["plan"]
    gate = result["zero_residual_dynamic_gate"]
    assert plan["headroom"]["base_linear_velocity_mps"] <= 1e-9
    assert plan["headroom"]["base_yaw_rate_radps"] <= 1e-9
    assert plan["headroom"]["proxy_rate_radps"] <= 1e-9
    assert gate["position_error_p95_m"] < 0.15
    assert gate["dynamic_margins"]["position_error_p95_m"] > 0.0
    assert gate["camera_lever_arm_correction_saturation_ratio"] > 0.9
    assert min(gate["normalized_residual_label_headroom"]) > 0.0
    contract = result["profile_window_contract"]
    assert contract["bounded_window_found"] is True
    assert len(contract["windows"]) == 1
    assert contract["windows"][0]["duration_s"] >= 0.1 - 1e-9


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selected_cases", [30, 23, 6, 7]),
        ("runtime_authorized", True),
        ("valid_for_training", True),
    ],
)
def test_readiness_rejects_selection_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = json.loads(MODULE.DEFAULT_SELECTION.read_text())
    payload[field] = value
    changed = tmp_path / "selection.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="selection checks failed"):
        _audit(selection_path=changed)


def test_readiness_rejects_dynamic_gate_regression(tmp_path: Path) -> None:
    payload = json.loads(MODULE.DEFAULT_GATE.read_text())
    payload["dynamic_quality_passed"] = False
    changed = tmp_path / "gate.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="selection checks failed"):
        _audit(gate_path=changed)


def test_cli_writes_portable_lf_evidence(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    raw = output.read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r\n" not in raw
    payload = json.loads(raw)
    assert payload["passed"] is True
    assert payload["pair_profile_cpu_ready"] is False
