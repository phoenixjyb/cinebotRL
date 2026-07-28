import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "audit_model_based_corrective_case32_validation_pair_readiness.py"
)
EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260728_case32_validation_pair_readiness_cpu_v1/summary.json"
)
SPEC = importlib.util.spec_from_file_location("case32_readiness", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _audit():
    return MODULE.audit_readiness()


def test_case32_readiness_preserves_exact_source_and_separate_clocks() -> None:
    result = _audit()
    assert result["passed"] is True
    assert result["case"] == 32
    assert result["split"] == "validation"
    assert all(result["plan_checks"].values())
    assert all(result["source_execution_clock_checks"].values())
    assert result["plan"]["sample_count"] == 1099
    assert result["plan"]["transition_count"] == 1098
    assert result["plan"]["source_duration_s"] == 21.648708
    assert result["plan"]["execution_duration_s"] == pytest.approx(
        29.592866387237176
    )


def test_case32_uses_structural_natural_error_not_a_pulse() -> None:
    result = _audit()
    assert result["profile_window_contract"]["windows"] == []
    assert result["profile_window_contract"]["bounded_window_found"] is False
    assert result["case_specific_profile_required"] is True
    assert result["safe_window_absent_requires_structural_profile"] is True
    assert result["plan"]["headroom"]["base_linear_velocity_mps"] == 0.0
    assert result["plan"]["headroom"]["base_yaw_rate_radps"] == 0.0
    assert result["plan"]["headroom"]["proxy_rate_radps"] == 0.0


def test_case32_historical_gate_is_healthy_but_not_current_admission() -> None:
    result = _audit()
    gate = result["zero_residual_dynamic_gate"]
    assert gate["position_error_p95_m"] == pytest.approx(
        0.10241882262348277
    )
    assert gate["position_error_max_m"] == pytest.approx(0.133996309804517)
    assert gate["normalized_residual_label_abs_max"][0] < 0.95
    assert all(result["gate_checks"].values())
    assert result["pair_profile_cpu_ready"] is False
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


def test_readiness_rejects_selection_role_drift(tmp_path: Path) -> None:
    selection = json.loads(MODULE.DEFAULT_SELECTION.read_text())
    selection["selected_rows"][1]["selection_role"] = "teacher"
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pair selection checks failed"):
        MODULE.audit_readiness(selection_path=path)


def test_committed_readiness_matches_auditor() -> None:
    assert json.loads(EVIDENCE.read_text(encoding="utf-8")) == _audit()


def test_cli_regenerates_lf_readiness(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == EVIDENCE.read_bytes()
    assert b"\r\n" not in output.read_bytes()
