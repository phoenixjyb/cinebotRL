import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "audit_model_based_corrective_case16_validation_pair_readiness.py"
)
EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case16_validation_pair_readiness_cpu_v1/summary.json"
)
SPEC = importlib.util.spec_from_file_location("case16_readiness", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _audit():
    return MODULE.audit_readiness(
        MODULE.DEFAULT_SELECTION,
        MODULE.DEFAULT_PLAN,
        MODULE.DEFAULT_GATE,
    )


def test_case16_readiness_preserves_validation_identity_and_clocks() -> None:
    result = _audit()
    assert result["passed"] is True
    assert result["case"] == 16
    assert result["split"] == "validation"
    assert result["selection_checks"]["case16_role"] is True
    assert result["selection_checks"]["case16_checks"] is True
    assert all(result["plan_checks"].values())
    assert all(result["source_execution_clock_checks"].values())
    assert result["plan"]["sample_count"] == 896
    assert result["plan"]["transition_count"] == 895
    assert result["plan"]["source_duration_s"] == 17.548706
    assert result["plan"]["execution_duration_s"] == pytest.approx(
        26.028629743189363
    )


def test_case16_requires_structural_profile_without_pulse_window() -> None:
    result = _audit()
    assert result["profile_window_contract"]["windows"] == []
    assert result["profile_window_contract"]["bounded_window_found"] is False
    assert result["case_specific_profile_required"] is True
    assert result["safe_window_absent_requires_structural_profile"] is True
    assert result["next_bounded_action"] == (
        "design_case16_validation_structural_natural_error_profile_cpu_only"
    )
    headroom = result["plan"]["headroom"]
    assert headroom["base_linear_velocity_mps"] == 0.0
    assert headroom["base_yaw_rate_radps"] == 0.0
    assert headroom["proxy_rate_radps"] == 0.0


def test_case16_zero_residual_gate_passes_but_stays_closed() -> None:
    result = _audit()
    gate = result["zero_residual_dynamic_gate"]
    assert gate["position_error_p95_m"] == pytest.approx(0.08059952749099941)
    assert gate["position_error_max_m"] == pytest.approx(0.081492189372026)
    assert gate["camera_lever_arm_correction_saturation_ratio"] > 0.95
    assert all(result["gate_checks"].values())
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


def test_committed_evidence_matches_auditor() -> None:
    assert json.loads(EVIDENCE.read_text(encoding="utf-8")) == _audit()


def test_auditor_rejects_wrong_plan_identity(tmp_path: Path) -> None:
    engine = MODULE._load_engine()
    metadata, arrays = engine._load_plan(MODULE.DEFAULT_PLAN)
    metadata["case"] = 8
    output = tmp_path / "wrong.npz"
    np.savez_compressed(
        output,
        metadata_json=np.array(json.dumps(metadata)),
        **arrays,
    )
    with pytest.raises(ValueError, match="selection checks failed"):
        MODULE.audit_readiness(
            MODULE.DEFAULT_SELECTION,
            output,
            MODULE.DEFAULT_GATE,
        )


def test_cli_regenerates_lf_evidence(tmp_path: Path) -> None:
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
