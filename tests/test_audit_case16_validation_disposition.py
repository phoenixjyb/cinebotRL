import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/audit_case16_validation_disposition.py"
)
EVIDENCE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260728_case16_validation_disposition_cpu_v1/summary.json"
)
SPEC = importlib.util.spec_from_file_location("case16_disposition", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _audit():
    return MODULE.audit_disposition()


def test_case16_is_ceiling_limited_not_intrinsically_hard() -> None:
    result = _audit()
    case16 = result["case16"]
    assert result["passed"] is True
    assert case16["baseline_position_p95_m"] == pytest.approx(
        0.08059952749099941
    )
    assert case16["candidate_position_p95_m"] == pytest.approx(
        0.07803205319222484
    )
    assert case16["dynamic_position_p95_margin_m"] == pytest.approx(
        0.06940047250900058
    )
    assert case16["absolute_improvement_shortfall_m"] == pytest.approx(
        0.00043252570122542525
    )
    assert case16["relative_gate_passed"] is True
    assert case16["estimated_saturated_action_samples"] == 2
    assert case16["ceiling_limited"] is True
    assert case16["intrinsically_hard_in_realized_dynamics"] is False
    assert case16["further_case_specific_tuning_recommended"] is False
    assert case16["teacher_capture_recommended"] is False


def test_case32_is_selected_over_saturating_case22() -> None:
    result = _audit()
    case22 = result["replacement_candidates"]["22"]
    case32 = result["replacement_candidates"]["32"]
    assert case22["historical_dynamic_gate"]["dynamic_quality_passed"] is True
    assert case22["historical_dynamic_gate"]["action_saturation_ratio"] > 0.0
    assert case32["historical_dynamic_gate"]["dynamic_quality_passed"] is True
    assert case32["historical_dynamic_gate"]["action_saturation_ratio"] == 0.0
    assert case32["historical_dynamic_gate"]["position_error_p95_m"] == (
        pytest.approx(0.10241882262348277)
    )
    assert case32["plan"]["source_path_length_m"] == pytest.approx(
        9.575589358765875
    )
    assert all(case32["selection_checks"].values())
    assert result["selected_replacement_case"] == 32
    assert result["selected_candidate_is_currently_admitted"] is False
    assert result["fresh_readiness_and_provenance_review_required"] is True


def test_disposition_keeps_runtime_and_learning_closed() -> None:
    result = _audit()
    assert result["next_bounded_action"] == (
        "cpu_only_prepare_case32_validation_pair_readiness_and_profile"
    )
    for field in (
        "runtime_authorized",
        "gpu_launch_authorized",
        "label_capture_authorized",
        "cpu_conversion_authorized",
        "dataset_merge_authorized",
        "bc_authorized",
        "ppo_authorized",
        "training_started",
        "valid_for_training",
    ):
        assert result[field] is False


def test_audit_rejects_alternate_candidate_evidence(tmp_path: Path) -> None:
    alternate = tmp_path / "case_0032.json"
    alternate.write_bytes(MODULE.DEFAULT_CASE32_GATE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="case32_gate sha256 mismatch"):
        MODULE.audit_disposition(case32_gate_path=alternate)


def test_committed_evidence_matches_auditor() -> None:
    assert json.loads(EVIDENCE.read_text(encoding="utf-8")) == _audit()


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
