import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rl_platform.tasks.two_wheel_balance.riser_corrective_teacher import (
    load_corrective_teacher_profile,
)

ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "build_model_based_corrective_case32_validation_natural_error_profile.py"
)
READINESS = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260728_case32_validation_pair_readiness_cpu_v1/summary.json"
)
PLAN = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260728_case16_validation_disposition_cpu_v1/source/"
    "case_0032_smoothed_riser_plan_v1.npz"
)
GATE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260728_case16_validation_disposition_cpu_v1/source/"
    "case_0032_historical_dynamic_gate.json"
)
COMMITTED_PROFILE = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case32_validation_natural_error_"
    "profile_v1.json"
)
COMMITTED_PROPOSAL = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260728_case32_validation_natural_error_profile_cpu_v1/"
    "proposal.json"
)
SPEC = importlib.util.spec_from_file_location("case32_profile", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _inputs():
    metadata, arrays = MODULE._load_plan(PLAN)
    return {
        "readiness": MODULE._load_object(READINESS),
        "readiness_path": READINESS,
        "plan_metadata": metadata,
        "plan_arrays": arrays,
        "plan_path": PLAN,
        "gate": MODULE._load_object(GATE),
        "gate_path": GATE,
    }


def test_case32_profile_is_validation_only_and_closed() -> None:
    profile, proposal = MODULE.build_profile(**_inputs())
    assert proposal["passed"] is True
    assert proposal["case"] == profile["case"] == 32
    assert proposal["split"] == "validation"
    assert all(proposal["input_checks"].values())
    assert all(proposal["shape_checks"].values())
    assert all(proposal["gate_checks"].values())
    assert all(proposal["formula_checks"].values())
    assert all(proposal["validation_profile_checks"].values())
    assert proposal["validation_pair_profile_cpu_ready"] is True
    assert proposal["runtime_route_implemented"] is False
    for field in (
        "authorization_token_issued",
        "runtime_authorized",
        "gpu_launch_authorized",
        "teacher_admission_authorized",
        "label_capture_authorized",
        "dataset_creation_authorized",
        "dataset_conversion_authorized",
        "dataset_merge_authorized",
        "bc_authorized",
        "ppo_authorized",
        "training_started",
        "valid_for_training",
    ):
        assert proposal[field] is False


def test_profile_retention_is_bounded_by_case32_dynamic_margin() -> None:
    profile, proposal = MODULE.build_profile(**_inputs())
    formula = proposal["profile_formula"]
    expected_retention = (
        formula["position_p95_margin_m"] / formula["position_p95_gate_m"]
    )
    assert formula["retention_fraction"] == pytest.approx(expected_retention)
    observed = np.asarray(formula["observed_normalized_raw_envelope"])
    scales = np.asarray(formula["policy_residual_scales"])
    expected = observed * scales * expected_retention
    np.testing.assert_allclose(
        profile["maximum_residuals"], expected, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        profile["maximum_slew_rates"],
        expected / 0.40,
        rtol=0.0,
        atol=1e-12,
    )
    assert max(np.asarray(profile["maximum_residuals"]) / scales) < 0.29


def test_projection_is_contractive_at_case32_command_ceilings() -> None:
    _, proposal = MODULE.build_profile(**_inputs())
    directions = proposal["projection_envelope"]["directions"]
    assert directions["negative"]["command_clipped_transition_count"] == [
        371,
        179,
        0,
    ]
    assert directions["positive"]["command_clipped_transition_count"] == [
        309,
        103,
        0,
    ]
    assert proposal["projection_envelope"]["all_command_limits_passed"] is True
    assert proposal["projection_envelope"]["all_projections_contractive"] is True


def test_natural_error_replaces_external_wrench() -> None:
    _, proposal = MODULE.build_profile(**_inputs())
    contract = proposal["natural_error_contract"]
    assert contract["trace_sample_count"] == 67
    assert contract["samples_above_threshold"] == 59
    assert contract["external_wrench_required"] is False
    assert contract["external_perturbation_forbidden"] is True
    assert contract["requested_residual_is_not_a_training_label"] is True


def test_generated_profile_requires_case32_identity(tmp_path: Path) -> None:
    profile, _ = MODULE.build_profile(**_inputs())
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not the reviewed case"):
        load_corrective_teacher_profile(path)
    case, config, _ = load_corrective_teacher_profile(
        path, expected_case=32
    )
    assert case == 32
    assert max(config.maximum_residuals) < 0.015


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split", "train"),
        ("case8_profile_reuse_authorized", True),
        ("case16_profile_reuse_authorized", True),
        ("safe_window_absent_requires_structural_profile", False),
    ],
)
def test_builder_rejects_validation_contract_drift(
    field: str, value: object
) -> None:
    inputs = _inputs()
    inputs["readiness"][field] = value
    with pytest.raises(ValueError, match="validation profile checks failed"):
        MODULE.build_profile(**inputs)


def test_committed_outputs_match_builder() -> None:
    profile, proposal = MODULE.build_profile(**_inputs())
    assert json.loads(COMMITTED_PROFILE.read_text()) == profile
    assert json.loads(COMMITTED_PROPOSAL.read_text()) == proposal


def test_cli_writes_lf_outputs_and_refuses_overwrite(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    proposal = tmp_path / "proposal.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--readiness",
        str(READINESS),
        "--plan",
        str(PLAN),
        "--dynamic-gate",
        str(GATE),
        "--corrective-profile-output",
        str(profile),
        "--proposal-output",
        str(proposal),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert profile.read_bytes() == COMMITTED_PROFILE.read_bytes()
    assert proposal.read_bytes() == COMMITTED_PROPOSAL.read_bytes()
    retry = subprocess.run(command, check=False, capture_output=True, text=True)
    assert retry.returncode != 0
    assert "refusing to overwrite" in retry.stderr
