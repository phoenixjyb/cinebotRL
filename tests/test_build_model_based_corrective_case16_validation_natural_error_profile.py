import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from rl_platform.tasks.two_wheel_balance.riser_corrective_teacher import (
    load_corrective_teacher_profile,
)


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "build_model_based_corrective_case16_validation_natural_error_profile.py"
)
READINESS_ROOT = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case16_validation_pair_readiness_cpu_v1"
)
READINESS = READINESS_ROOT / "summary.json"
PLAN = READINESS_ROOT / "source/case_0016_smoothed_riser_plan_v1.npz"
GATE = READINESS_ROOT / "source/case_0016_dynamic_gate.json"
COMMITTED_PROFILE = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case16_validation_natural_error_"
    "profile_v1.json"
)
COMMITTED_PROPOSAL = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case16_validation_natural_error_profile_cpu_v1/"
    "proposal.json"
)
SPEC = importlib.util.spec_from_file_location(
    "case16_validation_natural_error_profile_builder", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _inputs():
    plan_metadata, plan_arrays = MODULE._load_plan(PLAN)
    return {
        "readiness": MODULE._load_object(READINESS),
        "readiness_path": READINESS,
        "plan_metadata": plan_metadata,
        "plan_arrays": plan_arrays,
        "plan_path": PLAN,
        "gate": MODULE._load_object(GATE),
        "gate_path": GATE,
    }


def test_case16_profile_is_structural_validation_only_and_closed() -> None:
    profile, proposal = MODULE.build_profile(**_inputs())
    assert proposal["passed"] is True
    assert proposal["case"] == 16
    assert proposal["split"] == "validation"
    assert profile["case"] == 16
    assert all(proposal["input_checks"].values())
    assert all(proposal["shape_checks"].values())
    assert all(proposal["gate_checks"].values())
    assert all(proposal["formula_checks"].values())
    assert all(proposal["validation_profile_checks"].values())
    assert proposal["natural_error_contract"]["validation_only"] is True
    assert (
        proposal["natural_error_contract"]["external_perturbation_forbidden"]
        is True
    )
    assert proposal["validation_pair_profile_cpu_ready"] is True
    assert proposal["runtime_route_implemented"] is False
    assert proposal["holdout_opened"] is False
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


def test_retention_is_derived_from_case16_dynamic_margin() -> None:
    profile, proposal = MODULE.build_profile(**_inputs())
    formula = proposal["profile_formula"]
    expected_retention = min(
        0.40,
        formula["position_p95_margin_m"] / formula["position_p95_gate_m"],
    )
    assert formula["retention_fraction"] == pytest.approx(
        expected_retention, abs=1e-12
    )
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
    assert max(np.asarray(profile["maximum_residuals"]) / scales) < 0.15


def test_projection_handles_saturated_case16_commands_without_expansion() -> None:
    _, proposal = MODULE.build_profile(**_inputs())
    projection = proposal["projection_envelope"]
    assert projection["directions"]["negative"][
        "command_clipped_transition_count"
    ] == [0, 20, 0]
    assert projection["directions"]["positive"][
        "command_clipped_transition_count"
    ] == [607, 174, 0]
    assert projection["all_command_limits_passed"] is True
    assert projection["all_projections_contractive"] is True
    assert proposal["formula_checks"][
        "negative_linear_projection_not_required"
    ]
    assert proposal["formula_checks"][
        "outward_positive_linear_projection_required"
    ]
    assert proposal["formula_checks"][
        "both_yaw_directions_require_projection"
    ]
    assert proposal["formula_checks"]["riser_projection_not_required"]


def test_natural_error_is_persistent_and_replaces_external_wrench() -> None:
    _, proposal = MODULE.build_profile(**_inputs())
    contract = proposal["natural_error_contract"]
    assert contract["trace_sample_count"] == 54
    assert contract["samples_above_threshold"] == 52
    assert contract["position_error_trace_max_m"] > 0.08
    assert contract["external_wrench_required"] is False
    assert contract["external_wrench_profile_created"] is False
    assert contract["capture_labels_must_use_effective_projected_residual"]
    assert contract["requested_residual_is_not_a_training_label"]


def test_generated_profile_requires_case16_identity(tmp_path: Path) -> None:
    profile, _ = MODULE.build_profile(**_inputs())
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not the reviewed case"):
        load_corrective_teacher_profile(path)
    case, config, _ = load_corrective_teacher_profile(path, expected_case=16)
    assert case == 16
    assert max(config.maximum_residuals) < 0.01


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split", "train"),
        ("case2_profile_reuse_authorized", True),
        ("case7_profile_reuse_authorized", True),
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


def test_builder_rejects_nonpositive_p95_margin() -> None:
    inputs = _inputs()
    inputs["readiness"]["zero_residual_dynamic_gate"]["dynamic_margins"][
        "position_error_p95_m"
    ] = 0.0
    with pytest.raises(ValueError, match="cannot support"):
        MODULE.build_profile(**inputs)


def test_builder_rejects_missing_natural_error_excitation() -> None:
    inputs = _inputs()
    for sample in inputs["gate"]["results"][0]["trace"]:
        sample["position_error_m"] = 0.0
    with pytest.raises(ValueError, match="profile gate checks failed"):
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
    for path in (profile, proposal):
        raw = path.read_bytes()
        assert raw.endswith(b"\n")
        assert b"\r\n" not in raw
    retry = subprocess.run(command, check=False, capture_output=True, text=True)
    assert retry.returncode != 0
    assert "refusing to overwrite" in retry.stderr
