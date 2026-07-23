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
    "build_model_based_corrective_case2_natural_error_profile.py"
)
READINESS = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case2_pair_readiness_cpu_v1/summary.json"
)
PLAN = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case2_pair_readiness_cpu_v1/source/"
    "case_0002_smoothed_riser_plan_v1.npz"
)
GATE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case2_pair_readiness_cpu_v1/source/"
    "case_0002_dynamic_gate.json"
)
COMMITTED_PROFILE = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case2_natural_error_profile_v1.json"
)
COMMITTED_PROPOSAL = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case2_natural_error_profile_cpu_v1/proposal.json"
)
SPEC = importlib.util.spec_from_file_location(
    "case2_natural_error_profile_builder", SCRIPT
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


def test_case2_natural_error_profile_is_formula_bound_and_closed() -> None:
    profile, proposal = MODULE.build_profile(**_inputs())
    assert proposal["passed"] is True
    assert all(proposal["input_checks"].values())
    assert all(proposal["shape_checks"].values())
    assert all(proposal["gate_checks"].values())
    assert all(proposal["formula_checks"].values())
    assert proposal["case"] == 2
    assert proposal["pair_profile_cpu_ready"] is True
    assert proposal["natural_error_pair_profile_cpu_ready"] is True
    assert proposal["runtime_route_implemented"] is False
    assert proposal["natural_error_contract"]["external_wrench_required"] is False
    assert (
        proposal["natural_error_contract"]["external_wrench_profile_created"]
        is False
    )
    assert profile["case"] == 2
    projection_identity = proposal["identities"]["safety_projection_source"]
    assert projection_identity["path"].endswith("riser_residual_policy.py")
    assert len(projection_identity["sha256"]) == 64
    for field in (
        "authorization_token_issued",
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
        assert proposal[field] is False


def test_profile_retains_one_quarter_of_observed_raw_envelope() -> None:
    profile, proposal = MODULE.build_profile(**_inputs())
    formula = proposal["profile_formula"]
    observed = np.asarray(formula["observed_normalized_raw_envelope"])
    scales = np.asarray(formula["policy_residual_scales"])
    expected = observed * scales * 0.25
    np.testing.assert_allclose(
        profile["maximum_residuals"], expected, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        profile["maximum_slew_rates"],
        expected / 0.40,
        rtol=0.0,
        atol=1e-12,
    )


def test_projection_neutralizes_only_outward_saturated_commands() -> None:
    _, proposal = MODULE.build_profile(**_inputs())
    projection = proposal["projection_envelope"]
    negative = projection["directions"]["negative"]
    positive = projection["directions"]["positive"]
    assert negative["command_clipped_transition_count"] == [430, 0, 0]
    assert positive["command_clipped_transition_count"] == [0, 103, 0]
    assert projection["all_command_limits_passed"] is True
    assert projection["all_projections_contractive"] is True
    assert proposal["formula_checks"]["outward_linear_projection_required"] is True
    assert proposal["formula_checks"]["outward_yaw_projection_required"] is True
    assert proposal["formula_checks"]["riser_projection_not_required"] is True


def test_natural_error_is_sufficient_without_an_external_wrench() -> None:
    _, proposal = MODULE.build_profile(**_inputs())
    contract = proposal["natural_error_contract"]
    assert contract["trace_sample_count"] == 47
    assert contract["samples_above_threshold"] == 42
    assert contract["position_error_trace_max_m"] > 0.15
    assert contract["capture_labels_must_use_effective_projected_residual"] is True


def test_generated_profile_loads_only_with_case2_identity(tmp_path: Path) -> None:
    profile, _ = MODULE.build_profile(**_inputs())
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not the reviewed case"):
        load_corrective_teacher_profile(path)
    case, config, _ = load_corrective_teacher_profile(path, expected_case=2)
    assert case == 2
    assert max(config.maximum_residuals) < 0.05


def test_committed_outputs_match_builder() -> None:
    profile, proposal = MODULE.build_profile(**_inputs())
    assert json.loads(COMMITTED_PROFILE.read_text()) == profile
    assert json.loads(COMMITTED_PROPOSAL.read_text()) == proposal


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runtime_authorized", True),
        ("case23_profile_reuse_authorized", True),
        ("case6_profile_reuse_authorized", True),
        ("safe_window_absent_requires_structural_profile", False),
    ],
)
def test_builder_rejects_readiness_drift(
    field: str, value: object
) -> None:
    inputs = _inputs()
    inputs["readiness"][field] = value
    with pytest.raises(ValueError, match="profile input checks failed"):
        MODULE.build_profile(**inputs)


def test_builder_rejects_missing_natural_error_excitation() -> None:
    inputs = _inputs()
    for sample in inputs["gate"]["results"][0]["trace"]:
        sample["position_error_m"] = 0.0
    with pytest.raises(ValueError, match="profile gate checks failed"):
        MODULE.build_profile(**inputs)


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
