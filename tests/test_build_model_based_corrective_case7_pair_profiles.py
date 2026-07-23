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
from rl_platform.tasks.two_wheel_balance.riser_perturbation import (
    load_deterministic_wrench_profile,
)


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "build_model_based_corrective_case7_pair_profiles.py"
)
READINESS = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case7_pair_readiness_cpu_v1/summary.json"
)
PLAN = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case7_pair_readiness_cpu_v1/source/"
    "case_0007_smoothed_riser_plan_v1.npz"
)
PLANT = (
    ROOT / "docs/03_training/two_wheel_balance/PLANT_PRIOR_PROVISIONAL_V1.json"
)
COMMITTED_CORRECTIVE = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case7_profile_v1.json"
)
COMMITTED_WRENCH = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case7_wrench_profile_v1.json"
)
COMMITTED_PROPOSAL = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case7_pair_profile_cpu_v1/proposal.json"
)
SPEC = importlib.util.spec_from_file_location("case7_profile_builder", SCRIPT)
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
        "plant": MODULE._load_object(PLANT),
        "plant_path": PLANT,
    }


def test_current_case7_profiles_are_formula_bound_and_closed() -> None:
    corrective, wrench, proposal = MODULE.build_profiles(**_inputs())
    assert proposal["passed"] is True
    assert all(proposal["input_checks"].values())
    assert all(proposal["shape_checks"].values())
    assert all(proposal["formula_checks"].values())
    assert proposal["case"] == corrective["case"] == wrench["case"] == 7
    assert proposal["case23_profile_reuse_authorized"] is False
    assert proposal["case6_profile_reuse_authorized"] is False
    assert proposal["case2_profile_reuse_authorized"] is False
    assert proposal["pair_profile_cpu_ready"] is True
    assert proposal["runtime_route_implemented"] is False
    assert proposal["next_bounded_action"] == (
        "implement_case7_pair_runtime_contract_cpu_only_without_authorization"
    )
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


def test_profile_retains_half_observed_envelope_with_slew_margin() -> None:
    corrective, _, proposal = MODULE.build_profiles(**_inputs())
    formula = proposal["profile_formula"]
    observed = np.asarray(formula["observed_normalized_raw_envelope"])
    scales = np.asarray(formula["policy_residual_scales"])
    expected = observed * scales * 0.5
    np.testing.assert_allclose(
        corrective["maximum_residuals"], expected, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        corrective["maximum_slew_rates"],
        expected / 0.35,
        rtol=0.0,
        atol=1e-12,
    )
    assert max(expected / scales) < 0.4


def test_pulse_uses_lowest_utilization_eligible_window() -> None:
    _, wrench, proposal = MODULE.build_profiles(**_inputs())
    window = proposal["pulse_window"]
    assert window["window_index"] == 1
    assert window["window_start_index"] == 27
    assert window["window_end_index"] == 125
    assert window["pulse_start_index"] == 92
    assert window["pulse_start_phase_time_s"] == pytest.approx(
        2.851306269086607
    )
    assert window["pulse_duration_steps"] == 20
    assert window["pulse_duration_s_at_policy_rate"] == 0.1
    assert window["pulse_source_sample_count"] == 4
    assert window["recovery_tail_s"] > 15.0
    assert window["local_headroom"][0] > 0.10
    assert window["local_headroom"][1] > 0.39
    assert wrench["force_body_x_n"] == 20.0
    assert proposal["pulse_lower_model"]["impulse_ns"] == 2.0


def test_projection_contract_preserves_effective_label_margin() -> None:
    _, _, proposal = MODULE.build_profiles(**_inputs())
    projection = proposal["projection_envelope"]
    full = projection["full_plan"]
    pulse = projection["pulse_window"]
    assert full["directions"]["negative"][
        "command_clipped_transition_count"
    ] == [0, 0, 4]
    assert full["directions"]["positive"][
        "command_clipped_transition_count"
    ] == [0, 0, 0]
    assert full["all_projections_contractive"] is True
    assert pulse["all_requested_commands_unclipped"] is True
    assert projection["effective_normalized_action_abs_max"] < 0.4
    assert projection["label_contract"] == (
        "effective_post_supervisor_residual_only"
    )


def test_generated_profiles_load_only_with_case7_identity(tmp_path: Path) -> None:
    corrective, wrench, _ = MODULE.build_profiles(**_inputs())
    corrective_path = tmp_path / "corrective.json"
    wrench_path = tmp_path / "wrench.json"
    corrective_path.write_text(json.dumps(corrective) + "\n", encoding="utf-8")
    wrench_path.write_text(json.dumps(wrench) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not the reviewed case"):
        load_corrective_teacher_profile(corrective_path)
    with pytest.raises(ValueError, match="invalid deterministic wrench profile"):
        load_deterministic_wrench_profile(wrench_path)
    case, config, _ = load_corrective_teacher_profile(
        corrective_path, expected_case=7
    )
    pulse, _ = load_deterministic_wrench_profile(
        wrench_path, expected_case=7
    )
    assert case == pulse.case == 7
    assert max(config.maximum_residuals) < 0.02


def test_committed_profiles_match_formula_bound_proposal() -> None:
    corrective, wrench, proposal = MODULE.build_profiles(**_inputs())
    assert json.loads(COMMITTED_CORRECTIVE.read_text()) == corrective
    assert json.loads(COMMITTED_WRENCH.read_text()) == wrench
    assert json.loads(COMMITTED_PROPOSAL.read_text()) == proposal
    _, _, corrective_identity = load_corrective_teacher_profile(
        COMMITTED_CORRECTIVE, expected_case=7
    )
    _, wrench_identity = load_deterministic_wrench_profile(
        COMMITTED_WRENCH, expected_case=7
    )
    assert proposal["identities"]["corrective_profile"]["sha256"] == (
        corrective_identity["sha256"]
    )
    assert proposal["identities"]["wrench_profile"]["sha256"] == (
        wrench_identity["sha256"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runtime_authorized", True),
        ("case23_profile_reuse_authorized", True),
        ("case6_profile_reuse_authorized", True),
        ("case2_profile_reuse_authorized", True),
        ("case_specific_profile_required", False),
    ],
)
def test_builder_rejects_readiness_contract_drift(
    field: str, value: object
) -> None:
    inputs = _inputs()
    inputs["readiness"][field] = value
    with pytest.raises(ValueError, match="profile input checks failed"):
        MODULE.build_profiles(**inputs)


def test_builder_rejects_missing_low_motion_window() -> None:
    inputs = _inputs()
    arrays = dict(inputs["plan_arrays"])
    feedforward = arrays["feedforward_v_wz"].copy()
    feedforward[:, 0] = 0.4
    arrays["feedforward_v_wz"] = feedforward
    inputs["plan_arrays"] = arrays
    with pytest.raises(ValueError, match="no eligible low-motion pulse"):
        MODULE.build_profiles(**inputs)


def test_builder_rejects_source_anchor_drift() -> None:
    inputs = _inputs()
    arrays = dict(inputs["plan_arrays"])
    anchors = arrays["source_anchor_execution_index"].copy()
    anchors[-1] -= 1
    arrays["source_anchor_execution_index"] = anchors
    inputs["plan_arrays"] = arrays
    with pytest.raises(ValueError, match="plan array checks failed"):
        MODULE.build_profiles(**inputs)


def test_cli_writes_lf_profiles_and_refuses_overwrite(tmp_path: Path) -> None:
    corrective = tmp_path / "corrective.json"
    wrench = tmp_path / "wrench.json"
    proposal = tmp_path / "proposal.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--readiness",
        str(READINESS),
        "--plan",
        str(PLAN),
        "--plant-prior",
        str(PLANT),
        "--corrective-profile-output",
        str(corrective),
        "--wrench-profile-output",
        str(wrench),
        "--proposal-output",
        str(proposal),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for path in (corrective, wrench, proposal):
        raw = path.read_bytes()
        assert raw.endswith(b"\n")
        assert b"\r\n" not in raw
    retry = subprocess.run(command, check=False, capture_output=True, text=True)
    assert retry.returncode != 0
    assert "refusing to overwrite" in retry.stderr
