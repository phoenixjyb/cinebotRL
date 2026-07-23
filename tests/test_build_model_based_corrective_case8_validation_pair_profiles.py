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
    "build_model_based_corrective_case8_validation_pair_profiles.py"
)
READINESS = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case8_validation_pair_readiness_cpu_v1/summary.json"
)
PLAN = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case8_validation_pair_readiness_cpu_v1/source/"
    "case_0008_smoothed_riser_plan_v1.npz"
)
PLANT = (
    ROOT / "docs/03_training/two_wheel_balance/PLANT_PRIOR_PROVISIONAL_V1.json"
)
COMMITTED_CORRECTIVE = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case8_validation_profile_v1.json"
)
COMMITTED_WRENCH = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case8_validation_wrench_profile_v1.json"
)
COMMITTED_PROPOSAL = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case8_validation_pair_profile_cpu_v1/proposal.json"
)
CASE7_CORRECTIVE = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case7_profile_v1.json"
)
CASE7_WRENCH = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case7_wrench_profile_v1.json"
)
SPEC = importlib.util.spec_from_file_location("case8_profile_builder", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _inputs():
    engine = MODULE._load_formula_engine()
    plan_metadata, plan_arrays = engine._load_plan(PLAN)
    return {
        "readiness": MODULE._load_object(READINESS),
        "readiness_path": READINESS,
        "plan_metadata": plan_metadata,
        "plan_arrays": plan_arrays,
        "plan_path": PLAN,
        "plant": MODULE._load_object(PLANT),
        "plant_path": PLANT,
    }


def test_current_case8_profiles_are_validation_only_and_closed() -> None:
    corrective, wrench, proposal = MODULE.build_profiles(**_inputs())
    assert proposal["passed"] is True
    assert all(proposal["input_checks"].values())
    assert all(proposal["shape_checks"].values())
    assert all(proposal["formula_checks"].values())
    assert all(proposal["validation_profile_checks"].values())
    assert proposal["case"] == corrective["case"] == wrench["case"] == 8
    assert proposal["split"] == "validation"
    assert proposal["case30_profile_reuse_authorized"] is False
    assert proposal["case23_profile_reuse_authorized"] is False
    assert proposal["case6_profile_reuse_authorized"] is False
    assert proposal["case2_profile_reuse_authorized"] is False
    assert proposal["case7_profile_reuse_authorized"] is False
    assert proposal["train_profile_reuse_authorized"] is False
    assert proposal["pair_profile_cpu_ready"] is True
    assert proposal["runtime_route_implemented"] is False
    assert proposal["next_bounded_action"] == (
        "implement_case8_validation_pair_runtime_contract_cpu_only_"
        "without_authorization"
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


def test_profile_uses_dedicated_conservative_validation_formula() -> None:
    corrective, _, proposal = MODULE.build_profiles(**_inputs())
    formula = proposal["profile_formula"]
    observed = np.asarray(formula["observed_normalized_raw_envelope"])
    scales = np.asarray(formula["policy_residual_scales"])
    expected = observed * scales * 0.40
    np.testing.assert_allclose(
        corrective["maximum_residuals"], expected, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        corrective["maximum_slew_rates"],
        expected / 0.40,
        rtol=0.0,
        atol=1e-12,
    )
    assert proposal["projection_envelope"][
        "effective_normalized_action_abs_max"
    ] < 0.31
    assert json.loads(CASE7_CORRECTIVE.read_text()) != corrective


def test_pulse_uses_audited_window_and_differs_from_train_case() -> None:
    _, wrench, proposal = MODULE.build_profiles(**_inputs())
    window = proposal["pulse_window"]
    assert window["window_index"] == 1
    assert window["window_start_index"] == 27
    assert window["window_end_index"] == 125
    assert window["pulse_start_index"] == 92
    assert window["pulse_start_phase_time_s"] == pytest.approx(
        2.8513062688185196
    )
    assert window["pulse_duration_steps"] == 20
    assert window["pulse_source_sample_count"] == 4
    assert window["recovery_tail_s"] > 15.0
    assert wrench["force_body_x_n"] == 18.0
    assert wrench != json.loads(CASE7_WRENCH.read_text())
    assert proposal["pulse_lower_model"]["impulse_ns"] == 1.8
    assert (
        proposal["pulse_lower_model"][
            "ideal_free_body_displacement_during_pulse_m"
        ]
        > 0.003
    )


def test_projection_contract_preserves_validation_label_margin() -> None:
    _, _, proposal = MODULE.build_profiles(**_inputs())
    full = proposal["projection_envelope"]["full_plan"]
    pulse = proposal["projection_envelope"]["pulse_window"]
    assert full["directions"]["negative"][
        "command_clipped_transition_count"
    ] == [0, 0, 4]
    assert full["directions"]["positive"][
        "command_clipped_transition_count"
    ] == [0, 0, 0]
    assert full["all_projections_contractive"] is True
    assert pulse["all_requested_commands_unclipped"] is True
    assert proposal["projection_envelope"]["label_contract"] == (
        "effective_post_supervisor_residual_only"
    )


def test_generated_profiles_load_only_with_case8_identity(tmp_path: Path) -> None:
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
        corrective_path, expected_case=8
    )
    pulse, _ = load_deterministic_wrench_profile(
        wrench_path, expected_case=8
    )
    assert case == pulse.case == 8
    assert max(config.maximum_residuals) < 0.016


def test_committed_profiles_match_formula_bound_proposal() -> None:
    corrective, wrench, proposal = MODULE.build_profiles(**_inputs())
    assert json.loads(COMMITTED_CORRECTIVE.read_text()) == corrective
    assert json.loads(COMMITTED_WRENCH.read_text()) == wrench
    assert json.loads(COMMITTED_PROPOSAL.read_text()) == proposal


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split", "train"),
        ("case30_profile_reuse_authorized", True),
        ("case7_profile_reuse_authorized", True),
    ],
)
def test_builder_rejects_validation_contract_drift(
    field: str, value: object
) -> None:
    inputs = _inputs()
    inputs["readiness"][field] = value
    with pytest.raises(ValueError, match="validation profile checks failed"):
        MODULE.build_profiles(**inputs)


def test_builder_rejects_selection_contract_drift() -> None:
    inputs = _inputs()
    inputs["readiness"]["selection_checks"]["case8_role"] = False
    with pytest.raises(ValueError, match="validation profile checks failed"):
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
