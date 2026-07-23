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
    "build_model_based_corrective_case6_pair_profiles.py"
)
READINESS = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case6_pair_readiness_cpu_v1/summary.json"
)
PLAN = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case6_pair_readiness_cpu_v1/source/"
    "case_0006_smoothed_riser_plan_v1.npz"
)
PLANT = (
    ROOT / "docs/03_training/two_wheel_balance/PLANT_PRIOR_PROVISIONAL_V1.json"
)
COMMITTED_CORRECTIVE = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case6_profile_v1.json"
)
COMMITTED_WRENCH = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case6_wrench_profile_v1.json"
)
COMMITTED_PROPOSAL = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case6_pair_profile_cpu_v1/proposal.json"
)
SPEC = importlib.util.spec_from_file_location("case6_profile_builder", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _inputs(tmp_path: Path):
    plan_metadata, plan_arrays = MODULE._load_plan(PLAN)
    return {
        "readiness": MODULE._load_object(READINESS),
        "readiness_path": READINESS,
        "plan_metadata": plan_metadata,
        "plan_arrays": plan_arrays,
        "plan_path": PLAN,
        "plant": MODULE._load_object(PLANT),
        "plant_path": PLANT,
        "corrective_profile_path": tmp_path / "case6_corrective.json",
        "wrench_profile_path": tmp_path / "case6_wrench.json",
    }


def test_current_case6_profiles_are_formula_bound_and_closed(
    tmp_path: Path,
) -> None:
    corrective, wrench, proposal = MODULE.build_profiles(**_inputs(tmp_path))
    assert proposal["passed"] is True
    assert all(proposal["input_checks"].values())
    assert all(proposal["shape_checks"].values())
    assert all(proposal["formula_checks"].values())
    assert proposal["case"] == 6
    assert proposal["case23_profile_reuse_authorized"] is False
    assert proposal["pair_profile_cpu_ready"] is True
    assert proposal["runtime_route_implemented"] is False
    assert proposal["next_bounded_action"] == (
        "implement_case6_pair_runtime_contract_cpu_only_without_authorization"
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
    assert corrective["case"] == 6
    assert wrench["case"] == 6


def test_profile_formula_retains_case6_envelope_with_policy_margin(
    tmp_path: Path,
) -> None:
    corrective, _, proposal = MODULE.build_profiles(**_inputs(tmp_path))
    formula = proposal["profile_formula"]
    observed = np.asarray(formula["observed_normalized_raw_envelope"])
    scales = np.asarray(formula["policy_residual_scales"])
    expected = observed * scales * 0.75
    np.testing.assert_allclose(
        corrective["maximum_residuals"], expected, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        corrective["maximum_slew_rates"],
        expected / 0.30,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.all(expected > 0.0)
    assert np.all(expected < scales)


def test_pulse_fits_unique_low_motion_window_and_recovery_tail(
    tmp_path: Path,
) -> None:
    _, wrench, proposal = MODULE.build_profiles(**_inputs(tmp_path))
    window = proposal["pulse_window"]
    assert window["window_start_index"] == 761
    assert window["window_end_index"] == 782
    assert window["pulse_duration_steps"] == 20
    assert window["pulse_duration_s_at_policy_rate"] == 0.1
    assert window["pulse_start_phase_time_s"] >= window[
        "window_start_phase_time_s"
    ]
    assert window["pulse_nominal_end_phase_time_s"] <= window[
        "window_end_phase_time_s"
    ]
    assert window["recovery_tail_s"] >= 0.4
    assert wrench["force_body_x_n"] == 20.0
    assert proposal["pulse_lower_model"]["impulse_ns"] == 2.0
    assert (
        proposal["pulse_lower_model"][
            "ideal_free_body_displacement_during_pulse_m"
        ]
        >= 0.003
    )


def test_generated_profiles_load_only_with_case6_identity(
    tmp_path: Path,
) -> None:
    corrective, wrench, _ = MODULE.build_profiles(**_inputs(tmp_path))
    corrective_path = tmp_path / "corrective.json"
    wrench_path = tmp_path / "wrench.json"
    corrective_path.write_text(json.dumps(corrective) + "\n", encoding="utf-8")
    wrench_path.write_text(json.dumps(wrench) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not the reviewed case"):
        load_corrective_teacher_profile(corrective_path)
    with pytest.raises(ValueError, match="invalid deterministic wrench profile"):
        load_deterministic_wrench_profile(wrench_path)
    case, config, _ = load_corrective_teacher_profile(
        corrective_path, expected_case=6
    )
    pulse, _ = load_deterministic_wrench_profile(
        wrench_path, expected_case=6
    )
    assert case == 6
    assert pulse.case == 6
    assert max(config.maximum_residuals) < 0.05


def test_committed_profiles_match_the_formula_bound_proposal(
    tmp_path: Path,
) -> None:
    corrective, wrench, proposal = MODULE.build_profiles(
        **(
            _inputs(tmp_path)
            | {
                "corrective_profile_path": COMMITTED_CORRECTIVE,
                "wrench_profile_path": COMMITTED_WRENCH,
            }
        )
    )
    assert json.loads(COMMITTED_CORRECTIVE.read_text()) == corrective
    assert json.loads(COMMITTED_WRENCH.read_text()) == wrench
    assert json.loads(COMMITTED_PROPOSAL.read_text()) == proposal
    case, _, corrective_identity = load_corrective_teacher_profile(
        COMMITTED_CORRECTIVE, expected_case=6
    )
    pulse, wrench_identity = load_deterministic_wrench_profile(
        COMMITTED_WRENCH, expected_case=6
    )
    assert case == pulse.case == 6
    assert (
        proposal["identities"]["corrective_profile"]["sha256"]
        == corrective_identity["sha256"]
    )
    assert (
        proposal["identities"]["wrench_profile"]["sha256"]
        == wrench_identity["sha256"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runtime_authorized", True),
        ("case23_profile_reuse_authorized", True),
        ("case_specific_profile_required", False),
    ],
)
def test_builder_rejects_readiness_contract_drift(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    inputs = _inputs(tmp_path)
    inputs["readiness"][field] = value
    with pytest.raises(ValueError, match="profile input checks failed"):
        MODULE.build_profiles(**inputs)


def test_builder_rejects_missing_low_motion_window(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    arrays = dict(inputs["plan_arrays"])
    feedforward = arrays["feedforward_v_wz"].copy()
    feedforward[:, 0] = 0.4
    arrays["feedforward_v_wz"] = feedforward
    inputs["plan_arrays"] = arrays
    with pytest.raises(ValueError, match="exactly one eligible"):
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
