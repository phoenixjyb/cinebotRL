import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/two_wheel_balance/audit_model_based_corrective_case23_pair_readiness.py"
)
SPEC = importlib.util.spec_from_file_location("case23_pair_readiness", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


PLAN_SHA = "a" * 64
GATE_SHA = "b" * 64


def _inputs() -> dict[str, object]:
    time_s = np.linspace(0.0, 10.0, 501)
    feedforward_v_wz = np.zeros((500, 2), dtype=np.float64)
    feedforward_v_wz[:, 0] = 0.23
    feedforward_v_wz[:, 1] = 0.01
    plan_arrays = {
        "time_s": time_s,
        "feedforward_v_wz": feedforward_v_wz,
        "feedforward_riser_velocity": np.full(500, 0.08),
        "feedforward_proxy_velocity": np.full((500, 3), 0.05),
    }
    pulse = {
        "schema": "cinebotrl_two_wheel_riser_deterministic_wrench_pulse_v1",
        "case": 23,
        "start_phase_time_s": 5.0,
        "duration_steps": 20,
        "force_body_x_n": 20.0,
        "application_height_m": 0.5,
    }
    gate_result = {
        "case": 23,
        "dynamic_quality_passed": True,
        "thermal_admission_passed": True,
        "position_error_p95_m": 0.105,
        "position_error_max_m": 0.106,
        "pitch_max_deg": 5.75,
        "attitude_error_max_deg": 0.21,
        "riser_servo_error_max_m": 0.011,
        "action_saturation_ratio": 0.001,
    }
    case30_pulse = dict(pulse, case=30, start_phase_time_s=15.0)
    rollout_common = {
        "cases": [30],
        "passed": True,
        "results": [
            {
                "case": 30,
                "deterministic_wrench_perturbation": {"profile": case30_pulse},
            }
        ],
    }
    return {
        "proposal": {
            "case": 23,
            "passed": True,
            "runtime_authorized": False,
            "gpu_launch_authorized": False,
            "label_capture_authorized": False,
            "dataset_created": False,
            "bc_authorized": False,
            "ppo_authorized": False,
            "proposed_perturbation": pulse,
            "identities": {"plan": {"sha256": PLAN_SHA}},
        },
        "selection": {
            "passed": True,
            "selected_rows": [
                {
                    "case": 23,
                    "selection_role": "same_seed_paired_canary_required",
                    "plan_sha256": PLAN_SHA,
                    "dynamic_gate_sha256": GATE_SHA,
                }
            ],
        },
        "plan_metadata": {
            "case": 23,
            "trajectory_integrity_passed": True,
            "timing_transition_kinematic_gate_passed": True,
        },
        "plan_arrays": plan_arrays,
        "plan_sha256": PLAN_SHA,
        "dynamic_gate": {"passed": True, "results": [gate_result]},
        "dynamic_gate_sha256": GATE_SHA,
        "plant": {
            "nominal": {"total_mass_kg": 28.0},
            "provisional_operating_envelope": {
                "accepted_signed_push_impulse_ns": [-2.0, 2.0]
            },
        },
        "case30_final": {
            "passed": True,
            "paired_admission": {
                "corrective_target_admission_passed": True,
                "position_p95_absolute_improvement_m": 0.0063,
                "position_p95_relative_improvement": 0.044,
            },
            "dataset_created": False,
            "bc_authorized": False,
            "ppo_authorized": False,
            "training_started": False,
        },
        "case30_baseline": rollout_common,
        "case30_candidate": rollout_common,
        "corrective_profile": {
            "case": 23,
            "longitudinal_gain_s_inv": 0.2,
            "deadbands_m": [0.01, 0.01, 0.005],
            "maximum_residuals": [0.045, 0.045, 0.018],
        },
        "wrench_profile": pulse,
    }


def test_readiness_recommends_only_one_bounded_measurement() -> None:
    result = MODULE.audit_readiness(**_inputs())
    assert result["passed"] is True
    assert result["decision"] == "recommend_exactly_one_bounded_case23_pair_canary"
    assert result["pulse"]["duration_s"] == 0.1
    assert result["pulse"]["impulse_ns"] == 2.0
    assert result["runtime_authorized"] is False
    assert result["label_capture_authorized"] is False
    assert result["valid_for_training"] is False
    json.dumps(result)


def test_readiness_rejects_midpoint_speed_without_headroom() -> None:
    inputs = _inputs()
    inputs["plan_arrays"]["feedforward_v_wz"][:, 0] = 0.35
    result = MODULE.audit_readiness(**inputs)
    assert result["checks"]["midpoint_motion_headroom"] is False
    assert result["passed"] is False
    assert result["decision"] == "do_not_authorize_case23_pair_canary"


def test_readiness_rejects_unproven_dynamic_margin() -> None:
    inputs = _inputs()
    inputs["dynamic_gate"]["results"][0]["pitch_max_deg"] = 12.1
    result = MODULE.audit_readiness(**inputs)
    assert result["checks"]["dynamic_gate_passed"] is False
    assert result["passed"] is False


def test_readiness_rejects_impulse_beyond_plant_envelope() -> None:
    inputs = _inputs()
    inputs["plant"]["provisional_operating_envelope"][
        "accepted_signed_push_impulse_ns"
    ] = [-1.0, 1.0]
    result = MODULE.audit_readiness(**inputs)
    assert result["checks"]["pulse_within_provisional_impulse_envelope"] is False
    assert result["passed"] is False


def test_readiness_rejects_weak_case30_precedent() -> None:
    inputs = _inputs()
    inputs["case30_final"]["paired_admission"][
        "position_p95_absolute_improvement_m"
    ] = 0.002
    result = MODULE.audit_readiness(**inputs)
    assert result["checks"]["case30_same_pulse_precedent"] is False
    assert result["passed"] is False
