import copy

import pytest

from scripts.two_wheel_balance.audit_initial_teacher41_case78_failure import audit


def _trace(position: float, phase: float, progress: float, vx: float, wz: float):
    return {
        "phase_time_s": phase,
        "position_error_m": position,
        "progress_scale": progress,
        "phase_feedforward_v_mps": 0.1,
        "phase_feedforward_wz_rad_s": 0.05,
        "vx_reference_mps": vx,
        "wz_reference_rad_s": wz,
    }


def _payloads():
    checks = {"completed_reference": True, "position_p95_bounded": False}
    learned_result = {
        "completed_phase_time_s": 10.0,
        "execution_duration_s": 10.0,
        "position_error_p95_m": 0.165,
        "position_error_max_m": 0.23,
        "completed_steps": 100,
        "total_simulated_duration_s": 12.0,
        "progress_scale_mean": 0.4,
        "progress_hold_step_count": 2,
        "termination": None,
        "action_saturation_ratio": 0.0,
        "riser_saturation_ratio": 0.0,
        "proxy_saturation_ratio": 0.0,
        "checks": checks,
        "trace": [
            _trace(0.1, 0.0, 0.8, 0.12, 0.07),
            _trace(0.18, 1.0, 0.3, 0.08, 0.02),
        ],
    }
    teacher_result = {
        "passed": True,
        "position_error_p95_m": 0.11,
        "position_error_max_m": 0.18,
        "completed_steps": 90,
        "total_simulated_duration_s": 11.0,
        "progress_scale_mean": 0.5,
        "progress_hold_step_count": 0,
        "trace": [
            _trace(0.09, 0.0, 0.9, 0.13, 0.08),
            _trace(0.14, 1.0, 0.5, 0.11, 0.06),
        ],
    }
    learned = {"results": [learned_result]}
    teacher = {"passed": True, "results": [teacher_result]}
    final = {
        "passed": False,
        "remaining_validation_cases_authorized": False,
        "dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
    }
    policy = {
        "training_method": "offline_behavior_cloning",
        "policy_architecture": "state_shared_lookahead_fusion_previous_action_masked_v1",
        "masked_observation_indices": [23, 24, 25],
    }
    playback = """
raw_residual_command = build_raw_residual_command(
policy_output = residual_policy(
apply_residual_action(
                phase_feedforward_v_mps,
"""
    residual = """
commanded_vx_m_s - feedforward_vx_m_s
commanded_wz_rad_s - feedforward_wz_rad_s
"""
    return learned, teacher, final, policy, playback, residual


def _audit(payloads=None):
    learned, teacher, final, policy, playback, residual = payloads or _payloads()
    return audit(
        learned,
        teacher,
        final,
        policy,
        playback_source=playback,
        residual_contract_source=residual,
    )


def test_classifies_planner_imitation_as_not_final_residual_policy() -> None:
    result = _audit()
    assert result["passed"]
    assert result["failed_dynamic_gate"] == "position_p95_bounded"
    assert len(result["high_error_trace_intervals"]) == 1
    assert not result["architecture_audit"]["required_contract_satisfied"]
    assert not result["decision"]["case16_22_32_tranche_authorized"]
    assert not result["decision"]["threshold_relaxation_authorized"]
    assert not result["decision"]["ppo_authorized"]


def test_rejects_additional_dynamic_failure() -> None:
    payloads = list(_payloads())
    payloads[0] = copy.deepcopy(payloads[0])
    payloads[0]["results"][0]["checks"]["pitch_bounded"] = False
    with pytest.raises(ValueError, match="input contract failed"):
        _audit(tuple(payloads))


def test_rejects_threshold_relaxation_disguised_as_pass() -> None:
    payloads = list(_payloads())
    payloads[0] = copy.deepcopy(payloads[0])
    payloads[0]["results"][0]["position_error_p95_m"] = 0.149
    with pytest.raises(ValueError, match="input contract failed"):
        _audit(tuple(payloads))


def test_rejects_missing_architecture_evidence() -> None:
    payloads = list(_payloads())
    payloads[4] = payloads[4].replace("phase_feedforward_v_mps", "vx_ref")
    with pytest.raises(ValueError, match="input contract failed"):
        _audit(tuple(payloads))
