import copy

from scripts.two_wheel_balance.summarize_model_based_zero_residual_case8_canary import (
    summarize,
)
from scripts.two_wheel_balance.validate_model_based_zero_residual_case8_contract import (
    ADMISSION_SCHEMA,
    EXPECTED_CONTROLLER,
    EXPECTED_PLAN,
    NAMESPACE,
    ZERO_POLICY_TORCHSCRIPT_SHA256,
)


def _rollout(*, checkpoint: bool):
    metrics = {
        "position_error_p95_m": 0.13,
        "position_error_max_m": 0.14,
        "attitude_error_p95_deg": 0.15,
        "attitude_error_max_deg": 0.22,
        "pitch_p95_deg": 5.9,
        "pitch_max_deg": 6.1,
        "riser_servo_error_p95_m": 0.011,
        "riser_servo_error_max_m": 0.012,
        "proxy_servo_error_p95_deg": 0.12,
        "proxy_servo_error_max_deg": 0.23,
    }
    return {
        "cases": [8],
        "trajectory_command_source": (
            "model_based_planner_plus_torchscript_residual"
            if checkpoint
            else "model_based_planner_plus_zero_policy_residual"
        ),
        "policy_command_base": EXPECTED_CONTROLLER["policy_command_base"],
        "policy_residual_contract": EXPECTED_CONTROLLER[
            "policy_residual_contract"
        ],
        "residual_action_scales": EXPECTED_CONTROLLER["residual_action_scales"],
        "residual_policy": "policy.pt" if checkpoint else None,
        "raw_teacher_capture_started": False,
        "normalized_dataset_capture_started": False,
        "policy_trace_started": False,
        "shadow_teacher_trace_started": False,
        "dynamic_quality_passed": True,
        "passed": True,
        "results": [
            {
                "case": 8,
                "source_duration_s": EXPECTED_PLAN["source_duration_s"],
                "execution_duration_s": EXPECTED_PLAN["execution_duration_s"],
                "completed_phase_time_s": EXPECTED_PLAN["execution_duration_s"],
                "dynamic_quality_passed": True,
                "residual_action_abs_max": [0.0, 0.0, 0.0],
                "executed_residual_dataset": None,
                "passed": True,
                **metrics,
            }
        ],
    }


def _admission():
    return {
        "schema": ADMISSION_SCHEMA,
        "passed": True,
        "cpu_contract_ready": True,
        "namespace": NAMESPACE,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "dynamic_canary_authorized": False,
        "dataset_creation_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "holdout_opened": False,
    }


def _summary(baseline=None, candidate=None, **overrides):
    values = {
        "explicit_zero": baseline or _rollout(checkpoint=False),
        "zero_checkpoint": candidate or _rollout(checkpoint=True),
        "cpu_admission": _admission(),
        "explicit_zero_exit_code": 0,
        "zero_checkpoint_exit_code": 0,
        "policy_sha256": ZERO_POLICY_TORCHSCRIPT_SHA256,
        "dataset_files_present": False,
    }
    values.update(overrides)
    return summarize(**values)


def test_healthy_zero_checkpoint_preserves_model_based_teacher() -> None:
    result = _summary()
    assert result["passed"] is True
    assert result["zero_residual_preservation_passed"] is True
    assert result["case78_authorized"] is False
    assert result["training_authorized"] is False


def test_rejects_nonzero_residual_or_wrong_command_base() -> None:
    candidate = _rollout(checkpoint=True)
    candidate["policy_command_base"] = "phase_feedforward"
    candidate["results"][0]["residual_action_abs_max"] = [0.001, 0.0, 0.0]
    result = _summary(candidate=candidate)
    assert result["passed"] is False
    assert not result["checks"]["zero_checkpoint_result_passes"]


def test_rejects_metric_regression_even_when_physics_gate_passes() -> None:
    candidate = _rollout(checkpoint=True)
    candidate["results"][0]["position_error_p95_m"] += 0.006
    result = _summary(candidate=candidate)
    assert result["passed"] is False
    assert not result["checks"]["position_metrics_preserved"]


def test_rejects_incomplete_rollout_or_dataset_side_effect() -> None:
    baseline = copy.deepcopy(_rollout(checkpoint=False))
    baseline["results"][0]["completed_phase_time_s"] -= 0.01
    result = _summary(baseline=baseline, dataset_files_present=True)
    assert result["passed"] is False
    assert not result["checks"]["explicit_zero_result_passes"]
    assert not result["checks"]["dataset_absent"]


def test_rejects_wrong_policy_identity_or_failed_exit() -> None:
    result = _summary(policy_sha256="forged", zero_checkpoint_exit_code=2)
    assert result["passed"] is False
    assert not result["checks"]["zero_checkpoint_identity_exact"]
    assert not result["checks"]["rollout_exit_codes_zero"]
