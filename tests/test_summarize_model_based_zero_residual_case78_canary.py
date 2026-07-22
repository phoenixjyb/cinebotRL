import copy

from scripts.two_wheel_balance.build_model_based_zero_residual_case78_contract import (
    EXPECTED_CONTROLLER,
    EXPECTED_HASHES,
    EXPECTED_PLAN,
    NAMESPACE,
)
from scripts.two_wheel_balance.summarize_model_based_zero_residual_case78_canary import (
    CONTRACT_SCHEMA,
    summarize,
)


def _rollout(*, checkpoint: bool):
    metrics = {
        "position_error_p95_m": 0.1166,
        "position_error_max_m": 0.1842,
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
        "cases": [78],
        "trajectory_command_source": (
            "model_based_planner_plus_torchscript_residual"
            if checkpoint
            else "model_based_planner_plus_zero_policy_residual"
        ),
        "policy_command_base": "model_based_planner",
        "policy_residual_contract": EXPECTED_CONTROLLER["policy_residual_contract"],
        "residual_action_scales": [0.05, 0.05, 0.02],
        "maximum_camera_lever_arm_correction_m": 0.1,
        "residual_policy": "policy.pt" if checkpoint else None,
        "raw_teacher_capture_started": False,
        "normalized_dataset_capture_started": False,
        "policy_trace_started": False,
        "shadow_teacher_trace_started": False,
        "dynamic_quality_passed": True,
        "passed": True,
        "results": [
            {
                "case": 78,
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


def _contract():
    return {
        "schema": CONTRACT_SCHEMA,
        "cpu_contract_ready": True,
        "namespace": NAMESPACE,
        "case": 78,
        "controller_contract": EXPECTED_CONTROLLER,
        "runtime_authorization_token_issued": False,
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
        "cpu_contract": _contract(),
        "explicit_zero_exit_code": 0,
        "zero_checkpoint_exit_code": 0,
        "policy_sha256": EXPECTED_HASHES["zero_policy_torchscript"],
        "dataset_files_present": False,
    }
    values.update(overrides)
    return summarize(**values)


def test_healthy_case78_preservation_passes_but_keeps_next_stage_closed() -> None:
    result = _summary()
    assert result["passed"] is True
    assert result["zero_residual_preservation_passed"] is True
    assert result["case16_22_32_authorized"] is False
    assert result["training_authorized"] is False


def test_rejects_wrong_camera_cap_or_nonzero_residual() -> None:
    candidate = _rollout(checkpoint=True)
    candidate["maximum_camera_lever_arm_correction_m"] = 0.05
    candidate["results"][0]["residual_action_abs_max"] = [0.001, 0.0, 0.0]
    result = _summary(candidate=candidate)
    assert result["passed"] is False
    assert not result["checks"]["zero_checkpoint_result_passes"]


def test_rejects_metric_regression_or_incomplete_reference() -> None:
    candidate = _rollout(checkpoint=True)
    candidate["results"][0]["position_error_p95_m"] += 0.006
    candidate["results"][0]["completed_phase_time_s"] -= 0.01
    result = _summary(candidate=candidate)
    assert result["passed"] is False
    assert not result["checks"]["position_metrics_preserved"]
    assert not result["checks"]["zero_checkpoint_result_passes"]


def test_rejects_dataset_policy_or_exit_drift() -> None:
    result = _summary(
        policy_sha256="forged",
        dataset_files_present=True,
        zero_checkpoint_exit_code=2,
    )
    assert result["passed"] is False
    assert not result["checks"]["zero_checkpoint_identity_exact"]
    assert not result["checks"]["dataset_absent"]
    assert not result["checks"]["rollout_exit_codes_zero"]


def test_rejects_reopened_runtime_contract() -> None:
    contract = copy.deepcopy(_contract())
    contract["runtime_authorized"] = True
    result = summarize(
        explicit_zero=_rollout(checkpoint=False),
        zero_checkpoint=_rollout(checkpoint=True),
        cpu_contract=contract,
        explicit_zero_exit_code=0,
        zero_checkpoint_exit_code=0,
        policy_sha256=EXPECTED_HASHES["zero_policy_torchscript"],
        dataset_files_present=False,
    )
    assert result["passed"] is False
    assert not result["checks"]["cpu_contract_exact"]
