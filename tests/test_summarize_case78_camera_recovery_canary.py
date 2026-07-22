import json
from pathlib import Path

from scripts.two_wheel_balance.summarize_case78_camera_recovery_canary import (
    EXECUTION_DURATION_S,
    MAXIMUM_STEPS,
    NAMESPACE,
    SOURCE_DURATION_S,
    summarize,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def setup_case(root: Path, commit: str, recovery_enabled: bool = True) -> None:
    write_json(
        root / "admission.json",
        {
            "runtime_authorized": True,
            "dynamic_qualification_authorized": True,
            "runtime_commit": commit,
            "case": 78,
            "current_split": "unused",
            "namespace": NAMESPACE,
            "split_change_authorized": False,
            "dataset_creation_authorized": False,
            "bc_authorized": False,
            "ppo_authorized": False,
        },
    )
    write_json(
        root / "runtime_heartbeat.json",
        {
            "schema": "cinebotrl_two_wheel_riser_runtime_heartbeat_v1",
            "emitted_epoch_s": 190.0,
            "case": 78,
            "policy_hz": 200,
            "maximum_steps": MAXIMUM_STEPS,
            "completed_steps": 88_000,
            "phase_time_s": EXECUTION_DURATION_S,
            "source_duration_s": SOURCE_DURATION_S,
            "execution_duration_s": EXECUTION_DURATION_S,
            "capture_outputs_enabled": False,
            "dataset_created": False,
            "valid_for_training": False,
        },
    )
    result = {
        "case": 78,
        "source_duration_s": SOURCE_DURATION_S,
        "execution_duration_s": EXECUTION_DURATION_S,
        "completed_phase_time_s": EXECUTION_DURATION_S,
        "camera_recovery_governor_enabled": recovery_enabled,
        "camera_recovery_error_range_m": [0.13, 0.155],
        "minimum_camera_recovery_scale": 0.2,
        "camera_recovery_telemetry_observed": True,
        "camera_recovery_activation_ratio": 0.05,
        "camera_recovery_progress_scale_min": 0.2,
        "dynamic_quality_passed": True,
        "thermal_admission_passed": True,
        "controller_evidence_passed": True,
        "termination": None,
        "position_error_p95_m": 0.145,
        "position_error_max_m": 0.23,
        "executed_residual_dataset": None,
        "executed_raw_teacher_capture": None,
        "executed_policy_trace": None,
        "executed_shadow_teacher_trace": None,
        "passed": True,
    }
    write_json(
        root / "gates/case_0078.json",
        {
            "cases": [78],
            "trajectory_command_source": "deterministic_teacher",
            "residual_policy": None,
            "camera_recovery_governor_enabled": recovery_enabled,
            "camera_recovery_error_range_m": [0.13, 0.155],
            "minimum_camera_recovery_scale": 0.2,
            "raw_teacher_capture_started": False,
            "normalized_dataset_capture_started": False,
            "policy_trace_started": False,
            "shadow_teacher_trace_started": False,
            "training_started": False,
            "dagger_authorized": False,
            "ppo_authorized": False,
            "results": [result],
            "passed": True,
        },
    )
    (root / "logs").mkdir()
    (root / "logs/playback.log").write_text("ok\n", encoding="utf-8")


def test_recovery_canary_success_requires_exact_governor(tmp_path: Path) -> None:
    commit = "d" * 40
    setup_case(tmp_path, commit)
    result = summarize(
        tmp_path,
        runtime_commit=commit,
        playback_exit_code=0,
        wall_started_epoch_s=100.0,
        wall_finished_epoch_s=200.0,
        gpu_release_passed=True,
    )
    assert result["passed"] is True
    assert result["dynamic_qualification_passed"] is True
    assert result["case78_validation_admitted"] is False
    assert result["dataset_created"] is False


def test_recovery_canary_rejects_missing_governor(tmp_path: Path) -> None:
    commit = "e" * 40
    setup_case(tmp_path, commit, recovery_enabled=False)
    result = summarize(
        tmp_path,
        runtime_commit=commit,
        playback_exit_code=0,
        wall_started_epoch_s=100.0,
        wall_finished_epoch_s=200.0,
        gpu_release_passed=True,
    )
    assert result["gate_contract_checks"]["recovery_governor_enabled"] is False
    assert result["passed"] is False

