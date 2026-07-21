import json
from pathlib import Path

from scripts.two_wheel_balance.summarize_riser_case78_dynamic_canary_v2 import (
    EXECUTION_DURATION_S,
    MAXIMUM_STEPS,
    NAMESPACE,
    SOURCE_DURATION_S,
    summarize,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def admission(commit: str) -> dict:
    return {
        "runtime_authorized": True,
        "dynamic_qualification_authorized": True,
        "runtime_commit": commit,
        "case": 78,
        "current_split": "unused",
        "namespace": NAMESPACE,
        "split_change_authorized": False,
        "dataset_creation_authorized": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
    }


def heartbeat(emitted: float, steps: int = 4000) -> dict:
    return {
        "schema": "cinebotrl_two_wheel_riser_runtime_heartbeat_v1",
        "emitted_epoch_s": emitted,
        "case": 78,
        "policy_hz": 200,
        "maximum_steps": MAXIMUM_STEPS,
        "completed_steps": steps,
        "phase_time_s": 18.0,
        "source_duration_s": SOURCE_DURATION_S,
        "execution_duration_s": EXECUTION_DURATION_S,
        "capture_outputs_enabled": False,
        "dataset_created": False,
        "valid_for_training": False,
    }


def passing_gate() -> dict:
    result = {
        "case": 78,
        "source_duration_s": SOURCE_DURATION_S,
        "execution_duration_s": EXECUTION_DURATION_S,
        "completed_phase_time_s": EXECUTION_DURATION_S,
        "dynamic_quality_passed": True,
        "thermal_admission_passed": True,
        "controller_evidence_passed": True,
        "termination": None,
        "executed_residual_dataset": None,
        "executed_raw_teacher_capture": None,
        "executed_policy_trace": None,
        "executed_shadow_teacher_trace": None,
        "passed": True,
    }
    return {
        "cases": [78],
        "trajectory_command_source": "deterministic_teacher",
        "residual_policy": None,
        "raw_teacher_capture_started": False,
        "normalized_dataset_capture_started": False,
        "policy_trace_started": False,
        "shadow_teacher_trace_started": False,
        "training_started": False,
        "dagger_authorized": False,
        "ppo_authorized": False,
        "results": [result],
        "passed": True,
    }


def test_v2_success_requires_gate_heartbeat_and_no_data(tmp_path: Path) -> None:
    commit = "a" * 40
    write_json(tmp_path / "admission.json", admission(commit))
    write_json(tmp_path / "runtime_heartbeat.json", heartbeat(150.0, 50_000))
    write_json(tmp_path / "gates/case_0078.json", passing_gate())
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/playback.log").write_text("ok\n", encoding="utf-8")

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
    assert result["dataset_created"] is False
    assert result["case78_validation_admitted"] is False


def test_v2_timeout_seals_healthy_heartbeat_without_claiming_pass(
    tmp_path: Path,
) -> None:
    commit = "b" * 40
    write_json(tmp_path / "admission.json", admission(commit))
    write_json(tmp_path / "runtime_heartbeat.json", heartbeat(5390.0, 110_000))
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/playback.log").write_text("timeout\n", encoding="utf-8")

    result = summarize(
        tmp_path,
        runtime_commit=commit,
        playback_exit_code=124,
        wall_started_epoch_s=0.0,
        wall_finished_epoch_s=5400.02,
        gpu_release_passed=True,
    )

    assert result["timed_out"] is True
    assert result["timeout_evidence_preserved"] is True
    assert result["heartbeat_completed_steps"] == 110_000
    assert result["dynamic_qualification_passed"] is False
    assert result["dataset_created"] is False
    assert result["passed"] is False


def test_v2_rejects_heartbeat_outside_runtime_clock(tmp_path: Path) -> None:
    commit = "c" * 40
    write_json(tmp_path / "admission.json", admission(commit))
    write_json(tmp_path / "runtime_heartbeat.json", heartbeat(99.0))
    write_json(tmp_path / "gates/case_0078.json", passing_gate())
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/playback.log").write_text("ok\n", encoding="utf-8")

    result = summarize(
        tmp_path,
        runtime_commit=commit,
        playback_exit_code=0,
        wall_started_epoch_s=100.0,
        wall_finished_epoch_s=200.0,
        gpu_release_passed=True,
    )

    assert result["heartbeat_checks"]["emitted_during_runtime"] is False
    assert result["passed"] is False

