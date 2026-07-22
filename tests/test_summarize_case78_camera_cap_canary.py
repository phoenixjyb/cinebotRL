import json
from pathlib import Path

from scripts.two_wheel_balance.summarize_case78_camera_cap_canary import (
    EXECUTION_DURATION_S,
    HEARTBEAT_SCHEMA,
    MAXIMUM_STEPS,
    NAMESPACE,
    SOURCE_DURATION_S,
    summarize,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def healthy_root(tmp_path: Path) -> Path:
    root = tmp_path / NAMESPACE
    write_json(root / "admission.json", {
        "runtime_authorized": True,
        "dynamic_qualification_authorized": True,
        "runtime_commit": "abc",
        "case": 78,
        "current_split": "unused",
        "namespace": NAMESPACE,
        "split_change_authorized": False,
        "dataset_creation_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
    })
    write_json(root / "runtime_heartbeat.json", {
        "schema": HEARTBEAT_SCHEMA,
        "case": 78,
        "policy_hz": 200,
        "maximum_steps": MAXIMUM_STEPS,
        "completed_steps": 80000,
        "source_duration_s": SOURCE_DURATION_S,
        "execution_duration_s": EXECUTION_DURATION_S,
        "emitted_epoch_s": 150.0,
        "capture_outputs_enabled": False,
        "dataset_created": False,
        "valid_for_training": False,
        "phase_time_s": EXECUTION_DURATION_S,
    })
    result = {
        "case": 78,
        "source_duration_s": SOURCE_DURATION_S,
        "execution_duration_s": EXECUTION_DURATION_S,
        "completed_phase_time_s": EXECUTION_DURATION_S,
        "completed_steps": 80000,
        "camera_lever_arm_compensation_enabled": True,
        "camera_lever_arm_compensation_gain": 1.0,
        "maximum_camera_lever_arm_correction_m": 0.10,
        "camera_lever_arm_telemetry_observed": True,
        "camera_lever_arm_telemetry_sample_count": 80000,
        "camera_lever_arm_correction_max_m": 0.10,
        "camera_lever_arm_correction_saturation_ratio": 0.5,
        "camera_recovery_governor_enabled": False,
        "dynamic_quality_passed": True,
        "thermal_admission_passed": True,
        "controller_evidence_passed": True,
        "termination": None,
        "passed": True,
        "position_error_p95_m": 0.14,
        "position_error_max_m": 0.23,
        "residual_label_envelope_passed": False,
        "executed_residual_dataset": None,
        "executed_raw_teacher_capture": None,
        "executed_policy_trace": None,
        "executed_shadow_teacher_trace": None,
    }
    write_json(root / "gates/case_0078.json", {
        "cases": [78],
        "trajectory_command_source": "deterministic_teacher",
        "residual_policy": None,
        "camera_lever_arm_compensation_enabled": True,
        "camera_lever_arm_compensation_gain": 1.0,
        "maximum_camera_lever_arm_correction_m": 0.10,
        "camera_recovery_governor_enabled": False,
        "raw_teacher_capture_started": False,
        "normalized_dataset_capture_started": False,
        "policy_trace_started": False,
        "shadow_teacher_trace_started": False,
        "training_started": False,
        "dagger_authorized": False,
        "ppo_authorized": False,
        "passed": True,
        "results": [result],
    })
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "logs/playback.log").write_text("complete\n", encoding="utf-8")
    return root


def run_summary(root: Path) -> dict:
    return summarize(
        root,
        runtime_commit="abc",
        playback_exit_code=0,
        wall_started_epoch_s=100.0,
        wall_finished_epoch_s=200.0,
        gpu_release_passed=True,
    )


def test_camera_cap_summary_passes_independently_of_label_envelope(
    tmp_path: Path,
) -> None:
    result = run_summary(healthy_root(tmp_path))
    assert result["passed"] is True
    assert result["dynamic_qualification_passed"] is True
    assert result["residual_label_envelope_passed"] is False
    assert result["dataset_created"] is False
    assert result["case78_validation_admitted"] is False


def test_camera_cap_summary_rejects_recovery_or_wrong_cap(tmp_path: Path) -> None:
    root = healthy_root(tmp_path)
    gate_path = root / "gates/case_0078.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["camera_recovery_governor_enabled"] = True
    gate["results"][0]["maximum_camera_lever_arm_correction_m"] = 0.05
    write_json(gate_path, gate)

    result = run_summary(root)
    assert result["passed"] is False
    assert not result["gate_contract_checks"]["camera_recovery_disabled"]
    assert not result["gate_contract_checks"]["camera_correction_cap_exact"]
