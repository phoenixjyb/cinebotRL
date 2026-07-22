import json
from pathlib import Path

import numpy as np

from scripts.two_wheel_balance.summarize_case78_shadow_label_measurement import (
    ACTION_SCALES,
    EXECUTION_DURATION_S,
    HEARTBEAT_SCHEMA,
    MAXIMUM_STEPS,
    NAMESPACE,
    SOURCE_DURATION_S,
    TRACE_SCHEMA,
    summarize,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _healthy_root(tmp_path: Path) -> Path:
    root = tmp_path / NAMESPACE
    _write_json(
        root / "admission.json",
        {
            "runtime_authorized": True,
            "gpu_launch_authorized": True,
            "shadow_measurement_authorized": True,
            "runtime_commit": "abc",
            "case": 78,
            "current_split": "validation",
            "namespace": NAMESPACE,
            "label_capture_authorized": False,
            "dataset_creation_authorized": False,
            "bc_authorized": False,
            "ppo_authorized": False,
        },
    )
    _write_json(
        root / "runtime_heartbeat.json",
        {
            "schema": HEARTBEAT_SCHEMA,
            "case": 78,
            "policy_hz": 200,
            "maximum_steps": MAXIMUM_STEPS,
            "completed_steps": 3,
            "source_duration_s": SOURCE_DURATION_S,
            "execution_duration_s": EXECUTION_DURATION_S,
            "emitted_epoch_s": 150.0,
            "capture_outputs_enabled": True,
            "dataset_created": False,
            "valid_for_training": False,
            "phase_time_s": EXECUTION_DURATION_S,
        },
    )
    trace = root / "traces/case_0078_shadow_teacher_trace_v1.npz"
    trace.parent.mkdir(parents=True)
    names = ["feedforward_vx_m_s", "feedforward_wz_rad_s", "riser_position_m"]
    observations = np.asarray([[0.1, -0.1, 0.8]] * 3, dtype=np.float32)
    raw = np.asarray(
        [[0.01, 0.02, 0.001], [-0.02, 0.03, -0.002], [0.03, -0.04, 0.003]],
        dtype=np.float32,
    )
    commands = observations + raw
    metadata = {
        "schema": TRACE_SCHEMA,
        "observation_names": names,
        "visited_state_source": "deterministic_controller",
        "action_scales": ACTION_SCALES.tolist(),
        "trace_only": True,
        "shadow_teacher_applied_to_commands": False,
        "shadow_teacher_labels_admitted_for_training": False,
        "valid_for_training": False,
        "training_started": False,
        "bc_authorized": False,
        "dagger_authorized": False,
        "ppo_authorized": False,
    }
    np.savez_compressed(
        trace,
        metadata_json=np.asarray(json.dumps(metadata)),
        observations=observations,
        applied_residual_actions=np.zeros((3, 3), dtype=np.float32),
        final_high_level_commands=commands,
        case_ids=np.full(3, 78, dtype=np.int16),
        elapsed_time_s=np.asarray([0.0, 0.01, 0.02]),
        phase_time_s=np.asarray([0.0, 0.01, 0.02]),
        shadow_teacher_raw_residual_commands=raw,
        shadow_teacher_normalized_residual_actions=raw / ACTION_SCALES,
        shadow_teacher_high_level_commands=commands,
    )
    result = {
        "case": 78,
        "source_duration_s": SOURCE_DURATION_S,
        "execution_duration_s": EXECUTION_DURATION_S,
        "completed_phase_time_s": EXECUTION_DURATION_S,
        "completed_steps": 3,
        "maximum_camera_lever_arm_correction_m": 0.1,
        "camera_recovery_governor_enabled": False,
        "executed_shadow_teacher_trace": str(trace.resolve()),
        "executed_residual_dataset": None,
        "executed_raw_teacher_capture": None,
        "executed_policy_trace": None,
        "residual_action_abs_max": [0.0, 0.0, 0.0],
        "dynamic_quality_passed": True,
        "thermal_admission_passed": True,
        "controller_evidence_passed": True,
        "termination": None,
        "passed": True,
    }
    _write_json(
        root / "gates/case_0078.json",
        {
            "cases": [78],
            "trajectory_command_source": "deterministic_teacher",
            "residual_policy": None,
            "residual_action_scales": ACTION_SCALES.tolist(),
            "maximum_camera_lever_arm_correction_m": 0.1,
            "camera_recovery_governor_enabled": False,
            "shadow_teacher_trace_started": True,
            "raw_teacher_capture_started": False,
            "normalized_dataset_capture_started": False,
            "policy_trace_started": False,
            "passed": True,
            "results": [result],
        },
    )
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "logs/playback.log").write_text("complete\n", encoding="utf-8")
    return root


def _run(root: Path) -> dict:
    return summarize(
        root,
        runtime_commit="abc",
        playback_exit_code=0,
        wall_started_epoch_s=100.0,
        wall_finished_epoch_s=200.0,
        gpu_release_passed=True,
    )


def test_shadow_label_summary_passes_zero_overflow_deterministic_trace(
    tmp_path: Path,
) -> None:
    result = _run(_healthy_root(tmp_path))
    assert result["passed"] is True
    assert result["physical_quality_passed"] is True
    assert result["shadow_trace_passed"] is True
    assert result["candidate_scale_overflow_passed"] is True
    assert result["shadow_label_statistics"]["overflow_sample_count"] == [0, 0, 0]
    assert result["dataset_created"] is False
    assert result["bc_authorized"] is False


def test_shadow_label_summary_rejects_applied_action_or_overflow(tmp_path: Path) -> None:
    root = _healthy_root(tmp_path)
    path = root / "traces/case_0078_shadow_teacher_trace_v1.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    arrays["applied_residual_actions"] = np.ones((3, 3), dtype=np.float32) * 0.1
    raw = np.asarray(arrays["shadow_teacher_raw_residual_commands"])
    raw[1, 0] = ACTION_SCALES[0] + 0.001
    arrays["shadow_teacher_raw_residual_commands"] = raw
    arrays["shadow_teacher_normalized_residual_actions"] = raw / ACTION_SCALES
    observations = np.asarray(arrays["observations"])
    commands = observations + raw
    arrays["shadow_teacher_high_level_commands"] = commands
    arrays["final_high_level_commands"] = commands
    np.savez_compressed(path, **arrays)
    result = _run(root)
    assert result["passed"] is False
    assert result["trace_checks"]["zero_applied_residual"] is False
    assert result["trace_checks"]["candidate_scale_has_zero_overflow"] is False
