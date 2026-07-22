#!/usr/bin/env python3
"""Seal one deterministic case-78 shadow-label measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


NAMESPACE = "20260722_case78_shadow_label_measurement_v1_exclusive"
HEARTBEAT_SCHEMA = "cinebotrl_two_wheel_riser_runtime_heartbeat_v1"
TRACE_SCHEMA = "cinebotrl_two_wheel_riser_shadow_teacher_trace_v1"
SOURCE_DURATION_S = 135.487646
EXECUTION_DURATION_S = 192.29956737098348
MAXIMUM_STEPS = 115381
MAXIMUM_WALL_DURATION_S = 5400.0
ACTION_SCALES = np.asarray([0.35, 0.4, 0.1], dtype=np.float64)
PERCENTILES = (50.0, 90.0, 95.0, 99.0, 99.9)


def identity(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_trace(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not path.is_file():
        return {}, {}
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {
            name: np.asarray(data[name])
            for name in data.files
            if name != "metadata_json"
        }
    return metadata, arrays


def trace_statistics(
    metadata: dict[str, Any], arrays: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, bool]]:
    required = {
        "observations",
        "applied_residual_actions",
        "final_high_level_commands",
        "case_ids",
        "elapsed_time_s",
        "phase_time_s",
        "shadow_teacher_raw_residual_commands",
        "shadow_teacher_normalized_residual_actions",
        "shadow_teacher_high_level_commands",
    }
    present = required.issubset(arrays)
    if not present:
        return {}, {"required_arrays_present": False}
    raw = np.asarray(arrays["shadow_teacher_raw_residual_commands"], dtype=np.float64)
    normalized = np.asarray(
        arrays["shadow_teacher_normalized_residual_actions"], dtype=np.float64
    )
    applied = np.asarray(arrays["applied_residual_actions"], dtype=np.float64)
    observations = np.asarray(arrays["observations"], dtype=np.float64)
    commands = np.asarray(arrays["final_high_level_commands"], dtype=np.float64)
    teacher_commands = np.asarray(
        arrays["shadow_teacher_high_level_commands"], dtype=np.float64
    )
    elapsed = np.asarray(arrays["elapsed_time_s"], dtype=np.float64)
    phase = np.asarray(arrays["phase_time_s"], dtype=np.float64)
    cases = np.asarray(arrays["case_ids"])
    count = len(raw)
    shapes = (
        raw.shape == normalized.shape == applied.shape == commands.shape
        == teacher_commands.shape == (count, 3)
        and observations.ndim == 2
        and len(observations) == len(elapsed) == len(phase) == len(cases) == count
    )
    finite = bool(
        shapes
        and all(
            np.isfinite(value).all()
            for value in (
                raw,
                normalized,
                applied,
                observations,
                commands,
                teacher_commands,
                elapsed,
                phase,
            )
        )
    )
    names = metadata.get("observation_names", [])
    required_names = (
        "feedforward_vx_m_s",
        "feedforward_wz_rad_s",
        "riser_position_m",
    )
    named = all(name in names for name in required_names)
    increasing = bool(
        count >= 2
        and abs(float(elapsed[0])) <= 1e-9
        and np.all(np.diff(elapsed) > 0.0)
        and np.all(np.diff(phase) >= 0.0)
    )
    if not (finite and named and increasing):
        return {}, {
            "required_arrays_present": present,
            "array_shapes_match": shapes,
            "arrays_finite": finite,
            "observation_names_present": named,
            "timestamps_monotonic": increasing,
        }
    reconstructed = np.column_stack(
        (
            observations[:, names.index("feedforward_vx_m_s")] + raw[:, 0],
            observations[:, names.index("feedforward_wz_rad_s")] + raw[:, 1],
            observations[:, names.index("riser_position_m")] + raw[:, 2],
        )
    )
    raw_normalization_error = float(
        np.max(np.abs(raw - normalized * ACTION_SCALES))
    )
    teacher_reconstruction_error = float(
        np.max(np.abs(reconstructed - teacher_commands))
    )
    deterministic_command_error = float(np.max(np.abs(commands - teacher_commands)))
    applied_abs_max = np.max(np.abs(applied), axis=0)
    absolute = np.abs(raw)
    raw_abs_max = np.max(absolute, axis=0)
    overflow = absolute >= ACTION_SCALES
    delta = np.diff(elapsed)
    overflow_duration = np.sum(overflow[:-1] * delta[:, None], axis=0)
    total_duration = float(np.sum(delta))
    statistics = {
        "row_count": count,
        "observed_duration_s": total_duration,
        "raw_residual_signed_min": np.min(raw, axis=0).tolist(),
        "raw_residual_signed_max": np.max(raw, axis=0).tolist(),
        "raw_residual_abs_max": raw_abs_max.tolist(),
        "raw_residual_abs_percentiles": {
            str(percentile): np.percentile(absolute, percentile, axis=0).tolist()
            for percentile in PERCENTILES
        },
        "normalized_residual_abs_max": (raw_abs_max / ACTION_SCALES).tolist(),
        "overflow_comparison": "absolute_raw_residual_greater_than_or_equal_to_scale",
        "overflow_duration_contract": "left_sample_forward_interval_v1",
        "overflow_sample_count": np.sum(overflow, axis=0).tolist(),
        "overflow_sample_ratio": np.mean(overflow, axis=0).tolist(),
        "overflow_duration_s": overflow_duration.tolist(),
        "overflow_duration_ratio": (
            overflow_duration / total_duration
            if total_duration > 0.0
            else overflow_duration
        ).tolist(),
        "raw_normalization_max_error": raw_normalization_error,
        "teacher_command_reconstruction_max_error": teacher_reconstruction_error,
        "deterministic_command_match_max_error": deterministic_command_error,
        "applied_residual_action_abs_max": applied_abs_max.tolist(),
    }
    checks = {
        "required_arrays_present": True,
        "array_shapes_match": shapes,
        "arrays_finite": finite,
        "observation_names_present": named,
        "timestamps_monotonic": increasing,
        "single_case_78": bool(np.array_equal(np.unique(cases), [78])),
        "metadata_schema": metadata.get("schema") == TRACE_SCHEMA,
        "deterministic_visited_states": metadata.get("visited_state_source")
        == "deterministic_controller",
        "candidate_scales_exact": bool(
            np.allclose(metadata.get("action_scales"), ACTION_SCALES, atol=1e-12)
        ),
        "trace_only": metadata.get("trace_only") is True,
        "labels_unapplied": metadata.get("shadow_teacher_applied_to_commands")
        is False,
        "labels_unadmitted": metadata.get(
            "shadow_teacher_labels_admitted_for_training"
        )
        is False,
        "not_trainable": metadata.get("valid_for_training") is False,
        "training_closed": metadata.get("training_started") is False
        and metadata.get("bc_authorized") is False
        and metadata.get("dagger_authorized") is False
        and metadata.get("ppo_authorized") is False,
        "zero_applied_residual": bool(np.all(applied_abs_max <= 1e-12)),
        "raw_normalization_exact": raw_normalization_error <= 2e-7,
        "teacher_command_reconstruction_exact": teacher_reconstruction_error <= 2e-6,
        "deterministic_commands_unchanged": deterministic_command_error <= 2e-6,
        "candidate_scale_has_zero_overflow": bool(np.all(overflow == 0)),
    }
    return statistics, checks


def summarize(
    root: Path,
    *,
    runtime_commit: str,
    playback_exit_code: int,
    wall_started_epoch_s: float,
    wall_finished_epoch_s: float,
    gpu_release_passed: bool,
) -> dict[str, Any]:
    admission_path = root / "admission.json"
    gate_path = root / "gates/case_0078.json"
    heartbeat_path = root / "runtime_heartbeat.json"
    trace_path = root / "traces/case_0078_shadow_teacher_trace_v1.npz"
    log_path = root / "logs/playback.log"
    admission = load_json(admission_path)
    gate = load_json(gate_path)
    heartbeat = load_json(heartbeat_path)
    metadata, arrays = load_trace(trace_path)
    statistics, trace_checks = trace_statistics(metadata, arrays)
    results = gate.get("results", [])
    result = results[0] if len(results) == 1 and isinstance(results[0], dict) else {}
    observed_wall_s = max(0.0, wall_finished_epoch_s - wall_started_epoch_s)
    heartbeat_emitted_s = float(heartbeat.get("emitted_epoch_s", 0.0))
    completed_steps = int(heartbeat.get("completed_steps", 0))
    admission_checks = {
        "runtime_authorized": admission.get("runtime_authorized") is True,
        "gpu_authorized": admission.get("gpu_launch_authorized") is True,
        "shadow_measurement_authorized": admission.get(
            "shadow_measurement_authorized"
        )
        is True,
        "runtime_commit": admission.get("runtime_commit") == runtime_commit,
        "case_and_split": admission.get("case") == 78
        and admission.get("current_split") == "validation",
        "namespace": admission.get("namespace") == NAMESPACE,
        "learning_closed": admission.get("dataset_creation_authorized") is False
        and admission.get("label_capture_authorized") is False
        and admission.get("bc_authorized") is False
        and admission.get("ppo_authorized") is False,
    }
    heartbeat_checks = {
        "present": heartbeat_path.is_file(),
        "schema": heartbeat.get("schema") == HEARTBEAT_SCHEMA,
        "case": heartbeat.get("case") == 78,
        "policy_hz": heartbeat.get("policy_hz") == 200,
        "maximum_steps": heartbeat.get("maximum_steps") == MAXIMUM_STEPS,
        "step_range": 0 <= completed_steps <= MAXIMUM_STEPS,
        "source_clock": heartbeat.get("source_duration_s") == SOURCE_DURATION_S,
        "execution_clock": heartbeat.get("execution_duration_s")
        == EXECUTION_DURATION_S,
        "emitted_during_runtime": wall_started_epoch_s
        <= heartbeat_emitted_s
        <= wall_finished_epoch_s,
        "shadow_output_enabled": heartbeat.get("capture_outputs_enabled") is True,
        "non_training": heartbeat.get("dataset_created") is False
        and heartbeat.get("valid_for_training") is False,
    }
    gate_checks = {
        "single_case": gate.get("cases") == [78] and result.get("case") == 78,
        "deterministic_teacher": gate.get("trajectory_command_source")
        == "deterministic_teacher",
        "no_residual_policy": gate.get("residual_policy") is None,
        "candidate_scales": bool(
            np.allclose(gate.get("residual_action_scales"), ACTION_SCALES, atol=1e-12)
        ),
        "source_clock": result.get("source_duration_s") == SOURCE_DURATION_S,
        "execution_clock": result.get("execution_duration_s")
        == EXECUTION_DURATION_S,
        "complete_reference": result.get("completed_phase_time_s")
        == EXECUTION_DURATION_S,
        "camera_cap_exact": gate.get("maximum_camera_lever_arm_correction_m")
        == result.get("maximum_camera_lever_arm_correction_m")
        == 0.1,
        "camera_recovery_disabled": gate.get("camera_recovery_governor_enabled")
        is False
        and result.get("camera_recovery_governor_enabled") is False,
        "only_shadow_trace_enabled": gate.get("shadow_teacher_trace_started") is True
        and gate.get("raw_teacher_capture_started") is False
        and gate.get("normalized_dataset_capture_started") is False
        and gate.get("policy_trace_started") is False,
        "trace_path_bound": result.get("executed_shadow_teacher_trace")
        == str(trace_path.resolve()),
        "no_other_runtime_output": result.get("executed_residual_dataset") is None
        and result.get("executed_raw_teacher_capture") is None
        and result.get("executed_policy_trace") is None,
        "zero_runtime_residual": bool(
            np.allclose(result.get("residual_action_abs_max"), 0.0, atol=1e-12)
        ),
        "wall_clock_ordered": wall_finished_epoch_s >= wall_started_epoch_s,
        "wall_timeout_bounded": observed_wall_s <= MAXIMUM_WALL_DURATION_S + 1.0,
        "gpu_released": gpu_release_passed,
    }
    physical_checks = {
        "dynamic_quality": result.get("dynamic_quality_passed") is True,
        "thermal_admission": result.get("thermal_admission_passed") is True,
        "controller_evidence": result.get("controller_evidence_passed") is True,
        "no_termination": result.get("termination") is None,
        "gate_passed": gate.get("passed") is True and result.get("passed") is True,
    }
    filesystem_checks = {
        "one_trace_only": trace_path.is_file()
        and len(list((root / "traces").glob("*.npz"))) == 1,
        "no_dataset_dir": not (root / "datasets").exists(),
        "no_raw_teacher_dir": not (root / "raw_teacher").exists(),
        "no_policy_trace_dir": not (root / "policy_traces").exists(),
    }
    heartbeat_passed = all(heartbeat_checks.values())
    physical_passed = all(physical_checks.values()) and all(gate_checks.values())
    trace_passed = bool(trace_checks) and all(trace_checks.values())
    passed = (
        playback_exit_code == 0
        and all(admission_checks.values())
        and heartbeat_passed
        and physical_passed
        and trace_passed
        and all(filesystem_checks.values())
    )
    return {
        "schema": "cinebotrl_two_wheel_riser_case78_shadow_label_final_v1",
        "runtime_commit": runtime_commit,
        "case": 78,
        "current_split": "validation",
        "namespace": NAMESPACE,
        "playback_exit_code": playback_exit_code,
        "timed_out": playback_exit_code == 124,
        "wall_started_epoch_s": wall_started_epoch_s,
        "wall_finished_epoch_s": wall_finished_epoch_s,
        "observed_wall_duration_s": observed_wall_s,
        "maximum_wall_duration_s": MAXIMUM_WALL_DURATION_S,
        "source_duration_s": result.get("source_duration_s"),
        "execution_duration_s": result.get("execution_duration_s"),
        "heartbeat_completed_steps": completed_steps,
        "heartbeat_phase_time_s": heartbeat.get("phase_time_s"),
        "shadow_label_statistics": statistics,
        "admission_checks": admission_checks,
        "heartbeat_checks": heartbeat_checks,
        "gate_contract_checks": gate_checks,
        "physical_checks": physical_checks,
        "trace_checks": trace_checks,
        "filesystem_checks": filesystem_checks,
        "physical_quality_passed": physical_passed,
        "shadow_trace_passed": trace_passed,
        "candidate_scale_overflow_passed": trace_checks.get(
            "candidate_scale_has_zero_overflow", False
        ),
        "admission": identity(admission_path),
        "heartbeat": identity(heartbeat_path),
        "gate": identity(gate_path),
        "shadow_trace": identity(trace_path),
        "playback_log": identity(log_path),
        "runtime_authorized": all(admission_checks.values()),
        "runtime_started": heartbeat_path.is_file() or playback_exit_code != 99,
        "shadow_measurement_completed": trace_passed,
        "labels_applied_to_commands": False,
        "label_capture_admitted": False,
        "dataset_created": False,
        "valid_for_training": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--playback-exit-code", type=int, required=True)
    parser.add_argument("--wall-started-epoch-s", type=float, required=True)
    parser.add_argument("--wall-finished-epoch-s", type=float, required=True)
    parser.add_argument("--gpu-release-passed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.root,
        runtime_commit=args.runtime_commit,
        playback_exit_code=args.playback_exit_code,
        wall_started_epoch_s=args.wall_started_epoch_s,
        wall_finished_epoch_s=args.wall_finished_epoch_s,
        gpu_release_passed=bool(args.gpu_release_passed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
