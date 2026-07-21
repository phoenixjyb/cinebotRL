#!/usr/bin/env python3
"""Localize coarse physical-state divergence between teacher and learned rollout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    OBSERVATION_NAMES,
)


FIELD_SPECS = {
    "pitch_deg": ("balance", 0.5, "deg", False),
    "actual_yaw_rate_rad_s": ("balance", 0.05, "rad_s", False),
    "actual_root_velocity_mps": ("balance", 0.03, "m_s", False),
    "wheel_derived_velocity_mps": ("balance", 0.03, "m_s", False),
    "root_wheel_velocity_mismatch_mps": ("balance", 0.03, "m_s", False),
    "pitch_reference_rad": ("balance", 0.02, "rad", False),
    "total_pitch_reference_rad": ("balance", 0.02, "rad", False),
    "common_wheel_action": ("balance", 0.05, "normalized", False),
    "base_xy_error_m": ("base", 0.01, "m", False),
    "base_yaw_error_deg": ("base", 1.0, "deg", False),
    "actual_base_x_m": ("base", 0.01, "m", False),
    "actual_base_y_m": ("base", 0.01, "m", False),
    "actual_base_yaw_rad": ("base", math.radians(1.0), "rad", True),
    "camera_position_error_x_m": ("camera", 0.01, "m", False),
    "camera_position_error_y_m": ("camera", 0.01, "m", False),
    "camera_position_error_z_m": ("camera", 0.01, "m", False),
    "actual_camera_x_m": ("camera", 0.01, "m", False),
    "actual_camera_y_m": ("camera", 0.01, "m", False),
    "actual_camera_z_m": ("camera", 0.01, "m", False),
    "camera_lever_correction_x_m": ("camera", 0.01, "m", False),
    "camera_lever_correction_y_m": ("camera", 0.01, "m", False),
    "riser_error_m": ("riser", 0.003, "m", False),
    "proxy_error_x_deg": ("gimbal", 0.5, "deg", False),
    "proxy_error_y_deg": ("gimbal", 0.5, "deg", False),
    "proxy_error_z_deg": ("gimbal", 0.5, "deg", False),
    "proxy_velocity_x_deg_s": ("gimbal", 5.0, "deg_s", False),
    "proxy_velocity_y_deg_s": ("gimbal", 5.0, "deg_s", False),
    "proxy_velocity_z_deg_s": ("gimbal", 5.0, "deg_s", False),
    "effective_velocity_reference_mps": ("command", 0.03, "m_s", False),
    "phase_feedforward_v_mps": ("command", 0.03, "m_s", False),
    "phase_feedforward_wz_rad_s": ("command", 0.05, "rad_s", False),
    "vx_reference_mps": ("command", 0.03, "m_s", False),
    "wz_reference_rad_s": ("command", 0.05, "rad_s", False),
}


VECTOR_FIELDS = {
    "actual_base_xy_yaw": ("actual_base_x_m", "actual_base_y_m", "actual_base_yaw_rad"),
    "camera_position_error_xyz_m": (
        "camera_position_error_x_m",
        "camera_position_error_y_m",
        "camera_position_error_z_m",
    ),
    "actual_camera_position_world_m": (
        "actual_camera_x_m",
        "actual_camera_y_m",
        "actual_camera_z_m",
    ),
    "camera_lever_arm_correction_xy_m": (
        "camera_lever_correction_x_m",
        "camera_lever_correction_y_m",
    ),
    "proxy_signed_error_deg": (
        "proxy_error_x_deg",
        "proxy_error_y_deg",
        "proxy_error_z_deg",
    ),
    "proxy_velocity_deg_s": (
        "proxy_velocity_x_deg_s",
        "proxy_velocity_y_deg_s",
        "proxy_velocity_z_deg_s",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def flatten_trace(rows: list[dict]) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {
        "phase_time_s": np.asarray([row["phase_time_s"] for row in rows], dtype=np.float64),
        "elapsed_s": np.asarray([row["elapsed_s"] for row in rows], dtype=np.float64),
        "position_error_m": np.asarray(
            [row["position_error_m"] for row in rows], dtype=np.float64
        ),
    }
    for name in FIELD_SPECS:
        if name in rows[0]:
            values[name] = np.asarray([row[name] for row in rows], dtype=np.float64)
    for source, names in VECTOR_FIELDS.items():
        array = np.asarray([row[source] for row in rows], dtype=np.float64)
        if array.shape != (len(rows), len(names)):
            raise ValueError(f"trace vector shape mismatch: {source}")
        for index, name in enumerate(names):
            values[name] = array[:, index]
    missing = sorted(set(FIELD_SPECS) - set(values))
    if missing:
        raise ValueError(f"trace fields are missing: {missing}")
    if np.any(np.diff(values["phase_time_s"]) <= 0.0):
        raise ValueError("trace phase time must be strictly increasing")
    return values


def wrapped_delta(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(first - second), np.cos(first - second))


def align_teacher_to_learned(
    teacher: dict[str, np.ndarray], learned: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    phase = learned["phase_time_s"]
    if phase[0] < teacher["phase_time_s"][0] or phase[-1] > teacher["phase_time_s"][-1]:
        raise ValueError("learned trace lies outside teacher phase coverage")
    aligned = {"phase_time_s": phase.copy()}
    for name, values in teacher.items():
        if name != "phase_time_s":
            aligned[name] = np.interp(phase, teacher["phase_time_s"], values)
    return aligned


def safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def analyze_fields(
    teacher: dict[str, np.ndarray], learned: dict[str, np.ndarray]
) -> tuple[list[dict], dict[str, dict], int, np.ndarray]:
    aligned = align_teacher_to_learned(teacher, learned)
    excess = learned["position_error_m"] - aligned["position_error_m"]
    peak_index = int(np.argmax(learned["position_error_m"]))
    rows = []
    group_series: dict[str, list[np.ndarray]] = {}
    for name, (group, floor, unit, angle) in FIELD_SPECS.items():
        delta = (
            wrapped_delta(learned[name], aligned[name])
            if angle
            else learned[name] - aligned[name]
        )
        scale = max(float(np.std(aligned[name])), floor)
        normalized = np.abs(delta) / scale
        pre_peak = normalized[: peak_index + 1]
        crossings = np.flatnonzero(pre_peak >= 1.0)
        onset_index = int(crossings[0]) if len(crossings) else None
        rows.append(
            {
                "field": name,
                "group": group,
                "unit": unit,
                "normalization_floor": floor,
                "teacher_std": float(np.std(aligned[name])),
                "delta_rms": float(np.sqrt(np.mean(np.square(delta)))),
                "delta_abs_max": float(np.max(np.abs(delta))),
                "normalized_delta_rms": float(np.sqrt(np.mean(np.square(normalized)))),
                "normalized_delta_pre_peak_max": float(np.max(pre_peak)),
                "delta_at_tracking_peak": float(delta[peak_index]),
                "abs_delta_correlation_with_positive_excess_error": safe_correlation(
                    np.abs(delta), np.maximum(excess, 0.0)
                ),
                "onset_phase_time_s": (
                    None if onset_index is None else float(learned["phase_time_s"][onset_index])
                ),
                "lead_time_to_tracking_peak_s": (
                    None
                    if onset_index is None
                    else float(
                        learned["phase_time_s"][peak_index]
                        - learned["phase_time_s"][onset_index]
                    )
                ),
            }
        )
        group_series.setdefault(group, []).append(normalized)
    groups = {}
    for group, series in group_series.items():
        envelope = np.max(np.stack(series), axis=0)
        crossings = np.flatnonzero(envelope[: peak_index + 1] >= 1.0)
        onset = int(crossings[0]) if len(crossings) else None
        groups[group] = {
            "normalized_envelope_rms": float(np.sqrt(np.mean(np.square(envelope)))),
            "normalized_envelope_pre_peak_max": float(np.max(envelope[: peak_index + 1])),
            "correlation_with_positive_excess_error": safe_correlation(
                envelope, np.maximum(excess, 0.0)
            ),
            "onset_phase_time_s": (
                None if onset is None else float(learned["phase_time_s"][onset])
            ),
            "lead_time_to_tracking_peak_s": (
                None
                if onset is None
                else float(learned["phase_time_s"][peak_index] - learned["phase_time_s"][onset])
            ),
            "normalized_envelope": envelope,
        }
    rows.sort(key=lambda item: item["normalized_delta_pre_peak_max"], reverse=True)
    return rows, groups, peak_index, excess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-gate", type=Path, required=True)
    parser.add_argument("--learned-gate", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument("--expected-learned-sha256", required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--case", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def result_item(path: Path, case: int) -> tuple[dict, dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    matches = [item for item in document.get("results", []) if item.get("case") == case]
    if document.get("cases") != [case] or len(matches) != 1:
        raise ValueError(f"gate does not contain exactly case {case}: {path}")
    return document, matches[0]


def main() -> int:
    args = parse_args()
    for path, expected in (
        (args.teacher_gate, args.expected_teacher_sha256),
        (args.learned_gate, args.expected_learned_sha256),
        (args.dataset, args.expected_dataset_sha256),
    ):
        if sha256(path) != expected:
            raise ValueError(f"input identity mismatch: {path}")
    if args.output_dir.exists():
        raise ValueError("refusing to overwrite diagnosis output")
    teacher_document, teacher_item = result_item(args.teacher_gate, args.case)
    learned_document, learned_item = result_item(args.learned_gate, args.case)
    source_checks = {
        "same_tracking_profile": teacher_document.get("tracking_profile")
        == learned_document.get("tracking_profile"),
        "same_execution_duration": teacher_item.get("execution_duration_s")
        == learned_item.get("execution_duration_s"),
        "teacher_source": teacher_document.get("trajectory_command_source")
        == "deterministic_teacher",
        "learned_source": learned_document.get("trajectory_command_source")
        == "torchscript_residual_policy",
        "both_dynamic_pass": teacher_item.get("dynamic_quality_passed") is True
        and learned_item.get("dynamic_quality_passed") is True,
        "runtime_closed": teacher_document.get("training_started") is False
        and learned_document.get("training_started") is False,
        "ppo_closed": teacher_document.get("ppo_authorized") is False
        and learned_document.get("ppo_authorized") is False,
    }
    if not all(source_checks.values()):
        raise ValueError(f"rollout comparison contract failed: {source_checks}")
    with np.load(args.dataset, allow_pickle=False) as dataset:
        metadata = json.loads(str(dataset["metadata_json"].item()))
        mask = dataset["case_ids"] == args.case
        if args.case not in metadata["split_cases"]["validation"] or not np.any(mask):
            raise ValueError("diagnostic case is not in validation split")
        teacher_dataset_rows = int(np.sum(mask))
    teacher = flatten_trace(teacher_item["trace"])
    learned = flatten_trace(learned_item["trace"])
    field_rows, groups, peak_index, excess = analyze_fields(teacher, learned)
    ranked_groups = sorted(
        groups,
        key=lambda name: groups[name]["normalized_envelope_pre_peak_max"],
        reverse=True,
    )
    dominant_group = ranked_groups[0]
    peak_phase = float(learned["phase_time_s"][peak_index])
    peak_error = float(learned["position_error_m"][peak_index])
    aligned_teacher = align_teacher_to_learned(teacher, learned)
    args.output_dir.mkdir(parents=True)
    csv_path = args.output_dir / "case_0004_coarse_trace_alignment.csv"
    group_names = sorted(groups)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "phase_time_s",
                "learned_position_error_m",
                "teacher_position_error_m",
                "excess_position_error_m",
                *[f"{name}_normalized_envelope" for name in group_names],
            ],
        )
        writer.writeheader()
        for index, phase in enumerate(learned["phase_time_s"]):
            writer.writerow(
                {
                    "phase_time_s": phase,
                    "learned_position_error_m": learned["position_error_m"][index],
                    "teacher_position_error_m": aligned_teacher["position_error_m"][index],
                    "excess_position_error_m": excess[index],
                    **{
                        f"{name}_normalized_envelope": groups[name][
                            "normalized_envelope"
                        ][index]
                        for name in group_names
                    },
                }
            )
    plot_path = args.output_dir / "case_0004_coarse_covariate_shift.png"
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    phase = learned["phase_time_s"]
    axes[0].plot(phase, learned["position_error_m"], label="masked policy")
    axes[0].plot(phase, aligned_teacher["position_error_m"], label="teacher")
    axes[0].axvline(peak_phase, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("position error (m)")
    axes[0].legend()
    for name in ranked_groups:
        axes[1].plot(phase, groups[name]["normalized_envelope"], label=name)
    axes[1].axhline(1.0, color="black", linestyle=":", linewidth=1)
    axes[1].set_ylabel("normalized group envelope")
    axes[1].legend(ncol=4)
    for row in field_rows[:5]:
        name = row["field"]
        _, floor, _, angle = FIELD_SPECS[name]
        delta = (
            wrapped_delta(learned[name], aligned_teacher[name])
            if angle
            else learned[name] - aligned_teacher[name]
        )
        scale = max(float(np.std(aligned_teacher[name])), floor)
        axes[2].plot(phase, delta / scale, label=name)
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_ylabel("normalized learned-teacher delta")
    axes[2].set_xlabel("execution phase time (s)")
    axes[2].legend(ncol=2)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    for metrics in groups.values():
        metrics.pop("normalized_envelope")
    report = {
        "schema": "cinebotrl_two_wheel_riser_coarse_physical_covariate_shift_v1",
        "case": args.case,
        "split": "validation",
        "inputs": {
            "teacher_gate": identity(args.teacher_gate),
            "learned_gate": identity(args.learned_gate),
            "dataset": identity(args.dataset),
        },
        "source_checks": source_checks,
        "trace_contract": {
            "alignment": "linear_interpolation_by_execution_phase_time_v1",
            "teacher_trace_rows": len(teacher["phase_time_s"]),
            "learned_trace_rows": len(learned["phase_time_s"]),
            "teacher_dataset_policy_rate_rows": teacher_dataset_rows,
            "trace_nominal_resolution_hz": 1.0,
            "full_policy_observation_count": len(OBSERVATION_NAMES),
            "full_learned_policy_observations_available": False,
            "lookahead_observations_available": False,
            "previous_action_observations_available": False,
            "coarse_physical_groups_available": sorted(groups),
        },
        "tracking": {
            "learned_position_p95_m": learned_item["position_error_p95_m"],
            "teacher_position_p95_m": teacher_item["position_error_p95_m"],
            "teacher_relative_budget_m": teacher_item["position_error_p95_m"] * 1.05,
            "trace_peak_phase_time_s": peak_phase,
            "trace_peak_position_error_m": peak_error,
            "trace_peak_excess_position_error_m": float(excess[peak_index]),
        },
        "dominant_pre_peak_group": dominant_group,
        "ranked_groups": ranked_groups,
        "group_metrics": groups,
        "ranked_field_metrics": field_rows,
        "classification": "coarse_physical_state_covariate_shift_observed",
        "limitations": [
            "The runtime JSON trace is nominally 1 Hz, not policy rate.",
            "The full 65-D learned observation, lookahead vectors, and policy-rate actions were not persisted.",
            "Correlation and onset timing are diagnostic evidence, not causal proof.",
        ],
        "recommended_next_change": (
            "Add trace-only policy-rate logging for the exact 65-D pre-action observation, "
            "policy output, final supervised command, phase time, and tracking error; rerun "
            "teacher and masked case 4 without creating a training dataset."
        ),
        "artifacts": {"alignment_csv": identity(csv_path), "plot": identity(plot_path)},
        "valid_for_training": False,
        "dagger_capture_authorized": False,
        "isaac_launched": False,
        "holdout_opened": False,
        "ppo_authorized": False,
        "ppo_started": False,
        "passed": True,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
