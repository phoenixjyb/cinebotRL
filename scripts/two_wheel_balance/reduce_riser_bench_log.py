#!/usr/bin/env python3
"""Reduce calibrated riser bench telemetry into auditable numeric measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "cinebotrl_two_wheel_riser_bench_log_reduction_v1"
CANDIDATE_BOUND_SCHEMA = "cinebotrl_two_wheel_riser_bench_log_reduction_v2"
CANDIDATE_PROFILES = {
    "leadshine_400w_engineering_sample_v1",
    "leadshine_750w_production_candidate_v1",
}
EXPECTED_COLUMNS = (
    "time_s",
    "phase",
    "trial_id",
    "active_command",
    "steady_state",
    "stop_trigger_event",
    "commanded_velocity_mps",
    "measured_velocity_mps",
    "position_m",
    "phase_current_a",
    "dc_input_current_a",
    "dc_bus_voltage_v",
    "ambient_temperature_c",
    "motor_housing_temperature_c",
    "drive_temperature_c",
    "fault_active",
    "tooth_jump_detected",
    "position_loss_detected",
)
NUMERIC_COLUMNS = (
    "time_s",
    "commanded_velocity_mps",
    "measured_velocity_mps",
    "position_m",
    "phase_current_a",
    "dc_input_current_a",
    "dc_bus_voltage_v",
    "ambient_temperature_c",
    "motor_housing_temperature_c",
    "drive_temperature_c",
)
BOOLEAN_COLUMNS = (
    "active_command",
    "steady_state",
    "stop_trigger_event",
    "fault_active",
    "tooth_jump_detected",
    "position_loss_detected",
)
PHASES = {"continuous", "emergency_stop"}
STOPPED_SPEED_MPS = 0.02
FINAL_THERMAL_WINDOW_S = 300.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_float(row: dict[str, str], name: str, row_number: int) -> float:
    try:
        value = float(row[name])
    except (TypeError, ValueError) as error:
        raise ValueError(f"row {row_number} has invalid {name}") from error
    if not math.isfinite(value):
        raise ValueError(f"row {row_number} has non-finite {name}")
    return value


def _parse_int(row: dict[str, str], name: str, row_number: int) -> int:
    try:
        value = int(row[name])
    except (TypeError, ValueError) as error:
        raise ValueError(f"row {row_number} has invalid {name}") from error
    return value


def _parse_bool(row: dict[str, str], name: str, row_number: int) -> bool:
    value = _parse_int(row, name, row_number)
    if value not in (0, 1):
        raise ValueError(f"row {row_number} has non-boolean {name}")
    return bool(value)


def load_log(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError("riser bench log columns do not match the v1 contract")
        rows = list(reader)
    if not rows:
        raise ValueError("riser bench log has no samples")

    payload: dict[str, Any] = {
        name: [] for name in EXPECTED_COLUMNS
    }
    for row_number, row in enumerate(rows, start=2):
        phase = row["phase"]
        if phase not in PHASES:
            raise ValueError(f"row {row_number} has invalid phase")
        payload["phase"].append(phase)
        trial_id = _parse_int(row, "trial_id", row_number)
        if trial_id < 0:
            raise ValueError(f"row {row_number} has negative trial_id")
        payload["trial_id"].append(trial_id)
        for name in NUMERIC_COLUMNS:
            payload[name].append(_parse_float(row, name, row_number))
        for name in BOOLEAN_COLUMNS:
            payload[name].append(_parse_bool(row, name, row_number))

    arrays = {
        "phase": np.asarray(payload["phase"], dtype="U16"),
        "trial_id": np.asarray(payload["trial_id"], dtype=np.int64),
    }
    arrays.update(
        {
            name: np.asarray(payload[name], dtype=np.float64)
            for name in NUMERIC_COLUMNS
        }
    )
    arrays.update(
        {
            name: np.asarray(payload[name], dtype=bool)
            for name in BOOLEAN_COLUMNS
        }
    )
    if not np.all(np.diff(arrays["time_s"]) > 0.0):
        raise ValueError("riser bench log time must increase strictly")
    if np.any(arrays["phase_current_a"] < 0.0) or np.any(
        arrays["dc_input_current_a"] < 0.0
    ):
        raise ValueError("riser bench log currents must be absolute non-negative values")
    return arrays


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _thermal_slope_c_per_minute(
    time_s: np.ndarray, temperature_c: np.ndarray
) -> float:
    duration = float(time_s[-1] - time_s[0])
    window = min(FINAL_THERMAL_WINDOW_S, duration)
    mask = time_s >= time_s[-1] - window
    if np.count_nonzero(mask) < 3 or time_s[mask][-1] - time_s[mask][0] <= 0.0:
        raise ValueError("continuous phase lacks a final thermal slope window")
    slope_c_per_s = float(np.polyfit(time_s[mask], temperature_c[mask], 1)[0])
    return slope_c_per_s * 60.0


def _continuous_metrics(data: dict[str, np.ndarray]) -> dict[str, object]:
    mask = data["phase"] == "continuous"
    indices = np.flatnonzero(mask)
    if len(indices) < 3 or not np.array_equal(
        indices, np.arange(indices[0], indices[-1] + 1)
    ):
        raise ValueError("continuous phase must be one contiguous block")
    time_s = data["time_s"][mask]
    duration_s = float(time_s[-1] - time_s[0])
    if duration_s <= 0.0:
        raise ValueError("continuous phase duration must be positive")
    active = data["active_command"][mask]
    steady = data["steady_state"][mask] & active
    if not np.any(active) or np.count_nonzero(steady) < 3:
        raise ValueError("continuous phase lacks active steady-state samples")
    intervals = np.diff(time_s)
    duty_cycle = float(np.sum(intervals[active[:-1]]) / duration_s)
    motor_slope = _thermal_slope_c_per_minute(
        time_s, data["motor_housing_temperature_c"][mask]
    )
    drive_slope = _thermal_slope_c_per_minute(
        time_s, data["drive_temperature_c"][mask]
    )
    return {
        "duration_s": duration_s,
        "duty_cycle_fraction": duty_cycle,
        "commanded_speed_mps": float(
            np.max(np.abs(data["commanded_velocity_mps"][mask][active]))
        ),
        "minimum_achieved_speed_mps": float(
            np.min(np.abs(data["measured_velocity_mps"][mask][steady]))
        ),
        "phase_current_rms_a": _rms(data["phase_current_a"][mask][active]),
        "phase_current_peak_a": float(np.max(data["phase_current_a"][mask])),
        "dc_input_current_rms_a": _rms(
            data["dc_input_current_a"][mask][active]
        ),
        "dc_bus_voltage_max_v": float(np.max(data["dc_bus_voltage_v"][mask])),
        "ambient_temperature_c": float(
            np.median(data["ambient_temperature_c"][mask])
        ),
        "motor_housing_temperature_max_c": float(
            np.max(data["motor_housing_temperature_c"][mask])
        ),
        "drive_temperature_max_c": float(
            np.max(data["drive_temperature_c"][mask])
        ),
        "final_thermal_slope_c_per_min": max(motor_slope, drive_slope),
        "motor_final_thermal_slope_c_per_min": motor_slope,
        "drive_final_thermal_slope_c_per_min": drive_slope,
        "no_fault_or_tooth_jump": bool(
            not np.any(data["fault_active"][mask])
            and not np.any(data["tooth_jump_detected"][mask])
        ),
    }


def _emergency_stop_metrics(data: dict[str, np.ndarray]) -> dict[str, object]:
    mask = data["phase"] == "emergency_stop"
    if np.count_nonzero(mask) < 3:
        raise ValueError("emergency-stop phase has too few samples")
    trial_ids = np.unique(data["trial_id"][mask])
    if len(trial_ids) == 0 or np.any(trial_ids <= 0):
        raise ValueError("emergency-stop trials need positive trial IDs")
    initial_speeds = []
    stopping_distances = []
    for trial_id in trial_ids:
        trial = np.flatnonzero(mask & (data["trial_id"] == trial_id))
        if not np.array_equal(trial, np.arange(trial[0], trial[-1] + 1)):
            raise ValueError(f"emergency-stop trial {trial_id} is not contiguous")
        trigger_rows = trial[data["stop_trigger_event"][trial]]
        if len(trigger_rows) != 1:
            raise ValueError(
                f"emergency-stop trial {trial_id} needs exactly one trigger"
            )
        trigger = int(trigger_rows[0])
        after = trial[trial > trigger]
        stopped = after[
            np.abs(data["measured_velocity_mps"][after]) <= STOPPED_SPEED_MPS
        ]
        if not len(stopped):
            raise ValueError(f"emergency-stop trial {trial_id} never reaches stop")
        stop = int(stopped[0])
        initial_speeds.append(abs(float(data["measured_velocity_mps"][trigger])))
        stopping_distances.append(
            abs(float(data["position_m"][stop] - data["position_m"][trigger]))
        )
    return {
        "repetitions": int(len(trial_ids)),
        "initial_speed_abs_min_mps": min(initial_speeds),
        "worst_stopping_distance_m": max(stopping_distances),
        "phase_current_peak_a": float(
            np.max(data["phase_current_a"][mask])
        ),
        "dc_bus_voltage_max_v": float(
            np.max(data["dc_bus_voltage_v"][mask])
        ),
        "no_fault_or_position_loss": bool(
            not np.any(data["fault_active"][mask])
            and not np.any(data["position_loss_detected"][mask])
        ),
        "per_trial": [
            {
                "trial_id": int(trial_id),
                "initial_speed_abs_mps": initial_speeds[index],
                "stopping_distance_m": stopping_distances[index],
            }
            for index, trial_id in enumerate(trial_ids)
        ],
    }


def reduce_log(
    path: Path,
    *,
    candidate_profile: str | None = None,
) -> dict[str, object]:
    if candidate_profile is not None and candidate_profile not in CANDIDATE_PROFILES:
        raise ValueError(f"unsupported riser candidate profile: {candidate_profile}")
    data = load_log(path)
    continuous = _continuous_metrics(data)
    emergency = _emergency_stop_metrics(data)
    return {
        "schema": CANDIDATE_BOUND_SCHEMA if candidate_profile else SCHEMA,
        "candidate_profile": candidate_profile,
        "raw_log": {
            "path": path.as_posix(),
            "sha256": sha256(path),
            "sample_count": int(len(data["time_s"])),
            "columns": list(EXPECTED_COLUMNS),
        },
        "continuous_duty": continuous,
        "emergency_stop": emergency,
        "checks": {
            "time_strictly_increasing": True,
            "numeric_values_finite": True,
            "boolean_values_are_zero_or_one": True,
            "continuous_metrics_derived": True,
            "emergency_trials_structurally_complete": True,
            "raw_log_hash_recorded": True,
        },
        "passed": True,
        "valid_for_bench_measurement_numeric_merge": True,
        "valid_for_candidate_bound_bench_merge": candidate_profile is not None,
        "manual_calibration_supplier_and_safety_fields_still_required": True,
        "ready_for_production_design_review": False,
        "valid_for_production_procurement": False,
        "valid_for_hardware_transfer": False,
        "valid_for_training": False,
        "runtime_authorized": False,
        "gpu_work_started": False,
        "bc_authorized": False,
        "ppo_authorized": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-profile", choices=tuple(sorted(CANDIDATE_PROFILES)))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = reduce_log(args.input, candidate_profile=args.candidate_profile)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite bench reduction: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
