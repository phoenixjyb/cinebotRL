#!/usr/bin/env python3
"""Audit measured riser bench evidence against the provisional hardware gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEASUREMENTS = (
    PROJECT_ROOT
    / "docs/03_training/two_wheel_balance/"
    "RISER_BENCH_MEASUREMENT_TEMPLATE_20260723.json"
)
DEFAULT_PROCUREMENT_AUDIT = (
    PROJECT_ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_hardware_procurement_candidate_v1/summary.json"
)
DEFAULT_VENDOR_SNAPSHOT = (
    PROJECT_ROOT
    / "docs/03_training/two_wheel_balance/"
    "RISER_VENDOR_SPEC_SNAPSHOT_20260723.json"
)

MIN_PRODUCTION_FORCE_MARGIN = 1.15
MIN_CONTINUOUS_TEST_DURATION_S = 1800.0
MIN_DUTY_CYCLE = 0.60
MIN_ACHIEVED_SPEED_MPS = 0.95
MAX_COMMAND_SPEED_MPS = 1.0
MAX_MOTOR_HOUSING_RISE_C = 50.0
MAX_DRIVE_RISE_C = 40.0
MAX_FINAL_THERMAL_SLOPE_C_PER_MIN = 1.0
MAX_DC_BUS_VOLTAGE_V = 65.0
MIN_STOP_REPETITIONS = 10
MAX_STOPPING_DISTANCE_M = 0.12
MIN_BRAKE_HOLD_DURATION_S = 600.0
MAX_BRAKE_HOLD_DISPLACEMENT_M = 0.002
MIN_ANTI_FALL_TESTS = 10
MAX_ANTI_FALL_CATCH_DISTANCE_M = 0.03
CAMERA_HEIGHT_TOLERANCE_M = 0.01


def _identity(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _get(payload: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for key in dotted_path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _finite_number(value: Any, *, minimum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and (minimum is None or numeric >= minimum)


def _at_most(value: Any, limit: float) -> bool:
    return _finite_number(value, minimum=0.0) and float(value) <= limit


def _at_least(value: Any, limit: float) -> bool:
    return _finite_number(value) and float(value) >= limit


def _number_or(value: Any, fallback: float) -> float:
    return float(value) if _finite_number(value) else fallback


NUMERIC_FIELDS = (
    "configuration.complete_moving_mass_kg",
    "configuration.measured_friction_force_n",
    "configuration.counterbalance_force_n",
    "configuration.mechanical_stroke_m",
    "configuration.software_camera_height_min_m",
    "configuration.software_camera_height_max_m",
    "continuous_duty.duration_s",
    "continuous_duty.duty_cycle_fraction",
    "continuous_duty.commanded_speed_mps",
    "continuous_duty.minimum_achieved_speed_mps",
    "continuous_duty.phase_current_rms_a",
    "continuous_duty.phase_current_peak_a",
    "continuous_duty.dc_input_current_rms_a",
    "continuous_duty.dc_bus_voltage_max_v",
    "continuous_duty.ambient_temperature_c",
    "continuous_duty.motor_housing_temperature_max_c",
    "continuous_duty.drive_temperature_max_c",
    "continuous_duty.final_thermal_slope_c_per_min",
    "emergency_stop.repetitions",
    "emergency_stop.initial_speed_abs_min_mps",
    "emergency_stop.worst_stopping_distance_m",
    "emergency_stop.phase_current_peak_a",
    "emergency_stop.dc_bus_voltage_max_v",
    "power_loss_safety.motor_brake_hold_duration_s",
    "power_loss_safety.motor_brake_hold_displacement_m",
    "power_loss_safety.independent_anti_fall_test_count",
    "power_loss_safety.independent_anti_fall_worst_catch_distance_m",
)

BOOLEAN_FIELDS = (
    "instrumentation.force_calibration_valid",
    "instrumentation.current_calibration_valid",
    "instrumentation.temperature_calibration_valid",
    "instrumentation.position_calibration_valid",
    "supplier_evidence.vertical_mobile_axis_duty_approved",
    "supplier_evidence.gearbox_continuous_speed_approved",
    "supplier_evidence.gearbox_emergency_braking_torque_approved",
    "supplier_evidence.belt_tooth_jump_margin_approved",
    "continuous_duty.no_fault_or_tooth_jump",
    "emergency_stop.no_fault_or_position_loss",
    "power_loss_safety.independent_anti_fall_installed",
    "power_loss_safety.independent_anti_fall_no_damage",
    "limits.lower_hard_limit_passed",
    "limits.upper_hard_limit_passed",
    "limits.lower_absorbing_end_stop_passed",
    "limits.upper_absorbing_end_stop_passed",
    "limits.safety_rated_power_removal_passed",
)

HASH_FIELDS = (
    "evidence.raw_log_sha256",
    "evidence.force_calibration_record_sha256",
    "evidence.current_calibration_record_sha256",
    "evidence.temperature_calibration_record_sha256",
    "evidence.position_calibration_record_sha256",
    "evidence.supplier_approval_package_sha256",
    "evidence.safety_test_video_sha256",
)


def _sha256_identity(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def build_report(
    measurements: Mapping[str, Any],
    procurement: Mapping[str, Any],
    vendor: Mapping[str, Any],
) -> dict[str, Any]:
    missing_fields = [
        field
        for field in NUMERIC_FIELDS
        if not _finite_number(_get(measurements, field), minimum=0.0)
    ]
    missing_fields.extend(
        field
        for field in BOOLEAN_FIELDS
        if not isinstance(_get(measurements, field), bool)
    )
    missing_fields.extend(
        field for field in HASH_FIELDS if not _sha256_identity(_get(measurements, field))
    )
    measurement_complete = not missing_fields

    mass = _get(measurements, "configuration.complete_moving_mass_kg")
    friction = _get(measurements, "configuration.measured_friction_force_n")
    counterbalance = _get(measurements, "configuration.counterbalance_force_n")
    if all(_finite_number(value, minimum=0.0) for value in (mass, friction, counterbalance)):
        emergency_design_force_n = 2.0 * max(
            0.0,
            float(mass) * (9.81 + 5.0) + float(friction) - float(counterbalance),
        )
        rated_force_n = float(procurement["calculated"]["rated_linear_force_n"])
        measured_force_margin = (
            rated_force_n / emergency_design_force_n
            if emergency_design_force_n > 0.0
            else math.inf
        )
    else:
        emergency_design_force_n = None
        measured_force_margin = None

    motor = vendor["leadshine_motor"]
    drive = vendor["leadshine_drive"]
    motor_rise = None
    drive_rise = None
    ambient = _get(measurements, "continuous_duty.ambient_temperature_c")
    motor_temperature = _get(
        measurements, "continuous_duty.motor_housing_temperature_max_c"
    )
    drive_temperature = _get(
        measurements, "continuous_duty.drive_temperature_max_c"
    )
    if _finite_number(ambient) and _finite_number(motor_temperature):
        motor_rise = float(motor_temperature) - float(ambient)
    if _finite_number(ambient) and _finite_number(drive_temperature):
        drive_rise = float(drive_temperature) - float(ambient)

    checks = {
        "measurement_schema_matches": measurements.get("schema")
        == "cinebotrl_two_wheel_riser_bench_measurements_v1",
        "test_id_is_not_template": isinstance(measurements.get("test_id"), str)
        and bool(measurements["test_id"].strip())
        and not measurements["test_id"].startswith("UNMEASURED_TEMPLATE"),
        "measurement_complete": measurement_complete,
        "raw_and_supporting_evidence_hashes_present": all(
            _sha256_identity(_get(measurements, field)) for field in HASH_FIELDS
        ),
        "candidate_motor_matches": _get(measurements, "candidate.motor_model")
        == motor["model"],
        "candidate_drive_matches": _get(measurements, "candidate.drive_model")
        == drive["model"],
        "candidate_reduction_is_3_to_1": math.isclose(
            _number_or(_get(measurements, "candidate.reduction_ratio"), 0.0), 3.0
        ),
        "candidate_lead_is_70mm_per_rev": math.isclose(
            _number_or(
                _get(measurements, "candidate.linear_lead_m_per_rev"), 0.0
            ),
            0.07,
        ),
        "all_instrumentation_calibrated": all(
            _get(measurements, field) is True
            for field in BOOLEAN_FIELDS
            if field.startswith("instrumentation.")
        ),
        "camera_minimum_is_0_60m": _at_most(
            _get(measurements, "configuration.software_camera_height_min_m"),
            0.60 + CAMERA_HEIGHT_TOLERANCE_M,
        )
        and _at_least(
            _get(measurements, "configuration.software_camera_height_min_m"),
            0.60 - CAMERA_HEIGHT_TOLERANCE_M,
        ),
        "camera_maximum_is_1_80m": _at_most(
            _get(measurements, "configuration.software_camera_height_max_m"),
            1.80 + CAMERA_HEIGHT_TOLERANCE_M,
        )
        and _at_least(
            _get(measurements, "configuration.software_camera_height_max_m"),
            1.80 - CAMERA_HEIGHT_TOLERANCE_M,
        ),
        "mechanical_stroke_is_at_least_1_50m": _at_least(
            _get(measurements, "configuration.mechanical_stroke_m"), 1.50
        ),
        "measured_emergency_force_margin_at_least_15_percent": (
            measured_force_margin is not None
            and measured_force_margin >= MIN_PRODUCTION_FORCE_MARGIN
        ),
        "supplier_vertical_and_transmission_approvals_present": all(
            _get(measurements, field) is True
            for field in BOOLEAN_FIELDS
            if field.startswith("supplier_evidence.")
        ),
        "continuous_test_duration_passed": _at_least(
            _get(measurements, "continuous_duty.duration_s"),
            MIN_CONTINUOUS_TEST_DURATION_S,
        ),
        "continuous_test_duty_cycle_passed": _at_least(
            _get(measurements, "continuous_duty.duty_cycle_fraction"),
            MIN_DUTY_CYCLE,
        ),
        "one_mps_speed_passed": _at_most(
            _get(measurements, "continuous_duty.commanded_speed_mps"),
            MAX_COMMAND_SPEED_MPS,
        )
        and _at_least(
            _get(measurements, "continuous_duty.commanded_speed_mps"),
            MAX_COMMAND_SPEED_MPS,
        )
        and _at_least(
            _get(measurements, "continuous_duty.minimum_achieved_speed_mps"),
            MIN_ACHIEVED_SPEED_MPS,
        ),
        "continuous_phase_current_passed": _at_most(
            _get(measurements, "continuous_duty.phase_current_rms_a"),
            float(motor["rated_current_a"]),
        ),
        "dc_input_current_measured_for_battery_branch": _finite_number(
            _get(measurements, "continuous_duty.dc_input_current_rms_a"), minimum=0.0
        ),
        "peak_phase_current_passed": _at_most(
            max(
                _number_or(
                    _get(measurements, "continuous_duty.phase_current_peak_a"),
                    math.inf,
                ),
                _number_or(
                    _get(measurements, "emergency_stop.phase_current_peak_a"),
                    math.inf,
                ),
            ),
            min(float(motor["peak_current_a"]), float(drive["peak_current_apeak"])),
        ),
        "regenerative_bus_voltage_passed": _at_most(
            max(
                _number_or(
                    _get(measurements, "continuous_duty.dc_bus_voltage_max_v"),
                    math.inf,
                ),
                _number_or(
                    _get(measurements, "emergency_stop.dc_bus_voltage_max_v"),
                    math.inf,
                ),
            ),
            MAX_DC_BUS_VOLTAGE_V,
        ),
        "motor_temperature_rise_passed": motor_rise is not None
        and 0.0 <= motor_rise <= MAX_MOTOR_HOUSING_RISE_C,
        "drive_temperature_rise_passed": drive_rise is not None
        and 0.0 <= drive_rise <= MAX_DRIVE_RISE_C,
        "thermal_steady_state_passed": _at_most(
            _get(measurements, "continuous_duty.final_thermal_slope_c_per_min"),
            MAX_FINAL_THERMAL_SLOPE_C_PER_MIN,
        ),
        "continuous_run_had_no_fault": _get(
            measurements, "continuous_duty.no_fault_or_tooth_jump"
        )
        is True,
        "emergency_stop_repetition_and_distance_passed": _at_least(
            _get(measurements, "emergency_stop.repetitions"), MIN_STOP_REPETITIONS
        )
        and _at_least(
            _get(measurements, "emergency_stop.initial_speed_abs_min_mps"),
            MIN_ACHIEVED_SPEED_MPS,
        )
        and _at_most(
            _get(measurements, "emergency_stop.worst_stopping_distance_m"),
            MAX_STOPPING_DISTANCE_M,
        )
        and _get(measurements, "emergency_stop.no_fault_or_position_loss") is True,
        "motor_brake_static_hold_passed": _at_least(
            _get(measurements, "power_loss_safety.motor_brake_hold_duration_s"),
            MIN_BRAKE_HOLD_DURATION_S,
        )
        and _at_most(
            _get(measurements, "power_loss_safety.motor_brake_hold_displacement_m"),
            MAX_BRAKE_HOLD_DISPLACEMENT_M,
        ),
        "independent_anti_fall_passed": _get(
            measurements, "power_loss_safety.independent_anti_fall_installed"
        )
        is True
        and _at_least(
            _get(measurements, "power_loss_safety.independent_anti_fall_test_count"),
            MIN_ANTI_FALL_TESTS,
        )
        and _at_most(
            _get(
                measurements,
                "power_loss_safety.independent_anti_fall_worst_catch_distance_m",
            ),
            MAX_ANTI_FALL_CATCH_DISTANCE_M,
        )
        and _get(
            measurements, "power_loss_safety.independent_anti_fall_no_damage"
        )
        is True,
        "limits_and_safety_power_removal_passed": all(
            _get(measurements, field) is True
            for field in BOOLEAN_FIELDS
            if field.startswith("limits.")
        ),
    }
    passed = all(checks.values())
    if not measurement_complete:
        decision = "collect_complete_calibrated_bench_measurements"
    elif not checks["measured_emergency_force_margin_at_least_15_percent"]:
        decision = "resize_to_750w_class_or_reduce_load_then_repeat_full_bench_gate"
    elif not all(
        checks[name]
        for name in (
            "supplier_vertical_and_transmission_approvals_present",
            "motor_brake_static_hold_passed",
            "independent_anti_fall_passed",
            "limits_and_safety_power_removal_passed",
        )
    ):
        decision = "complete_mechanical_and_functional_safety_architecture"
    elif not passed:
        decision = "repair_failed_bench_gate_and_repeat_without_threshold_relaxation"
    else:
        decision = "400w_candidate_ready_for_production_design_review"

    return {
        "schema": "cinebotrl_two_wheel_riser_bench_acceptance_audit_v1",
        "threshold_provenance": (
            "project_engineering_gate_v1_not_vendor_product_rating_or_safety_certification"
        ),
        "missing_or_invalid_measurement_fields": sorted(missing_fields),
        "thresholds": {
            "minimum_production_force_margin_ratio": MIN_PRODUCTION_FORCE_MARGIN,
            "minimum_continuous_test_duration_s": MIN_CONTINUOUS_TEST_DURATION_S,
            "minimum_duty_cycle_fraction": MIN_DUTY_CYCLE,
            "minimum_achieved_speed_mps": MIN_ACHIEVED_SPEED_MPS,
            "maximum_command_speed_mps": MAX_COMMAND_SPEED_MPS,
            "maximum_motor_housing_temperature_rise_c": MAX_MOTOR_HOUSING_RISE_C,
            "maximum_drive_temperature_rise_c": MAX_DRIVE_RISE_C,
            "maximum_final_thermal_slope_c_per_min": (
                MAX_FINAL_THERMAL_SLOPE_C_PER_MIN
            ),
            "maximum_dc_bus_voltage_v": MAX_DC_BUS_VOLTAGE_V,
            "minimum_emergency_stop_repetitions": MIN_STOP_REPETITIONS,
            "maximum_stopping_distance_m": MAX_STOPPING_DISTANCE_M,
            "minimum_motor_brake_hold_duration_s": MIN_BRAKE_HOLD_DURATION_S,
            "maximum_motor_brake_hold_displacement_m": (
                MAX_BRAKE_HOLD_DISPLACEMENT_M
            ),
            "minimum_independent_anti_fall_tests": MIN_ANTI_FALL_TESTS,
            "maximum_independent_anti_fall_catch_distance_m": (
                MAX_ANTI_FALL_CATCH_DISTANCE_M
            ),
            "camera_height_tolerance_m": CAMERA_HEIGHT_TOLERANCE_M,
        },
        "calculated": {
            "emergency_design_force_n": emergency_design_force_n,
            "rated_linear_force_n": procurement["calculated"]["rated_linear_force_n"],
            "measured_force_margin_ratio": measured_force_margin,
            "required_force_margin_ratio": MIN_PRODUCTION_FORCE_MARGIN,
            "motor_housing_temperature_rise_c": motor_rise,
            "drive_temperature_rise_c": drive_rise,
            "measured_dc_input_current_rms_a": _get(
                measurements, "continuous_duty.dc_input_current_rms_a"
            ),
        },
        "checks": checks,
        "decision": decision,
        "passed": passed,
        "ready_for_production_design_review": passed,
        "valid_for_production_procurement": False,
        "valid_for_hardware_transfer": False,
        "valid_for_training": False,
        "runtime_authorized": False,
        "gpu_work_started": False,
        "bc_authorized": False,
        "ppo_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, default=DEFAULT_MEASUREMENTS)
    parser.add_argument(
        "--procurement-audit", type=Path, default=DEFAULT_PROCUREMENT_AUDIT
    )
    parser.add_argument("--vendor-snapshot", type=Path, default=DEFAULT_VENDOR_SNAPSHOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    measurements = json.loads(args.measurements.read_text(encoding="utf-8"))
    procurement = json.loads(args.procurement_audit.read_text(encoding="utf-8"))
    vendor = json.loads(args.vendor_snapshot.read_text(encoding="utf-8"))
    report = build_report(measurements, procurement, vendor)
    report["inputs"] = {
        "measurements": _identity(args.measurements),
        "procurement_audit": _identity(args.procurement_audit),
        "vendor_snapshot": _identity(args.vendor_snapshot),
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite bench audit: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
