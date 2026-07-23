import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/audit_riser_bench_measurements.py"
SPEC = importlib.util.spec_from_file_location("riser_bench_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sources():
    procurement = json.loads(
        MODULE.DEFAULT_PROCUREMENT_AUDIT.read_text(encoding="utf-8")
    )
    vendor = json.loads(MODULE.DEFAULT_VENDOR_SNAPSHOT.read_text(encoding="utf-8"))
    return procurement, vendor


def _production_sources():
    procurement = json.loads(
        MODULE.PRODUCTION_CANDIDATE_AUDIT.read_text(encoding="utf-8")
    )
    vendor = json.loads(
        MODULE.PRODUCTION_VENDOR_SNAPSHOT.read_text(encoding="utf-8")
    )
    return procurement, vendor


def _healthy_measurements() -> dict:
    return {
        "schema": "cinebotrl_two_wheel_riser_bench_measurements_v1",
        "test_id": "synthetic_test_only",
        "candidate": {
            "motor_model": "ELVM6040V48EH-M17-HD",
            "drive_model": "ELD2-CAN7010B",
            "reduction_ratio": 3.0,
            "linear_lead_m_per_rev": 0.07,
        },
        "evidence": {
            "raw_log_sha256": "1" * 64,
            "force_calibration_record_sha256": "2" * 64,
            "current_calibration_record_sha256": "3" * 64,
            "temperature_calibration_record_sha256": "4" * 64,
            "position_calibration_record_sha256": "5" * 64,
            "supplier_approval_package_sha256": "6" * 64,
            "safety_test_video_sha256": "7" * 64,
        },
        "configuration": {
            "complete_moving_mass_kg": 6.5,
            "measured_friction_force_n": 15.0,
            "counterbalance_force_n": 20.0,
            "mechanical_stroke_m": 1.5,
            "software_camera_height_min_m": 0.6,
            "software_camera_height_max_m": 1.8,
        },
        "instrumentation": {
            "force_calibration_valid": True,
            "current_calibration_valid": True,
            "temperature_calibration_valid": True,
            "position_calibration_valid": True,
        },
        "supplier_evidence": {
            "vertical_mobile_axis_duty_approved": True,
            "gearbox_continuous_speed_approved": True,
            "gearbox_emergency_braking_torque_approved": True,
            "belt_tooth_jump_margin_approved": True,
        },
        "continuous_duty": {
            "duration_s": 1800.0,
            "duty_cycle_fraction": 0.6,
            "commanded_speed_mps": 1.0,
            "minimum_achieved_speed_mps": 0.97,
            "phase_current_rms_a": 8.0,
            "phase_current_peak_a": 20.0,
            "dc_input_current_rms_a": 12.0,
            "dc_bus_voltage_max_v": 58.0,
            "ambient_temperature_c": 25.0,
            "motor_housing_temperature_max_c": 66.0,
            "drive_temperature_max_c": 57.0,
            "final_thermal_slope_c_per_min": 0.5,
            "no_fault_or_tooth_jump": True,
        },
        "emergency_stop": {
            "repetitions": 10,
            "initial_speed_abs_min_mps": 0.96,
            "worst_stopping_distance_m": 0.1,
            "phase_current_peak_a": 25.0,
            "dc_bus_voltage_max_v": 62.0,
            "no_fault_or_position_loss": True,
        },
        "power_loss_safety": {
            "motor_brake_hold_duration_s": 600.0,
            "motor_brake_hold_displacement_m": 0.001,
            "independent_anti_fall_installed": True,
            "independent_anti_fall_test_count": 10,
            "independent_anti_fall_worst_catch_distance_m": 0.02,
            "independent_anti_fall_no_damage": True,
        },
        "limits": {
            "lower_hard_limit_passed": True,
            "upper_hard_limit_passed": True,
            "lower_absorbing_end_stop_passed": True,
            "upper_absorbing_end_stop_passed": True,
            "safety_rated_power_removal_passed": True,
        },
    }


def _report(measurements: dict) -> dict:
    return MODULE.build_report(measurements, *_sources())


def _production_report(measurements: dict) -> dict:
    return MODULE.build_report(measurements, *_production_sources())


def _healthy_production_measurements() -> dict:
    measurements = _healthy_measurements()
    measurements["candidate"].update(
        {
            "motor_model": "ELVM8075V48EH-M17-HD",
            "drive_model": "ELD2-CAN7020B",
            "drive_profile": "leadshine_750w_production_candidate_v1",
        }
    )
    measurements["configuration"].update(
        {
            "complete_moving_mass_kg": 8.0,
            "measured_friction_force_n": 20.0,
            "counterbalance_force_n": 0.0,
        }
    )
    measurements["continuous_duty"].update(
        {
            "phase_current_rms_a": 15.0,
            "phase_current_peak_a": 45.0,
        }
    )
    measurements["emergency_stop"]["phase_current_peak_a"] = 55.0
    return measurements


def test_complete_healthy_measurement_reaches_review_not_procurement() -> None:
    report = _report(_healthy_measurements())
    assert report["passed"] is True
    assert report["ready_for_production_design_review"] is True
    assert report["decision"] == "400w_candidate_ready_for_production_design_review"
    assert report["calculated"]["measured_force_margin_ratio"] > 1.15
    assert report["valid_for_production_procurement"] is False
    assert report["valid_for_hardware_transfer"] is False
    assert report["valid_for_training"] is False
    assert report["thresholds"]["maximum_stopping_distance_m"] == 0.12


def test_750w_candidate_has_a_separate_passing_bench_route() -> None:
    report = _production_report(_healthy_production_measurements())
    assert report["passed"] is True
    assert report["candidate_profile"] == (
        "leadshine_750w_production_candidate_v1"
    )
    assert report["decision"] == "750w_candidate_ready_for_production_design_review"
    assert report["calculated"]["measured_force_margin_ratio"] > 1.9
    assert report["ready_for_production_design_review"] is True
    assert report["valid_for_production_procurement"] is False
    assert report["valid_for_hardware_transfer"] is False
    assert report["valid_for_training"] is False


def test_400w_and_750w_evidence_cannot_cross_candidate_routes() -> None:
    production_on_400w = _report(_healthy_production_measurements())
    engineering_on_750w = _production_report(_healthy_measurements())
    for report in (production_on_400w, engineering_on_750w):
        assert report["checks"]["candidate_motor_matches"] is False
        assert report["checks"]["candidate_drive_matches"] is False
        assert report["checks"]["candidate_profile_matches_source_route"] is False
        assert report["passed"] is False
        assert report["ready_for_production_design_review"] is False


def test_vendor_and_force_calculation_routes_cannot_be_crossed() -> None:
    engineering_procurement, engineering_vendor = _sources()
    production_procurement, production_vendor = _production_sources()
    production_with_engineering_calculation = MODULE.build_report(
        _healthy_production_measurements(),
        engineering_procurement,
        production_vendor,
    )
    engineering_with_production_calculation = MODULE.build_report(
        _healthy_measurements(),
        production_procurement,
        engineering_vendor,
    )
    for report in (
        production_with_engineering_calculation,
        engineering_with_production_calculation,
    ):
        assert report["checks"][
            "candidate_calculation_profile_matches_source_route"
        ] is False
        assert report["passed"] is False


def test_legacy_400w_template_profile_omission_is_candidate_bound() -> None:
    engineering = _report(_healthy_measurements())
    production = _production_report(_healthy_measurements())
    assert engineering["checks"]["candidate_profile_matches_source_route"] is True
    assert production["checks"]["candidate_profile_matches_source_route"] is False
    assert engineering["candidate_profile"] == (
        "leadshine_400w_engineering_sample_v1"
    )


def test_unmeasured_template_fails_closed() -> None:
    measurements = json.loads(
        MODULE.DEFAULT_MEASUREMENTS.read_text(encoding="utf-8")
    )
    report = _report(measurements)
    assert report["passed"] is False
    assert report["checks"]["measurement_complete"] is False
    assert report["missing_or_invalid_measurement_fields"]
    assert report["decision"] == "collect_complete_calibrated_bench_measurements"


def test_8kg_without_counterbalance_selects_resize_decision() -> None:
    measurements = _healthy_measurements()
    measurements["configuration"]["complete_moving_mass_kg"] = 8.0
    measurements["configuration"]["measured_friction_force_n"] = 20.0
    measurements["configuration"]["counterbalance_force_n"] = 0.0
    report = _report(measurements)
    assert report["checks"][
        "measured_emergency_force_margin_at_least_15_percent"
    ] is False
    assert report["decision"].startswith("resize_to_750w_class")


@pytest.mark.parametrize(
    ("section", "field", "bad_value", "check"),
    [
        ("continuous_duty", "dc_bus_voltage_max_v", 66.0, "regenerative_bus_voltage_passed"),
        ("emergency_stop", "worst_stopping_distance_m", 0.121, "emergency_stop_repetition_and_distance_passed"),
        ("continuous_duty", "motor_housing_temperature_max_c", 76.0, "motor_temperature_rise_passed"),
        ("power_loss_safety", "independent_anti_fall_test_count", 9, "independent_anti_fall_passed"),
    ],
)
def test_safety_and_duty_regressions_fail_closed(
    section: str, field: str, bad_value: float, check: str
) -> None:
    measurements = _healthy_measurements()
    measurements[section][field] = bad_value
    report = _report(measurements)
    assert report["checks"][check] is False
    assert report["passed"] is False


def test_supplier_approval_is_not_optional() -> None:
    measurements = _healthy_measurements()
    measurements["supplier_evidence"]["vertical_mobile_axis_duty_approved"] = False
    report = _report(measurements)
    assert report["checks"][
        "supplier_vertical_and_transmission_approvals_present"
    ] is False
    assert report["decision"] == "complete_mechanical_and_functional_safety_architecture"


def test_raw_evidence_hashes_are_required() -> None:
    measurements = _healthy_measurements()
    measurements["evidence"]["raw_log_sha256"] = "not-a-hash"
    report = _report(measurements)
    assert report["checks"]["measurement_complete"] is False
    assert report["checks"]["raw_and_supporting_evidence_hashes_present"] is False
    assert report["passed"] is False


def test_malformed_numeric_value_produces_rejection_report() -> None:
    measurements = _healthy_measurements()
    measurements["continuous_duty"]["phase_current_peak_a"] = "bad-reading"
    report = _report(measurements)
    assert report["checks"]["measurement_complete"] is False
    assert report["checks"]["peak_phase_current_passed"] is False
    assert report["passed"] is False


def test_cli_template_writes_lf_rejection_without_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    command = [sys.executable, str(SCRIPT), "--output", str(output)]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 6
    payload = output.read_bytes()
    assert payload.endswith(b"\n")
    assert b"\r\n" not in payload
    report = json.loads(payload)
    assert report["passed"] is False
    second = subprocess.run(command, check=False, capture_output=True, text=True)
    assert second.returncode != 0


def test_cli_750w_template_uses_the_production_route(tmp_path: Path) -> None:
    output = tmp_path / "audit-750w.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--candidate-profile",
        "leadshine_750w_production_candidate_v1",
        "--output",
        str(output),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 6
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["candidate_profile"] == (
        "leadshine_750w_production_candidate_v1"
    )
    assert report["checks"]["candidate_motor_matches"] is True
    assert report["checks"]["candidate_drive_matches"] is True
    assert report["checks"]["candidate_profile_matches_source_route"] is True
    assert report["passed"] is False


def test_input_mutation_does_not_change_sources() -> None:
    measurements = _healthy_measurements()
    original = copy.deepcopy(measurements)
    _report(measurements)
    assert measurements == original
