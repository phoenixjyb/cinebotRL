import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT / "scripts/two_wheel_balance/assemble_riser_750w_bench_evidence.py"
)
TEMPLATE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "RISER_750W_BENCH_MEASUREMENT_TEMPLATE_20260723.json"
)
ENGINEERING_TEMPLATE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "RISER_BENCH_MEASUREMENT_TEMPLATE_20260723.json"
)
SPEC = importlib.util.spec_from_file_location("riser_750w_assembler", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sources() -> tuple[dict, dict]:
    procurement = json.loads(
        MODULE.DEFAULT_PROCUREMENT_AUDIT.read_text(encoding="utf-8")
    )
    vendor = json.loads(
        MODULE.DEFAULT_VENDOR_SNAPSHOT.read_text(encoding="utf-8")
    )
    return procurement, vendor


def _manual() -> dict:
    manual = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    manual["test_id"] = "synthetic_750w_assembly_test"
    manual["evidence"].update(
        {
            "force_calibration_record_sha256": "1" * 64,
            "current_calibration_record_sha256": "2" * 64,
            "temperature_calibration_record_sha256": "3" * 64,
            "position_calibration_record_sha256": "4" * 64,
            "safety_test_video_sha256": "5" * 64,
        }
    )
    manual["configuration"].update(
        {
            "complete_moving_mass_kg": 8.0,
            "measured_friction_force_n": 20.0,
            "counterbalance_force_n": 0.0,
            "mechanical_stroke_m": 1.5,
            "software_camera_height_min_m": 0.6,
            "software_camera_height_max_m": 1.8,
        }
    )
    manual["instrumentation"] = {
        "force_calibration_valid": True,
        "current_calibration_valid": True,
        "temperature_calibration_valid": True,
        "position_calibration_valid": True,
    }
    manual["power_loss_safety"] = {
        "motor_brake_hold_duration_s": 600.0,
        "motor_brake_hold_displacement_m": 0.001,
        "independent_anti_fall_installed": True,
        "independent_anti_fall_test_count": 10,
        "independent_anti_fall_worst_catch_distance_m": 0.02,
        "independent_anti_fall_no_damage": True,
    }
    manual["limits"] = {
        "lower_hard_limit_passed": True,
        "upper_hard_limit_passed": True,
        "lower_absorbing_end_stop_passed": True,
        "upper_absorbing_end_stop_passed": True,
        "safety_rated_power_removal_passed": True,
    }
    return manual


def _reduction() -> dict:
    return {
        "schema": MODULE.REDUCTION_SCHEMA,
        "candidate_profile": MODULE.PROFILE,
        "raw_log": {"sha256": "6" * 64},
        "continuous_duty": {
            "duration_s": 1800.0,
            "duty_cycle_fraction": 0.6,
            "commanded_speed_mps": 1.0,
            "minimum_achieved_speed_mps": 0.97,
            "phase_current_rms_a": 15.0,
            "phase_current_peak_a": 45.0,
            "dc_input_current_rms_a": 18.0,
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
            "worst_stopping_distance_m": 0.10,
            "phase_current_peak_a": 55.0,
            "dc_bus_voltage_max_v": 62.0,
            "no_fault_or_position_loss": True,
        },
        "passed": True,
        "valid_for_bench_measurement_numeric_merge": True,
        "valid_for_candidate_bound_bench_merge": True,
        "valid_for_production_procurement": False,
        "valid_for_hardware_transfer": False,
        "valid_for_training": False,
        "runtime_authorized": False,
        "gpu_work_started": False,
        "bc_authorized": False,
        "ppo_authorized": False,
    }


def _supplier() -> dict:
    return {
        "schema": MODULE.SUPPLIER_AUDIT_SCHEMA,
        "passed": True,
        "valid_for_bench_supplier_evidence_merge": False,
        "valid_for_current_400w_bench_supplier_evidence_merge": False,
        "valid_for_750w_bench_supplier_evidence_merge": True,
        "candidate_identity_match_required_before_merge": True,
        "bench_measurement_merge_fragment": {
            "required_candidate": copy.deepcopy(MODULE.EXPECTED_CANDIDATE),
            "evidence": {"supplier_approval_package_sha256": "7" * 64},
            "supplier_evidence": {
                "vertical_mobile_axis_duty_approved": True,
                "gearbox_continuous_speed_approved": True,
                "gearbox_emergency_braking_torque_approved": True,
                "belt_tooth_jump_margin_approved": True,
            },
        },
        "inputs": {"supplier_response": {"sha256": "7" * 64}},
        "valid_for_production_procurement": False,
        "valid_for_hardware_transfer": False,
        "valid_for_training": False,
        "runtime_authorized": False,
        "gpu_work_started": False,
        "bc_authorized": False,
        "ppo_authorized": False,
    }


def _assemble(
    manual: dict | None = None,
    reduction: dict | None = None,
    supplier: dict | None = None,
) -> tuple[dict, dict]:
    procurement, vendor = _sources()
    return MODULE.build_assembly(
        manual or _manual(),
        reduction or _reduction(),
        supplier or _supplier(),
        procurement,
        vendor,
    )


def test_healthy_candidate_bound_fragments_pass_final_bench_gate() -> None:
    assembled, report = _assemble()
    assert report["passed"] is True
    assert report["input_contract_passed"] is True
    assert report["ready_for_production_design_review"] is True
    assert report["valid_for_production_procurement"] is False
    assert report["valid_for_hardware_transfer"] is False
    assert report["simulation_motor_model_updated"] is False
    assert report["valid_for_training"] is False
    assert report["runtime_authorized"] is False
    assert report["bc_authorized"] is False
    assert report["ppo_authorized"] is False
    assert assembled["evidence"]["raw_log_sha256"] == "6" * 64
    assert assembled["evidence"]["supplier_approval_package_sha256"] == "7" * 64
    assert assembled["candidate"]["drive_profile"] == MODULE.PROFILE
    assert assembled["continuous_duty"]["phase_current_rms_a"] == 15.0
    assert all(assembled["supplier_evidence"].values())


def test_legacy_or_cross_candidate_inputs_are_rejected() -> None:
    engineering = json.loads(ENGINEERING_TEMPLATE.read_text(encoding="utf-8"))
    engineering["test_id"] = "crossed"
    with pytest.raises(ValueError, match="input contract failed"):
        _assemble(manual=engineering)

    legacy_reduction = _reduction()
    legacy_reduction["schema"] = "cinebotrl_two_wheel_riser_bench_log_reduction_v1"
    legacy_reduction["candidate_profile"] = None
    legacy_reduction["valid_for_candidate_bound_bench_merge"] = False
    with pytest.raises(ValueError, match="input contract failed"):
        _assemble(reduction=legacy_reduction)


def test_supplier_required_candidate_must_match_exactly() -> None:
    supplier = _supplier()
    supplier["bench_measurement_merge_fragment"]["required_candidate"][
        "drive_model"
    ] = "ELD2-CAN7010B"
    with pytest.raises(ValueError, match="supplier_required_candidate_matches"):
        _assemble(supplier=supplier)


def test_supplier_merge_hash_must_match_audited_response() -> None:
    supplier = _supplier()
    supplier["bench_measurement_merge_fragment"]["evidence"][
        "supplier_approval_package_sha256"
    ] = "8" * 64
    with pytest.raises(
        ValueError,
        match="supplier_package_hash_matches_audited_response",
    ):
        _assemble(supplier=supplier)


def test_manual_cannot_prepopulate_automated_fields() -> None:
    manual = _manual()
    manual["continuous_duty"]["duration_s"] = 1800.0
    with pytest.raises(ValueError, match="manual_automation_slots_are_empty"):
        _assemble(manual=manual)


def test_missing_manual_safety_evidence_reaches_rejected_final_gate() -> None:
    manual = _manual()
    manual["limits"]["upper_hard_limit_passed"] = False
    assembled, report = _assemble(manual=manual)
    assert report["input_contract_passed"] is True
    assert report["passed"] is False
    assert report["ready_for_production_design_review"] is False
    assert report["decision"] == (
        "assembled_750w_bench_evidence_rejected_by_final_gate"
    )
    assert report["bench_audit"]["checks"][
        "limits_and_safety_power_removal_passed"
    ] is False
    assert assembled["assembly_provenance"]["valid_for_hardware_transfer"] is False


def test_inputs_are_not_mutated() -> None:
    manual = _manual()
    reduction = _reduction()
    supplier = _supplier()
    originals = copy.deepcopy((manual, reduction, supplier))
    _assemble(manual, reduction, supplier)
    assert (manual, reduction, supplier) == originals


def test_cli_writes_hash_bound_outputs_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    manual_path = tmp_path / "manual.json"
    reduction_path = tmp_path / "reduction.json"
    supplier_path = tmp_path / "supplier.json"
    measurements_path = tmp_path / "assembled.json"
    audit_path = tmp_path / "audit.json"
    for path, payload in (
        (manual_path, _manual()),
        (reduction_path, _reduction()),
        (supplier_path, _supplier()),
    ):
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    command = [
        sys.executable,
        str(SCRIPT),
        "--manual-measurements",
        str(manual_path),
        "--numeric-reduction",
        str(reduction_path),
        "--supplier-audit",
        str(supplier_path),
        "--output-measurements",
        str(measurements_path),
        "--output-audit",
        str(audit_path),
    ]
    first = subprocess.run(command, check=False, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert measurements_path.read_bytes().endswith(b"\n")
    assert audit_path.read_bytes().endswith(b"\n")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["passed"] is True
    assert audit["assembled_measurements"]["sha256"]
    assert audit["inputs"]["numeric_reduction"]["sha256"]
    second = subprocess.run(command, check=False, capture_output=True, text=True)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr
