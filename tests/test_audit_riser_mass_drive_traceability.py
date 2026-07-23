import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/audit_riser_mass_drive_traceability.py"
)
SPEC = importlib.util.spec_from_file_location("riser_mass_drive_traceability", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _inputs():
    hardware_module = MODULE._load_module("trace_test_hardware", MODULE.HARDWARE_SCRIPT)
    production_module = MODULE._load_module(
        "trace_test_production", MODULE.PRODUCTION_SCRIPT
    )
    drive_module = MODULE._load_module("trace_test_drive", MODULE.DRIVE_SCRIPT)
    hardware = hardware_module.build_report()
    vendor = json.loads(MODULE.VENDOR_PATH.read_text(encoding="utf-8"))
    production = production_module.build_report(vendor, hardware)
    drive = drive_module.build_report(
        production,
        urdf_path=MODULE.URDF_PATH,
        config_path=MODULE.CONFIG_PATH,
        thermal_control_path=MODULE.THERMAL_PATH,
    )
    return {
        "urdf": MODULE.read_urdf_contract(MODULE.URDF_PATH),
        "plant": json.loads(MODULE.PLANT_PATH.read_text(encoding="utf-8")),
        "bench": json.loads(MODULE.BENCH_PATH.read_text(encoding="utf-8")),
        "hardware": hardware,
        "production": production,
        "drive": drive,
    }


def test_mass_contract_distinguishes_robot_payload_and_sizing_mass() -> None:
    report = MODULE.build_report(**_inputs())
    mass = report["mass_contract"]

    assert report["passed"] is True
    assert mass["whole_robot_mass_kg"] == pytest.approx(28.0)
    assert mass["modeled_riser_moving_mass_kg"] == pytest.approx(4.342)
    assert mass["conservative_drive_sizing_mass_kg"] == pytest.approx(8.0)
    assert mass["sizing_mass_over_modeled_mass_ratio"] > 1.8
    assert (
        mass["calculated_maximum_moving_mass_at_15pct_force_margin_kg"] > 14.8
    )
    assert mass["whole_robot_mass_is_not_a_vertical_axis_payload"] is True


def test_force_power_and_stopping_chain_is_calculable() -> None:
    report = MODULE.build_report(**_inputs())
    force = report["force_power_contract"]
    safety = report["stroke_and_safety_contract"]

    assert force["counterbalance_force_credited_in_sizing_n"] == 0.0
    assert force["emergency_8kg_design_force_n"] == pytest.approx(276.96)
    assert force["rated_linear_force_n"] == pytest.approx(550.258929255)
    assert force["emergency_rated_force_margin_ratio"] > 1.98
    assert force["rated_linear_power_at_1mps_w"] == pytest.approx(
        force["motor_shaft_power_at_1mps_w"]
        * force["combined_transmission_efficiency"]
    )
    assert safety["software_camera_height_range_m"] == [0.6, 1.8]
    assert safety["full_speed_stopping_distance_m"] == pytest.approx(0.12)
    assert safety["recommended_mechanical_stroke_m"] == pytest.approx(1.5)
    assert safety[
        "maximum_single_descent_mechanical_energy_for_regen_review_j"
    ] == pytest.approx(98.176)


def test_report_keeps_hardware_and_learning_boundaries_closed() -> None:
    report = MODULE.build_report(**_inputs())
    classification = report["classification"]

    assert classification["calculation_traceability_passed"] is True
    assert classification["candidate_ready_for_supplier_and_bench_review"] is True
    assert classification["valid_for_production_procurement"] is False
    assert classification["valid_for_hardware_transfer"] is False
    assert classification["physical_riser_bench_qualification_passed"] is False
    assert classification["simulation_profile_changed"] is False
    assert classification["runtime_authorized"] is False
    assert classification["capture_authorized"] is False
    assert classification["valid_for_training"] is False
    assert classification["bc_authorized"] is False
    assert classification["ppo_authorized"] is False


def test_report_rejects_conflating_whole_robot_and_moving_mass() -> None:
    inputs = _inputs()
    inputs["urdf"] = copy.deepcopy(inputs["urdf"])
    inputs["urdf"]["riser_moving_subtree_mass_kg"] = 28.0
    inputs["urdf"]["riser_stationary_mass_kg"] = 0.0

    report = MODULE.build_report(**inputs)

    assert report["checks"]["riser_moving_mass_is_derived_from_subtree"] is False
    assert (
        report["checks"]["whole_robot_mass_is_not_used_as_riser_moving_mass"]
        is False
    )
    assert report["passed"] is False


def test_report_rejects_1p9m_camera_ceiling() -> None:
    inputs = _inputs()
    inputs["hardware"] = copy.deepcopy(inputs["hardware"])
    inputs["hardware"]["recommendation"]["software_camera_height_range_m"] = [
        0.6,
        1.9,
    ]

    report = MODULE.build_report(**inputs)

    assert report["checks"]["camera_height_ceiling_remains_1p8m"] is False
    assert report["passed"] is False


def test_cli_writes_host_independent_lf_evidence(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = output.read_bytes()
    assert payload.endswith(b"\n")
    assert b"\r\n" not in payload
    report = json.loads(payload)
    assert report["passed"] is True
    assert report["inputs"]["auditor"]["path"] == (
        "scripts/two_wheel_balance/audit_riser_mass_drive_traceability.py"
    )
    assert report["inputs"]["urdf"]["path"].endswith(
        "recomoProto2_two_wheel_riser.urdf"
    )
