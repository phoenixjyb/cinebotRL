#!/usr/bin/env python3
"""Audit the pinned 750 W riser production-design-review candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HARDWARE_SCRIPT = (
    PROJECT_ROOT / "scripts/two_wheel_balance/calculate_riser_hardware_envelope.py"
)
DEFAULT_VENDOR_SNAPSHOT = (
    PROJECT_ROOT
    / "docs/03_training/two_wheel_balance/"
    "RISER_PRODUCTION_CANDIDATE_VENDOR_SNAPSHOT_20260723.json"
)
PRODUCTION_EMERGENCY_FORCE_MARGIN = 1.15


def _load_hardware_module():
    spec = importlib.util.spec_from_file_location(
        "riser_hardware_envelope_production", HARDWARE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load riser hardware envelope module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _case(report: Mapping[str, Any], section: str, ratio: float, mass: float):
    for row in report[section]:
        if math.isclose(float(row["reduction_ratio"]), ratio) and math.isclose(
            float(row["moving_mass_kg"]), mass
        ):
            return row
    raise ValueError(f"missing {section} ratio={ratio} mass={mass}")


def build_report(
    vendor: Mapping[str, Any], hardware: Mapping[str, Any]
) -> dict[str, Any]:
    motor = vendor["motor"]
    drive = vendor["drive"]
    mechanism = vendor["mechanism_contract"]
    inputs = hardware["inputs"]
    ratio = float(mechanism["reduction_ratio"])
    lead = float(mechanism["linear_lead_m_per_rev"])
    combined_efficiency = float(inputs["transmission_efficiency"]) * float(
        inputs["reduction_efficiency"]
    )
    emergency_8kg = _case(hardware, "emergency_braking_cases", ratio, 8.0)
    emergency_force_n = float(emergency_8kg["design_force_n"])
    rated_force_n = (
        float(motor["rated_torque_nm"])
        * ratio
        * combined_efficiency
        * 2.0
        * math.pi
        / lead
    )
    peak_force_n = (
        float(motor["peak_torque_nm"])
        * ratio
        * combined_efficiency
        * 2.0
        * math.pi
        / lead
    )
    motor_speed_at_1mps_rpm = (
        float(mechanism["maximum_linear_speed_mps"]) / lead * 60.0 * ratio
    )
    mechanical_power_from_rating_w = (
        float(motor["rated_torque_nm"])
        * float(motor["rated_speed_rpm"])
        * 2.0
        * math.pi
        / 60.0
    )
    emergency_margin = rated_force_n / emergency_force_n
    maximum_moving_mass_at_required_margin_kg = (
        rated_force_n
        / PRODUCTION_EMERGENCY_FORCE_MARGIN
        / float(inputs["force_safety_factor"])
        - float(inputs["friction_force_n"])
    ) / (
        float(inputs["gravity_mps2"])
        + float(inputs["emergency_deceleration_mps2"])
    )
    checks = {
        "snapshot_schema": vendor.get("schema")
        == "cinebotrl_two_wheel_riser_production_candidate_vendor_snapshot_v1",
        "snapshot_current": vendor.get("verified_date") == "2026-07-23",
        "motor_is_pinned_48v_750w_brake_absolute": (
            motor.get("model") == "ELVM8075V48EH-M17-HD"
            and float(motor["rated_voltage_vdc"]) == 48.0
            and float(motor["rated_power_w"]) == 750.0
            and motor.get("brake") is True
            and motor.get("encoder") == "17_bit_magnetic_multi_turn_absolute"
        ),
        "motor_rating_self_consistent": math.isclose(
            mechanical_power_from_rating_w,
            float(motor["rated_power_w"]),
            rel_tol=0.01,
        ),
        "drive_is_official_match": (
            drive.get("model") == "ELD2-CAN7020B"
            and drive.get("official_motor_match_confirmed") is True
        ),
        "drive_power_covers_motor": float(drive["rated_power_w"])
        >= float(motor["rated_power_w"]),
        "drive_continuous_current_covers_motor": float(
            drive["rated_current_arms_at_or_below_48v"]
        )
        >= float(motor["rated_current_a"]),
        "drive_peak_current_covers_motor": float(drive["peak_current_apeak"])
        >= float(motor["peak_current_a"]),
        "drive_voltage_covers_48v": (
            float(drive["main_power_voltage_vdc"][0]) <= 48.0
            <= float(drive["main_power_voltage_vdc"][1])
        ),
        "external_regeneration_still_required": drive.get(
            "external_brake_resistor_required_for_regeneration"
        )
        is True,
        "drive_safety_function_still_absent": drive.get(
            "published_safe_function"
        )
        is False,
        "mechanism_ratio_and_lead_match_envelope": math.isclose(
            ratio, float(hardware["recommendation"]["preferred_reduction_ratio"])
        )
        and math.isclose(lead, float(inputs["linear_lead_m_per_rev"])),
        "camera_height_ceiling_is_1p8m": mechanism.get(
            "software_camera_height_range_m"
        )
        == [0.6, 1.8],
        "mechanical_stroke_preserves_stopping_space": math.isclose(
            float(mechanism["recommended_mechanical_stroke_m"]),
            float(hardware["stopping_envelope"]["recommended_mechanical_stroke_m"]),
        ),
        "motor_speed_covers_1mps": motor_speed_at_1mps_rpm
        <= float(motor["rated_speed_rpm"]),
        "rated_force_exceeds_required_emergency_margin": emergency_margin
        >= PRODUCTION_EMERGENCY_FORCE_MARGIN,
        "hardware_envelope_passed": hardware.get("passed") is True,
        "limitations_remain_fail_closed": len(vendor.get("limitations", [])) >= 8,
    }
    passed = all(checks.values())
    return {
        "schema": "cinebotrl_two_wheel_riser_hardware_production_candidate_v1",
        "checks": checks,
        "calculated": {
            "motor_mechanical_power_from_rating_w": mechanical_power_from_rating_w,
            "rated_linear_force_n": rated_force_n,
            "peak_linear_force_n": peak_force_n,
            "motor_speed_at_1mps_rpm": motor_speed_at_1mps_rpm,
            "emergency_8kg_design_force_n": emergency_force_n,
            "emergency_8kg_motor_torque_nm": emergency_8kg["motor_torque_nm"],
            "emergency_8kg_rated_force_margin_ratio": emergency_margin,
            "required_emergency_force_margin_ratio": (
                PRODUCTION_EMERGENCY_FORCE_MARGIN
            ),
            "maximum_moving_mass_at_required_margin_kg": (
                maximum_moving_mass_at_required_margin_kg
            ),
            "full_speed_stopping_distance_m": hardware["stopping_envelope"][
                "full_speed_stopping_distance_m"
            ],
            "recommended_mechanical_stroke_m": hardware["stopping_envelope"][
                "recommended_mechanical_stroke_m"
            ],
        },
        "recommendation": {
            "production_design_review_candidate": (
                "ELVM8075V48EH-M17-HD_plus_ELD2-CAN7020B"
            ),
            "mechanism": (
                "3_to_1_reduction_plus_70mm_per_rev_supplier_qualified_guided_"
                "belt_axis_or_two_stage_synchronized_telescoping_mast"
            ),
            "engineering_sample_predecessor": (
                "retain_400w_candidate_for_one_instrumented_bench_sample_only"
            ),
            "selection_reason": (
                "48v_750w_candidate_covers_8kg_emergency_force_with_at_least_"
                "15_percent_calculated_margin_under_current_assumptions"
            ),
            "safety_boundary": (
                "holding_brake_is_not_dynamic_brake; external_regeneration_"
                "independent_anti_fall_hard_limits_absorbing_end_stops_and_"
                "safety_rated_power_removal_remain_required"
            ),
        },
        "required_before_production_design_review": [
            "measure complete moving mass friction counterbalance and eccentric moments",
            "obtain supplier vertical mobile-axis duty and tooth-jump approval",
            "select and verify gearbox continuous speed braking torque efficiency and backlash",
            "design and test regenerative resistor or battery absorption path",
            "select and test independent anti-fall and safety-rated power removal",
            "complete the calibrated 1mps thermal force stop and limit bench campaign",
        ],
        "candidate_ready_for_supplier_and_bench_review": passed,
        "valid_for_production_procurement": False,
        "valid_for_hardware_transfer": False,
        "simulation_motor_model_updated": False,
        "runtime_authorized": False,
        "gpu_work_started": False,
        "valid_for_training": False,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendor-snapshot", type=Path, default=DEFAULT_VENDOR_SNAPSHOT
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    vendor = json.loads(args.vendor_snapshot.read_text(encoding="utf-8"))
    hardware = _load_hardware_module().build_report()
    report = build_report(vendor, hardware)
    report["inputs"] = {
        "vendor_snapshot": _identity(args.vendor_snapshot),
        "hardware_envelope_script": _identity(HARDWARE_SCRIPT),
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite hardware audit: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
