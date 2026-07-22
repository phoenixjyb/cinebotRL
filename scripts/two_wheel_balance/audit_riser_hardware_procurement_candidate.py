#!/usr/bin/env python3
"""Audit the provisional 1 m/s riser as a bench, not production, candidate."""

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
    / "docs/03_training/two_wheel_balance/RISER_VENDOR_SPEC_SNAPSHOT_20260723.json"
)
PRODUCTION_EMERGENCY_TORQUE_MARGIN = 1.15


def _load_hardware_module():
    spec = importlib.util.spec_from_file_location("riser_hardware_envelope", HARDWARE_SCRIPT)
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
    motor = vendor["leadshine_motor"]
    drive = vendor["leadshine_drive"]
    axis = vendor["igus_axis"]
    safety = vendor["sitema_safety_principle"]
    ratio = float(hardware["recommendation"]["preferred_reduction_ratio"])
    normal_8kg = _case(hardware, "sizing_cases", ratio, 8.0)
    emergency_8kg = _case(hardware, "emergency_braking_cases", ratio, 8.0)
    emergency_margin = float(emergency_8kg["rated_torque_margin_ratio"])
    inputs = hardware["inputs"]
    combined_efficiency = float(inputs["transmission_efficiency"]) * float(
        inputs["reduction_efficiency"]
    )
    rated_linear_force = (
        float(motor["rated_torque_nm"])
        * ratio
        * combined_efficiency
        * 2.0
        * math.pi
        / float(inputs["linear_lead_m_per_rev"])
    )
    force_at_production_margin = rated_linear_force / PRODUCTION_EMERGENCY_TORQUE_MARGIN
    maximum_mass_at_production_margin_kg = (
        force_at_production_margin / float(inputs["force_safety_factor"])
        - float(inputs["friction_force_n"])
    ) / (
        float(inputs["gravity_mps2"])
        + float(inputs["emergency_deceleration_mps2"])
    )
    mechanical_power_from_rating_w = (
        float(motor["rated_torque_nm"])
        * float(motor["rated_speed_rpm"])
        * 2.0
        * math.pi
        / 60.0
    )
    checks = {
        "vendor_snapshot_current": vendor.get("verified_date") == "2026-07-23",
        "motor_rating_self_consistent": math.isclose(
            mechanical_power_from_rating_w,
            float(motor["rated_power_w"]),
            rel_tol=0.01,
        ),
        "drive_continuous_current_matches_motor": float(drive["rated_current_arms"])
        >= float(motor["rated_current_a"]),
        "drive_peak_current_covers_motor": float(drive["peak_current_apeak"])
        >= float(motor["peak_current_a"]),
        "drive_requires_external_regen_design": drive.get(
            "external_brake_resistor_required_for_regeneration"
        )
        is True,
        "drive_has_no_published_safe_function": drive.get(
            "published_safe_function"
        )
        is False,
        "axis_matches_70mm_transmission": math.isclose(
            float(axis["transmission_mm_per_rev"]),
            float(inputs["linear_lead_m_per_rev"]) * 1000.0,
        ),
        "axis_catalog_speed_exceeds_target": float(
            axis["maximum_speed_mps_at_60_percent_duty"]
        )
        >= float(inputs["maximum_velocity_mps"]),
        "axis_catalog_stroke_covers_recommendation": float(
            axis["maximum_standard_stroke_m"]
        )
        >= float(hardware["stopping_envelope"]["recommended_mechanical_stroke_m"]),
        "axis_vertical_application_not_yet_approved": axis.get(
            "vertical_mobile_robot_application_approved"
        )
        is False,
        "hardware_envelope_passed": hardware.get("passed") is True,
        "three_to_one_8kg_normal_passes": normal_8kg.get("passed") is True,
        "three_to_one_8kg_emergency_passes_nominal_rating": emergency_8kg.get(
            "passed"
        )
        is True,
        "emergency_margin_below_production_target": emergency_margin
        < PRODUCTION_EMERGENCY_TORQUE_MARGIN,
        "safety_principle_is_fail_safe": safety.get(
            "fail_safe_on_power_pressure_or_emergency_stop"
        )
        is True
        and float(safety["holding_force_minimum_multiple_of_admissible_load"])
        >= 2.0,
        "specific_safety_catcher_not_selected": safety.get(
            "specific_product_selected"
        )
        is False,
    }
    passed = all(checks.values())
    return {
        "schema": "cinebotrl_two_wheel_riser_hardware_procurement_candidate_v1",
        "checks": checks,
        "calculated": {
            "motor_mechanical_power_from_rating_w": mechanical_power_from_rating_w,
            "rated_linear_force_n": rated_linear_force,
            "normal_8kg_motor_torque_nm": normal_8kg["motor_torque_nm"],
            "emergency_8kg_motor_torque_nm": emergency_8kg["motor_torque_nm"],
            "emergency_8kg_rated_torque_margin_ratio": emergency_margin,
            "production_emergency_margin_requirement": (
                PRODUCTION_EMERGENCY_TORQUE_MARGIN
            ),
            "maximum_moving_mass_at_production_margin_kg": (
                maximum_mass_at_production_margin_kg
            ),
            "motor_speed_at_1mps_rpm": normal_8kg["motor_speed_rpm"],
            "full_speed_stopping_distance_m": hardware["stopping_envelope"][
                "full_speed_stopping_distance_m"
            ],
            "recommended_mechanical_stroke_m": hardware["stopping_envelope"][
                "recommended_mechanical_stroke_m"
            ],
        },
        "recommendation": {
            "engineering_sample": (
                "one Leadshine 400W brake motor plus ELD2-CAN7010B for an "
                "instrumented non-riding bench prototype"
            ),
            "prototype_mechanism": (
                "70mm_per_rev_guided_belt_axis_with_3_to_1_reduction"
            ),
            "production_default_until_measurement": (
                "750W_class_or_resized_drive_unless_measured_mass_counterbalance_"
                "friction_and_regeneration_prove_at_least_15_percent_emergency_margin"
            ),
            "fixed_vs_telescoping": (
                "fixed_1p5m_mechanical_stroke_if_collapsed_height_allows; otherwise "
                "supplier-qualified_two_stage_synchronized_telescoping_mast"
            ),
            "safety": (
                "external_regen_path_plus_independent_anti_fall_hard_limits_"
                "absorbing_end_stops_and_safety_rated_power_removal"
            ),
            "battery_branch": (
                "do_not_size_fuse_or_cable from motor phase current; use drive "
                "input-current and regenerative-energy measurements"
            ),
        },
        "required_before_production_procurement": [
            "measure complete moving mass including cables and carriage",
            "measure eccentric camera and gimbal moments",
            "select and verify gearbox continuous input speed braking torque and backlash",
            "obtain supplier vertical-axis duty life and tooth-jump approval",
            "measure 1mps duty cycle temperature and carriage force",
            "select external regenerative resistor or battery absorption path",
            "select independent anti-fall device and safety-rated stop architecture",
            "verify 1p5m mechanical stroke or implement the position-dependent governor",
        ],
        "single_engineering_sample_purchase_recommended": passed,
        "valid_for_production_procurement": False,
        "valid_for_hardware_transfer": False,
        "valid_for_training": False,
        "gpu_work_started": False,
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
