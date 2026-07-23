#!/usr/bin/env python3
"""Bind riser mass identities to the current drive and safety calculation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping
from xml.etree import ElementTree


PROJECT_ROOT = Path(__file__).resolve().parents[2]
URDF_PATH = (
    PROJECT_ROOT
    / "assets_own/recomoProto2_two_wheel_riser/"
    "recomoProto2_two_wheel_riser.urdf"
)
PLANT_PATH = (
    PROJECT_ROOT
    / "docs/03_training/two_wheel_balance/PLANT_PRIOR_PROVISIONAL_V1.json"
)
VENDOR_PATH = (
    PROJECT_ROOT
    / "docs/03_training/two_wheel_balance/"
    "RISER_PRODUCTION_CANDIDATE_VENDOR_SNAPSHOT_20260723.json"
)
BENCH_PATH = (
    PROJECT_ROOT
    / "docs/03_training/two_wheel_balance/"
    "RISER_750W_BENCH_MEASUREMENT_TEMPLATE_20260723.json"
)
CONFIG_PATH = (
    PROJECT_ROOT / "src/rl_platform/robots/two_wheel_balance/config.py"
)
THERMAL_PATH = (
    PROJECT_ROOT
    / "src/rl_platform/tasks/two_wheel_balance/riser_control.py"
)
HARDWARE_SCRIPT = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/calculate_riser_hardware_envelope.py"
)
PRODUCTION_SCRIPT = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/audit_riser_hardware_production_candidate.py"
)
DRIVE_SCRIPT = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/audit_riser_drive_profile_selection.py"
)
AUDITOR_PATH = Path(__file__).resolve()

DESIGN_MOVING_MASS_KG = 8.0
PRODUCTION_FORCE_MARGIN_RATIO = 1.15
INDEPENDENT_ANTI_FALL_STATIC_FACTOR = 2.0


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _identity(path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _float(value: str | None, *, field: str) -> float:
    if value is None:
        raise ValueError(f"missing numeric field: {field}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric field: {field}")
    return number


def read_urdf_contract(path: Path) -> dict[str, Any]:
    root = ElementTree.parse(path).getroot()
    masses: dict[str, float] = {}
    for link in root.findall("link"):
        name = link.get("name")
        mass_node = link.find("./inertial/mass")
        if not name or mass_node is None:
            raise ValueError(f"link lacks explicit inertial mass: {name!r}")
        masses[name] = _float(mass_node.get("value"), field=f"{name}.mass")

    joints: dict[str, dict[str, Any]] = {}
    children_by_parent: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        parent = joint.find("parent")
        child = joint.find("child")
        if not name or parent is None or child is None:
            raise ValueError("joint lacks name, parent, or child")
        parent_link = parent.get("link")
        child_link = child.get("link")
        if not parent_link or not child_link:
            raise ValueError(f"joint has empty parent/child: {name}")
        limit = joint.find("limit")
        joints[name] = {
            "type": joint.get("type"),
            "parent": parent_link,
            "child": child_link,
            "limit": {} if limit is None else dict(limit.attrib),
        }
        children_by_parent.setdefault(parent_link, []).append(child_link)

    riser = joints.get("riser_joint")
    if riser is None:
        raise ValueError("missing riser_joint")
    moving_links: set[str] = set()
    pending = [str(riser["child"])]
    while pending:
        link = pending.pop()
        if link in moving_links:
            continue
        moving_links.add(link)
        pending.extend(children_by_parent.get(link, ()))

    limit = riser["limit"]
    return {
        "total_authored_mass_kg": sum(masses.values()),
        "riser_moving_subtree_mass_kg": sum(masses[name] for name in moving_links),
        "riser_moving_links": sorted(moving_links),
        "riser_stationary_mass_kg": sum(masses.values())
        - sum(masses[name] for name in moving_links),
        "riser_joint": {
            "parent": riser["parent"],
            "child": riser["child"],
            "lower_m": _float(limit.get("lower"), field="riser_joint.lower"),
            "upper_m": _float(limit.get("upper"), field="riser_joint.upper"),
            "effort_limit_n": _float(
                limit.get("effort"), field="riser_joint.effort"
            ),
            "velocity_limit_mps": _float(
                limit.get("velocity"), field="riser_joint.velocity"
            ),
        },
    }


def _find_case(
    report: Mapping[str, Any], section: str, ratio: float, mass: float
) -> Mapping[str, Any]:
    for row in report[section]:
        if math.isclose(float(row["reduction_ratio"]), ratio) and math.isclose(
            float(row["moving_mass_kg"]), mass
        ):
            return row
    raise ValueError(f"missing {section} ratio={ratio} mass={mass}")


def build_report(
    *,
    urdf: Mapping[str, Any],
    plant: Mapping[str, Any],
    bench: Mapping[str, Any],
    hardware: Mapping[str, Any],
    production: Mapping[str, Any],
    drive: Mapping[str, Any],
) -> dict[str, Any]:
    total_mass = float(urdf["total_authored_mass_kg"])
    moving_mass = float(urdf["riser_moving_subtree_mass_kg"])
    stationary_mass = float(urdf["riser_stationary_mass_kg"])
    plant_total_mass = float(plant["nominal"]["total_mass_kg"])
    riser_joint = urdf["riser_joint"]
    inputs = hardware["inputs"]
    ratio = float(hardware["recommendation"]["preferred_reduction_ratio"])
    emergency_case = _find_case(
        hardware, "emergency_braking_cases", ratio, DESIGN_MOVING_MASS_KG
    )
    normal_case = _find_case(
        hardware, "sizing_cases", ratio, DESIGN_MOVING_MASS_KG
    )
    calculated = production["calculated"]
    target_speed = float(inputs["maximum_velocity_mps"])
    working_stroke = float(inputs["usable_software_stroke_m"])
    gravity = float(inputs["gravity_mps2"])
    rated_force = float(calculated["rated_linear_force_n"])
    peak_force = float(calculated["peak_linear_force_n"])
    motor_speed = float(calculated["motor_speed_at_1mps_rpm"])
    vendor_motor = production["recommendation"][
        "production_design_review_candidate"
    ]
    combined_efficiency = float(inputs["transmission_efficiency"]) * float(
        inputs["reduction_efficiency"]
    )
    rated_torque = rated_force * float(inputs["linear_lead_m_per_rev"]) / (
        ratio * combined_efficiency * 2.0 * math.pi
    )
    motor_power_at_target = (
        rated_torque * motor_speed * 2.0 * math.pi / 60.0
    )
    linear_power_at_target = rated_force * target_speed
    expected_emergency_force = float(inputs["force_safety_factor"]) * (
        DESIGN_MOVING_MASS_KG
        * (gravity + float(inputs["emergency_deceleration_mps2"]))
        + float(inputs["friction_force_n"])
    )
    maximum_mass = float(
        calculated["maximum_moving_mass_at_required_margin_kg"]
    )
    stopping_distance = float(
        hardware["stopping_envelope"]["full_speed_stopping_distance_m"]
    )
    mechanical_stroke = float(
        hardware["stopping_envelope"]["recommended_mechanical_stroke_m"]
    )
    anti_fall_static_force = (
        INDEPENDENT_ANTI_FALL_STATIC_FACTOR
        * DESIGN_MOVING_MASS_KG
        * gravity
    )
    maximum_regenerative_mechanical_energy = (
        DESIGN_MOVING_MASS_KG * gravity * working_stroke
        + 0.5 * DESIGN_MOVING_MASS_KG * target_speed**2
    )

    bench_unmeasured = (
        bench["configuration"]["complete_moving_mass_kg"] is None
        and bench["configuration"]["measured_friction_force_n"] is None
        and bench["continuous_duty"]["duration_s"] is None
        and bench["emergency_stop"]["repetitions"] is None
    )
    checks = {
        "urdf_total_mass_matches_28kg_plant": math.isclose(
            total_mass, 28.0, abs_tol=1e-9
        )
        and math.isclose(total_mass, plant_total_mass, abs_tol=1e-9),
        "riser_moving_mass_is_derived_from_subtree": math.isclose(
            moving_mass + stationary_mass, total_mass, abs_tol=1e-9
        )
        and math.isclose(moving_mass, 4.342, abs_tol=1e-9),
        "whole_robot_mass_is_not_used_as_riser_moving_mass": (
            moving_mass < DESIGN_MOVING_MASS_KG < maximum_mass < total_mass
        ),
        "design_mass_conservatively_covers_current_model": (
            DESIGN_MOVING_MASS_KG / moving_mass >= 1.8
        ),
        "riser_joint_preserves_1p2m_working_stroke": math.isclose(
            float(riser_joint["upper_m"]) - float(riser_joint["lower_m"]),
            working_stroke,
            abs_tol=1e-12,
        ),
        "riser_joint_preserves_1mps_limit": math.isclose(
            float(riser_joint["velocity_limit_mps"]), target_speed, abs_tol=1e-12
        ),
        "camera_height_ceiling_remains_1p8m": hardware["recommendation"][
            "software_camera_height_range_m"
        ]
        == [0.6, 1.8],
        "mechanical_stroke_is_not_camera_height_extension": (
            math.isclose(working_stroke, 1.2, abs_tol=1e-12)
            and math.isclose(mechanical_stroke, 1.5, abs_tol=1e-12)
            and math.isclose(stopping_distance, 0.12, abs_tol=1e-12)
        ),
        "sizing_credits_no_counterbalance": math.isclose(
            float(emergency_case["design_force_n"]),
            expected_emergency_force,
            rel_tol=1e-12,
        ),
        "750w_candidate_closes_8kg_emergency_margin": (
            float(calculated["emergency_8kg_rated_force_margin_ratio"])
            >= PRODUCTION_FORCE_MARGIN_RATIO
            and rated_force
            >= PRODUCTION_FORCE_MARGIN_RATIO
            * float(emergency_case["design_force_n"])
        ),
        "750w_candidate_power_chain_is_consistent_at_1mps": math.isclose(
            linear_power_at_target,
            motor_power_at_target * combined_efficiency,
            rel_tol=1e-12,
        ),
        "active_simulation_remains_explicit_400w_profile": (
            drive["active_simulation_profile"]["name"]
            == "leadshine_400w_engineering_sample_v1"
            and drive["production_design_candidate"]["name"]
            == "leadshine_750w_production_candidate_v1"
            and drive["production_design_candidate"]["simulation_enabled"] is False
        ),
        "production_candidate_is_design_review_only": (
            production["candidate_ready_for_supplier_and_bench_review"] is True
            and production["valid_for_production_procurement"] is False
            and production["valid_for_hardware_transfer"] is False
            and production["runtime_authorized"] is False
            and production["valid_for_training"] is False
        ),
        "bench_and_safety_evidence_remain_unmeasured": (
            bench_unmeasured
            and bench["power_loss_safety"]["independent_anti_fall_installed"]
            is False
            and bench["limits"]["safety_rated_power_removal_passed"] is False
        ),
        "upstream_calculation_audits_pass": (
            hardware["passed"] is True
            and production["passed"] is True
            and drive["passed"] is True
        ),
    }
    passed = all(checks.values())
    return {
        "schema": "cinebotrl_two_wheel_riser_mass_drive_traceability_v1",
        "mass_contract": {
            "whole_robot_mass_kg": total_mass,
            "stationary_mass_kg": stationary_mass,
            "modeled_riser_moving_mass_kg": moving_mass,
            "modeled_riser_moving_links": urdf["riser_moving_links"],
            "conservative_drive_sizing_mass_kg": DESIGN_MOVING_MASS_KG,
            "sizing_mass_over_modeled_mass_ratio": DESIGN_MOVING_MASS_KG
            / moving_mass,
            "calculated_maximum_moving_mass_at_15pct_force_margin_kg": (
                maximum_mass
            ),
            "whole_robot_mass_is_a_balance_plant_parameter": True,
            "whole_robot_mass_is_not_a_vertical_axis_payload": True,
        },
        "force_power_contract": {
            "candidate": vendor_motor,
            "mechanism": production["recommendation"]["mechanism"],
            "target_speed_mps": target_speed,
            "normal_8kg_design_force_n": normal_case["design_force_n"],
            "emergency_8kg_design_force_n": emergency_case["design_force_n"],
            "rated_linear_force_n": rated_force,
            "peak_linear_force_n": peak_force,
            "emergency_rated_force_margin_ratio": calculated[
                "emergency_8kg_rated_force_margin_ratio"
            ],
            "rated_linear_power_at_1mps_w": linear_power_at_target,
            "motor_shaft_power_at_1mps_w": motor_power_at_target,
            "combined_transmission_efficiency": combined_efficiency,
            "counterbalance_force_credited_in_sizing_n": 0.0,
        },
        "stroke_and_safety_contract": {
            "software_camera_height_range_m": [0.6, 1.8],
            "software_working_stroke_m": working_stroke,
            "full_speed_stopping_distance_m": stopping_distance,
            "hard_margin_each_end_m": inputs["hard_margin_m"],
            "recommended_mechanical_stroke_m": mechanical_stroke,
            "mechanical_overtravel_does_not_raise_camera_height_limit": True,
            "minimum_independent_anti_fall_static_force_at_2x_8kg_weight_n": (
                anti_fall_static_force
            ),
            "maximum_single_descent_mechanical_energy_for_regen_review_j": (
                maximum_regenerative_mechanical_energy
            ),
            "holding_brake_is_not_dynamic_or_independent_anti_fall": True,
        },
        "measurement_boundary": {
            "measured_complete_moving_mass_available": False,
            "measured_friction_available": False,
            "measured_counterbalance_available": False,
            "measured_duty_and_thermal_available": False,
            "measured_regeneration_available": False,
            "supplier_vertical_axis_approval_available": False,
            "independent_anti_fall_qualified": False,
            "bench_template_schema": bench["schema"],
        },
        "checks": checks,
        "passed": passed,
        "classification": {
            "calculation_traceability_passed": passed,
            "motor_and_mechanism_recommendation_available": passed,
            "candidate_ready_for_supplier_and_bench_review": passed,
            "valid_for_production_procurement": False,
            "valid_for_hardware_transfer": False,
            "physical_riser_bench_qualification_passed": False,
            "simulation_profile_changed": False,
            "runtime_authorized": False,
            "capture_authorized": False,
            "valid_for_training": False,
            "bc_authorized": False,
            "ppo_authorized": False,
        },
    }


def build_current_report() -> dict[str, Any]:
    hardware_module = _load_module("riser_trace_hardware", HARDWARE_SCRIPT)
    production_module = _load_module("riser_trace_production", PRODUCTION_SCRIPT)
    drive_module = _load_module("riser_trace_drive", DRIVE_SCRIPT)
    plant = json.loads(PLANT_PATH.read_text(encoding="utf-8"))
    vendor = json.loads(VENDOR_PATH.read_text(encoding="utf-8"))
    bench = json.loads(BENCH_PATH.read_text(encoding="utf-8"))
    hardware = hardware_module.build_report()
    production = production_module.build_report(vendor, hardware)
    drive = drive_module.build_report(
        production,
        urdf_path=URDF_PATH,
        config_path=CONFIG_PATH,
        thermal_control_path=THERMAL_PATH,
    )
    report = build_report(
        urdf=read_urdf_contract(URDF_PATH),
        plant=plant,
        bench=bench,
        hardware=hardware,
        production=production,
        drive=drive,
    )
    report["inputs"] = {
        "auditor": _identity(AUDITOR_PATH),
        "urdf": _identity(URDF_PATH),
        "plant_prior": _identity(PLANT_PATH),
        "production_vendor_snapshot": _identity(VENDOR_PATH),
        "bench_measurement_template": _identity(BENCH_PATH),
        "isaac_config": _identity(CONFIG_PATH),
        "thermal_control": _identity(THERMAL_PATH),
        "hardware_calculator": _identity(HARDWARE_SCRIPT),
        "production_candidate_auditor": _identity(PRODUCTION_SCRIPT),
        "drive_profile_auditor": _identity(DRIVE_SCRIPT),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_current_report()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite traceability audit: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "whole_robot_mass_kg": report["mass_contract"][
                    "whole_robot_mass_kg"
                ],
                "modeled_riser_moving_mass_kg": report["mass_contract"][
                    "modeled_riser_moving_mass_kg"
                ],
                "design_moving_mass_kg": report["mass_contract"][
                    "conservative_drive_sizing_mass_kg"
                ],
                "hardware_transfer_ready": report["classification"][
                    "valid_for_hardware_transfer"
                ],
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
