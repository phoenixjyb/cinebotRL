#!/usr/bin/env python3
"""Calculate the provisional motor, reduction, and stopping envelope."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_control import (
    required_stopping_distance,
    safe_velocity_for_stopping_distance,
)


@dataclass(frozen=True)
class SizingInputs:
    gravity_mps2: float = 9.81
    upward_acceleration_mps2: float = 2.0
    maximum_velocity_mps: float = 1.0
    friction_force_n: float = 20.0
    force_safety_factor: float = 2.0
    linear_lead_m_per_rev: float = 0.07
    transmission_efficiency: float = 0.90
    reduction_efficiency: float = 0.95
    motor_rated_torque_nm: float = 1.27
    motor_rated_speed_rpm: float = 3000.0
    motor_rated_power_w: float = 400.0
    emergency_deceleration_mps2: float = 5.0
    response_delay_s: float = 0.02
    hard_margin_m: float = 0.03
    usable_software_stroke_m: float = 1.20
    camera_height_min_m: float = 0.60
    camera_height_max_m: float = 1.80


def size_case(
    moving_mass_kg: float,
    ratio: float,
    inputs: SizingInputs,
    *,
    upward_acceleration_mps2: float | None = None,
    scenario: str = "normal_upward_motion",
) -> dict:
    acceleration = (
        inputs.upward_acceleration_mps2
        if upward_acceleration_mps2 is None
        else upward_acceleration_mps2
    )
    nominal_force = (
        moving_mass_kg * (inputs.gravity_mps2 + acceleration)
        + inputs.friction_force_n
    )
    design_force = inputs.force_safety_factor * nominal_force
    pulley_torque = (
        design_force
        * inputs.linear_lead_m_per_rev
        / (2.0 * math.pi * inputs.transmission_efficiency)
    )
    motor_torque = pulley_torque / (ratio * inputs.reduction_efficiency)
    pulley_speed_rpm = (
        inputs.maximum_velocity_mps / inputs.linear_lead_m_per_rev * 60.0
    )
    motor_speed_rpm = pulley_speed_rpm * ratio
    motor_shaft_power = (
        motor_torque * motor_speed_rpm * 2.0 * math.pi / 60.0
    )
    checks = {
        "rated_torque_passed": motor_torque <= inputs.motor_rated_torque_nm,
        "rated_speed_passed": motor_speed_rpm <= inputs.motor_rated_speed_rpm,
        "rated_power_passed": motor_shaft_power <= inputs.motor_rated_power_w,
    }
    return {
        "scenario": scenario,
        "moving_mass_kg": moving_mass_kg,
        "reduction_ratio": ratio,
        "upward_acceleration_mps2": acceleration,
        "nominal_upward_force_n": nominal_force,
        "design_force_n": design_force,
        "load_mechanical_power_w": design_force * inputs.maximum_velocity_mps,
        "pulley_torque_nm": pulley_torque,
        "pulley_speed_rpm": pulley_speed_rpm,
        "motor_torque_nm": motor_torque,
        "motor_speed_rpm": motor_speed_rpm,
        "motor_shaft_power_w": motor_shaft_power,
        "rated_torque_margin_ratio": inputs.motor_rated_torque_nm / motor_torque,
        "rated_speed_margin_ratio": inputs.motor_rated_speed_rpm / motor_speed_rpm,
        "rated_power_margin_ratio": inputs.motor_rated_power_w / motor_shaft_power,
        "checks": checks,
        "passed": all(checks.values()),
    }


def build_report() -> dict:
    inputs = SizingInputs()
    masses = (4.342, 6.0, 8.0)
    ratios = (2.0, 3.0)
    cases = [size_case(mass, ratio, inputs) for ratio in ratios for mass in masses]
    emergency_cases = [
        size_case(
            mass,
            ratio,
            inputs,
            upward_acceleration_mps2=inputs.emergency_deceleration_mps2,
            scenario="stop_downward_motion_with_upward_deceleration",
        )
        for ratio in ratios
        for mass in masses
    ]
    stopping_distance = required_stopping_distance(
        inputs.maximum_velocity_mps,
        inputs.emergency_deceleration_mps2,
        inputs.response_delay_s,
    )
    recommended_mechanical_stroke = inputs.usable_software_stroke_m + 2.0 * (
        stopping_distance + inputs.hard_margin_m
    )
    stopping_samples = [0.0, 0.02, 0.05, stopping_distance, 0.15]
    velocity_envelope = [
        {
            "distance_to_hard_margin_m": distance,
            "safe_velocity_mps": min(
                inputs.maximum_velocity_mps,
                safe_velocity_for_stopping_distance(
                    distance,
                    inputs.emergency_deceleration_mps2,
                    inputs.response_delay_s,
                ),
            ),
        }
        for distance in stopping_samples
    ]
    by_ratio_mass = {
        (item["reduction_ratio"], item["moving_mass_kg"]): item for item in cases
    }
    emergency_by_ratio_mass = {
        (item["reduction_ratio"], item["moving_mass_kg"]): item
        for item in emergency_cases
    }
    checks = {
        "camera_height_max_is_exactly_1_8_m": inputs.camera_height_max_m == 1.8,
        "usable_stroke_matches_camera_range": math.isclose(
            inputs.camera_height_max_m - inputs.camera_height_min_m,
            inputs.usable_software_stroke_m,
        ),
        "full_speed_stop_requires_0_12_m": math.isclose(stopping_distance, 0.12),
        "buffered_mechanical_stroke_is_1_50_m": math.isclose(
            recommended_mechanical_stroke, 1.50
        ),
        "three_to_one_passes_all_mass_cases": all(
            item["passed"] for item in cases if item["reduction_ratio"] == 3.0
        ),
        "two_to_one_fails_8kg_rated_torque": not by_ratio_mass[(2.0, 8.0)][
            "checks"
        ]["rated_torque_passed"],
        "three_to_one_passes_8kg_rated_torque": by_ratio_mass[(3.0, 8.0)][
            "checks"
        ]["rated_torque_passed"],
        "three_to_one_passes_all_emergency_braking_cases": all(
            item["passed"]
            for item in emergency_cases
            if item["reduction_ratio"] == 3.0
        ),
        "three_to_one_emergency_8kg_torque_margin_above_one": (
            emergency_by_ratio_mass[(3.0, 8.0)]["rated_torque_margin_ratio"] > 1.0
        ),
    }
    return {
        "schema": "cinebotrl_two_wheel_riser_hardware_envelope_v1",
        "inputs": asdict(inputs),
        "motor_candidate": {
            "model": "Leadshine ELVM6040V48EH-M17-HD",
            "source": (
                "https://www.leadshine.com/product-detail/"
                "ELVM6040V48EH-M17-HD.html"
            ),
            "rated_voltage_vdc": 48.0,
            "rated_power_w": 400.0,
            "rated_torque_nm": 1.27,
            "peak_torque_nm": 3.81,
            "rated_speed_rpm": 3000.0,
            "peak_speed_rpm": 4000.0,
            "brake": True,
        },
        "sizing_cases": cases,
        "emergency_braking_cases": emergency_cases,
        "stopping_envelope": {
            "full_speed_stopping_distance_m": stopping_distance,
            "hard_margin_each_end_m": inputs.hard_margin_m,
            "usable_software_stroke_m": inputs.usable_software_stroke_m,
            "recommended_mechanical_stroke_m": recommended_mechanical_stroke,
            "velocity_samples": velocity_envelope,
        },
        "recommendation": {
            "mechanism": "AT10 belt-driven guided carriage or synchronized telescope",
            "preferred_reduction_ratio": 3.0,
            "two_to_one_boundary": (
                "prototype-only for measured low moving mass or effective "
                "counterbalance"
            ),
            "software_camera_height_range_m": [0.6, 1.8],
            "physical_stroke_if_full_speed_at_software_limits_m": 1.50,
            "emergency_8kg_three_to_one_torque_margin_ratio": (
                emergency_by_ratio_mass[(3.0, 8.0)]["rated_torque_margin_ratio"]
            ),
            "if_physical_stroke_remains_1_2_m": (
                "position-dependent direction-aware velocity governor is mandatory"
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "valid_for_procurement": False,
        "gpu_work_started": False,
        "training_started": False,
        "bc_started": False,
        "ppo_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "preferred_reduction_ratio": report["recommendation"][
                    "preferred_reduction_ratio"
                ],
                "recommended_mechanical_stroke_m": report["stopping_envelope"][
                    "recommended_mechanical_stroke_m"
                ],
                "passed": report["passed"],
                "valid_for_procurement": report["valid_for_procurement"],
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
