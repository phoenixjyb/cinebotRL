#!/usr/bin/env python3
"""Audit whether the provisional belt-drive sizing matches the Isaac riser plant."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = (
    PROJECT_ROOT
    / "assets_own/recomoProto2_two_wheel_riser/recomoProto2_two_wheel_riser.urdf"
)
DEFAULT_BUILD_AUDIT = DEFAULT_URDF.with_name("build_audit.json")
DEFAULT_CONFIG = (
    PROJECT_ROOT / "src/rl_platform/robots/two_wheel_balance/config.py"
)
DEFAULT_THERMAL_CONTROL = (
    PROJECT_ROOT / "src/rl_platform/tasks/two_wheel_balance/riser_control.py"
)
DEFAULT_PLAYBACK_RUNNER = (
    PROJECT_ROOT / "scripts/two_wheel_balance/smoke_riser_reference_playback.py"
)
DEFAULT_HARDWARE_ENVELOPE = (
    PROJECT_ROOT
    / "artifacts/two_wheel_riser/20260717_hardware_envelope_v1/summary.json"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _riser_config_values(path: Path) -> dict[str, float]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for statement in module.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Subscript):
            continue
        if not (
            isinstance(target.value, ast.Attribute)
            and target.value.attr == "actuators"
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "TWO_WHEEL_RISER_CFG"
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "riser_position"
            and isinstance(statement.value, ast.Call)
        ):
            continue
        keywords = {
            item.arg: ast.literal_eval(item.value)
            for item in statement.value.keywords
            if item.arg is not None
        }
        return {
            "effort_limit_sim_n": float(keywords["effort_limit_sim"]),
            "velocity_limit_sim_mps": float(keywords["velocity_limit_sim"]),
            "stiffness_n_per_m": float(keywords["stiffness"]),
            "damping_ns_per_m": float(keywords["damping"]),
            "armature_kg": float(keywords["armature"]),
        }
    raise ValueError("riser actuator configuration was not found")


def _riser_urdf_values(path: Path) -> dict[str, float]:
    root = ET.parse(path).getroot()
    joint = root.find("./joint[@name='riser_joint']")
    if joint is None:
        raise ValueError("riser_joint is absent from URDF")
    limit = joint.find("limit")
    if limit is None:
        raise ValueError("riser_joint limit is absent from URDF")
    return {
        "lower_m": float(limit.attrib["lower"]),
        "upper_m": float(limit.attrib["upper"]),
        "effort_limit_n": float(limit.attrib["effort"]),
        "velocity_limit_mps": float(limit.attrib["velocity"]),
    }


def _thermal_monitor_values(path: Path) -> dict[str, float | str]:
    module = ast.parse(path.read_text(encoding="utf-8"))
    contract = None
    monitor = None
    for statement in module.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == "RISER_THERMAL_FORCE_CONTRACT"
        ):
            contract = ast.literal_eval(statement.value)
        if isinstance(statement, ast.ClassDef) and statement.name == (
            "RiserMotorThermalMonitor"
        ):
            monitor = statement
    if not isinstance(contract, str) or monitor is None:
        raise ValueError("riser thermal monitor contract was not found")
    defaults = {
        statement.target.id: ast.literal_eval(statement.value)
        for statement in monitor.body
        if isinstance(statement, ast.AnnAssign)
        and isinstance(statement.target, ast.Name)
        and statement.value is not None
    }
    return {
        "contract": contract,
        "continuous_force_n": float(defaults["continuous_force_n"]),
        "peak_force_n": float(defaults["peak_force_n"]),
        "thermal_time_constant_s": float(defaults["thermal_time_constant_s"]),
    }


def _runner_thermal_checks(path: Path) -> dict[str, bool]:
    source = path.read_text(encoding="utf-8")
    return {
        "monitor_is_instantiated": "RiserMotorThermalMonitor()" in source,
        "applied_force_is_sampled_each_step": (
            "riser_thermal_monitor.step(riser_effort, 1.0 / POLICY_HZ)" in source
        ),
        "force_observation_is_gated": '"riser_thermal_force_observed"' in source,
        "thermal_load_is_gated": '"riser_thermal_load_bounded"' in source,
        "peak_force_is_gated": '"riser_peak_force_bounded"' in source,
        "contract_is_reported": (
            '"riser_thermal_force_contract": RISER_THERMAL_FORCE_CONTRACT'
            in source
        ),
    }


def _case(report: dict, section: str, ratio: float, mass: float) -> dict:
    for row in report[section]:
        if (
            math.isclose(float(row["reduction_ratio"]), ratio)
            and math.isclose(float(row["moving_mass_kg"]), mass)
        ):
            return row
    raise ValueError(f"missing {section} case for ratio={ratio}, mass={mass}")


def build_report(
    urdf_path: Path = DEFAULT_URDF,
    build_audit_path: Path = DEFAULT_BUILD_AUDIT,
    config_path: Path = DEFAULT_CONFIG,
    hardware_envelope_path: Path = DEFAULT_HARDWARE_ENVELOPE,
    thermal_control_path: Path = DEFAULT_THERMAL_CONTROL,
    playback_runner_path: Path = DEFAULT_PLAYBACK_RUNNER,
) -> dict:
    urdf = _riser_urdf_values(urdf_path)
    config = _riser_config_values(config_path)
    build_audit = json.loads(build_audit_path.read_text(encoding="utf-8"))
    hardware = json.loads(hardware_envelope_path.read_text(encoding="utf-8"))
    thermal = _thermal_monitor_values(thermal_control_path)
    thermal_runner_checks = _runner_thermal_checks(playback_runner_path)

    inputs = hardware["inputs"]
    motor = hardware["motor_candidate"]
    ratio = float(hardware["recommendation"]["preferred_reduction_ratio"])
    lead = float(inputs["linear_lead_m_per_rev"])
    efficiency = float(inputs["transmission_efficiency"]) * float(
        inputs["reduction_efficiency"]
    )
    rated_force = (
        float(motor["rated_torque_nm"])
        * ratio
        * efficiency
        * 2.0
        * math.pi
        / lead
    )
    peak_force = (
        float(motor["peak_torque_nm"])
        * ratio
        * efficiency
        * 2.0
        * math.pi
        / lead
    )
    rated_speed = (
        float(motor["rated_speed_rpm"]) / 60.0 * lead / ratio
    )
    peak_speed = float(motor["peak_speed_rpm"]) / 60.0 * lead / ratio
    motor_speed_at_sim_limit = urdf["velocity_limit_mps"] / lead * 60.0 * ratio
    heavy_normal = _case(hardware, "sizing_cases", ratio, 8.0)
    heavy_emergency = _case(hardware, "emergency_braking_cases", ratio, 8.0)
    simulated_force_ratio = urdf["effort_limit_n"] / rated_force

    checks = {
        "hardware_envelope_passed": hardware.get("passed") is True,
        "urdf_build_passed": build_audit.get("passed") is True,
        "urdf_and_isaac_force_limits_match": math.isclose(
            urdf["effort_limit_n"], config["effort_limit_sim_n"]
        ),
        "urdf_and_isaac_speed_limits_match": math.isclose(
            urdf["velocity_limit_mps"], config["velocity_limit_sim_mps"]
        ),
        "camera_range_is_exactly_0_6_to_1_8_m": build_audit["riser"][
            "camera_height_range_m"
        ]
        == [0.6, 1.8],
        "simulated_speed_is_below_motor_rated_speed": (
            motor_speed_at_sim_limit <= float(motor["rated_speed_rpm"])
        ),
        "simulated_force_is_below_motor_peak_force": (
            urdf["effort_limit_n"] <= peak_force
        ),
        "simulated_force_is_within_five_percent_of_rated_force": (
            simulated_force_ratio <= 1.05
        ),
        "rated_force_covers_8kg_normal_design_force": (
            rated_force >= float(heavy_normal["design_force_n"])
        ),
        "rated_force_covers_8kg_emergency_design_force": (
            rated_force >= float(heavy_emergency["design_force_n"])
        ),
        "counterbalance_not_required_for_stated_force_checks": (
            rated_force >= float(heavy_normal["design_force_n"])
            and rated_force >= float(heavy_emergency["design_force_n"])
        ),
        "thermal_monitor_continuous_force_matches_drive": math.isclose(
            float(thermal["continuous_force_n"]), rated_force, rel_tol=1e-12
        ),
        "thermal_monitor_peak_force_matches_drive": math.isclose(
            float(thermal["peak_force_n"]), peak_force, rel_tol=1e-12
        ),
        "thermal_monitor_contract_matches": thermal["contract"]
        == "leadshine_400w_first_order_monitor_v1",
        "thermal_monitor_is_wired_into_dynamic_admission": all(
            thermal_runner_checks.values()
        ),
    }
    continuous_force_parity = urdf["effort_limit_n"] <= rated_force
    concept_screening_passed = all(checks.values())
    return {
        "schema": "cinebotrl_two_wheel_riser_hardware_sim_parity_v2",
        "sources": {
            "urdf": str(urdf_path.resolve()),
            "urdf_sha256": sha256_file(urdf_path),
            "build_audit": str(build_audit_path.resolve()),
            "build_audit_sha256": sha256_file(build_audit_path),
            "isaac_config": str(config_path.resolve()),
            "isaac_config_sha256": sha256_file(config_path),
            "hardware_envelope": str(hardware_envelope_path.resolve()),
            "hardware_envelope_sha256": sha256_file(hardware_envelope_path),
            "thermal_control": str(thermal_control_path.resolve()),
            "thermal_control_sha256": sha256_file(thermal_control_path),
            "playback_runner": str(playback_runner_path.resolve()),
            "playback_runner_sha256": sha256_file(playback_runner_path),
        },
        "plant": {
            "urdf": urdf,
            "isaac_actuator": config,
            "provisional_moving_mass_kg": build_audit[
                "provisional_moving_mass_kg"
            ],
            "camera_height_range_m": build_audit["riser"][
                "camera_height_range_m"
            ],
        },
        "drive": {
            "motor_model": motor["model"],
            "reduction_ratio": ratio,
            "linear_lead_m_per_rev": lead,
            "combined_efficiency": efficiency,
            "rated_linear_force_n": rated_force,
            "peak_linear_force_n": peak_force,
            "rated_linear_speed_mps": rated_speed,
            "peak_linear_speed_mps": peak_speed,
            "motor_speed_at_sim_velocity_limit_rpm": motor_speed_at_sim_limit,
            "simulated_to_rated_force_ratio": simulated_force_ratio,
            "rated_force_margin_over_8kg_emergency": (
                rated_force / float(heavy_emergency["design_force_n"])
            ),
        },
        "thermal_admission": {
            **thermal,
            "runner_checks": thermal_runner_checks,
            "active_force_derating": False,
            "model_parameter_status": "provisional_until_bench_identification",
        },
        "classification": {
            "concept_screening_passed": concept_screening_passed,
            "continuous_rated_force_parity_passed": continuous_force_parity,
            "simulated_300n_interpretation": (
                "transient_actuator_cap_not_continuous_rated_force"
            ),
            "counterbalance_modeled": False,
            "counterbalance_required_for_current_sizing_pass": False,
            "training_admission_thermal_monitor_present": all(
                thermal_runner_checks.values()
            ),
            "active_thermal_force_derater_present": False,
            "valid_for_procurement": False,
            "valid_for_hardware_transfer": False,
            "valid_for_training": False,
        },
        "required_before_hardware_transfer": [
            "measure complete moving mass and carriage friction",
            "measure continuous and transient output force at the carriage",
            "model or disable the final counterbalance design consistently",
            "replace provisional thermal constants from bench measurements",
            "add an active continuous-current or thermal force derater",
            "validate regeneration, brake, hard limits, and independent anti-fall",
        ],
        "checks": checks,
        "passed": concept_screening_passed,
        "gpu_work_started": False,
        "residual_capture_started": False,
        "bc_started": False,
        "ppo_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--build-audit", type=Path, default=DEFAULT_BUILD_AUDIT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--hardware-envelope", type=Path, default=DEFAULT_HARDWARE_ENVELOPE
    )
    parser.add_argument(
        "--thermal-control", type=Path, default=DEFAULT_THERMAL_CONTROL
    )
    parser.add_argument(
        "--playback-runner", type=Path, default=DEFAULT_PLAYBACK_RUNNER
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        args.urdf,
        args.build_audit,
        args.config,
        args.hardware_envelope,
        args.thermal_control,
        args.playback_runner,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["classification"], indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
