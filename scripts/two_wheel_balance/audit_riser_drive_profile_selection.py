#!/usr/bin/env python3
"""Audit the active riser drive profile without silently upgrading the plant."""

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
PARITY_SCRIPT = (
    PROJECT_ROOT / "scripts/two_wheel_balance/audit_riser_hardware_sim_parity.py"
)
HARDWARE_SCRIPT = (
    PROJECT_ROOT / "scripts/two_wheel_balance/calculate_riser_hardware_envelope.py"
)
DEFAULT_PRODUCTION_AUDIT = (
    PROJECT_ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_hardware_production_candidate_v1/summary.json"
)
ACTIVE_PROFILE = "leadshine_400w_engineering_sample_v1"
PRODUCTION_CANDIDATE_PROFILE = "leadshine_750w_production_candidate_v1"


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


def _linear_force(
    torque_nm: float, ratio: float, efficiency: float, lead_m_per_rev: float
) -> float:
    return torque_nm * ratio * efficiency * 2.0 * math.pi / lead_m_per_rev


def build_report(
    production: Mapping[str, Any],
    *,
    urdf_path: Path | None = None,
    config_path: Path | None = None,
    thermal_control_path: Path | None = None,
) -> dict[str, Any]:
    parity = _load_module("riser_profile_parity", PARITY_SCRIPT)
    hardware_module = _load_module("riser_profile_hardware", HARDWARE_SCRIPT)
    hardware = hardware_module.build_report()
    urdf = parity._riser_urdf_values(urdf_path or parity.DEFAULT_URDF)
    config = parity._riser_config_values(config_path or parity.DEFAULT_CONFIG)
    thermal = parity._thermal_monitor_values(
        thermal_control_path or parity.DEFAULT_THERMAL_CONTROL
    )

    inputs = hardware["inputs"]
    motor = hardware["motor_candidate"]
    ratio = float(hardware["recommendation"]["preferred_reduction_ratio"])
    lead = float(inputs["linear_lead_m_per_rev"])
    efficiency = float(inputs["transmission_efficiency"]) * float(
        inputs["reduction_efficiency"]
    )
    engineering_rated_force = _linear_force(
        float(motor["rated_torque_nm"]), ratio, efficiency, lead
    )
    engineering_peak_force = _linear_force(
        float(motor["peak_torque_nm"]), ratio, efficiency, lead
    )
    candidate = production["calculated"]

    checks = {
        "engineering_envelope_passed": hardware.get("passed") is True,
        "active_urdf_force_is_300n": math.isclose(
            urdf["effort_limit_n"], 300.0
        ),
        "active_urdf_speed_is_1mps": math.isclose(
            urdf["velocity_limit_mps"], 1.0
        ),
        "active_isaac_matches_urdf": math.isclose(
            config["effort_limit_sim_n"], urdf["effort_limit_n"]
        )
        and math.isclose(
            config["velocity_limit_sim_mps"], urdf["velocity_limit_mps"]
        ),
        "active_thermal_contract_is_400w": thermal["contract"]
        == "leadshine_400w_first_order_monitor_v1",
        "active_thermal_force_matches_400w": math.isclose(
            float(thermal["continuous_force_n"]),
            engineering_rated_force,
            rel_tol=1e-12,
        )
        and math.isclose(
            float(thermal["peak_force_n"]),
            engineering_peak_force,
            rel_tol=1e-12,
        ),
        "production_candidate_audit_passed": production.get("passed") is True
        and production.get("candidate_ready_for_supplier_and_bench_review")
        is True,
        "production_candidate_not_simulation_enabled": production.get(
            "simulation_motor_model_updated"
        )
        is False,
        "production_candidate_not_runtime_authorized": production.get(
            "runtime_authorized"
        )
        is False,
        "production_candidate_not_training_authorized": production.get(
            "valid_for_training"
        )
        is False,
        "production_candidate_not_hardware_transfer_ready": production.get(
            "valid_for_hardware_transfer"
        )
        is False,
        "current_plant_does_not_claim_750w_force": not math.isclose(
            urdf["effort_limit_n"],
            float(candidate["rated_linear_force_n"]),
            rel_tol=0.01,
        ),
        "camera_height_ceiling_remains_1p8m": [
            float(inputs["camera_height_min_m"]),
            float(inputs["camera_height_max_m"]),
        ]
        == [0.6, 1.8],
        "riser_speed_target_remains_1mps": math.isclose(
            float(inputs["maximum_velocity_mps"]), 1.0
        ),
    }
    passed = all(checks.values())
    return {
        "schema": "cinebotrl_two_wheel_riser_drive_profile_selection_v1",
        "checks": checks,
        "active_simulation_profile": {
            "name": ACTIVE_PROFILE,
            "status": "active_provisional_engineering_profile",
            "motor_model": motor["model"],
            "simulation_effort_limit_n": urdf["effort_limit_n"],
            "simulation_velocity_limit_mps": urdf["velocity_limit_mps"],
            "continuous_force_reference_n": engineering_rated_force,
            "peak_force_reference_n": engineering_peak_force,
            "thermal_contract": thermal["contract"],
            "valid_for_hardware_transfer": False,
        },
        "production_design_candidate": {
            "name": PRODUCTION_CANDIDATE_PROFILE,
            "status": "supplier_and_bench_review_only",
            "motor_and_drive": production["recommendation"][
                "production_design_review_candidate"
            ],
            "rated_linear_force_n": candidate["rated_linear_force_n"],
            "peak_linear_force_n": candidate["peak_linear_force_n"],
            "motor_speed_at_1mps_rpm": candidate["motor_speed_at_1mps_rpm"],
            "simulation_enabled": False,
            "runtime_authorized": False,
            "valid_for_training": False,
            "valid_for_hardware_transfer": False,
        },
        "profile_switch_contract": {
            "environment_or_cli_switch_supported": False,
            "source_change_and_review_required": True,
            "required_changes": [
                "supplier_approve_vertical_axis_gearbox_and_tooth_jump_duty",
                "bench_measure_moving_mass_friction_force_current_temperature_"
                "and_regeneration",
                "pass_independent_anti_fall_hard_limit_and_safety_power_removal_tests",
                "update_urdf_and_isaac_effort_limits_together",
                "replace_400w_thermal_contract_with_bench_identified_750w_contract",
                "rebuild_usd_and_reseal_all_asset_hashes",
                "rerun_static_dynamic_exact_source_and_full79_controller_gates",
                "recapture_corrective_teacher_data_under_the_new_plant_identity",
            ],
            "existing_dynamic_evidence_reusable_after_switch": False,
            "existing_corrective_captures_reusable_after_switch": False,
            "existing_bc_checkpoint_reusable_after_switch": False,
        },
        "classification": {
            "active_profile_is_explicit": passed,
            "silent_750w_simulation_upgrade_rejected": passed,
            "candidate_ready_for_supplier_and_bench_review": (
                production.get("candidate_ready_for_supplier_and_bench_review")
                is True
            ),
            "valid_for_production_procurement": False,
            "valid_for_hardware_transfer": False,
            "runtime_authorized": False,
            "valid_for_training": False,
        },
        "passed": passed,
        "gpu_work_started": False,
        "isaac_started": False,
        "capture_started": False,
        "bc_started": False,
        "ppo_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production-audit", type=Path, default=DEFAULT_PRODUCTION_AUDIT
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    production = json.loads(args.production_audit.read_text(encoding="utf-8"))
    report = build_report(production)
    report["inputs"] = {
        "production_audit": _identity(args.production_audit),
        "parity_script": _identity(PARITY_SCRIPT),
        "hardware_script": _identity(HARDWARE_SCRIPT),
        "urdf": _identity(
            PROJECT_ROOT
            / "assets_own/recomoProto2_two_wheel_riser/"
            "recomoProto2_two_wheel_riser.urdf"
        ),
        "isaac_config": _identity(
            PROJECT_ROOT / "src/rl_platform/robots/two_wheel_balance/config.py"
        ),
        "thermal_control": _identity(
            PROJECT_ROOT
            / "src/rl_platform/tasks/two_wheel_balance/riser_control.py"
        ),
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite profile audit: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
