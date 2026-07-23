#!/usr/bin/env python3
"""Audit a supplier response for the two-wheel riser production candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESPONSE = (
    PROJECT_ROOT
    / "docs/03_training/two_wheel_balance/"
    "RISER_750W_SUPPLIER_RESPONSE_TEMPLATE_20260723.json"
)
DEFAULT_CANDIDATE = (
    PROJECT_ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_hardware_production_candidate_v1/summary.json"
)

SCHEMA = "cinebotrl_two_wheel_riser_supplier_response_v1"
AUDIT_SCHEMA = "cinebotrl_two_wheel_riser_supplier_response_audit_v1"
REQUEST_ID = "cinebotrl_two_wheel_riser_750w_axis_20260723"
MOTOR_MODEL = "ELVM8075V48EH-M17-HD"
DRIVE_MODEL = "ELD2-CAN7020B"
ALLOWED_ARCHITECTURES = {
    "guided_belt_axis",
    "two_stage_synchronized_telescoping_mast",
}

DESIGN_MOVING_MASS_KG = 8.0
CAMERA_HEIGHT_MIN_M = 0.60
CAMERA_HEIGHT_MAX_M = 1.80
SOFTWARE_WORKING_STROKE_M = 1.20
MIN_MECHANICAL_STROKE_M = 1.50
MAXIMUM_LINEAR_SPEED_MPS = 1.0
MIN_LINEAR_ACCELERATION_MPS2 = 2.0
MIN_LINEAR_JERK_MPS3 = 8.0
MIN_EMERGENCY_DECELERATION_MPS2 = 5.0
MIN_DUTY_CYCLE_FRACTION = 0.60
MIN_CONTINUOUS_DURATION_S = 1800.0
MIN_AXIS_CONTINUOUS_VERTICAL_FORCE_N = 300.0
MIN_ANTI_FALL_HOLDING_FORCE_N = 2.0 * DESIGN_MOVING_MASS_KG * 9.81
MAX_ANTI_FALL_CATCH_DISTANCE_M = 0.03

STRING_FIELDS = (
    "supplier.company",
    "supplier.engineer",
    "supplier.response_date",
    "configuration.axis_model",
    "configuration.axis_architecture",
    "configuration.gearbox_model",
    "safety.regenerative_absorption_model",
    "safety.independent_anti_fall_model",
    "safety.safety_power_removal_architecture",
    "safety.safety_category_or_standard",
)
NUMBER_FIELDS = (
    "configuration.reduction_ratio",
    "configuration.linear_lead_m_per_rev",
    "configuration.mechanical_stroke_m",
    "configuration.software_working_stroke_m",
    "configuration.software_camera_height_min_m",
    "configuration.software_camera_height_max_m",
    "ratings.rated_moving_mass_kg",
    "ratings.maximum_linear_speed_mps",
    "ratings.maximum_linear_acceleration_mps2",
    "ratings.maximum_linear_jerk_mps3",
    "ratings.minimum_emergency_deceleration_mps2",
    "ratings.continuous_vertical_force_n",
    "ratings.rated_eccentric_pitch_moment_nm",
    "ratings.rated_eccentric_yaw_moment_nm",
    "ratings.vertical_service_duty_cycle_fraction",
    "ratings.vertical_service_continuous_duration_s",
    "ratings.declared_cycle_life",
    "transmission.gearbox_continuous_input_speed_rpm",
    "transmission.gearbox_emergency_input_torque_nm",
    "transmission.gearbox_efficiency_fraction",
    "transmission.gearbox_backlash_deg",
    "transmission.belt_tooth_jump_margin_ratio",
    "safety.regenerative_bus_voltage_limit_v",
    "safety.independent_anti_fall_rated_holding_force_n",
    "safety.independent_anti_fall_maximum_catch_distance_m",
)
BOOLEAN_FIELDS = (
    "approvals.vertical_mobile_axis_duty_approved",
    "approvals.gearbox_continuous_speed_approved",
    "approvals.gearbox_emergency_braking_torque_approved",
    "approvals.belt_tooth_jump_margin_approved",
    "approvals.regenerative_absorption_approved",
    "approvals.independent_anti_fall_approved",
    "approvals.independent_hard_limits_and_end_stops_approved",
    "approvals.safety_power_removal_approved",
    "safety.motor_holding_brake_is_static_only",
    "safety.motor_holding_brake_used_for_dynamic_stop",
)
HASH_FIELDS = (
    "evidence.axis_datasheet_sha256",
    "evidence.gearbox_datasheet_sha256",
    "evidence.selection_calculation_sha256",
    "evidence.vertical_duty_approval_sha256",
    "evidence.regeneration_calculation_sha256",
    "evidence.safety_architecture_sha256",
    "evidence.signed_supplier_response_sha256",
)


def _get(payload: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for key in dotted_path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _number(value: Any, *, minimum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and (minimum is None or numeric >= minimum)


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _close(value: Any, expected: float, tolerance: float = 1e-9) -> bool:
    return _number(value) and math.isclose(
        float(value), expected, rel_tol=0.0, abs_tol=tolerance
    )


def _at_least(value: Any, minimum: float) -> bool:
    return _number(value) and float(value) >= minimum


def _at_most(value: Any, maximum: float) -> bool:
    return _number(value, minimum=0.0) and float(value) <= maximum


def _identity(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display = str(resolved)
    return {
        "path": display,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build_report(
    response: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    response_sha256: str,
) -> dict[str, Any]:
    missing_fields = [
        field for field in STRING_FIELDS if not _string(_get(response, field))
    ]
    missing_fields.extend(
        field for field in NUMBER_FIELDS if not _number(_get(response, field))
    )
    missing_fields.extend(
        field
        for field in BOOLEAN_FIELDS
        if not isinstance(_get(response, field), bool)
    )
    missing_fields.extend(
        field for field in HASH_FIELDS if not _sha256(_get(response, field))
    )

    calculated = candidate.get("calculated", {})
    required_gearbox_speed_rpm = float(calculated["motor_speed_at_1mps_rpm"])
    required_gearbox_emergency_torque_nm = (
        float(calculated["emergency_8kg_motor_torque_nm"])
        * float(calculated["required_emergency_force_margin_ratio"])
    )

    checks = {
        "response_schema_matches": response.get("schema") == SCHEMA,
        "request_identity_is_pinned": response.get("request_id") == REQUEST_ID,
        "response_identity_complete": not any(
            field.startswith("supplier.") for field in missing_fields
        ),
        "production_candidate_source_passed": candidate.get("passed") is True
        and candidate.get("candidate_ready_for_supplier_and_bench_review") is True
        and candidate.get("simulation_motor_model_updated") is False,
        "candidate_motor_is_pinned": _get(response, "configuration.motor_model")
        == MOTOR_MODEL,
        "candidate_drive_is_pinned": _get(response, "configuration.drive_model")
        == DRIVE_MODEL,
        "axis_architecture_is_supported": _get(
            response, "configuration.axis_architecture"
        )
        in ALLOWED_ARCHITECTURES,
        "transmission_ratio_is_3_to_1": _close(
            _get(response, "configuration.reduction_ratio"), 3.0
        ),
        "linear_lead_is_70mm_per_rev": _close(
            _get(response, "configuration.linear_lead_m_per_rev"), 0.07
        ),
        "mechanical_stroke_preserves_full_speed_stopping_space": _at_least(
            _get(response, "configuration.mechanical_stroke_m"),
            MIN_MECHANICAL_STROKE_M,
        ),
        "software_working_stroke_is_1p2m": _close(
            _get(response, "configuration.software_working_stroke_m"),
            SOFTWARE_WORKING_STROKE_M,
        ),
        "camera_height_minimum_is_0p6m": _close(
            _get(response, "configuration.software_camera_height_min_m"),
            CAMERA_HEIGHT_MIN_M,
        ),
        "camera_height_ceiling_is_1p8m": _close(
            _get(response, "configuration.software_camera_height_max_m"),
            CAMERA_HEIGHT_MAX_M,
        ),
        "axis_is_rated_for_8kg_moving_mass": _at_least(
            _get(response, "ratings.rated_moving_mass_kg"),
            DESIGN_MOVING_MASS_KG,
        ),
        "axis_speed_covers_1mps": _at_least(
            _get(response, "ratings.maximum_linear_speed_mps"),
            MAXIMUM_LINEAR_SPEED_MPS,
        ),
        "axis_acceleration_and_jerk_cover_contract": _at_least(
            _get(response, "ratings.maximum_linear_acceleration_mps2"),
            MIN_LINEAR_ACCELERATION_MPS2,
        )
        and _at_least(
            _get(response, "ratings.maximum_linear_jerk_mps3"),
            MIN_LINEAR_JERK_MPS3,
        ),
        "emergency_deceleration_covers_contract": _at_least(
            _get(response, "ratings.minimum_emergency_deceleration_mps2"),
            MIN_EMERGENCY_DECELERATION_MPS2,
        ),
        "axis_continuous_force_covers_contract": _at_least(
            _get(response, "ratings.continuous_vertical_force_n"),
            MIN_AXIS_CONTINUOUS_VERTICAL_FORCE_N,
        ),
        "eccentric_moment_ratings_are_declared": _at_least(
            _get(response, "ratings.rated_eccentric_pitch_moment_nm"), 0.0
        )
        and _at_least(
            _get(response, "ratings.rated_eccentric_yaw_moment_nm"), 0.0
        ),
        "vertical_duty_covers_bench_contract": _at_least(
            _get(response, "ratings.vertical_service_duty_cycle_fraction"),
            MIN_DUTY_CYCLE_FRACTION,
        )
        and _at_least(
            _get(response, "ratings.vertical_service_continuous_duration_s"),
            MIN_CONTINUOUS_DURATION_S,
        ),
        "cycle_life_is_declared": _at_least(
            _get(response, "ratings.declared_cycle_life"), 1.0
        ),
        "gearbox_speed_covers_1mps": _at_least(
            _get(response, "transmission.gearbox_continuous_input_speed_rpm"),
            required_gearbox_speed_rpm,
        ),
        "gearbox_emergency_torque_has_15_percent_margin": _at_least(
            _get(response, "transmission.gearbox_emergency_input_torque_nm"),
            required_gearbox_emergency_torque_nm,
        ),
        "gearbox_efficiency_matches_model": _at_least(
            _get(response, "transmission.gearbox_efficiency_fraction"), 0.95
        ),
        "gearbox_backlash_is_declared": _at_least(
            _get(response, "transmission.gearbox_backlash_deg"), 0.0
        ),
        "belt_tooth_jump_margin_exceeds_unity": _at_least(
            _get(response, "transmission.belt_tooth_jump_margin_ratio"),
            1.0 + 1e-9,
        ),
        "supplier_approvals_are_all_positive": all(
            _get(response, field) is True
            for field in BOOLEAN_FIELDS
            if field.startswith("approvals.")
        ),
        "holding_brake_is_static_only": _get(
            response, "safety.motor_holding_brake_is_static_only"
        )
        is True
        and _get(response, "safety.motor_holding_brake_used_for_dynamic_stop")
        is False,
        "regenerative_bus_limit_is_within_drive_contract": _at_most(
            _get(response, "safety.regenerative_bus_voltage_limit_v"), 65.0
        ),
        "independent_anti_fall_has_two_x_static_capacity": _at_least(
            _get(response, "safety.independent_anti_fall_rated_holding_force_n"),
            MIN_ANTI_FALL_HOLDING_FORCE_N,
        ),
        "independent_anti_fall_catch_distance_is_at_most_30mm": _at_most(
            _get(response, "safety.independent_anti_fall_maximum_catch_distance_m"),
            MAX_ANTI_FALL_CATCH_DISTANCE_M,
        ),
        "safety_architecture_is_named": all(
            _string(_get(response, field))
            for field in (
                "safety.regenerative_absorption_model",
                "safety.independent_anti_fall_model",
                "safety.safety_power_removal_architecture",
                "safety.safety_category_or_standard",
            )
        ),
        "all_required_evidence_hashes_present": all(
            _sha256(_get(response, field)) for field in HASH_FIELDS
        ),
    }
    complete = not missing_fields
    passed = complete and all(checks.values())

    if not complete:
        decision = "collect_complete_signed_supplier_response"
    elif not all(
        checks[name]
        for name in (
            "candidate_motor_is_pinned",
            "candidate_drive_is_pinned",
            "request_identity_is_pinned",
            "axis_architecture_is_supported",
            "transmission_ratio_is_3_to_1",
            "linear_lead_is_70mm_per_rev",
            "mechanical_stroke_preserves_full_speed_stopping_space",
            "software_working_stroke_is_1p2m",
            "camera_height_minimum_is_0p6m",
            "camera_height_ceiling_is_1p8m",
        )
    ):
        decision = "reject_supplier_configuration_contract_mismatch"
    elif not passed:
        decision = "repair_supplier_rating_safety_or_evidence_gap"
    else:
        decision = "supplier_response_qualified_for_bench_input_only"

    supplier_merge = {
        "required_candidate": {
            "motor_model": MOTOR_MODEL,
            "drive_model": DRIVE_MODEL,
            "drive_profile": "leadshine_750w_production_candidate_v1",
            "reduction_ratio": 3.0,
            "linear_lead_m_per_rev": 0.07,
        },
        "evidence": {"supplier_approval_package_sha256": response_sha256},
        "supplier_evidence": {
            "vertical_mobile_axis_duty_approved": bool(
                passed
                and _get(
                    response, "approvals.vertical_mobile_axis_duty_approved"
                )
                is True
            ),
            "gearbox_continuous_speed_approved": bool(
                passed
                and _get(
                    response, "approvals.gearbox_continuous_speed_approved"
                )
                is True
            ),
            "gearbox_emergency_braking_torque_approved": bool(
                passed
                and _get(
                    response,
                    "approvals.gearbox_emergency_braking_torque_approved",
                )
                is True
            ),
            "belt_tooth_jump_margin_approved": bool(
                passed
                and _get(
                    response, "approvals.belt_tooth_jump_margin_approved"
                )
                is True
            ),
        },
    }
    return {
        "schema": AUDIT_SCHEMA,
        "missing_or_invalid_response_fields": sorted(missing_fields),
        "requirements": {
            "design_moving_mass_kg": DESIGN_MOVING_MASS_KG,
            "software_camera_height_range_m": [
                CAMERA_HEIGHT_MIN_M,
                CAMERA_HEIGHT_MAX_M,
            ],
            "software_working_stroke_m": SOFTWARE_WORKING_STROKE_M,
            "minimum_mechanical_stroke_m": MIN_MECHANICAL_STROKE_M,
            "maximum_linear_speed_mps": MAXIMUM_LINEAR_SPEED_MPS,
            "minimum_linear_acceleration_mps2": MIN_LINEAR_ACCELERATION_MPS2,
            "minimum_linear_jerk_mps3": MIN_LINEAR_JERK_MPS3,
            "minimum_emergency_deceleration_mps2": (
                MIN_EMERGENCY_DECELERATION_MPS2
            ),
            "minimum_axis_continuous_vertical_force_n": (
                MIN_AXIS_CONTINUOUS_VERTICAL_FORCE_N
            ),
            "minimum_gearbox_continuous_input_speed_rpm": (
                required_gearbox_speed_rpm
            ),
            "minimum_gearbox_emergency_input_torque_nm": (
                required_gearbox_emergency_torque_nm
            ),
            "minimum_anti_fall_holding_force_n": MIN_ANTI_FALL_HOLDING_FORCE_N,
            "maximum_anti_fall_catch_distance_m": (
                MAX_ANTI_FALL_CATCH_DISTANCE_M
            ),
        },
        "checks": checks,
        "decision": decision,
        "passed": passed,
        "supplier_response_complete": complete,
        "valid_for_bench_supplier_evidence_merge": False,
        "valid_for_current_400w_bench_supplier_evidence_merge": False,
        "valid_for_750w_bench_supplier_evidence_merge": passed,
        "candidate_identity_match_required_before_merge": True,
        "bench_measurement_merge_fragment": supplier_merge,
        "ready_for_production_design_review": False,
        "valid_for_production_procurement": False,
        "valid_for_hardware_transfer": False,
        "simulation_motor_model_updated": False,
        "valid_for_training": False,
        "runtime_authorized": False,
        "gpu_work_started": False,
        "bc_authorized": False,
        "ppo_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, default=DEFAULT_RESPONSE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    response = json.loads(args.response.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    response_identity = _identity(args.response)
    report = build_report(
        response,
        candidate,
        response_sha256=response_identity["sha256"],
    )
    report["inputs"] = {
        "supplier_response": response_identity,
        "production_candidate": _identity(args.candidate),
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite supplier audit: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 7


if __name__ == "__main__":
    raise SystemExit(main())
