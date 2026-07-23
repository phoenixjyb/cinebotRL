import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/audit_riser_supplier_response.py"
BENCH_750W_TEMPLATE = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "RISER_750W_BENCH_MEASUREMENT_TEMPLATE_20260723.json"
)
SPEC = importlib.util.spec_from_file_location("riser_supplier_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _candidate() -> dict:
    return json.loads(MODULE.DEFAULT_CANDIDATE.read_text(encoding="utf-8"))


def _healthy_response() -> dict:
    return {
        "schema": MODULE.SCHEMA,
        "request_id": MODULE.REQUEST_ID,
        "supplier": {
            "company": "Synthetic Axis Supplier",
            "engineer": "Test Engineer",
            "response_date": "2026-07-23",
        },
        "configuration": {
            "axis_model": "TEST-AXIS-1500",
            "axis_architecture": "guided_belt_axis",
            "motor_model": MODULE.MOTOR_MODEL,
            "drive_model": MODULE.DRIVE_MODEL,
            "gearbox_model": "TEST-GBX-3",
            "reduction_ratio": 3.0,
            "linear_lead_m_per_rev": 0.07,
            "mechanical_stroke_m": 1.5,
            "software_working_stroke_m": 1.2,
            "software_camera_height_min_m": 0.6,
            "software_camera_height_max_m": 1.8,
        },
        "ratings": {
            "rated_moving_mass_kg": 8.0,
            "maximum_linear_speed_mps": 1.2,
            "maximum_linear_acceleration_mps2": 2.5,
            "maximum_linear_jerk_mps3": 10.0,
            "minimum_emergency_deceleration_mps2": 5.5,
            "continuous_vertical_force_n": 350.0,
            "rated_eccentric_pitch_moment_nm": 20.0,
            "rated_eccentric_yaw_moment_nm": 15.0,
            "vertical_service_duty_cycle_fraction": 0.7,
            "vertical_service_continuous_duration_s": 3600.0,
            "declared_cycle_life": 100000.0,
        },
        "transmission": {
            "gearbox_continuous_input_speed_rpm": 3000.0,
            "gearbox_emergency_input_torque_nm": 1.5,
            "gearbox_efficiency_fraction": 0.96,
            "gearbox_backlash_deg": 0.2,
            "belt_tooth_jump_margin_ratio": 1.5,
        },
        "approvals": {
            "vertical_mobile_axis_duty_approved": True,
            "gearbox_continuous_speed_approved": True,
            "gearbox_emergency_braking_torque_approved": True,
            "belt_tooth_jump_margin_approved": True,
            "regenerative_absorption_approved": True,
            "independent_anti_fall_approved": True,
            "independent_hard_limits_and_end_stops_approved": True,
            "safety_power_removal_approved": True,
        },
        "safety": {
            "motor_holding_brake_is_static_only": True,
            "motor_holding_brake_used_for_dynamic_stop": False,
            "regenerative_absorption_model": "TEST-REGEN-1",
            "regenerative_bus_voltage_limit_v": 62.0,
            "independent_anti_fall_model": "TEST-CATCHER-1",
            "independent_anti_fall_rated_holding_force_n": 200.0,
            "independent_anti_fall_maximum_catch_distance_m": 0.02,
            "safety_power_removal_architecture": "external dual channel removal",
            "safety_category_or_standard": "supplier declared test contract",
        },
        "evidence": {
            "axis_datasheet_sha256": "1" * 64,
            "gearbox_datasheet_sha256": "2" * 64,
            "selection_calculation_sha256": "3" * 64,
            "vertical_duty_approval_sha256": "4" * 64,
            "regeneration_calculation_sha256": "5" * 64,
            "safety_architecture_sha256": "6" * 64,
            "signed_supplier_response_sha256": "7" * 64,
        },
    }


def _report(response: dict) -> dict:
    return MODULE.build_report(
        response,
        _candidate(),
        response_sha256="a" * 64,
    )


def test_complete_response_allows_only_supplier_evidence_merge() -> None:
    report = _report(_healthy_response())
    assert report["passed"] is True
    assert report["valid_for_bench_supplier_evidence_merge"] is False
    assert report["valid_for_current_400w_bench_supplier_evidence_merge"] is False
    assert report["valid_for_750w_bench_supplier_evidence_merge"] is True
    assert report["candidate_identity_match_required_before_merge"] is True
    assert report["decision"] == "supplier_response_qualified_for_bench_input_only"
    assert report["bench_measurement_merge_fragment"]["required_candidate"] == {
        "motor_model": MODULE.MOTOR_MODEL,
        "drive_model": MODULE.DRIVE_MODEL,
        "drive_profile": "leadshine_750w_production_candidate_v1",
        "reduction_ratio": 3.0,
        "linear_lead_m_per_rev": 0.07,
    }
    assert report["bench_measurement_merge_fragment"]["evidence"][
        "supplier_approval_package_sha256"
    ] == "a" * 64
    assert all(
        report["bench_measurement_merge_fragment"]["supplier_evidence"].values()
    )
    assert report["ready_for_production_design_review"] is False
    assert report["valid_for_production_procurement"] is False
    assert report["valid_for_hardware_transfer"] is False
    assert report["simulation_motor_model_updated"] is False
    assert report["valid_for_training"] is False
    assert report["runtime_authorized"] is False
    assert report["bc_authorized"] is False
    assert report["ppo_authorized"] is False


def test_supplier_merge_identity_matches_only_the_750w_bench_template() -> None:
    report = _report(_healthy_response())
    template = json.loads(BENCH_750W_TEMPLATE.read_text(encoding="utf-8"))
    required = report["bench_measurement_merge_fragment"]["required_candidate"]
    assert all(template["candidate"][key] == value for key, value in required.items())
    assert template["candidate"]["drive_profile"] == (
        "leadshine_750w_production_candidate_v1"
    )


def test_unanswered_template_fails_closed() -> None:
    response = json.loads(MODULE.DEFAULT_RESPONSE.read_text(encoding="utf-8"))
    report = _report(response)
    assert report["passed"] is False
    assert report["missing_or_invalid_response_fields"]
    assert report["decision"] == "collect_complete_signed_supplier_response"
    assert not any(
        report["bench_measurement_merge_fragment"]["supplier_evidence"].values()
    )


@pytest.mark.parametrize(
    ("path", "bad_value", "check"),
    [
        (
            ("configuration", "software_camera_height_max_m"),
            1.8001,
            "camera_height_ceiling_is_1p8m",
        ),
        (
            ("configuration", "mechanical_stroke_m"),
            1.2,
            "mechanical_stroke_preserves_full_speed_stopping_space",
        ),
        (
            ("ratings", "maximum_linear_speed_mps"),
            0.9,
            "axis_speed_covers_1mps",
        ),
        (
            ("ratings", "minimum_emergency_deceleration_mps2"),
            4.9,
            "emergency_deceleration_covers_contract",
        ),
        (
            ("transmission", "gearbox_emergency_input_torque_nm"),
            1.2,
            "gearbox_emergency_torque_has_15_percent_margin",
        ),
    ],
)
def test_contract_regressions_fail_closed(
    path: tuple[str, str], bad_value: float, check: str
) -> None:
    response = _healthy_response()
    response[path[0]][path[1]] = bad_value
    report = _report(response)
    assert report["checks"][check] is False
    assert report["passed"] is False
    assert report["valid_for_750w_bench_supplier_evidence_merge"] is False


def test_holding_brake_cannot_be_dynamic_stop() -> None:
    response = _healthy_response()
    response["safety"]["motor_holding_brake_used_for_dynamic_stop"] = True
    report = _report(response)
    assert report["checks"]["holding_brake_is_static_only"] is False
    assert report["passed"] is False


def test_wrong_request_identity_and_long_catch_fail_closed() -> None:
    response = _healthy_response()
    response["request_id"] = "different-project"
    response["safety"]["independent_anti_fall_maximum_catch_distance_m"] = 0.031
    report = _report(response)
    assert report["checks"]["request_identity_is_pinned"] is False
    assert report["checks"][
        "independent_anti_fall_catch_distance_is_at_most_30mm"
    ] is False
    assert report["passed"] is False


def test_missing_signed_evidence_hash_fails_closed() -> None:
    response = _healthy_response()
    response["evidence"]["signed_supplier_response_sha256"] = None
    report = _report(response)
    assert "evidence.signed_supplier_response_sha256" in report[
        "missing_or_invalid_response_fields"
    ]
    assert report["checks"]["all_required_evidence_hashes_present"] is False
    assert report["passed"] is False


def test_input_is_not_mutated() -> None:
    response = _healthy_response()
    original = copy.deepcopy(response)
    _report(response)
    assert response == original


def test_cli_template_writes_lf_rejection_without_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    command = [sys.executable, str(SCRIPT), "--output", str(output)]
    first = subprocess.run(command, check=False, capture_output=True, text=True)
    assert first.returncode == 7
    payload = output.read_bytes()
    assert payload.endswith(b"\n")
    assert b"\r\n" not in payload
    report = json.loads(payload)
    assert report["passed"] is False
    second = subprocess.run(command, check=False, capture_output=True, text=True)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr
