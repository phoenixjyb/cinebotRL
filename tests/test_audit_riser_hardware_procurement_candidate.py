import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/audit_riser_hardware_procurement_candidate.py"
SPEC = importlib.util.spec_from_file_location("riser_hardware_procurement", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _inputs():
    vendor = json.loads(MODULE.DEFAULT_VENDOR_SNAPSHOT.read_text(encoding="utf-8"))
    hardware = MODULE._load_hardware_module().build_report()
    return vendor, hardware


def test_400w_is_bench_candidate_but_not_production_default() -> None:
    report = MODULE.build_report(*_inputs())
    calculated = report["calculated"]
    assert report["passed"] is True
    assert report["single_engineering_sample_purchase_recommended"] is True
    assert report["valid_for_production_procurement"] is False
    assert calculated["motor_mechanical_power_from_rating_w"] == pytest.approx(
        398.982267, rel=1e-6
    )
    assert calculated["emergency_8kg_rated_torque_margin_ratio"] == pytest.approx(
        1.0557373059
    )
    assert calculated["maximum_moving_mass_at_production_margin_kg"] < 8.0


def test_vendor_snapshot_requires_external_regen_and_safety_architecture() -> None:
    report = MODULE.build_report(*_inputs())
    assert report["checks"]["drive_requires_external_regen_design"]
    assert report["checks"]["drive_has_no_published_safe_function"]
    assert report["checks"]["specific_safety_catcher_not_selected"]
    assert "independent_anti_fall" in report["recommendation"]["safety"]


def test_audit_rejects_claim_of_vertical_application_approval() -> None:
    vendor, hardware = _inputs()
    vendor["igus_axis"]["vertical_mobile_robot_application_approved"] = True
    report = MODULE.build_report(vendor, hardware)
    assert report["checks"]["axis_vertical_application_not_yet_approved"] is False
    assert report["passed"] is False
    assert report["single_engineering_sample_purchase_recommended"] is False


def test_audit_rejects_undersized_drive_peak_current() -> None:
    vendor, hardware = _inputs()
    vendor["leadshine_drive"]["peak_current_apeak"] = 20.0
    report = MODULE.build_report(vendor, hardware)
    assert report["checks"]["drive_peak_current_covers_motor"] is False
    assert report["passed"] is False


def test_report_preserves_camera_and_stroke_contract() -> None:
    report = MODULE.build_report(*_inputs())
    assert report["calculated"]["full_speed_stopping_distance_m"] == pytest.approx(
        0.12
    )
    assert report["calculated"]["recommended_mechanical_stroke_m"] == pytest.approx(
        1.5
    )
    assert report["valid_for_training"] is False
    assert report["gpu_work_started"] is False


def test_repository_input_identities_are_host_independent() -> None:
    identity = MODULE._identity(MODULE.DEFAULT_VENDOR_SNAPSHOT)
    assert identity["path"] == (
        "docs/03_training/two_wheel_balance/"
        "RISER_VENDOR_SPEC_SNAPSHOT_20260723.json"
    )
    assert not str(identity["path"]).startswith("/")
