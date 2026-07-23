import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/audit_riser_hardware_production_candidate.py"
SPEC = importlib.util.spec_from_file_location(
    "riser_hardware_production_candidate", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _inputs():
    vendor = json.loads(MODULE.DEFAULT_VENDOR_SNAPSHOT.read_text(encoding="utf-8"))
    hardware = MODULE._load_hardware_module().build_report()
    return vendor, hardware


def test_750w_candidate_closes_calculated_8kg_margin_only() -> None:
    report = MODULE.build_report(*_inputs())
    calculated = report["calculated"]
    assert report["passed"] is True
    assert report["candidate_ready_for_supplier_and_bench_review"] is True
    assert calculated["motor_mechanical_power_from_rating_w"] == pytest.approx(
        750.840644, rel=1e-6
    )
    assert calculated["emergency_8kg_rated_force_margin_ratio"] > 1.15
    assert calculated["maximum_moving_mass_at_required_margin_kg"] > 8.0
    assert report["valid_for_production_procurement"] is False
    assert report["valid_for_hardware_transfer"] is False
    assert report["simulation_motor_model_updated"] is False
    assert report["valid_for_training"] is False


def test_official_drive_match_covers_750w_current_and_voltage() -> None:
    report = MODULE.build_report(*_inputs())
    assert report["checks"]["drive_is_official_match"]
    assert report["checks"]["drive_power_covers_motor"]
    assert report["checks"]["drive_continuous_current_covers_motor"]
    assert report["checks"]["drive_peak_current_covers_motor"]
    assert report["checks"]["drive_voltage_covers_48v"]
    assert report["checks"]["external_regeneration_still_required"]
    assert report["checks"]["drive_safety_function_still_absent"]


@pytest.mark.parametrize(
    ("field", "value", "check"),
    [
        ("official_motor_match_confirmed", False, "drive_is_official_match"),
        (
            "rated_current_arms_at_or_below_48v",
            18.0,
            "drive_continuous_current_covers_motor",
        ),
        ("peak_current_apeak", 58.0, "drive_peak_current_covers_motor"),
        ("published_safe_function", True, "drive_safety_function_still_absent"),
    ],
)
def test_candidate_rejects_drive_contract_drift(field, value, check) -> None:
    vendor, hardware = _inputs()
    vendor["drive"][field] = value
    report = MODULE.build_report(vendor, hardware)
    assert report["checks"][check] is False
    assert report["passed"] is False
    assert report["candidate_ready_for_supplier_and_bench_review"] is False


def test_candidate_rejects_1p9m_camera_ceiling() -> None:
    vendor, hardware = _inputs()
    vendor["mechanism_contract"]["software_camera_height_range_m"] = [0.6, 1.9]
    report = MODULE.build_report(vendor, hardware)
    assert report["checks"]["camera_height_ceiling_is_1p8m"] is False
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
    assert report["inputs"]["vendor_snapshot"]["path"] == (
        "docs/03_training/two_wheel_balance/"
        "RISER_PRODUCTION_CANDIDATE_VENDOR_SNAPSHOT_20260723.json"
    )
