import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "build_riser_750w_external_evidence_checklist.py"
)
SPEC = importlib.util.spec_from_file_location("riser_external_checklist", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _build() -> dict:
    paths = {
        "supplier_response": MODULE.DEFAULT_SUPPLIER_RESPONSE,
        "production_candidate": MODULE.DEFAULT_PRODUCTION_CANDIDATE,
        "bench_measurements": MODULE.DEFAULT_BENCH_MEASUREMENTS,
        "vendor_snapshot": MODULE.DEFAULT_VENDOR_SNAPSHOT,
    }
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    return MODULE.build_checklist(
        payloads["supplier_response"],
        payloads["production_candidate"],
        payloads["bench_measurements"],
        payloads["vendor_snapshot"],
        input_identities={
            name: MODULE._identity(path) for name, path in paths.items()
        },
    )


def test_checklist_preserves_all_missing_external_fields_and_hashes() -> None:
    report = _build()
    assert report["schema"] == (
        "cinebotrl_two_wheel_riser_750w_external_evidence_checklist_v1"
    )
    assert report["supplier_collection"]["missing_or_invalid_field_count"] == 52
    assert report["bench_collection"]["missing_or_invalid_field_count"] == 34
    assert sum(
        len(fields)
        for fields in report["supplier_collection"][
            "missing_or_invalid_fields_by_section"
        ].values()
    ) == 52
    assert sum(
        len(fields)
        for fields in report["bench_collection"][
            "missing_or_invalid_fields_by_section"
        ].values()
    ) == 34
    for identity in report["inputs"].values():
        assert identity["path"]
        assert len(identity["sha256"]) == 64


def test_checklist_pins_motion_safety_and_collection_contract() -> None:
    report = _build()
    contract = report["immutable_contract"]
    assert contract["motor_model"] == "ELVM8075V48EH-M17-HD"
    assert contract["drive_model"] == "ELD2-CAN7020B"
    assert contract["reduction_ratio"] == 3.0
    assert contract["linear_lead_m_per_rev"] == 0.07
    assert contract["mechanical_stroke_m_min"] == 1.5
    assert contract["software_camera_height_range_m"] == [0.6, 1.8]
    assert contract["software_working_stroke_m"] == 1.2
    assert contract["design_moving_mass_kg"] == 8.0
    assert contract["maximum_linear_speed_mps"] == 1.0
    assert contract["minimum_linear_acceleration_mps2"] == 2.0
    assert contract["minimum_linear_jerk_mps3"] == 8.0
    assert contract["minimum_emergency_deceleration_mps2"] == 5.0
    assert contract["minimum_continuous_vertical_force_n"] == 300.0
    assert contract["minimum_vertical_duty_cycle_fraction"] == 0.6
    assert contract["minimum_continuous_test_duration_s"] == 1800.0
    assert contract["motor_holding_brake_is_static_only"] is True
    assert contract["independent_anti_fall_required"] is True
    assert contract["safety_rated_power_removal_required"] is True
    assert [
        step["order"] for step in report["collection_sequence"]
    ] == [1, 2, 3, 4, 5]


def test_package_readiness_never_implies_hardware_or_learning_authority() -> None:
    report = _build()
    assert report["external_collection_package_ready"] is True
    assert report["checks"]["template_state_is_fail_closed"] is True
    assert report["checks"]["supplier_or_bench_approval_not_synthesized"] is True
    for key in (
        "real_supplier_evidence_collected",
        "real_bench_evidence_collected",
        "hardware_qualified",
        "ready_for_production_design_review",
        "valid_for_production_procurement",
        "valid_for_hardware_transfer",
        "simulation_motor_model_updated",
        "runtime_authorized",
        "gpu_work_started",
        "capture_authorized",
        "dataset_creation_authorized",
        "valid_for_training",
        "bc_authorized",
        "ppo_authorized",
        "training_started",
    ):
        assert report[key] is False


def test_changed_template_or_forged_identity_fails_package_readiness() -> None:
    paths = {
        "supplier_response": MODULE.DEFAULT_SUPPLIER_RESPONSE,
        "production_candidate": MODULE.DEFAULT_PRODUCTION_CANDIDATE,
        "bench_measurements": MODULE.DEFAULT_BENCH_MEASUREMENTS,
        "vendor_snapshot": MODULE.DEFAULT_VENDOR_SNAPSHOT,
    }
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    identities = {
        name: MODULE._identity(path) for name, path in paths.items()
    }
    changed_supplier = copy.deepcopy(payloads["supplier_response"])
    changed_supplier["supplier"]["company"] = "not-the-pinned-template"
    changed = MODULE.build_checklist(
        changed_supplier,
        payloads["production_candidate"],
        payloads["bench_measurements"],
        payloads["vendor_snapshot"],
        input_identities=identities,
    )
    assert changed["external_collection_package_ready"] is False
    assert changed["checks"]["expected_missing_fields_are_preserved"] is False

    forged_identities = copy.deepcopy(identities)
    forged_identities["vendor_snapshot"]["sha256"] = "not-a-sha"
    forged = MODULE.build_checklist(
        payloads["supplier_response"],
        payloads["production_candidate"],
        payloads["bench_measurements"],
        payloads["vendor_snapshot"],
        input_identities=forged_identities,
    )
    assert forged["external_collection_package_ready"] is False
    assert forged["checks"]["input_paths_and_hashes_present"] is False


def test_cli_writes_lf_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    command = [sys.executable, str(SCRIPT), "--output", str(output)]
    first = subprocess.run(command, check=False, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert output.read_bytes().endswith(b"\n")
    second = subprocess.run(command, check=False, capture_output=True, text=True)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr
