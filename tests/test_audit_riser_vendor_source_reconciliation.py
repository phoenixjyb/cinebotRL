import copy
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "audit_riser_vendor_source_reconciliation.py"
)
RECONCILIATION = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "RISER_VENDOR_SOURCE_RECONCILIATION_20260724.json"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "riser_vendor_source_reconciliation",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _module()


def _payload():
    return json.loads(RECONCILIATION.read_text(encoding="utf-8"))


def test_current_vendor_source_reconciliation_passes() -> None:
    result = MODULE.build_report(_payload(), ROOT)
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["selected_drive"] == "ELD2-CAN7020B"
    assert result["selected_drive_has_dedicated_cn6_sto"] is False
    assert result["external_safety_rated_power_removal_required"] is True
    assert result["camera_height_ceiling_m"] == 1.8
    assert result["target_riser_speed_mps"] == 1.0
    assert result["ready_for_production_procurement"] is False
    assert result["valid_for_hardware_transfer"] is False
    assert result["runtime_authorized"] is False
    assert result["training_started"] is False


def test_rejects_misattributed_7040b_sto_on_selected_7020b() -> None:
    payload = _payload()
    payload["drive_contract"]["selected_model_has_dedicated_cn6_sto"] = True
    result = MODULE.build_report(payload, ROOT)
    assert result["checks"]["selected_7020b_has_no_dedicated_sto"] is False
    assert result["passed"] is False


def test_rejects_catalog_axis_as_vertical_production_approval() -> None:
    payload = _payload()
    payload["fixed_axis_reference"][
        "vertical_mobile_axis_supplier_approval_available"
    ] = True
    payload["fixed_axis_reference"]["approved_as_production_axis"] = True
    result = MODULE.build_report(payload, ROOT)
    assert result["checks"]["axis_vertical_boundary_fail_closed"] is False
    assert result["passed"] is False


def test_rejects_camera_height_or_mechanical_stroke_expansion() -> None:
    payload = _payload()
    payload["consolidated_recommendation"][
        "software_camera_height_range_m"
    ] = [0.6, 1.9]
    payload["consolidated_recommendation"][
        "recommended_mechanical_stroke_m"
    ] = 1.6
    result = MODULE.build_report(payload, ROOT)
    assert result["checks"][
        "recommendation_matches_mass_and_external_contracts"
    ] is False
    assert result["passed"] is False


def test_rejects_unselected_component_promoted_to_selected() -> None:
    payload = _payload()
    payload["consolidated_recommendation"][
        "exact_gearbox_model_selected"
    ] = True
    result = MODULE.build_report(payload, ROOT)
    assert result["checks"]["unselected_components_remain_explicit"] is False
    assert result["passed"] is False


def test_rejects_identity_drift() -> None:
    payload = copy.deepcopy(_payload())
    payload["inputs"]["vendor_snapshot"]["sha256"] = "0" * 64
    result = MODULE.build_report(payload, ROOT)
    assert result["checks"]["vendor_snapshot_identity"] is False
    assert result["passed"] is False
