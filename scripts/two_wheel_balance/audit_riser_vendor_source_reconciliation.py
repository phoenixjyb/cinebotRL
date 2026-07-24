#!/usr/bin/env python3
"""Audit the model-specific vendor evidence behind the riser recommendation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECONCILIATION = (
    PROJECT_ROOT
    / "docs/03_training/two_wheel_balance/"
    "RISER_VENDOR_SOURCE_RECONCILIATION_20260724.json"
)
SCHEMA = "cinebotrl_two_wheel_riser_vendor_source_reconciliation_v1"
MOTOR_MODEL = "ELVM8075V48EH-M17-HD"
DRIVE_MODEL = "ELD2-CAN7020B"


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_checks(
    reconciliation: Mapping[str, Any],
    repo: Path,
) -> tuple[dict[str, bool], dict[str, dict[str, Any]]]:
    checks: dict[str, bool] = {}
    identities: dict[str, dict[str, Any]] = {}
    inputs = reconciliation.get("inputs", {})
    if not isinstance(inputs, Mapping):
        return {"inputs_are_mapping": False}, identities
    checks["inputs_are_mapping"] = True
    for name in (
        "vendor_snapshot",
        "mass_drive_traceability",
        "external_evidence_checklist",
    ):
        row = inputs.get(name, {})
        relative = row.get("path") if isinstance(row, Mapping) else None
        path = (repo / str(relative)).resolve()
        inside = path.is_relative_to(repo.resolve())
        exists = inside and path.is_file()
        actual_sha = _sha256(path) if exists else None
        row_checks = {
            "path_is_relative": isinstance(relative, str)
            and not Path(relative).is_absolute(),
            "inside_repository": inside,
            "file_exists": exists,
            "sha256_matches": actual_sha
            == (row.get("sha256") if isinstance(row, Mapping) else None),
        }
        identities[name] = {
            "path": relative,
            "sha256": actual_sha,
            "checks": row_checks,
            "passed": all(row_checks.values()),
        }
        checks[f"{name}_identity"] = identities[name]["passed"]
    return checks, identities


def _source_checks(sources: Mapping[str, Any]) -> dict[str, bool]:
    expected_hosts = {
        "motor_product_page": "https://www.leadshine.com/",
        "drive_family_manual": "https://www.leadshine.com/",
        "fixed_axis_technical_page": "https://www.igus.com/",
    }
    checks: dict[str, bool] = {}
    for name, prefix in expected_hosts.items():
        row = sources.get(name, {})
        checks[f"{name}_official_url"] = isinstance(row, Mapping) and str(
            row.get("url", "")
        ).startswith(prefix)
        checks[f"{name}_retrieval_hash"] = isinstance(row, Mapping) and (
            re.fullmatch(r"[0-9a-f]{64}", str(row.get("retrieved_sha256", "")))
            is not None
        )
    drive = sources.get("drive_family_manual", {})
    checks["drive_manual_shape"] = (
        isinstance(drive, Mapping)
        and drive.get("pdf_pages") == 182
        and drive.get("pdf_creation_date") == "2025-08-15"
        and int(drive.get("retrieved_size_bytes", 0)) > 7_000_000
    )
    return checks


def build_report(
    reconciliation: Mapping[str, Any],
    repo: Path,
) -> dict[str, Any]:
    input_checks, identities = _input_checks(reconciliation, repo)
    vendor_path = repo / str(identities["vendor_snapshot"]["path"])
    mass_path = repo / str(identities["mass_drive_traceability"]["path"])
    checklist_path = repo / str(
        identities["external_evidence_checklist"]["path"]
    )
    vendor = _load_object(vendor_path) if vendor_path.is_file() else {}
    mass = _load_object(mass_path) if mass_path.is_file() else {}
    checklist = _load_object(checklist_path) if checklist_path.is_file() else {}

    motor = reconciliation.get("motor_contract", {})
    drive = reconciliation.get("drive_contract", {})
    axis = reconciliation.get("fixed_axis_reference", {})
    recommendation = reconciliation.get("consolidated_recommendation", {})
    classification = reconciliation.get("classification", {})
    vendor_motor = vendor.get("motor", {})
    vendor_drive = vendor.get("drive", {})
    mass_force = mass.get("force_power_contract", {})
    mass_stroke = mass.get("stroke_and_safety_contract", {})
    immutable = checklist.get("immutable_contract", {})
    sources = reconciliation.get("official_sources", {})

    checks = {
        "schema": reconciliation.get("schema") == SCHEMA,
        "retrieval_date": reconciliation.get("retrieved_date") == "2026-07-24",
        **input_checks,
        **_source_checks(sources if isinstance(sources, Mapping) else {}),
        "motor_matches_vendor_snapshot": (
            isinstance(motor, Mapping)
            and isinstance(vendor_motor, Mapping)
            and motor.get("model") == vendor_motor.get("model") == MOTOR_MODEL
            and all(
                motor.get(name) == vendor_motor.get(name)
                for name in (
                    "rated_voltage_vdc",
                    "rated_power_w",
                    "rated_torque_nm",
                    "peak_torque_nm",
                    "rated_speed_rpm",
                    "peak_speed_rpm",
                    "rated_current_a",
                    "peak_current_a",
                    "brake",
                    "encoder",
                    "protection",
                )
            )
        ),
        "drive_matches_vendor_snapshot": (
            isinstance(drive, Mapping)
            and isinstance(vendor_drive, Mapping)
            and drive.get("selected_model")
            == vendor_drive.get("model")
            == DRIVE_MODEL
            and drive.get("rated_current_arms")
            == vendor_drive.get("rated_current_arms_at_or_below_48v")
            and drive.get("peak_current_apeak")
            == vendor_drive.get("peak_current_apeak")
            and drive.get("main_power_voltage_vdc")
            == vendor_drive.get("main_power_voltage_vdc")
        ),
        "selected_7020b_has_no_dedicated_sto": (
            isinstance(drive, Mapping)
            and DRIVE_MODEL in drive.get("models_without_dedicated_cn6_sto", [])
            and DRIVE_MODEL not in drive.get("models_with_published_cn6_sto", [])
            and drive.get("selected_model_has_dedicated_cn6_sto") is False
            and vendor_drive.get("published_safe_function") is False
            and drive.get("external_safety_rated_power_removal_required") is True
        ),
        "larger_drive_sto_not_misattributed": (
            isinstance(drive, Mapping)
            and drive.get("models_with_published_cn6_sto")
            == ["ELD2-CAN7040B", "ELD2-CAN7060B"]
            and drive.get("manual_interpretation")
            == "family_table_sto_sil3_and_cn6_apply_to_7040b_7060b_group_not_selected_7020b"
        ),
        "axis_reference_matches_concept": (
            isinstance(axis, Mapping)
            and axis.get("model") == "igus_drylin_ZLW_1080_standard"
            and axis.get("linear_lead_m_per_rev") == 0.07
            and axis.get("maximum_catalog_stroke_m") == 2.0
            and axis.get("maximum_catalog_speed_mps_at_60pct_duty") == 5.0
            and axis.get("maximum_catalog_radial_load_n") == 300.0
        ),
        "axis_vertical_boundary_fail_closed": (
            isinstance(axis, Mapping)
            and axis.get("catalog_load_evidence_is_horizontal_installation_only")
            is True
            and axis.get("vertical_mobile_axis_supplier_approval_available")
            is False
            and axis.get("approved_as_production_axis") is False
        ),
        "recommendation_matches_mass_and_external_contracts": (
            isinstance(recommendation, Mapping)
            and recommendation.get("motor_and_drive")
            == "ELVM8075V48EH-M17-HD_plus_ELD2-CAN7020B"
            and recommendation.get("reduction_ratio") == 3.0
            and str(mass_force.get("mechanism", "")).startswith("3_to_1")
            and recommendation.get("linear_lead_m_per_rev") == 0.07
            and recommendation.get("software_camera_height_range_m")
            == mass_stroke.get("software_camera_height_range_m")
            == immutable.get("software_camera_height_range_m")
            == [0.6, 1.8]
            and recommendation.get("software_working_stroke_m")
            == mass_stroke.get("software_working_stroke_m")
            == immutable.get("software_working_stroke_m")
            == 1.2
            and recommendation.get("recommended_mechanical_stroke_m")
            == mass_stroke.get("recommended_mechanical_stroke_m")
            == immutable.get("mechanical_stroke_m_min")
            == 1.5
            and recommendation.get("maximum_linear_speed_mps")
            == immutable.get("maximum_linear_speed_mps")
            == 1.0
        ),
        "unselected_components_remain_explicit": (
            isinstance(recommendation, Mapping)
            and all(
                recommendation.get(name) is False
                for name in (
                    "exact_gearbox_model_selected",
                    "exact_vertical_axis_model_selected",
                    "regenerative_absorption_selected",
                    "independent_anti_fall_selected",
                    "safety_power_removal_selected",
                )
            )
        ),
        "external_evidence_remains_missing": (
            checklist.get("real_supplier_evidence_collected") is False
            and checklist.get("real_bench_evidence_collected") is False
            and checklist.get("hardware_qualified") is False
        ),
        "classification_is_fail_closed": (
            isinstance(classification, Mapping)
            and classification.get("vendor_source_reconciliation_passed") is True
            and classification.get("ready_for_production_procurement") is False
            and classification.get("valid_for_hardware_transfer") is False
            and classification.get("simulation_profile_changed") is False
            and classification.get("runtime_authorized") is False
            and classification.get("capture_authorized") is False
            and classification.get("valid_for_training") is False
            and classification.get("bc_authorized") is False
            and classification.get("ppo_authorized") is False
            and classification.get("training_started") is False
        ),
    }
    passed = all(checks.values())
    return {
        "schema": "cinebotrl_two_wheel_riser_vendor_source_reconciliation_audit_v1",
        "checks": checks,
        "identities": identities,
        "selected_motor": MOTOR_MODEL,
        "selected_drive": DRIVE_MODEL,
        "selected_drive_has_dedicated_cn6_sto": False,
        "external_safety_rated_power_removal_required": True,
        "fixed_axis_reference": axis.get("model")
        if isinstance(axis, Mapping)
        else None,
        "camera_height_ceiling_m": 1.8,
        "target_riser_speed_mps": 1.0,
        "supplier_vertical_axis_approval_collected": False,
        "real_bench_evidence_collected": False,
        "ready_for_production_procurement": False,
        "valid_for_hardware_transfer": False,
        "simulation_profile_changed": False,
        "runtime_authorized": False,
        "capture_authorized": False,
        "valid_for_training": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reconciliation",
        type=Path,
        default=DEFAULT_RECONCILIATION,
    )
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(
        _load_object(args.reconciliation),
        args.repo_root.resolve(),
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
