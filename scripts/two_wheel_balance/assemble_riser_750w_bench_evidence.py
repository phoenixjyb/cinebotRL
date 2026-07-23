#!/usr/bin/env python3
"""Assemble candidate-bound 750 W riser bench evidence and run the final gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCH_AUDIT_PATH = Path(__file__).with_name("audit_riser_bench_measurements.py")
DEFAULT_PROCUREMENT_AUDIT = (
    PROJECT_ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_hardware_production_candidate_v1/summary.json"
)
DEFAULT_VENDOR_SNAPSHOT = (
    PROJECT_ROOT
    / "docs/03_training/two_wheel_balance/"
    "RISER_PRODUCTION_CANDIDATE_VENDOR_SNAPSHOT_20260723.json"
)

SCHEMA = "cinebotrl_two_wheel_riser_750w_bench_assembly_v1"
MEASUREMENT_SCHEMA = "cinebotrl_two_wheel_riser_bench_measurements_v1"
REDUCTION_SCHEMA = "cinebotrl_two_wheel_riser_bench_log_reduction_v2"
SUPPLIER_AUDIT_SCHEMA = "cinebotrl_two_wheel_riser_supplier_response_audit_v1"
PROFILE = "leadshine_750w_production_candidate_v1"
EXPECTED_CANDIDATE = {
    "motor_model": "ELVM8075V48EH-M17-HD",
    "drive_model": "ELD2-CAN7020B",
    "drive_profile": PROFILE,
    "reduction_ratio": 3.0,
    "linear_lead_m_per_rev": 0.07,
}
EXPECTED_PROCUREMENT_SCHEMA = (
    "cinebotrl_two_wheel_riser_hardware_production_candidate_v1"
)
EXPECTED_VENDOR_SCHEMA = (
    "cinebotrl_two_wheel_riser_production_candidate_vendor_snapshot_v1"
)

CONTINUOUS_FIELDS = (
    "duration_s",
    "duty_cycle_fraction",
    "commanded_speed_mps",
    "minimum_achieved_speed_mps",
    "phase_current_rms_a",
    "phase_current_peak_a",
    "dc_input_current_rms_a",
    "dc_bus_voltage_max_v",
    "ambient_temperature_c",
    "motor_housing_temperature_max_c",
    "drive_temperature_max_c",
    "final_thermal_slope_c_per_min",
    "no_fault_or_tooth_jump",
)
EMERGENCY_FIELDS = (
    "repetitions",
    "initial_speed_abs_min_mps",
    "worst_stopping_distance_m",
    "phase_current_peak_a",
    "dc_bus_voltage_max_v",
    "no_fault_or_position_loss",
)
SUPPLIER_FIELDS = (
    "vertical_mobile_axis_duty_approved",
    "gearbox_continuous_speed_approved",
    "gearbox_emergency_braking_torque_approved",
    "belt_tooth_jump_margin_approved",
)
FALSE_BOUNDARY_FIELDS = (
    "valid_for_production_procurement",
    "valid_for_hardware_transfer",
    "valid_for_training",
    "runtime_authorized",
    "gpu_work_started",
    "bc_authorized",
    "ppo_authorized",
)


def _load_bench_audit_module():
    spec = importlib.util.spec_from_file_location(
        "riser_bench_audit_for_assembly", BENCH_AUDIT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load riser bench audit module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BENCH_AUDIT = _load_bench_audit_module()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": _sha256(path)}


def _sha256_value(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _candidate_matches(candidate: Any) -> bool:
    return isinstance(candidate, Mapping) and all(
        candidate.get(key) == value for key, value in EXPECTED_CANDIDATE.items()
    )


def _boundary_closed(payload: Mapping[str, Any]) -> bool:
    return all(payload.get(field) is False for field in FALSE_BOUNDARY_FIELDS)


def _automation_slots_are_empty(manual: Mapping[str, Any]) -> bool:
    continuous = manual.get("continuous_duty")
    emergency = manual.get("emergency_stop")
    supplier = manual.get("supplier_evidence")
    evidence = manual.get("evidence")
    if not all(isinstance(item, Mapping) for item in (
        continuous,
        emergency,
        supplier,
        evidence,
    )):
        return False
    for field in CONTINUOUS_FIELDS:
        expected = False if field == "no_fault_or_tooth_jump" else None
        if continuous.get(field) is not expected:
            return False
    for field in EMERGENCY_FIELDS:
        expected = False if field == "no_fault_or_position_loss" else None
        if emergency.get(field) is not expected:
            return False
    if any(supplier.get(field) is not False for field in SUPPLIER_FIELDS):
        return False
    return (
        evidence.get("raw_log_sha256") is None
        and evidence.get("supplier_approval_package_sha256") is None
    )


def _selected_fields(
    source: Mapping[str, Any],
    fields: tuple[str, ...],
    *,
    section: str,
) -> dict[str, Any]:
    missing = [field for field in fields if field not in source]
    if missing:
        raise ValueError(f"{section} is missing fields: {missing}")
    return {field: copy.deepcopy(source[field]) for field in fields}


def build_assembly(
    manual: Mapping[str, Any],
    reduction: Mapping[str, Any],
    supplier_audit: Mapping[str, Any],
    procurement: Mapping[str, Any],
    vendor: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    supplier_fragment = supplier_audit.get("bench_measurement_merge_fragment")
    supplier_evidence = (
        supplier_fragment.get("supplier_evidence")
        if isinstance(supplier_fragment, Mapping)
        else None
    )
    supplier_hash = (
        supplier_fragment.get("evidence", {}).get(
            "supplier_approval_package_sha256"
        )
        if isinstance(supplier_fragment, Mapping)
        else None
    )
    supplier_input_hash = (
        supplier_audit.get("inputs", {})
        .get("supplier_response", {})
        .get("sha256")
    )
    reduction_hash = reduction.get("raw_log", {}).get("sha256")

    checks = {
        "manual_measurement_schema_matches": manual.get("schema")
        == MEASUREMENT_SCHEMA,
        "manual_candidate_matches_750w_route": _candidate_matches(
            manual.get("candidate")
        ),
        "manual_automation_slots_are_empty": _automation_slots_are_empty(manual),
        "reduction_schema_is_candidate_bound_v2": reduction.get("schema")
        == REDUCTION_SCHEMA,
        "reduction_candidate_matches_750w_route": reduction.get(
            "candidate_profile"
        )
        == PROFILE,
        "reduction_passed": reduction.get("passed") is True,
        "reduction_numeric_merge_allowed": reduction.get(
            "valid_for_bench_measurement_numeric_merge"
        )
        is True,
        "reduction_candidate_bound_merge_allowed": reduction.get(
            "valid_for_candidate_bound_bench_merge"
        )
        is True,
        "reduction_raw_log_hash_present": _sha256_value(reduction_hash),
        "reduction_authority_boundary_closed": _boundary_closed(reduction),
        "supplier_audit_schema_matches": supplier_audit.get("schema")
        == SUPPLIER_AUDIT_SCHEMA,
        "supplier_audit_passed": supplier_audit.get("passed") is True,
        "supplier_generic_merge_remains_closed": supplier_audit.get(
            "valid_for_bench_supplier_evidence_merge"
        )
        is False,
        "supplier_400w_merge_remains_closed": supplier_audit.get(
            "valid_for_current_400w_bench_supplier_evidence_merge"
        )
        is False,
        "supplier_750w_merge_allowed": supplier_audit.get(
            "valid_for_750w_bench_supplier_evidence_merge"
        )
        is True,
        "supplier_candidate_identity_required": supplier_audit.get(
            "candidate_identity_match_required_before_merge"
        )
        is True,
        "supplier_required_candidate_matches": isinstance(
            supplier_fragment, Mapping
        )
        and _candidate_matches(supplier_fragment.get("required_candidate")),
        "supplier_evidence_fields_positive": isinstance(
            supplier_evidence, Mapping
        )
        and all(supplier_evidence.get(field) is True for field in SUPPLIER_FIELDS),
        "supplier_package_hash_present": _sha256_value(supplier_hash),
        "supplier_package_hash_matches_audited_response": _sha256_value(
            supplier_input_hash
        )
        and supplier_hash == supplier_input_hash,
        "supplier_authority_boundary_closed": _boundary_closed(supplier_audit),
        "procurement_candidate_schema_matches": procurement.get("schema")
        == EXPECTED_PROCUREMENT_SCHEMA,
        "procurement_candidate_passed": procurement.get("passed") is True,
        "vendor_candidate_schema_matches": vendor.get("schema")
        == EXPECTED_VENDOR_SCHEMA,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"750 W bench assembly input contract failed: {failed}")

    assembled = copy.deepcopy(dict(manual))
    assembled["continuous_duty"] = _selected_fields(
        reduction["continuous_duty"],
        CONTINUOUS_FIELDS,
        section="continuous reduction",
    )
    assembled["emergency_stop"] = _selected_fields(
        reduction["emergency_stop"],
        EMERGENCY_FIELDS,
        section="emergency-stop reduction",
    )
    assembled["supplier_evidence"] = _selected_fields(
        supplier_evidence,
        SUPPLIER_FIELDS,
        section="supplier evidence",
    )
    assembled["evidence"]["raw_log_sha256"] = reduction_hash
    assembled["evidence"]["supplier_approval_package_sha256"] = supplier_hash
    assembled["assembly_provenance"] = {
        "schema": SCHEMA,
        "candidate_profile": PROFILE,
        "inputs": copy.deepcopy(dict(provenance or {})),
        "candidate_identity_verified": True,
        "numeric_reduction_candidate_bound": True,
        "supplier_candidate_bound": True,
        "valid_for_production_procurement": False,
        "valid_for_hardware_transfer": False,
        "valid_for_training": False,
        "runtime_authorized": False,
        "gpu_work_started": False,
        "bc_authorized": False,
        "ppo_authorized": False,
    }

    bench_report = BENCH_AUDIT.build_report(assembled, procurement, vendor)
    passed = bench_report.get("passed") is True
    result = {
        "schema": SCHEMA,
        "candidate_profile": PROFILE,
        "input_contract_checks": checks,
        "input_contract_passed": True,
        "bench_audit": bench_report,
        "passed": passed,
        "decision": (
            "assembled_750w_bench_evidence_ready_for_production_design_review"
            if passed
            else "assembled_750w_bench_evidence_rejected_by_final_gate"
        ),
        "valid_for_bench_audit_evidence": passed,
        "ready_for_production_design_review": passed,
        "valid_for_production_procurement": False,
        "valid_for_hardware_transfer": False,
        "simulation_motor_model_updated": False,
        "valid_for_training": False,
        "runtime_authorized": False,
        "gpu_work_started": False,
        "bc_authorized": False,
        "ppo_authorized": False,
    }
    return assembled, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-measurements", type=Path, required=True)
    parser.add_argument("--numeric-reduction", type=Path, required=True)
    parser.add_argument("--supplier-audit", type=Path, required=True)
    parser.add_argument(
        "--procurement-audit", type=Path, default=DEFAULT_PROCUREMENT_AUDIT
    )
    parser.add_argument(
        "--vendor-snapshot", type=Path, default=DEFAULT_VENDOR_SNAPSHOT
    )
    parser.add_argument("--output-measurements", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    args = parser.parse_args()

    for output in (args.output_measurements, args.output_audit):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite bench assembly: {output}")

    input_paths = {
        "manual_measurements": args.manual_measurements,
        "numeric_reduction": args.numeric_reduction,
        "supplier_audit": args.supplier_audit,
        "procurement_audit": args.procurement_audit,
        "vendor_snapshot": args.vendor_snapshot,
    }
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in input_paths.items()
    }
    provenance = {name: _identity(path) for name, path in input_paths.items()}
    assembled, result = build_assembly(
        payloads["manual_measurements"],
        payloads["numeric_reduction"],
        payloads["supplier_audit"],
        payloads["procurement_audit"],
        payloads["vendor_snapshot"],
        provenance=provenance,
    )

    args.output_measurements.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    with args.output_measurements.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(assembled, indent=2) + "\n")
    result["inputs"] = provenance
    result["assembled_measurements"] = _identity(args.output_measurements)
    with args.output_audit.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 8


if __name__ == "__main__":
    raise SystemExit(main())
