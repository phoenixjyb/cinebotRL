#!/usr/bin/env python3
"""Build the fail-closed collection checklist for the 750 W riser candidate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts/two_wheel_balance"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import audit_riser_bench_measurements as bench_audit  # noqa: E402
import audit_riser_supplier_response as supplier_audit  # noqa: E402


SCHEMA = "cinebotrl_two_wheel_riser_750w_external_evidence_checklist_v1"
PROFILE = "leadshine_750w_production_candidate_v1"
EXPECTED_SUPPLIER_MISSING_FIELDS = 52
EXPECTED_BENCH_MISSING_FIELDS = 34

DEFAULT_SUPPLIER_RESPONSE = supplier_audit.DEFAULT_RESPONSE
DEFAULT_PRODUCTION_CANDIDATE = supplier_audit.DEFAULT_CANDIDATE
DEFAULT_BENCH_MEASUREMENTS = bench_audit.PRODUCTION_MEASUREMENTS
DEFAULT_VENDOR_SNAPSHOT = bench_audit.PRODUCTION_VENDOR_SNAPSHOT
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_riser_750w_external_checklist_v1/summary.json"
)


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


def _group_fields(fields: list[str]) -> dict[str, list[str]]:
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for field in fields:
        section = field.split(".", maxsplit=1)[0]
        grouped[section].append(field)
    return {
        section: sorted(grouped[section])
        for section in sorted(grouped)
    }


def _valid_identity(identity: Mapping[str, str]) -> bool:
    return bool(identity.get("path")) and (
        re.fullmatch(r"[0-9a-f]{64}", identity.get("sha256", "")) is not None
    )


def build_checklist(
    supplier_response: Mapping[str, Any],
    production_candidate: Mapping[str, Any],
    bench_measurements: Mapping[str, Any],
    vendor_snapshot: Mapping[str, Any],
    *,
    input_identities: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    supplier_report = supplier_audit.build_report(
        supplier_response,
        production_candidate,
        response_sha256=input_identities["supplier_response"]["sha256"],
    )
    bench_report = bench_audit.build_report(
        bench_measurements,
        production_candidate,
        vendor_snapshot,
    )
    supplier_missing = supplier_report["missing_or_invalid_response_fields"]
    bench_missing = bench_report["missing_or_invalid_measurement_fields"]

    input_paths_present = all(
        _valid_identity(item) for item in input_identities.values()
    )
    template_state_is_fail_closed = (
        supplier_report["passed"] is False
        and supplier_report["supplier_response_complete"] is False
        and bench_report["passed"] is False
        and bench_report["checks"]["measurement_complete"] is False
    )
    expected_missing_fields_are_preserved = (
        len(supplier_missing) == EXPECTED_SUPPLIER_MISSING_FIELDS
        and len(bench_missing) == EXPECTED_BENCH_MISSING_FIELDS
    )
    manual_values_remain_unmeasured = (
        isinstance(bench_measurements.get("test_id"), str)
        and bench_measurements["test_id"].startswith("UNMEASURED_TEMPLATE")
        and bench_report["checks"]["measurement_complete"] is False
    )
    supplier_or_bench_approval_not_synthesized = (
        supplier_report["passed"] is False
        and supplier_report["valid_for_750w_bench_supplier_evidence_merge"]
        is False
        and bench_report["passed"] is False
        and bench_report["ready_for_production_design_review"] is False
    )
    package_ready = bool(
        input_paths_present
        and template_state_is_fail_closed
        and expected_missing_fields_are_preserved
        and manual_values_remain_unmeasured
        and supplier_or_bench_approval_not_synthesized
    )

    return {
        "schema": SCHEMA,
        "candidate_profile": PROFILE,
        "inputs": dict(input_identities),
        "immutable_contract": {
            "motor_model": supplier_audit.MOTOR_MODEL,
            "drive_model": supplier_audit.DRIVE_MODEL,
            "reduction_ratio": 3.0,
            "linear_lead_m_per_rev": 0.07,
            "mechanical_stroke_m_min": 1.50,
            "software_camera_height_range_m": [0.60, 1.80],
            "software_working_stroke_m": 1.20,
            "design_moving_mass_kg": 8.0,
            "maximum_linear_speed_mps": 1.0,
            "minimum_linear_acceleration_mps2": 2.0,
            "minimum_linear_jerk_mps3": 8.0,
            "minimum_emergency_deceleration_mps2": 5.0,
            "minimum_continuous_vertical_force_n": 300.0,
            "minimum_vertical_duty_cycle_fraction": 0.60,
            "minimum_continuous_test_duration_s": 1800.0,
            "motor_holding_brake_is_static_only": True,
            "independent_anti_fall_required": True,
            "safety_rated_power_removal_required": True,
        },
        "supplier_collection": {
            "template_complete": False,
            "audit_passed": False,
            "missing_or_invalid_field_count": len(supplier_missing),
            "missing_or_invalid_fields": supplier_missing,
            "missing_or_invalid_fields_by_section": _group_fields(
                supplier_missing
            ),
        },
        "bench_collection": {
            "template_complete": False,
            "audit_passed": False,
            "missing_or_invalid_field_count": len(bench_missing),
            "missing_or_invalid_fields": bench_missing,
            "missing_or_invalid_fields_by_section": _group_fields(bench_missing),
        },
        "collection_sequence": [
            {
                "order": 1,
                "action": "collect_signed_supplier_response_and_datasheets",
                "output": "<supplier_response.json>",
            },
            {
                "order": 2,
                "action": "assemble_complete_instrumented_vertical_axis",
                "output": "<physical_750w_riser_axis>",
            },
            {
                "order": 3,
                "action": "collect_calibrated_raw_bench_log_and_manual_safety_evidence",
                "output": "<raw_bench_log.csv>",
            },
            {
                "order": 4,
                "action": "reduce_candidate_bound_raw_bench_log",
                "output": "<numeric_reduction.json>",
            },
            {
                "order": 5,
                "action": "assemble_hash_bound_supplier_and_bench_evidence",
                "output": "<assembled_measurements.json> and <final_audit.json>",
            },
        ],
        "commands": {
            "audit_supplier_response": [
                "python3",
                "scripts/two_wheel_balance/audit_riser_supplier_response.py",
                "--response",
                "<supplier_response.json>",
                "--output",
                "<supplier_audit.json>",
            ],
            "reduce_raw_bench_log": [
                "python3",
                "scripts/two_wheel_balance/reduce_riser_bench_log.py",
                "--candidate-profile",
                PROFILE,
                "--input",
                "<raw_bench_log.csv>",
                "--output",
                "<numeric_reduction.json>",
            ],
            "assemble_final_evidence": [
                "python3",
                "scripts/two_wheel_balance/assemble_riser_750w_bench_evidence.py",
                "--manual-measurements",
                "<completed_manual_measurements.json>",
                "--numeric-reduction",
                "<numeric_reduction.json>",
                "--supplier-audit",
                "<supplier_audit.json>",
                "--output-measurements",
                "<assembled_measurements.json>",
                "--output-audit",
                "<final_audit.json>",
            ],
        },
        "checks": {
            "input_paths_and_hashes_present": input_paths_present,
            "template_state_is_fail_closed": template_state_is_fail_closed,
            "expected_missing_fields_are_preserved": (
                expected_missing_fields_are_preserved
            ),
            "manual_values_remain_unmeasured": manual_values_remain_unmeasured,
            "supplier_or_bench_approval_not_synthesized": (
                supplier_or_bench_approval_not_synthesized
            ),
        },
        "external_collection_package_ready": package_ready,
        "real_supplier_evidence_collected": False,
        "real_bench_evidence_collected": False,
        "hardware_qualified": False,
        "ready_for_production_design_review": False,
        "valid_for_production_procurement": False,
        "valid_for_hardware_transfer": False,
        "simulation_motor_model_updated": False,
        "runtime_authorized": False,
        "gpu_work_started": False,
        "capture_authorized": False,
        "dataset_creation_authorized": False,
        "valid_for_training": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supplier-response", type=Path, default=DEFAULT_SUPPLIER_RESPONSE
    )
    parser.add_argument(
        "--production-candidate", type=Path, default=DEFAULT_PRODUCTION_CANDIDATE
    )
    parser.add_argument(
        "--bench-measurements", type=Path, default=DEFAULT_BENCH_MEASUREMENTS
    )
    parser.add_argument(
        "--vendor-snapshot", type=Path, default=DEFAULT_VENDOR_SNAPSHOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite external evidence checklist: {args.output}"
        )
    input_paths = {
        "supplier_response": args.supplier_response,
        "production_candidate": args.production_candidate,
        "bench_measurements": args.bench_measurements,
        "vendor_snapshot": args.vendor_snapshot,
    }
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in input_paths.items()
    }
    report = build_checklist(
        payloads["supplier_response"],
        payloads["production_candidate"],
        payloads["bench_measurements"],
        payloads["vendor_snapshot"],
        input_identities={
            name: _identity(path) for name, path in input_paths.items()
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["external_collection_package_ready"] else 9


if __name__ == "__main__":
    raise SystemExit(main())
