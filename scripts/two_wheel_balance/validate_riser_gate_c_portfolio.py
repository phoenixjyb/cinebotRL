#!/usr/bin/env python3
"""Validate a frozen exact-source plan portfolio for Gate C execution only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "cinebotrl_two_wheel_riser_exact_source_plan_portfolio_v1"
CONTRACT = "exact_source_v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_false(value: object) -> bool:
    return value is False


def validate(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_source_manifest_sha256: str,
    expected_count: int,
    minimum_candidates: int,
    requested_cases: list[int],
) -> dict:
    manifest_sha256 = sha256_file(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    item_cases = [item.get("case") for item in items]
    accepted = payload.get("kinematic_accepted_cases")
    rejected = payload.get("kinematic_rejected_cases")
    accepted = accepted if isinstance(accepted, list) else []
    rejected = rejected if isinstance(rejected, list) else []
    expected_cases = list(range(1, expected_count + 1))
    requested_unique = list(dict.fromkeys(requested_cases))

    rows = []
    for item in items:
        case = item.get("case")
        plan = manifest_path.parent / str(item.get("file", ""))
        plan_exists = plan.is_file()
        plan_hash = sha256_file(plan) if plan_exists else None
        checks = {
            "plan_exists": plan_exists,
            "plan_hash_matches": plan_hash == item.get("plan_sha256"),
            "trajectory_integrity_passed": item.get("trajectory_integrity_passed") is True,
            "source_timestamps_preserved": item.get("source_timestamps_preserved") is True,
            "ordered_target_geometry_preserved": item.get("ordered_target_geometry_preserved") is True,
            "initialization_separated": item.get("initialization_separated") is True,
            "training_disabled": _is_false(item.get("valid_for_training"))
            and _is_false(item.get("quality_gate_passed")),
            "kinematic_declaration_consistent": (
                item.get("kinematic_gate_passed") is (case in accepted)
            ),
        }
        rows.append(
            {
                "case": case,
                "file": item.get("file"),
                "plan_sha256": plan_hash,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )

    top_checks = {
        "manifest_hash_matches": manifest_sha256 == expected_manifest_sha256,
        "schema_matches": payload.get("schema") == SCHEMA,
        "contract_matches": payload.get("trajectory_integrity_contract") == CONTRACT,
        "source_manifest_hash_matches": (
            payload.get("source_manifest_sha256") == expected_source_manifest_sha256
        ),
        "case_count_matches": payload.get("case_count") == expected_count
        and len(items) == expected_count,
        "cases_are_contiguous": item_cases == expected_cases,
        "exact_source_complete": payload.get("exact_source_pass_count") == expected_count,
        "accepted_count_sufficient": len(accepted) >= minimum_candidates
        and payload.get("kinematic_accepted_count") == len(accepted)
        and payload.get("gate_c_candidate_count_sufficient") is True,
        "accepted_rejected_partition": sorted(accepted + rejected) == expected_cases
        and not set(accepted).intersection(rejected)
        and payload.get("kinematic_rejected_count") == len(rejected),
        "requested_cases_unique": requested_cases == requested_unique,
        "requested_cases_admitted": bool(requested_cases)
        and set(requested_cases).issubset(accepted),
        "quarantined_lineage_absent": payload.get("quarantined_lineage_absent") is True,
        "dynamic_quality_not_predeclared": _is_false(
            payload.get("gate_c_dynamic_quality_started")
        )
        and _is_false(payload.get("quality_gate_passed")),
        "training_disabled": _is_false(payload.get("valid_for_training"))
        and _is_false(payload.get("training_started"))
        and _is_false(payload.get("ppo_started"))
        and _is_false(payload.get("base_manifest_training_started"))
        and _is_false(payload.get("recovery_manifest_training_started")),
        "all_plan_rows_pass": len(rows) == expected_count
        and all(row["passed"] for row in rows),
    }
    passed = all(top_checks.values())
    return {
        "schema": "cinebotrl_two_wheel_riser_gate_c_portfolio_admission_v1",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "source_manifest_sha256": payload.get("source_manifest_sha256"),
        "requested_cases": requested_cases,
        "accepted_case_count": len(accepted),
        "accepted_cases": accepted,
        "rejected_cases": rejected,
        "top_checks": top_checks,
        "rows": rows,
        "gate_c_execution_authorized": passed,
        "residual_capture_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-count", type=int, default=79)
    parser.add_argument("--minimum-candidates", type=int, default=70)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    requested_cases = [int(value) for value in args.cases.split(",") if value.strip()]
    result = validate(
        args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        expected_count=args.expected_count,
        minimum_candidates=args.minimum_candidates,
        requested_cases=requested_cases,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
