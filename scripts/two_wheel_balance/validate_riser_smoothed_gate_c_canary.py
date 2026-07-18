#!/usr/bin/env python3
"""Fail-closed admission for one smoothed-plan deterministic Gate C canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "cinebotrl_two_wheel_riser_smoothed_plan_export_v1"
PLAN_SCHEMA = "cinebotrl_two_wheel_riser_smoothed_plan_v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_source_manifest_sha256: str,
    expected_planner_commit: str,
    expected_count: int,
    minimum_candidates: int,
    requested_case: int,
) -> dict[str, object]:
    manifest_sha256 = sha256_file(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    passed_cases = payload.get("passed_cases")
    rejected_cases = payload.get("rejected_cases")
    passed_cases = passed_cases if isinstance(passed_cases, list) else []
    rejected_cases = rejected_cases if isinstance(rejected_cases, list) else []
    expected_cases = list(range(1, expected_count + 1))

    rows = []
    for item in items:
        case = item.get("case")
        plan = manifest_path.parent / str(item.get("file", ""))
        plan_exists = plan.is_file()
        plan_hash = sha256_file(plan) if plan_exists else None
        checks = item.get("checks")
        kinematic_checks = item.get("kinematic_checks")
        declared_pass = case in passed_cases
        row_checks = {
            "plan_exists": plan_exists,
            "plan_hash_matches": plan_hash == item.get("plan_sha256"),
            "source_json_hash_bound": isinstance(item.get("source_json_sha256"), str)
            and len(item["source_json_sha256"]) == 64,
            "training_disabled": item.get("valid_for_training") is False,
            "pass_declaration_consistent": item.get("passed") is declared_pass
            and item.get("timing_transition_kinematic_gate_passed") is declared_pass,
            "all_integrity_checks_pass_for_admitted": not declared_pass
            or (isinstance(checks, dict) and bool(checks) and all(checks.values())),
            "all_kinematic_checks_pass_for_admitted": not declared_pass
            or (
                isinstance(kinematic_checks, dict)
                and bool(kinematic_checks)
                and all(kinematic_checks.values())
            ),
            "duration_bound_for_admitted": not declared_pass
            or (
                isinstance(item.get("execution_source_duration_ratio"), (int, float))
                and item["execution_source_duration_ratio"] <= 2.0 + 1e-9
            ),
        }
        rows.append(
            {
                "case": case,
                "file": item.get("file"),
                "plan_sha256": plan_hash,
                "declared_pass": declared_pass,
                "checks": row_checks,
                "passed": all(row_checks.values()),
            }
        )

    selected = next((row for row in rows if row["case"] == requested_case), None)
    top_checks = {
        "manifest_hash_matches": manifest_sha256 == expected_manifest_sha256,
        "schema_matches": payload.get("schema") == SCHEMA,
        "plan_schema_matches": payload.get("plan_schema") == PLAN_SCHEMA,
        "source_manifest_hash_matches": (
            payload.get("source_manifest_sha256") == expected_source_manifest_sha256
        ),
        "case_count_matches": payload.get("source_package_case_count") == expected_count
        and len(items) == expected_count,
        "cases_are_contiguous": [item.get("case") for item in items] == expected_cases,
        "all_cases_attempted": payload.get("attempted_cases") == expected_cases
        and payload.get("requested_cases") == expected_cases,
        "accepted_count_sufficient": len(passed_cases) >= minimum_candidates
        and payload.get("minimum_passes_required") == minimum_candidates
        and payload.get("minimum_pass_count_met") is True
        and payload.get("portfolio_gate_passed") is True,
        "accepted_rejected_partition": sorted(passed_cases + rejected_cases)
        == expected_cases
        and not set(passed_cases).intersection(rejected_cases),
        "requested_case_admitted": requested_case in passed_cases
        and selected is not None
        and selected["passed"] is True,
        "planner_commit_bound": payload.get("code_commit") == expected_planner_commit
        and payload.get("upstream_commit") == expected_planner_commit,
        "tracked_state_was_clean": payload.get("tracked_state_clean") is True,
        "runtime_and_learning_not_started": payload.get("isaac_started") is False
        and payload.get("residual_capture_started") is False
        and payload.get("bc_started") is False
        and payload.get("ppo_started") is False
        and payload.get("differential_session_work_started") is False,
        "training_disabled": payload.get("valid_for_training") is False,
        "all_plan_rows_pass": len(rows) == expected_count
        and all(row["passed"] for row in rows),
    }
    passed = all(top_checks.values())
    return {
        "schema": "cinebotrl_two_wheel_riser_smoothed_gate_c_admission_v1",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "source_manifest_sha256": payload.get("source_manifest_sha256"),
        "planner_commit": payload.get("code_commit"),
        "requested_cases": [requested_case],
        "accepted_case_count": len(passed_cases),
        "accepted_cases": passed_cases,
        "rejected_cases": rejected_cases,
        "selected_plan": selected,
        "top_checks": top_checks,
        "rows": rows,
        "gate_c_execution_authorized": passed,
        "residual_capture_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "valid_for_training": False,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-planner-commit", required=True)
    parser.add_argument("--expected-count", type=int, default=79)
    parser.add_argument("--minimum-candidates", type=int, default=70)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        expected_planner_commit=args.expected_planner_commit,
        expected_count=args.expected_count,
        minimum_candidates=args.minimum_candidates,
        requested_case=args.case,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
