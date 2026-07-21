#!/usr/bin/env python3
"""Seal an admitted raw-teacher subset after a fail-closed capture campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARENT_SCHEMA = "cinebotrl_two_wheel_riser_smoothed_representative_admission_v1"
SUBSET_SCHEMA = "cinebotrl_two_wheel_riser_raw_teacher_subset_admission_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def identity(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True
    ).strip()


def parse_cases(value: str) -> list[int]:
    cases = [int(item) for item in value.split(",") if item.strip()]
    if not cases or len(cases) != len(set(cases)):
        raise argparse.ArgumentTypeError("cases must be a non-empty unique list")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-admission", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--progress-status", type=Path, required=True)
    parser.add_argument("--case-audit-dir", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--excluded-cases", type=parse_cases, required=True)
    parser.add_argument("--minimum-count", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite admission: {args.output}")
    if args.minimum_count < 40:
        raise ValueError("raw-teacher subset minimum cannot be below 40")

    head = git_value("rev-parse", "HEAD")
    upstream = git_value("rev-parse", "@{upstream}")
    tracked_status = git_value("status", "--porcelain", "--untracked-files=no")
    parent = load_json(args.parent_admission)
    selection = load_json(args.selection)
    progress = load_json(args.progress_status)
    excluded = sorted(args.excluded_cases)
    retained = sorted(int(case) for case in progress.get("completed_cases", []))
    parent_cases = [int(case) for case in parent.get("requested_cases", [])]
    parent_plans = {
        int(row["case"]): row for row in parent.get("selected_plans", [])
    }
    selection_rows = {
        int(row["case"]): row for row in selection.get("rows", [])
    }

    retained_evidence = []
    retained_checks = []
    for case in retained:
        audit_path = args.case_audit_dir / f"case_{case:04d}.json"
        audit = load_json(audit_path)
        gate_path = Path(audit.get("gate", ""))
        raw_path = Path(audit.get("raw_case", ""))
        checks = {
            "audit_case": audit.get("case") == case,
            "audit_passed": audit.get("passed") is True
            and audit.get("capture_admission_passed") is True,
            "parent_admission_hash": audit.get("admission_sha256")
            == sha256(args.parent_admission),
            "selection_hash": audit.get("selection_sha256")
            == sha256(args.selection),
            "gate_exists": gate_path.is_file(),
            "gate_hash": gate_path.is_file()
            and audit.get("gate_sha256") == sha256(gate_path),
            "raw_exists": raw_path.is_file(),
            "raw_hash": raw_path.is_file()
            and audit.get("raw_case_sha256") == sha256(raw_path),
            "plan_bound": case in parent_plans
            and case in selection_rows
            and parent_plans[case].get("plan_sha256")
            == selection_rows[case].get("plan_sha256"),
            "training_closed": audit.get("valid_for_training") is False
            and audit.get("bc_authorized") is False
            and audit.get("ppo_authorized") is False
            and audit.get("training_started") is False,
        }
        retained_checks.append(all(checks.values()))
        retained_evidence.append(
            {
                "case": case,
                "case_audit": identity(audit_path),
                "gate": identity(gate_path) if gate_path.is_file() else None,
                "raw_case": identity(raw_path) if raw_path.is_file() else None,
                "plan_sha256": parent_plans.get(case, {}).get("plan_sha256"),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )

    excluded_evidence = []
    excluded_checks = []
    for case in excluded:
        gate_path = args.gate_dir / f"case_{case:04d}.json"
        gate = load_json(gate_path)
        results = gate.get("results", [])
        result = results[0] if len(results) == 1 else {}
        raw_path = args.raw_dir / f"case_{case:04d}_executed_raw_teacher_v1.npz"
        failed_checks = sorted(
            key for key, value in result.get("checks", {}).items() if value is False
        )
        checks = {
            "single_result": len(results) == 1,
            "case": result.get("case") == case,
            "gate_rejected": gate.get("passed") is False
            and result.get("passed") is False,
            "dynamic_rejected": result.get("dynamic_quality_passed") is False,
            "raw_absent": not raw_path.exists(),
            "dataset_absent": result.get("executed_residual_dataset") is None,
            "training_closed": gate.get("training_started") is False
            and gate.get("ppo_authorized") is False,
        }
        excluded_checks.append(all(checks.values()))
        excluded_evidence.append(
            {
                "case": case,
                "gate": identity(gate_path),
                "failed_checks": failed_checks,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )

    checks = {
        "head_matches_upstream": head == upstream,
        "tracked_state_clean": tracked_status == "",
        "parent_schema": parent.get("schema") == PARENT_SCHEMA,
        "parent_passed": parent.get("passed") is True,
        "parent_capture_authorized": parent.get("raw_teacher_capture_authorized")
        is True,
        "parent_learning_closed": parent.get("bc_authorized") is False
        and parent.get("ppo_authorized") is False
        and parent.get("training_started") is False,
        "selection_schema": selection.get("schema")
        == "cinebotrl_two_wheel_riser_initial_teacher_selection_v1",
        "selection_hash_bound": parent.get("selection_sha256")
        == sha256(args.selection),
        "progress_schema": progress.get("schema")
        == "cinebotrl_two_wheel_riser_raw_teacher42_progress_v1",
        "progress_parent_commit": progress.get("runtime_commit")
        == parent.get("runtime_commit"),
        "progress_stopped_on_excluded": progress.get("stopped_case") in excluded,
        "progress_failed_closed": progress.get("reason")
        == "runtime_or_physical_reject"
        and progress.get("capture_admission_passed") is False,
        "retained_count_met": len(retained) >= args.minimum_count,
        "retained_unique": len(retained) == len(set(retained)),
        "partition_parent": set(retained).isdisjoint(excluded)
        and set(retained) | set(excluded) == set(parent_cases),
        "all_retained_evidence_passed": all(retained_checks),
        "all_excluded_evidence_passed": all(excluded_checks),
    }
    passed = all(checks.values())
    payload = {
        "schema": SUBSET_SCHEMA,
        "sealing_commit": head,
        "sealing_upstream_commit": upstream,
        "parent_runtime_commit": parent.get("runtime_commit"),
        "parent_admission": identity(args.parent_admission),
        "selection": identity(args.selection),
        "selection_sha256": sha256(args.selection),
        "progress_status": identity(args.progress_status),
        "requested_cases": retained,
        "selected_plans": [parent_plans[case] for case in retained],
        "excluded_cases": excluded,
        "retained_case_count": len(retained),
        "minimum_case_count": args.minimum_count,
        "retained_case_evidence": retained_evidence,
        "excluded_case_evidence": excluded_evidence,
        "checks": checks,
        "corpus_audit_authorized": passed,
        "raw_teacher_capture_authorized": passed,
        "new_raw_teacher_capture_authorized": False,
        "normalized_dataset_capture_authorized": False,
        "residual_action_application_authorized": False,
        "scale_freeze_authorized_after_corpus_gate_only": passed,
        "runtime_authorized": False,
        "valid_for_training": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
