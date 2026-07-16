#!/usr/bin/env python3
"""Seal a pass or fail Gate C canary summary from per-case runtime JSONs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    requested = [int(value) for value in args.cases.split(",")]
    passed_cases = []
    gate_rows = []
    first_reject = None
    for case in requested:
        gate = args.root / "gates" / f"case_{case:04d}.json"
        log = args.root / "logs" / f"case_{case:04d}.log"
        if not gate.is_file():
            first_reject = {
                "case": case,
                "classification": "missing_runtime_json",
                "log": str(log.resolve()) if log.is_file() else None,
                "log_sha256": sha256_file(log) if log.is_file() else None,
            }
            break
        payload = json.loads(gate.read_text(encoding="utf-8"))
        result = payload.get("results", [{}])[0]
        row = {
            "case": case,
            "gate": str(gate.resolve()),
            "gate_sha256": sha256_file(gate),
            "passed": payload.get("passed") is True and result.get("passed") is True,
            "source_duration_s": result.get("source_duration_s"),
            "execution_duration_s": result.get("execution_duration_s"),
            "completed_steps": result.get("completed_steps"),
        }
        gate_rows.append(row)
        if row["passed"]:
            passed_cases.append(case)
            continue
        first_reject = {
            "case": case,
            "classification": result.get("classification", "dynamic_gate_rejection"),
            "stage": result.get("stage", "dynamic_gate"),
            "exception_type": result.get("exception_type"),
            "exception_message": result.get("exception_message"),
            "normalized_action": result.get("normalized_action"),
            "gate_sha256": row["gate_sha256"],
            "log_sha256": sha256_file(log) if log.is_file() else None,
        }
        break

    not_started = requested[len(passed_cases) + (1 if first_reject else 0) :]
    admission = args.root / "admission.json"
    passed = first_reject is None and passed_cases == requested
    summary = {
        "schema": "cinebotrl_two_wheel_riser_gate_c_canary_v2",
        "git_commit": args.git_commit,
        "admission_sha256": sha256_file(admission),
        "requested_cases": requested,
        "dynamically_passed_cases": passed_cases,
        "first_dynamic_reject": first_reject,
        "not_started_cases": not_started,
        "gate_rows": gate_rows,
        "source_execution_timing_separated": all(
            row["source_duration_s"] is not None
            and row["execution_duration_s"] is not None
            for row in gate_rows
        ),
        "thresholds_relaxed": False,
        "actions_clipped": False,
        "residual_capture_started": False,
        "bc_started": False,
        "ppo_started": False,
        "passed": passed,
        "valid_for_final_gate_c": passed,
        "valid_for_training": False,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
