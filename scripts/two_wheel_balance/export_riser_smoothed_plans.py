#!/usr/bin/env python3
"""Export fail-closed CPU-only smoothed riser plan canaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_exact_source import (  # noqa: E402
    load_exact_source_package,
    sha256_file,
)
from rl_platform.tasks.two_wheel_balance.riser_kinematics import (  # noqa: E402
    UrdfRiserCameraKinematics,
)
from rl_platform.tasks.two_wheel_balance.riser_smoothed_plan import (  # noqa: E402
    SMOOTHED_PLAN_SCHEMA,
    audit_smoothed_riser_plan,
    build_smoothed_riser_plan,
    save_smoothed_riser_plan,
)


def parse_cases(value: str) -> list[int] | None:
    if value.strip().lower() == "all":
        return None
    cases = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not cases or len(cases) != len(set(cases)) or any(case <= 0 for case in cases):
        raise argparse.ArgumentTypeError("cases must be unique positive integers")
    return cases


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-count", type=int, default=79)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", type=parse_cases, default=parse_cases("74,77,52"))
    parser.add_argument("--continue-on-reject", action="store_true")
    parser.add_argument("--minimum-passes", type=int)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise ValueError(f"fresh output namespace already exists: {args.output_dir}")
    code_commit = _git_output("rev-parse", "HEAD")
    upstream_commit = _git_output("rev-parse", "@{upstream}")
    if code_commit != upstream_commit:
        raise ValueError("smoothed plan export requires HEAD equal to upstream")
    if _git_output("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("smoothed plan export requires clean tracked state")
    references = load_exact_source_package(
        args.source_manifest,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
        expected_count=args.expected_count,
    )
    cases = sorted(references) if args.cases is None else args.cases
    minimum_passes = len(cases) if args.minimum_passes is None else args.minimum_passes
    if not 1 <= minimum_passes <= len(cases):
        raise ValueError("minimum passes must be in [1, requested case count]")
    missing = [case for case in cases if case not in references]
    if missing:
        raise ValueError(f"cases absent from exact-source package: {missing}")
    kinematics = UrdfRiserCameraKinematics(args.urdf)
    args.output_dir.mkdir(parents=True)

    rows: list[dict[str, object]] = []
    requested_cases = list(cases)
    stopped_on_case: int | None = None
    for case in requested_cases:
        source = references[case]
        result = build_smoothed_riser_plan(source, kinematics)
        output = args.output_dir / f"case_{case:04d}_smoothed_riser_plan_v1.npz"
        save_smoothed_riser_plan(output, result, source)
        audit = audit_smoothed_riser_plan(output, source, kinematics)
        row = {
            **audit,
            "selected_smoothing_sigma_samples": result.smoothing_sigma_samples,
            "selected_lookahead_distance_m": result.lookahead_distance_m,
            "selected_heading_gain": result.heading_gain,
            "attempt_count": len(result.attempts),
            "attempts": list(result.attempts),
        }
        rows.append(row)
        _write_json(args.output_dir / f"case_{case:04d}.json", row)
        print(
            json.dumps(
                {
                    "case": case,
                    "passed": audit["passed"],
                    "execution_source_duration_ratio": audit[
                        "execution_source_duration_ratio"
                    ],
                    "path_length_relative_drift": audit["path_metrics"][
                        "path_length_relative_drift"
                    ],
                    "maximum_base_branch_step_rad": audit[
                        "transition_metrics"
                    ]["maximum_pre_densification_base_branch_step_rad"],
                    "maximum_proxy_branch_step_rad": audit[
                        "transition_metrics"
                    ]["maximum_pre_densification_proxy_branch_step_rad"],
                    "failed_checks": [
                        key for key, value in audit["checks"].items() if not value
                    ]
                    + [
                        key
                        for key, value in audit["kinematic_checks"].items()
                        if not value
                    ],
                }
            ),
            flush=True,
        )
        if not audit["passed"] and not args.continue_on_reject:
            stopped_on_case = case
            break

    complete_attempt = len(rows) == len(requested_cases)
    passed_count = sum(bool(row["passed"]) for row in rows)
    minimum_pass_count_met = complete_attempt and passed_count >= minimum_passes

    manifest = {
        "schema": "cinebotrl_two_wheel_riser_smoothed_plan_export_v1",
        "plan_schema": SMOOTHED_PLAN_SCHEMA,
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": args.expected_source_manifest_sha256,
        "source_package_case_count": len(references),
        "code_commit": code_commit,
        "upstream_commit": upstream_commit,
        "tracked_state_clean": True,
        "requested_cases": requested_cases,
        "attempted_cases": [int(row["case"]) for row in rows],
        "passed_cases": [int(row["case"]) for row in rows if row["passed"]],
        "rejected_cases": [int(row["case"]) for row in rows if not row["passed"]],
        "stopped_on_case": stopped_on_case,
        "continue_on_reject": args.continue_on_reject,
        "minimum_passes_required": minimum_passes,
        "minimum_pass_count_met": minimum_pass_count_met,
        "fail_fast_respected": stopped_on_case is None
        or len(rows) == requested_cases.index(stopped_on_case) + 1,
        "all_requested_passed": complete_attempt
        and passed_count == len(requested_cases),
        "portfolio_gate_passed": minimum_pass_count_met,
        "isaac_started": False,
        "residual_capture_started": False,
        "bc_started": False,
        "ppo_started": False,
        "differential_session_work_started": False,
        "valid_for_training": False,
        "items": rows,
    }
    manifest_path = args.output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    summary = {
        **manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "plan_sha256": {str(row["case"]): row["plan_sha256"] for row in rows},
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if manifest["portfolio_gate_passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
