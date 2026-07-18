#!/usr/bin/env python3
"""Compose a fresh all-79 smoothed portfolio from audited CPU replacements."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import statistics
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PORTFOLIO_SCHEMA = "cinebotrl_two_wheel_riser_smoothed_plan_export_v1"
PLAN_SCHEMA = "cinebotrl_two_wheel_riser_smoothed_plan_v1"
REPLACEMENT_SCHEMAS = {
    "cinebotrl_two_wheel_riser_dynamic_margin_retime_v1",
    "cinebotrl_two_wheel_riser_case74_localized_heading_relief_v1",
    "cinebotrl_two_wheel_riser_explicit_preview_derivation_v1",
}


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _windows_path_from_gitdir(value: str) -> str:
    if value.startswith("/mnt/") and len(value) >= 7 and value[6] == "/":
        return f"{value[5].upper()}:{value[6:]}"
    return value


def _git_command() -> list[str]:
    marker = PROJECT_ROOT / ".git"
    if marker.is_file():
        prefix, gitdir = marker.read_text(encoding="utf-8").strip().split(":", 1)
        _require(prefix == "gitdir" and bool(gitdir.strip()), "invalid .git marker")
        return [
            "git",
            "--git-dir",
            _windows_path_from_gitdir(gitdir.strip()),
            "--work-tree",
            str(PROJECT_ROOT),
        ]
    return ["git", "-C", str(PROJECT_ROOT)]


def _git_output(*args: str) -> str:
    return subprocess.check_output([*_git_command(), *args], text=True).strip()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_hash_bound(path: Path, expected_sha256: str) -> dict[str, object]:
    _require(path.is_file(), f"missing JSON: {path}")
    _require(sha256_file(path) == expected_sha256, f"hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"invalid JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--replacement-manifest", type=Path, required=True)
    parser.add_argument("--expected-replacement-sha256", required=True)
    parser.add_argument("--expected-count", type=int, default=79)
    parser.add_argument("--minimum-candidates", type=int, default=70)
    parser.add_argument("--maximum-duration-median", type=float, default=1.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    _require(not args.output_dir.exists(), "fresh output namespace already exists")
    commit = _git_output("rev-parse", "HEAD")
    upstream = _git_output("rev-parse", "@{upstream}")
    _require(commit == upstream, "composition requires HEAD equal to upstream")
    _require(
        not _git_output("status", "--porcelain", "--untracked-files=no"),
        "composition requires clean tracked state",
    )
    parent = _load_hash_bound(args.parent_manifest, args.expected_parent_sha256)
    replacement = _load_hash_bound(
        args.replacement_manifest, args.expected_replacement_sha256
    )
    _require(parent.get("schema") == PORTFOLIO_SCHEMA, "wrong parent schema")
    _require(parent.get("plan_schema") == PLAN_SCHEMA, "wrong parent plan schema")
    _require(
        replacement.get("schema") in REPLACEMENT_SCHEMAS
        and replacement.get("passed") is True
        and replacement.get("valid_for_training") is False,
        "replacement is not an admitted CPU-only derivation",
    )
    for name in (
        "isaac_started",
        "residual_capture_started",
        "bc_started",
        "ppo_started",
    ):
        _require(replacement.get(name) is False, f"replacement opened {name}")

    parent_items = parent.get("items")
    _require(
        isinstance(parent_items, list) and len(parent_items) == args.expected_count,
        "parent does not contain the expected item count",
    )
    expected_cases = list(range(1, args.expected_count + 1))
    _require(
        [item.get("case") for item in parent_items] == expected_cases,
        "parent cases are not ordered and contiguous",
    )
    replacement_item = replacement.get("item")
    _require(isinstance(replacement_item, dict), "replacement item is missing")
    replacement_case = replacement_item.get("case")
    _require(
        isinstance(replacement_case, int) and replacement_case in expected_cases,
        "invalid replacement case",
    )
    parent_item = parent_items[replacement_case - 1]
    _require(
        replacement_item.get("parent_plan_sha256") == parent_item.get("plan_sha256"),
        "replacement is not bound to its parent plan",
    )
    _require(
        replacement_item.get("passed") is True
        and replacement_item.get("valid_for_training") is False,
        "replacement case is not a closed CPU pass",
    )
    replacement_plan = (
        args.replacement_manifest.parent / str(replacement_item.get("file", ""))
    )
    _require(
        replacement_plan.is_file()
        and sha256_file(replacement_plan) == replacement_item.get("plan_sha256"),
        "replacement plan hash mismatch",
    )

    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for item in parent_items:
        case = int(item["case"])
        parent_plan = args.parent_manifest.parent / str(item.get("file", ""))
        _require(
            parent_plan.is_file()
            and sha256_file(parent_plan) == item.get("plan_sha256"),
            f"parent plan hash mismatch for case {case}",
        )
        destination = args.output_dir / parent_plan.name
        if case == replacement_case:
            shutil.copy2(replacement_plan, destination)
            row = copy.deepcopy(item)
            row.update(copy.deepcopy(replacement_item))
            row["file"] = destination.name
            row["replacement_selected"] = True
            row["replacement_manifest_sha256"] = args.expected_replacement_sha256
        else:
            shutil.copy2(parent_plan, destination)
            row = copy.deepcopy(item)
            row["replacement_selected"] = False
        row["plan_sha256"] = sha256_file(destination)
        row["valid_for_training"] = False
        _require(
            row.get("passed") is row.get("timing_transition_kinematic_gate_passed"),
            f"inconsistent pass declaration for case {case}",
        )
        rows.append(row)
        _write_json(args.output_dir / f"case_{case:04d}.json", row)

    passed_cases = [int(row["case"]) for row in rows if row.get("passed") is True]
    rejected_cases = [int(row["case"]) for row in rows if row.get("passed") is False]
    _require(
        len(passed_cases) >= args.minimum_candidates,
        "composed portfolio has too few admitted plans",
    )
    duration_median = float(
        statistics.median(
            float(row["execution_source_duration_ratio"])
            for row in rows
            if row.get("passed") is True
        )
    )
    _require(
        duration_median <= args.maximum_duration_median + 1e-12,
        "composed duration median exceeds the contract",
    )
    _require(
        parent.get("source_manifest_sha256")
        == replacement.get("source_manifest_sha256"),
        "replacement source package differs from parent",
    )

    manifest = {
        "schema": PORTFOLIO_SCHEMA,
        "plan_schema": PLAN_SCHEMA,
        "source_manifest": parent.get("source_manifest"),
        "source_manifest_sha256": parent.get("source_manifest_sha256"),
        "source_package_case_count": parent.get("source_package_case_count"),
        "code_commit": commit,
        "upstream_commit": upstream,
        "tracked_state_clean": True,
        "parent_manifest": str(args.parent_manifest.resolve()),
        "parent_manifest_sha256": args.expected_parent_sha256,
        "replacement_manifests": [
            {
                "case": replacement_case,
                "path": str(args.replacement_manifest.resolve()),
                "sha256": args.expected_replacement_sha256,
                "plan_sha256": replacement_item.get("plan_sha256"),
            }
        ],
        "requested_cases": expected_cases,
        "attempted_cases": expected_cases,
        "passed_cases": passed_cases,
        "rejected_cases": rejected_cases,
        "stopped_on_case": None,
        "continue_on_reject": True,
        "minimum_passes_required": args.minimum_candidates,
        "minimum_pass_count_met": True,
        "fail_fast_respected": True,
        "all_requested_passed": len(passed_cases) == args.expected_count,
        "portfolio_gate_passed": True,
        "accepted_duration_median": duration_median,
        "maximum_duration_median": args.maximum_duration_median,
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
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
