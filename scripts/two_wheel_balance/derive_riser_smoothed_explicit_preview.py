#!/usr/bin/env python3
"""Derive one hash-bound explicit-preview riser plan from a dynamic reject."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_exact_source import (  # noqa: E402
    load_exact_source_package,
)
from rl_platform.tasks.two_wheel_balance.riser_kinematics import (  # noqa: E402
    UrdfRiserCameraKinematics,
)
from rl_platform.tasks.two_wheel_balance.riser_smoothed_plan import (  # noqa: E402
    audit_smoothed_riser_plan,
    build_smoothed_riser_plan,
    save_smoothed_riser_plan,
)


SCHEMA = "cinebotrl_two_wheel_riser_explicit_preview_derivation_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _require_hash_bound_file(path: Path, expected: str) -> None:
    _require(path.is_file() and sha256_file(path) == expected, f"hash mismatch: {path}")


def _load_hash_bound_json(path: Path, expected: str) -> dict[str, object]:
    _require_hash_bound_file(path, expected)
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"invalid JSON: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-count", type=int, default=79)
    parser.add_argument("--parent-portfolio-manifest", type=Path, required=True)
    parser.add_argument("--expected-parent-portfolio-sha256", required=True)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--expected-parent-plan-sha256", required=True)
    parser.add_argument("--gate-json", type=Path, required=True)
    parser.add_argument("--expected-gate-sha256", required=True)
    parser.add_argument("--gate-summary", type=Path, required=True)
    parser.add_argument("--expected-gate-summary-sha256", required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--case", type=int, default=16)
    parser.add_argument("--lookahead-distance", type=float, default=0.15)
    parser.add_argument("--heading-gain", type=float, default=2.75)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    _require(not args.output_dir.exists(), "fresh output namespace already exists")
    commit = _git_output("rev-parse", "HEAD")
    upstream = _git_output("rev-parse", "@{upstream}")
    _require(commit == upstream, "derivation requires HEAD equal to upstream")
    _require(
        not _git_output("status", "--porcelain", "--untracked-files=no"),
        "derivation requires clean tracked state",
    )
    source_manifest = _load_hash_bound_json(
        args.source_manifest, args.expected_source_manifest_sha256
    )
    parent = _load_hash_bound_json(
        args.parent_portfolio_manifest, args.expected_parent_portfolio_sha256
    )
    _require_hash_bound_file(args.parent_plan, args.expected_parent_plan_sha256)
    gate = _load_hash_bound_json(args.gate_json, args.expected_gate_sha256)
    summary = _load_hash_bound_json(
        args.gate_summary, args.expected_gate_summary_sha256
    )

    _require(source_manifest.get("case_count") == args.expected_count, "bad source count")
    items = parent.get("items")
    _require(isinstance(items, list) and len(items) == args.expected_count, "bad parent")
    parent_item = next((item for item in items if item.get("case") == args.case), None)
    _require(
        isinstance(parent_item, dict)
        and parent_item.get("plan_sha256") == args.expected_parent_plan_sha256
        and parent_item.get("passed") is True,
        "parent case is not admitted and hash-bound",
    )
    results = gate.get("results")
    _require(gate.get("cases") == [args.case] and isinstance(results, list) and len(results) == 1, "bad gate case")
    gate_row = results[0]
    failed = [key for key, value in gate_row.get("checks", {}).items() if value is not True]
    _require(
        failed == ["position_p95_bounded"]
        and gate_row.get("checks", {}).get("completed_reference") is True
        and gate_row.get("thermal_admission_passed") is True
        and gate_row.get("controller_evidence_passed") is True,
        "gate is not a completed position-p95-only reject",
    )
    first_reject = summary.get("first_dynamic_reject")
    _require(
        isinstance(first_reject, dict)
        and first_reject.get("case") == args.case
        and first_reject.get("runtime_contract_passed") is True
        and summary.get("residual_capture_started") is False
        and summary.get("bc_started") is False
        and summary.get("ppo_started") is False,
        "summary does not seal the expected closed reject",
    )

    references = load_exact_source_package(
        args.source_manifest,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
        expected_count=args.expected_count,
    )
    source = references[args.case]
    kinematics = UrdfRiserCameraKinematics(args.urdf)
    result = build_smoothed_riser_plan(
        source,
        kinematics,
        smoothing_sigma_candidates=(0.0,),
        preview_configurations=((args.lookahead_distance, args.heading_gain),),
    )
    _require(result.passed, "explicit preview candidate failed static admission")
    _require(
        result.smoothing_sigma_samples == 0.0
        and result.lookahead_distance_m == args.lookahead_distance
        and result.heading_gain == args.heading_gain,
        "planner selected an unexpected configuration",
    )

    args.output_dir.mkdir(parents=True)
    output = args.output_dir / f"case_{args.case:04d}_explicit_preview_v1.npz"
    save_smoothed_riser_plan(output, result, source)
    audit = audit_smoothed_riser_plan(output, source, kinematics)
    with np.load(output, allow_pickle=False) as data:
        source_arrays_immutable = (
            np.array_equal(data["source_time_s"], source.source_time_s)
            and np.array_equal(
                data["source_target_position_world_m"],
                source.source_position_world_m,
            )
            and np.array_equal(
                data["source_target_semantic_dfr_quat_xyzw"],
                source.source_semantic_dfr_quat_xyzw,
            )
            and np.array_equal(
                data["smoothed_target_position_source_frame_m"],
                source.source_position_world_m,
            )
        )
    passed = audit.get("passed") is True and source_arrays_immutable
    row = {
        **audit,
        "parent_plan": str(args.parent_plan.resolve()),
        "parent_plan_sha256": args.expected_parent_plan_sha256,
        "gate_json_sha256": args.expected_gate_sha256,
        "gate_summary_sha256": args.expected_gate_summary_sha256,
        "explicit_preview": {
            "lookahead_distance_m": args.lookahead_distance,
            "heading_gain": args.heading_gain,
            "smoothing_sigma_samples": 0.0,
            "source_geometry_changed": False,
            "controller_changed": False,
            "thresholds_changed": False,
        },
        "source_arrays_immutable": source_arrays_immutable,
        "passed": passed,
        "valid_for_training": False,
    }
    manifest = {
        "schema": SCHEMA,
        "code_commit": commit,
        "upstream_commit": upstream,
        "tracked_state_clean": True,
        "source_manifest_sha256": args.expected_source_manifest_sha256,
        "parent_portfolio_manifest_sha256": args.expected_parent_portfolio_sha256,
        "case": args.case,
        "item": row,
        "isaac_started": False,
        "residual_capture_started": False,
        "bc_started": False,
        "ppo_started": False,
        "valid_for_training": False,
        "passed": passed,
    }
    manifest_path = args.output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(
        args.output_dir / "summary.json",
        {**manifest, "manifest_sha256": sha256_file(manifest_path)},
    )
    return 0 if passed else 6


if __name__ == "__main__":
    raise SystemExit(main())
