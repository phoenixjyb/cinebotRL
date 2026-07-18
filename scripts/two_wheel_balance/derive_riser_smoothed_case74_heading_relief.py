#!/usr/bin/env python3
"""Derive one CPU-only localized heading-relief candidate for case 74."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np


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
    CASE74_RELIEF_END_ANCHOR,
    CASE74_RELIEF_START_ANCHOR,
    CASE74_RELIEF_STRATEGY,
    audit_smoothed_riser_plan,
    build_case74_localized_heading_relief,
    load_smoothed_riser_plan,
    save_smoothed_riser_plan,
)


SCHEMA = "cinebotrl_two_wheel_riser_case74_localized_heading_relief_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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
    return subprocess.check_output(
        [*_git_command(), *args], cwd=PROJECT_ROOT, text=True
    ).strip()


def _load_hash_bound(path: Path, expected_sha256: str) -> dict[str, object]:
    _require(path.is_file(), f"missing JSON: {path}")
    _require(sha256_file(path) == expected_sha256, f"hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"invalid JSON object: {path}")
    return payload


def _validate_gate_reject(
    gate: dict[str, object], summary: dict[str, object]
) -> dict[str, object]:
    results = gate.get("results")
    _require(isinstance(results, list) and len(results) == 1, "invalid Gate C row")
    row = results[0]
    _require(isinstance(row, dict), "invalid Gate C result")
    checks = row.get("checks")
    _require(isinstance(checks, dict), "Gate C checks are missing")
    failed = [name for name, passed in checks.items() if passed is not True]
    _require(gate.get("cases") == [74] and row.get("case") == 74, "wrong case")
    _require(failed == ["position_p95_bounded"], "reject is not p95-only")
    _require(row.get("completed_phase_time_s") == row.get("execution_duration_s"),
             "case 74 did not complete")
    _require(row.get("thermal_admission_passed") is True, "thermal gate failed")
    _require(row.get("termination") is None, "case 74 terminated")
    _require(row.get("executed_residual_dataset") is None, "dataset leakage")
    _require(
        row.get("raw_residual_label_applied_to_commands") is False
        and row.get("residual_action_abs_max") == [0.0, 0.0, 0.0],
        "residual altered deterministic commands",
    )
    _require(
        summary.get("dynamic_quality_passed") is False
        and summary.get("residual_capture_started") is False
        and summary.get("bc_started") is False
        and summary.get("ppo_started") is False
        and summary.get("valid_for_training") is False,
        "Gate C summary did not keep learning closed",
    )
    return row


def _array_checks(parent_path: Path, output_path: Path) -> dict[str, bool]:
    immutable = (
        "target_semantic_dfr_quat_wxyz",
        "source_time_s",
        "source_target_position_world_m",
        "source_target_semantic_dfr_quat_xyzw",
        "source_anchor_execution_index",
        "initialization_time_s",
        "initialization_state",
    )
    with np.load(parent_path, allow_pickle=False) as parent, np.load(
        output_path, allow_pickle=False
    ) as output:
        checks = {
            f"{name}_unchanged": np.array_equal(parent[name], output[name])
            for name in immutable
        }
        start = CASE74_RELIEF_START_ANCHOR
        end = CASE74_RELIEF_END_ANCHOR
        outside = np.r_[0:start, end + 1 : len(parent["execution_time_s"])]
        checks.update(
            {
                "smoothed_geometry_before_window_unchanged": np.array_equal(
                    parent["smoothed_target_position_source_frame_m"][:start],
                    output["smoothed_target_position_source_frame_m"][:start],
                ),
                "smoothed_geometry_after_window_unchanged": np.array_equal(
                    parent["smoothed_target_position_source_frame_m"][end + 1 :],
                    output["smoothed_target_position_source_frame_m"][end + 1 :],
                ),
                "target_geometry_outside_window_unchanged": np.array_equal(
                    parent["target_position_world_m"][outside],
                    output["target_position_world_m"][outside],
                ),
                "window_boundary_start_unchanged": np.array_equal(
                    parent["target_position_world_m"][start],
                    output["target_position_world_m"][start],
                ),
                "window_boundary_end_unchanged": np.array_equal(
                    parent["target_position_world_m"][end],
                    output["target_position_world_m"][end],
                ),
                "target_z_unchanged": np.array_equal(
                    parent["target_position_world_m"][:, 2],
                    output["target_position_world_m"][:, 2],
                ),
                "localized_xy_changed": not np.array_equal(
                    parent["target_position_world_m"][start + 1 : end, :2],
                    output["target_position_world_m"][start + 1 : end, :2],
                ),
                "time_alias_unambiguous": np.array_equal(
                    output["time_s"], output["execution_time_s"]
                ),
                "execution_clock_strict": bool(
                    output["execution_time_s"][0] == 0.0
                    and np.all(np.diff(output["execution_time_s"]) > 0.0)
                ),
            }
        )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--parent-portfolio-manifest", type=Path, required=True)
    parser.add_argument("--expected-parent-portfolio-sha256", required=True)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--expected-parent-plan-sha256", required=True)
    parser.add_argument("--gate-json", type=Path, required=True)
    parser.add_argument("--expected-gate-sha256", required=True)
    parser.add_argument("--gate-summary", type=Path, required=True)
    parser.add_argument("--expected-gate-summary-sha256", required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=79)
    parser.add_argument("--minimum-candidates", type=int, default=70)
    parser.add_argument("--maximum-duration-median", type=float, default=1.5)
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
    source_manifest = _load_hash_bound(
        args.source_manifest, args.expected_source_manifest_sha256
    )
    portfolio = _load_hash_bound(
        args.parent_portfolio_manifest, args.expected_parent_portfolio_sha256
    )
    gate = _load_hash_bound(args.gate_json, args.expected_gate_sha256)
    gate_summary = _load_hash_bound(
        args.gate_summary, args.expected_gate_summary_sha256
    )
    _require(
        source_manifest.get("episode_count") == args.expected_count,
        "source package count mismatch",
    )
    gate_row = _validate_gate_reject(gate, gate_summary)
    _require(
        args.parent_plan.is_file()
        and sha256_file(args.parent_plan) == args.expected_parent_plan_sha256,
        "parent plan hash mismatch",
    )

    items = portfolio.get("items")
    _require(isinstance(items, list) and len(items) == args.expected_count,
             "parent portfolio count mismatch")
    parent_item = next((item for item in items if item.get("case") == 74), None)
    _require(
        isinstance(parent_item, dict)
        and parent_item.get("passed") is True
        and parent_item.get("plan_sha256") == args.expected_parent_plan_sha256,
        "case 74 is not an admitted parent plan",
    )
    references = load_exact_source_package(
        args.source_manifest,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
        expected_count=args.expected_count,
    )
    source = references[74]
    parent_plan, _ = load_smoothed_riser_plan(args.parent_plan)
    _require(parent_plan.case == 74, "parent plan is not case 74")
    with np.load(args.parent_plan, allow_pickle=False) as parent_arrays:
        parent_smoothed = np.asarray(
            parent_arrays["smoothed_target_position_source_frame_m"],
            dtype=np.float64,
        )

    kinematics = UrdfRiserCameraKinematics(args.urdf)
    result = build_case74_localized_heading_relief(
        source, kinematics, parent_smoothed
    )
    _require(result.plan.planning_strategy == CASE74_RELIEF_STRATEGY,
             "wrong localized relief strategy")
    args.output_dir.mkdir(parents=True)
    plan_path = args.output_dir / "case_0074_localized_heading_relief_v1.npz"
    save_smoothed_riser_plan(plan_path, result, source)
    audit = audit_smoothed_riser_plan(plan_path, source, kinematics)
    array_checks = _array_checks(args.parent_plan, plan_path)
    _require(audit.get("passed") is True, "localized candidate failed CPU audit")
    _require(all(array_checks.values()), f"array invariants failed: {array_checks}")

    ratios = [
        float(item["execution_source_duration_ratio"])
        for item in items
        if item.get("passed") is True and item.get("case") != 74
    ]
    ratios.append(float(audit["execution_source_duration_ratio"]))
    duration_median = float(statistics.median(ratios))
    passed_count = sum(item.get("passed") is True for item in items)
    _require(passed_count >= args.minimum_candidates, "too few parent candidates")
    _require(
        duration_median <= args.maximum_duration_median + 1e-12,
        "prospective portfolio median exceeds contract",
    )

    item = {
        **audit,
        "parent_plan_sha256": args.expected_parent_plan_sha256,
        "plan_sha256": sha256_file(plan_path),
        "gate_json_sha256": args.expected_gate_sha256,
        "gate_summary_sha256": args.expected_gate_summary_sha256,
        "gate_position_p95_m": gate_row["position_error_p95_m"],
        "array_derivation_checks": array_checks,
        "prospective_portfolio_passed_count": passed_count,
        "prospective_accepted_duration_median": duration_median,
        "valid_for_training": False,
    }
    manifest = {
        "schema": SCHEMA,
        "code_commit": commit,
        "upstream_commit": upstream,
        "tracked_state_clean": True,
        "case": 74,
        "source_manifest_sha256": args.expected_source_manifest_sha256,
        "parent_portfolio_sha256": args.expected_parent_portfolio_sha256,
        "parent_plan_sha256": args.expected_parent_plan_sha256,
        "gate_json_sha256": args.expected_gate_sha256,
        "gate_summary_sha256": args.expected_gate_summary_sha256,
        "passed": True,
        "isaac_started": False,
        "residual_capture_started": False,
        "bc_started": False,
        "ppo_started": False,
        "valid_for_training": False,
        "item": item,
    }
    manifest_path = args.output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    summary = {**manifest, "manifest_sha256": sha256_file(manifest_path)}
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
