#!/usr/bin/env python3
"""Derive one CPU-only dynamic-margin retime from sealed Gate C evidence."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
from rl_platform.tasks.two_wheel_balance.riser_playback import (  # noqa: E402
    riser_playback_kinematic_gate,
    riser_playback_kinematic_metrics,
)
from rl_platform.tasks.two_wheel_balance.riser_smoothed_plan import (  # noqa: E402
    MAXIMUM_EXECUTION_SOURCE_DURATION_RATIO,
    SmoothedPlanResult,
    audit_smoothed_riser_plan,
    load_smoothed_riser_plan,
    save_smoothed_riser_plan,
    transition_metrics,
    uniformly_retime_smoothed_plan,
)


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
        if prefix != "gitdir" or not gitdir.strip():
            raise ValueError("invalid linked-worktree .git marker")
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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parent_result(path: Path) -> tuple[SmoothedPlanResult, dict[str, object]]:
    plan, metadata = load_smoothed_riser_plan(path)
    target = metadata.get("smoothed_target")
    _require(isinstance(target, dict), "parent smoothed target metadata is missing")
    with np.load(path, allow_pickle=False) as data:
        smoothed = np.asarray(
            data["smoothed_target_position_source_frame_m"], dtype=np.float64
        )
    result = SmoothedPlanResult(
        plan=plan,
        smoothed_position_source_frame_m=smoothed,
        smoothing_sigma_samples=float(target["smoothing_sigma_samples"]),
        smoothing_blend_factor=float(target["smoothing_blend_factor"]),
        lookahead_distance_m=float(target["lookahead_distance_m"]),
        heading_gain=float(target["heading_gain"]),
        reset_yaw_mode=str(target["reset_yaw_mode"]),
        reset_yaw_rad=float(target["planned_reset_base_yaw_rad"]),
        path_metrics=dict(metadata["path_metrics"]),
        transition_metrics=dict(metadata["transition_metrics"]),
        kinematic_metrics=dict(metadata["kinematic_metrics"]),
        kinematic_checks=dict(metadata["kinematic_checks"]),
        checks=dict(metadata["checks"]),
        attempts=tuple(metadata.get("attempts", [])),
        dynamic_margin_retime=None,
    )
    _require(result.passed, "parent plan is not an admitted Gate B plan")
    return result, metadata


def _gate_c_rejection(
    gate_path: Path,
    summary_path: Path,
    *,
    case: int,
    reject_mode: str,
) -> dict[str, object]:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = gate.get("results")
    _require(isinstance(results, list) and len(results) == 1, "invalid Gate C row")
    row = results[0]
    checks = row.get("checks")
    _require(isinstance(checks, dict), "Gate C checks are missing")
    failed = [name for name, passed in checks.items() if passed is not True]
    _require(gate.get("cases") == [case] and row.get("case") == case, "wrong case")
    expected_failures = {
        "completion_only": ["completed_reference"],
        "completed_position_p95_only": ["position_p95_bounded"],
    }
    _require(reject_mode in expected_failures, "unknown Gate C reject mode")
    _require(
        failed == expected_failures[reject_mode],
        f"Gate C failure does not match {reject_mode}",
    )
    if reject_mode == "completed_position_p95_only":
        _require(checks.get("completed_reference") is True, "Gate C did not complete")
        _require(
            np.isclose(
                float(row.get("completed_phase_time_s", float("nan"))),
                float(row.get("execution_duration_s", float("nan"))),
                rtol=0.0,
                atol=1e-9,
            ),
            "completed Gate C clocks do not agree",
        )
    _require(row.get("dynamic_quality_passed") is False, "dynamic reject is missing")
    _require(row.get("thermal_admission_passed") is True, "thermal gate did not pass")
    _require(row.get("termination") is None, "Gate C terminated physically")
    _require(row.get("executed_residual_dataset") is None, "dataset leakage detected")
    _require(
        row.get("raw_residual_label_applied_to_commands") is False,
        "prospective residual altered commands",
    )
    _require(
        row.get("residual_action_abs_max") == [0.0, 0.0, 0.0],
        "executed residual was nonzero",
    )
    first_reject = summary.get("first_dynamic_reject")
    _require(
        isinstance(first_reject, dict)
        and first_reject.get("case") == case
        and first_reject.get("classification") == "dynamic_gate_rejection",
        "summary does not seal the expected dynamic reject",
    )
    _require(
        summary.get("residual_capture_started") is False
        and summary.get("bc_started") is False
        and summary.get("ppo_started") is False
        and summary.get("valid_for_training") is False,
        "learning stage was not closed",
    )
    if reject_mode == "completed_position_p95_only":
        requested_cases = summary.get("requested_cases")
        passed_cases = summary.get("dynamically_passed_cases")
        not_started_cases = summary.get("not_started_cases")
        _require(
            isinstance(requested_cases, list)
            and case in requested_cases
            and isinstance(passed_cases, list)
            and isinstance(not_started_cases, list),
            "completed p95-only reject lacks ordered fail-fast evidence",
        )
        reject_index = requested_cases.index(case)
        _require(
            passed_cases == requested_cases[:reject_index]
            and not_started_cases == requested_cases[reject_index + 1 :]
            and first_reject.get("stage") == "dynamic_gate"
            and first_reject.get("physical_dynamic_quality_passed") is False
            and first_reject.get("thermal_admission_passed") is True
            and first_reject.get("runtime_contract_passed") is True
            and row.get("controller_evidence_passed") is True,
            "completed p95-only reject lacks healthy runtime evidence",
        )
    return row


def _array_derivation_checks(
    parent_path: Path,
    output_path: Path,
    scale: float,
) -> dict[str, bool]:
    immutable = (
        "target_position_world_m",
        "smoothed_target_position_source_frame_m",
        "target_semantic_dfr_quat_wxyz",
        "base_xy_yaw",
        "riser_q",
        "proxy_gimbal_q",
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
        checks.update(
            {
                "execution_clock_uniformly_scaled": np.allclose(
                    output["execution_time_s"],
                    parent["execution_time_s"] * scale,
                    rtol=0.0,
                    atol=1e-12,
                ),
                "time_alias_unambiguous": np.array_equal(
                    output["time_s"], output["execution_time_s"]
                ),
                "base_feedforward_scaled": np.allclose(
                    output["feedforward_v_wz"],
                    parent["feedforward_v_wz"] / scale,
                    rtol=1e-11,
                    atol=1e-12,
                ),
                "riser_feedforward_scaled": np.allclose(
                    output["feedforward_riser_velocity"],
                    parent["feedforward_riser_velocity"] / scale,
                    rtol=1e-11,
                    atol=1e-12,
                ),
                "proxy_feedforward_scaled": np.allclose(
                    output["feedforward_proxy_velocity"],
                    parent["feedforward_proxy_velocity"] / scale,
                    rtol=1e-11,
                    atol=1e-12,
                ),
            }
        )
    return checks


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
    parser.add_argument("--case", type=int, default=77)
    parser.add_argument(
        "--gate-reject-mode",
        choices=("completion_only", "completed_position_p95_only"),
        default="completion_only",
    )
    parser.add_argument("--target-ratio", type=float, default=1.4)
    parser.add_argument("--maximum-portfolio-median", type=float, default=1.5)
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
    expected_hashes = {
        args.source_manifest: args.expected_source_manifest_sha256,
        args.parent_portfolio_manifest: args.expected_parent_portfolio_sha256,
        args.parent_plan: args.expected_parent_plan_sha256,
        args.gate_json: args.expected_gate_sha256,
        args.gate_summary: args.expected_gate_summary_sha256,
    }
    for path, expected in expected_hashes.items():
        _require(
            path.is_file() and sha256_file(path) == expected,
            f"hash mismatch: {path}",
        )

    references = load_exact_source_package(
        args.source_manifest,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
        expected_count=args.expected_count,
    )
    _require(args.case in references, "case is absent from source package")
    source = references[args.case]
    gate_row = _gate_c_rejection(
        args.gate_json,
        args.gate_summary,
        case=args.case,
        reject_mode=args.gate_reject_mode,
    )
    parent_result, parent_metadata = _parent_result(args.parent_plan)
    _require(parent_result.plan.case == args.case, "parent plan case mismatch")

    portfolio = json.loads(args.parent_portfolio_manifest.read_text(encoding="utf-8"))
    items = portfolio.get("items")
    _require(
        isinstance(items, list) and len(items) == args.expected_count,
        "bad portfolio",
    )
    passed_items = [item for item in items if item.get("passed") is True]
    _require(len(passed_items) >= 70, "parent portfolio has fewer than 70 passes")
    parent_item = next((item for item in items if item.get("case") == args.case), None)
    _require(
        isinstance(parent_item, dict)
        and parent_item.get("plan_sha256") == args.expected_parent_plan_sha256
        and parent_item.get("passed") is True,
        "parent case is not hash-bound and admitted",
    )

    source_duration_s = float(source.source_time_s[-1])
    old_ratio = float(parent_result.plan.time_s[-1] / source_duration_s)
    retimed_plan = uniformly_retime_smoothed_plan(
        parent_result.plan,
        source_duration_s,
        args.target_ratio,
    )
    scale = float(retimed_plan.time_s[-1] / parent_result.plan.time_s[-1])
    kinematics = UrdfRiserCameraKinematics(args.urdf)
    kinematic_metrics = riser_playback_kinematic_metrics(retimed_plan, kinematics)
    kinematic_checks = riser_playback_kinematic_gate(kinematic_metrics, kinematics)
    transitions = transition_metrics(retimed_plan)
    checks = dict(parent_result.checks)
    checks.update(
        {
            "execution_not_faster_than_source": args.target_ratio >= 1.0,
            "execution_duration_ratio_bounded": args.target_ratio
            <= MAXIMUM_EXECUTION_SOURCE_DURATION_RATIO,
            "execution_schedule_strict": bool(
                retimed_plan.time_s[0] == 0.0
                and np.all(np.diff(retimed_plan.time_s) > 0.0)
            ),
        }
    )
    ratios = [float(item["execution_source_duration_ratio"]) for item in passed_items]
    old_index = [item.get("case") for item in passed_items].index(args.case)
    ratios[old_index] = args.target_ratio
    prospective_median = float(statistics.median(ratios))
    _require(
        prospective_median <= args.maximum_portfolio_median + 1e-12,
        "derived candidate violates portfolio median target",
    )
    dynamic_retime = {
        "schema": "dynamic_margin_uniform_execution_retime_v1",
        "applied": True,
        "reason": {
            "completion_only": "gate_c_completed_reference_only_rejection",
            "completed_position_p95_only": (
                "gate_c_completed_position_p95_only_rejection"
            ),
        }[args.gate_reject_mode],
        "gate_reject_mode": args.gate_reject_mode,
        "parent_plan_sha256": args.expected_parent_plan_sha256,
        "gate_json_sha256": args.expected_gate_sha256,
        "gate_summary_sha256": args.expected_gate_summary_sha256,
        "old_execution_source_ratio": old_ratio,
        "target_execution_source_ratio": args.target_ratio,
        "uniform_execution_scale": scale,
        "controller_changed": False,
        "phase_governor_changed": False,
        "thresholds_changed": False,
        "source_geometry_changed": False,
        "gate_completed_phase_time_s": gate_row["completed_phase_time_s"],
        "gate_execution_duration_s": gate_row["execution_duration_s"],
    }
    attempt = {
        "derivation": dynamic_retime["schema"],
        "execution_source_duration_ratio": args.target_ratio,
        "failed_checks": [],
        "passed": True,
    }
    result = replace(
        parent_result,
        plan=retimed_plan,
        transition_metrics=transitions,
        kinematic_metrics=kinematic_metrics,
        kinematic_checks=kinematic_checks,
        checks=checks,
        attempts=(*parent_result.attempts, attempt),
        dynamic_margin_retime=dynamic_retime,
    )
    _require(result.passed, "retimed candidate failed kinematic admission")

    args.output_dir.mkdir(parents=True)
    output = args.output_dir / f"case_{args.case:04d}_dynamic_margin_retime_v1.npz"
    save_smoothed_riser_plan(output, result, source)
    audit = audit_smoothed_riser_plan(output, source, kinematics)
    derivation_checks = _array_derivation_checks(args.parent_plan, output, scale)
    with np.load(output, allow_pickle=False) as data:
        output_metadata = json.loads(str(data["metadata_json"].item()))
    metadata_checks = {
        "source_provenance_unchanged": output_metadata.get("source_provenance")
        == parent_metadata.get("source_provenance"),
        "path_metrics_unchanged": output_metadata.get("path_metrics")
        == parent_metadata.get("path_metrics"),
        "dynamic_retime_recorded": output_metadata.get("dynamic_margin_retime")
        == dynamic_retime,
        "learning_closed": output_metadata.get("valid_for_training") is False
        and output_metadata.get("residual_capture_started") is False
        and output_metadata.get("bc_started") is False
        and output_metadata.get("ppo_started") is False,
    }
    passed = (
        audit["passed"] is True
        and all(derivation_checks.values())
        and all(metadata_checks.values())
    )
    row = {
        **audit,
        "parent_plan": str(args.parent_plan.resolve()),
        "parent_plan_sha256": args.expected_parent_plan_sha256,
        "gate_json_sha256": args.expected_gate_sha256,
        "gate_summary_sha256": args.expected_gate_summary_sha256,
        "dynamic_margin_retime": dynamic_retime,
        "derivation_checks": derivation_checks,
        "metadata_checks": metadata_checks,
        "prospective_portfolio_accepted_count": len(passed_items),
        "prospective_portfolio_duration_median": prospective_median,
        "maximum_portfolio_duration_median": args.maximum_portfolio_median,
        "passed": passed,
        "valid_for_training": False,
    }
    manifest = {
        "schema": "cinebotrl_two_wheel_riser_dynamic_margin_retime_v1",
        "code_commit": commit,
        "upstream_commit": upstream,
        "tracked_state_clean": True,
        "source_manifest_sha256": args.expected_source_manifest_sha256,
        "parent_portfolio_manifest_sha256": args.expected_parent_portfolio_sha256,
        "case": args.case,
        "target_execution_source_ratio": args.target_ratio,
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
    summary = {**manifest, "manifest_sha256": sha256_file(manifest_path)}
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if passed else 6


if __name__ == "__main__":
    raise SystemExit(main())
