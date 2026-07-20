#!/usr/bin/env python3
"""Derive one CPU-only preview plan for a narrowly admitted static parent."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    build_smoothed_riser_plan_from_geometry,
    save_smoothed_riser_plan,
)


SCHEMA = "cinebotrl_two_wheel_riser_static_margin_preview_derivation_v1"
POSITION_P95_LIMIT_M = 0.15
PARENT_POSITION_P95_FLOOR_M = 0.14
MINIMUM_POSITION_P95_IMPROVEMENT_M = 0.03
MINIMUM_POSITION_MAX_IMPROVEMENT_M = 0.03


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


def _load_hash_bound_json(path: Path, expected_sha256: str) -> dict[str, object]:
    _require(path.is_file(), f"missing JSON: {path}")
    _require(sha256_file(path) == expected_sha256, f"hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"invalid JSON: {path}")
    return payload


def _require_hash_bound_file(path: Path, expected_sha256: str) -> None:
    _require(
        path.is_file() and sha256_file(path) == expected_sha256,
        f"hash mismatch: {path}",
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _validate_parent_static_margin(item: dict[str, object]) -> dict[str, float]:
    metrics = item.get("kinematic_metrics")
    _require(isinstance(metrics, dict), "parent kinematic metrics are missing")
    p95 = metrics.get("position_error_p95_m")
    maximum = metrics.get("position_error_max_m")
    _require(
        item.get("passed") is True
        and item.get("timing_transition_kinematic_gate_passed") is True
        and item.get("valid_for_training") is False,
        "parent is not a closed CPU admission",
    )
    _require(
        isinstance(p95, (int, float))
        and math.isfinite(p95)
        and PARENT_POSITION_P95_FLOOR_M <= p95 <= POSITION_P95_LIMIT_M,
        "parent is not inside the proactive p95 margin band",
    )
    _require(
        isinstance(maximum, (int, float)) and math.isfinite(maximum),
        "parent maximum position error is invalid",
    )
    return {"position_error_p95_m": float(p95), "position_error_max_m": float(maximum)}


def _validate_candidate_improvement(
    parent_metrics: dict[str, float], candidate: dict[str, object]
) -> dict[str, float]:
    metrics = candidate.get("kinematic_metrics")
    _require(isinstance(metrics, dict), "candidate kinematic metrics are missing")
    p95 = metrics.get("position_error_p95_m")
    maximum = metrics.get("position_error_max_m")
    _require(
        candidate.get("passed") is True
        and isinstance(p95, (int, float))
        and isinstance(maximum, (int, float))
        and math.isfinite(p95)
        and math.isfinite(maximum),
        "candidate is not a finite static admission",
    )
    _require(
        p95
        <= parent_metrics["position_error_p95_m"]
        - MINIMUM_POSITION_P95_IMPROVEMENT_M,
        "candidate p95 improvement is insufficient",
    )
    _require(
        maximum
        <= parent_metrics["position_error_max_m"]
        - MINIMUM_POSITION_MAX_IMPROVEMENT_M,
        "candidate maximum-error improvement is insufficient",
    )
    return {"position_error_p95_m": float(p95), "position_error_max_m": float(maximum)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-count", type=int, default=79)
    parser.add_argument("--parent-portfolio-manifest", type=Path, required=True)
    parser.add_argument("--expected-parent-portfolio-sha256", required=True)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--expected-parent-plan-sha256", required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--lookahead-distance", type=float, required=True)
    parser.add_argument("--heading-gain", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    _require(not args.output_dir.exists(), "fresh output namespace already exists")
    _require(
        math.isfinite(args.lookahead_distance) and args.lookahead_distance > 0.0,
        "lookahead distance must be finite and positive",
    )
    _require(
        math.isfinite(args.heading_gain) and args.heading_gain > 0.0,
        "heading gain must be finite and positive",
    )
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
    source_items = source_manifest.get("items")
    _require(
        source_manifest.get("episode_count") == args.expected_count
        and isinstance(source_items, list)
        and len(source_items) == args.expected_count,
        "bad source count",
    )
    parent = _load_hash_bound_json(
        args.parent_portfolio_manifest, args.expected_parent_portfolio_sha256
    )
    items = parent.get("items")
    _require(
        isinstance(items, list) and len(items) == args.expected_count,
        "bad parent portfolio",
    )
    parent_item = next((item for item in items if item.get("case") == args.case), None)
    _require(isinstance(parent_item, dict), "parent case is missing")
    _require(
        parent_item.get("plan_sha256") == args.expected_parent_plan_sha256,
        "parent plan is not bound to the portfolio",
    )
    parent_metrics = _validate_parent_static_margin(parent_item)
    _require_hash_bound_file(args.parent_plan, args.expected_parent_plan_sha256)

    with np.load(args.parent_plan, allow_pickle=False) as parent_data:
        parent_smoothed_geometry = np.asarray(
            parent_data["smoothed_target_position_source_frame_m"],
            dtype=np.float64,
        ).copy()
    smoothing_sigma = float(parent_item["selected_smoothing_sigma_samples"])
    smoothing_blend = float(parent_item["selected_smoothing_blend_factor"])
    reset_yaw_mode = str(parent_item["selected_reset_yaw_mode"])

    references = load_exact_source_package(
        args.source_manifest,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
        expected_count=args.expected_count,
    )
    _require(args.case in references, "case is absent from the source package")
    source = references[args.case]
    kinematics = UrdfRiserCameraKinematics(args.urdf)
    result = build_smoothed_riser_plan_from_geometry(
        source,
        kinematics,
        parent_smoothed_geometry,
        smoothing_sigma_samples=smoothing_sigma,
        smoothing_blend_factor=smoothing_blend,
        lookahead_distance_m=args.lookahead_distance,
        heading_gain=args.heading_gain,
        reset_yaw_mode=reset_yaw_mode,
    )

    args.output_dir.mkdir(parents=True)
    output = args.output_dir / f"case_{args.case:04d}_static_margin_preview_v1.npz"
    save_smoothed_riser_plan(output, result, source)
    audit = audit_smoothed_riser_plan(output, source, kinematics)
    candidate_metrics = _validate_candidate_improvement(parent_metrics, audit)

    with np.load(output, allow_pickle=False) as data, np.load(
        args.parent_plan, allow_pickle=False
    ) as parent_data:
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
        )
        parent_smoothed_geometry_preserved = np.array_equal(
            data["smoothed_target_position_source_frame_m"],
            parent_smoothed_geometry,
        )
        candidate_parent_deltas = {
            "base_state_abs_max": float(
                np.max(np.abs(data["base_xy_yaw"] - parent_data["base_xy_yaw"]))
            ),
            "base_feedforward_abs_max": float(
                np.max(
                    np.abs(
                        data["feedforward_v_wz"]
                        - parent_data["feedforward_v_wz"]
                    )
                )
            ),
            "riser_state_abs_max": float(
                np.max(np.abs(data["riser_q"] - parent_data["riser_q"]))
            ),
            "proxy_state_abs_max": float(
                np.max(
                    np.abs(
                        data["proxy_gimbal_q"] - parent_data["proxy_gimbal_q"]
                    )
                )
            ),
        }

    passed = (
        audit.get("passed") is True
        and source_arrays_immutable
        and parent_smoothed_geometry_preserved
        and any(value > 1e-12 for value in candidate_parent_deltas.values())
    )
    _require(passed, "static-margin derivation contract failed")
    row = {
        **audit,
        "parent_plan": str(args.parent_plan.resolve()),
        "parent_plan_sha256": args.expected_parent_plan_sha256,
        "parent_static_position_metrics": parent_metrics,
        "candidate_static_position_metrics": candidate_metrics,
        "static_margin_preview": {
            "lookahead_distance_m": args.lookahead_distance,
            "heading_gain": args.heading_gain,
            "smoothing_sigma_samples": smoothing_sigma,
            "smoothing_blend_factor": smoothing_blend,
            "reset_yaw_mode": reset_yaw_mode,
            "minimum_position_p95_improvement_m": (
                MINIMUM_POSITION_P95_IMPROVEMENT_M
            ),
            "minimum_position_max_improvement_m": (
                MINIMUM_POSITION_MAX_IMPROVEMENT_M
            ),
            "parent_smoothed_geometry_preserved": True,
            "source_geometry_changed": False,
            "controller_changed": False,
            "thresholds_changed": False,
        },
        "candidate_parent_deltas": candidate_parent_deltas,
        "source_arrays_immutable": source_arrays_immutable,
        "passed": True,
        "valid_for_training": False,
    }
    manifest = {
        "schema": SCHEMA,
        "code_commit": commit,
        "upstream_commit": upstream,
        "tracked_state_clean": True,
        "source_manifest_sha256": args.expected_source_manifest_sha256,
        "parent_portfolio_manifest_sha256": (
            args.expected_parent_portfolio_sha256
        ),
        "case": args.case,
        "item": row,
        "isaac_started": False,
        "residual_capture_started": False,
        "bc_started": False,
        "ppo_started": False,
        "valid_for_training": False,
        "passed": True,
    }
    manifest_path = args.output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(
        args.output_dir / "summary.json",
        {**manifest, "manifest_sha256": sha256_file(manifest_path)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
