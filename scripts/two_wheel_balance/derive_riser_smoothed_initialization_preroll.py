#!/usr/bin/env python3
"""Derive a separate rest-to-source initialization pre-roll for one plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_playback import (
    PLAYBACK_BASE_LINEAR_LIMIT_MPS,
    PLAYBACK_BASE_LATERAL_LIMIT_MPS,
    PLAYBACK_BASE_YAW_RATE_LIMIT_RAD_S,
    PLAYBACK_PROXY_RATE_LIMIT_RAD_S,
    PLAYBACK_RISER_RATE_LIMIT_MPS,
    load_riser_playback_plan,
)


SCHEMA = "cinebotrl_two_wheel_riser_initialization_preroll_derivation_v1"
STATE_ORDER = (
    "base_x_m",
    "base_y_m",
    "base_yaw_rad",
    "riser_m",
    "proxy_pitch_rad",
    "proxy_roll_rad",
    "proxy_continuous_yaw_rad",
)
IMMUTABLE_ARRAYS = (
    "time_s",
    "execution_time_s",
    "target_position_world_m",
    "smoothed_target_position_source_frame_m",
    "target_semantic_dfr_quat_wxyz",
    "base_xy_yaw",
    "riser_q",
    "proxy_gimbal_q",
    "feedforward_v_wz",
    "feedforward_riser_velocity",
    "feedforward_proxy_velocity",
    "source_time_s",
    "source_target_position_world_m",
    "source_target_semantic_dfr_quat_xyzw",
    "source_anchor_execution_index",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_hash_bound_json(path: Path, expected_sha256: str) -> dict[str, object]:
    require(path.is_file(), f"missing JSON: {path}")
    require(sha256_file(path) == expected_sha256, f"hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"invalid JSON object: {path}")
    return payload


def initialization_state_and_metrics(
    arrays: dict[str, np.ndarray], duration_s: float, policy_hz: float
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    require(math.isfinite(duration_s) and duration_s >= 1.0, "pre-roll is too short")
    require(math.isfinite(policy_hz) and policy_hz >= 50.0, "policy rate is too low")
    sample_count = int(round(duration_s * policy_hz)) + 1
    time_s = np.linspace(0.0, duration_s, sample_count, dtype=np.float64)
    u = time_s / duration_s
    q0 = np.concatenate(
        (
            arrays["base_xy_yaw"][0],
            np.array([arrays["riser_q"][0]]),
            arrays["proxy_gimbal_q"][0],
        )
    ).astype(np.float64)
    initial_v = float(arrays["feedforward_v_wz"][0, 0])
    initial_yaw = float(arrays["base_xy_yaw"][0, 2])
    terminal_rate = np.concatenate(
        (
            np.array(
                [
                    initial_v * math.cos(initial_yaw),
                    initial_v * math.sin(initial_yaw),
                    arrays["feedforward_v_wz"][0, 1],
                    arrays["feedforward_riser_velocity"][0],
                ]
            ),
            arrays["feedforward_proxy_velocity"][0],
        )
    ).astype(np.float64)
    # Integral of smoothstep velocity 3u^2-2u^3, shifted to end at q0.
    displacement_profile = u**3 - 0.5 * u**4 - 0.5
    state = q0[None, :] + (
        duration_s * displacement_profile[:, None] * terminal_rate[None, :]
    )
    state[-1] = q0

    dt = np.diff(time_s)
    rate = np.diff(state, axis=0) / dt[:, None]
    midpoint_yaw = 0.5 * (state[:-1, 2] + state[1:, 2])
    forward = np.cos(midpoint_yaw) * rate[:, 0] + np.sin(midpoint_yaw) * rate[:, 1]
    lateral = -np.sin(midpoint_yaw) * rate[:, 0] + np.cos(midpoint_yaw) * rate[:, 1]
    terminal_relative_error = np.abs(rate[-1] - terminal_rate) / np.maximum(
        np.abs(terminal_rate), 1e-9
    )
    start_rate_abs = np.abs(rate[0])
    metrics = {
        "duration_s": duration_s,
        "sample_count": sample_count,
        "maximum_abs_base_linear_velocity_mps": float(np.max(np.abs(forward))),
        "maximum_abs_base_lateral_velocity_mps": float(np.max(np.abs(lateral))),
        "maximum_abs_base_yaw_rate_rad_s": float(np.max(np.abs(rate[:, 2]))),
        "maximum_abs_riser_rate_mps": float(np.max(np.abs(rate[:, 3]))),
        "maximum_abs_proxy_rate_rad_s": float(np.max(np.abs(rate[:, 4:]))),
        "start_rate_abs_max": float(np.max(start_rate_abs)),
        "terminal_rate_relative_error_max": float(np.max(terminal_relative_error)),
        "terminal_rate_target": terminal_rate.tolist(),
        "initial_state": state[0].tolist(),
        "terminal_state": state[-1].tolist(),
    }
    checks = {
        "clock_starts_at_zero": time_s[0] == 0.0,
        "clock_strictly_increasing": bool(np.all(dt > 0.0)),
        "state_finite": bool(np.isfinite(state).all()),
        "terminal_state_matches_execution": bool(
            np.allclose(state[-1], q0, atol=1e-12, rtol=0.0)
        ),
        "starts_near_rest": metrics["start_rate_abs_max"] <= 0.01,
        "terminal_rate_matches_execution": (
            metrics["terminal_rate_relative_error_max"] <= 0.01
        ),
        "base_linear_velocity_bounded": (
            metrics["maximum_abs_base_linear_velocity_mps"]
            <= PLAYBACK_BASE_LINEAR_LIMIT_MPS + 1e-12
        ),
        "base_lateral_velocity_bounded": (
            metrics["maximum_abs_base_lateral_velocity_mps"]
            <= PLAYBACK_BASE_LATERAL_LIMIT_MPS + 1e-12
        ),
        "base_yaw_rate_bounded": (
            metrics["maximum_abs_base_yaw_rate_rad_s"]
            <= PLAYBACK_BASE_YAW_RATE_LIMIT_RAD_S + 1e-12
        ),
        "riser_rate_bounded": (
            metrics["maximum_abs_riser_rate_mps"]
            <= PLAYBACK_RISER_RATE_LIMIT_MPS + 1e-12
        ),
        "proxy_rate_bounded": (
            metrics["maximum_abs_proxy_rate_rad_s"]
            <= PLAYBACK_PROXY_RATE_LIMIT_RAD_S + 1e-12
        ),
    }
    require(all(checks.values()), f"pre-roll kinematic gate failed: {checks}")
    metrics["checks"] = checks
    return time_s, state, metrics


def derive(args: argparse.Namespace) -> dict[str, object]:
    require(not args.output_dir.exists(), "fresh output namespace already exists")
    parent_manifest = load_hash_bound_json(
        args.parent_manifest, args.expected_parent_manifest_sha256
    )
    gate = load_hash_bound_json(args.reject_gate, args.expected_reject_gate_sha256)
    require(args.parent_plan.is_file(), "missing parent plan")
    require(
        sha256_file(args.parent_plan) == args.expected_parent_plan_sha256,
        "parent plan hash mismatch",
    )
    results = gate.get("results")
    require(isinstance(results, list) and len(results) == 1, "expected one gate result")
    result = results[0]
    require(result.get("case") == args.expected_case, "gate case mismatch")
    require(result.get("dynamic_quality_passed") is False, "gate is not a reject")
    require(result.get("checks", {}).get("completed_reference") is True, "reject did not complete")
    require(result.get("executed_residual_dataset") is None, "reject wrote a dataset")
    require(gate.get("training_started") is False, "reject started training")

    items = parent_manifest.get("items")
    require(isinstance(items, list), "parent manifest has no items")
    rows = [item for item in items if item.get("case") == args.expected_case]
    require(len(rows) == 1, "parent case mismatch")
    parent_item = rows[0]
    require(parent_item.get("plan_sha256") == args.expected_parent_plan_sha256, "manifest plan hash mismatch")
    require(parent_item.get("passed") is True, "parent static gate failed")
    require(parent_item.get("valid_for_training") is False, "parent opened training")

    with np.load(args.parent_plan, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name]) for name in archive.files}
    require(all(name in arrays for name in IMMUTABLE_ARRAYS), "parent arrays incomplete")
    require(arrays["initialization_time_s"].shape == (0,), "parent initialization is not empty")
    require(arrays["initialization_state"].shape == (0, 7), "parent initialization state is not empty")
    initialization_time, initialization_state, metrics = initialization_state_and_metrics(
        arrays, args.duration_s, args.policy_hz
    )
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["initialization_preroll"] = {
        "schema": "rest_to_first_execution_derivative_v1",
        "state_order": list(STATE_ORDER),
        "duration_s": args.duration_s,
        "policy_hz": args.policy_hz,
        "source_clock_advanced": False,
        "execution_clock_advanced": False,
        "scored_as_source_tracking": False,
        "parent_plan_sha256": args.expected_parent_plan_sha256,
        "reject_gate_sha256": args.expected_reject_gate_sha256,
    }
    metadata["valid_for_training"] = False
    metadata["residual_capture_started"] = False
    metadata["bc_started"] = False
    metadata["ppo_started"] = False
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    arrays["initialization_time_s"] = initialization_time
    arrays["initialization_state"] = initialization_state

    args.output_dir.mkdir(parents=True)
    output_plan = args.output_dir / args.parent_plan.name
    np.savez_compressed(output_plan, **arrays)
    loaded = load_riser_playback_plan(output_plan)
    require(len(loaded.initialization_time_s) == len(initialization_time), "loader dropped pre-roll")
    with np.load(output_plan, allow_pickle=False) as candidate:
        immutable = {
            name: np.array_equal(candidate[name], arrays[name])
            for name in IMMUTABLE_ARRAYS
        }
    require(all(immutable.values()), "candidate mutated scored/source arrays")
    output_hash = sha256_file(output_plan)
    payload = {
        "schema": SCHEMA,
        "case": args.expected_case,
        "passed": True,
        "parent_manifest_sha256": args.expected_parent_manifest_sha256,
        "parent_plan_sha256": args.expected_parent_plan_sha256,
        "reject_gate_sha256": args.expected_reject_gate_sha256,
        "file": output_plan.name,
        "plan_sha256": output_hash,
        "source_and_scored_arrays_immutable": immutable,
        "initialization_metrics": metrics,
        "source_duration_s": float(arrays["source_time_s"][-1]),
        "execution_duration_s": float(arrays["execution_time_s"][-1]),
        "initialization_duration_s": args.duration_s,
        "source_clock_advanced": False,
        "execution_clock_advanced": False,
        "controller_changed": False,
        "thresholds_changed": False,
        "runtime_authorized": False,
        "isaac_started": False,
        "residual_capture_started": False,
        "bc_started": False,
        "ppo_started": False,
        "valid_for_training": False,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = {**payload, "manifest_sha256": sha256_file(manifest_path)}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--expected-parent-manifest-sha256", required=True)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--expected-parent-plan-sha256", required=True)
    parser.add_argument("--reject-gate", type=Path, required=True)
    parser.add_argument("--expected-reject-gate-sha256", required=True)
    parser.add_argument("--expected-case", type=int, default=42)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--policy-hz", type=float, default=200.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = derive(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
