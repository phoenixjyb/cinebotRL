#!/usr/bin/env python3
"""Validate GIK imitation demonstrations exported for CineBotRL.

This is intentionally lightweight: it checks schema, array shapes, finite values,
action ranges, and per-channel valid masks before the dataset is used by replay
or behavior-cloning code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ACTION_DIM = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=Path("data/gik_ik_demos"),
        help="Directory containing manifest.json and per-trajectory .npz demos.",
    )
    parser.add_argument(
        "--strict-arm",
        action="store_true",
        help="Fail if any arm/gimbal channel is clipped outside the current RL envelope.",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_item(item: dict, demo_dir: Path) -> dict:
    npz_path = Path(item["output_npz"])
    if not npz_path.is_absolute():
        npz_path = demo_dir / npz_path
    if not npz_path.exists():
        # The exporter stores absolute source-side paths.  When synced to another
        # machine, fall back to the basename inside demo_dir.
        npz_path = demo_dir / Path(item["output_npz"]).name
    require(npz_path.exists(), f"missing npz: {item['output_npz']}")

    data = np.load(npz_path)
    actions = data["actions"]
    q_current = data["q_current"]
    q_next = data["q_next"]
    dt = data["dt"]
    valid_mask = data["action_valid_mask"]
    target_pos = data["target_pos"] if "target_pos" in data else None
    target_quat = data["target_quat_wxyz"] if "target_quat_wxyz" in data else None
    actual_ee_pos = data["actual_ee_pos"] if "actual_ee_pos" in data else None
    actual_ee_quat = data["actual_ee_quat_wxyz"] if "actual_ee_quat_wxyz" in data else None

    require(actions.ndim == 2 and actions.shape[1] == ACTION_DIM, f"{npz_path}: bad actions shape {actions.shape}")
    require(q_current.shape == q_next.shape, f"{npz_path}: q_current/q_next shape mismatch")
    require(q_current.shape[0] == actions.shape[0], f"{npz_path}: q/action sample mismatch")
    require(valid_mask.shape == actions.shape, f"{npz_path}: valid mask mismatch")
    require(dt.shape[0] == actions.shape[0], f"{npz_path}: dt/action sample mismatch")
    require(np.isfinite(actions).all(), f"{npz_path}: non-finite actions")
    require(np.isfinite(q_current).all() and np.isfinite(q_next).all(), f"{npz_path}: non-finite q")
    require(np.isfinite(dt).all() and np.all(dt > 0), f"{npz_path}: invalid dt")
    require(np.max(np.abs(actions)) <= 1.000001, f"{npz_path}: clipped actions outside [-1,1]")
    for key, arr, width in (
        ("target_pos", target_pos, 3),
        ("target_quat_wxyz", target_quat, 4),
        ("actual_ee_pos", actual_ee_pos, 3),
        ("actual_ee_quat_wxyz", actual_ee_quat, 4),
    ):
        require(arr is not None, f"{npz_path}: missing {key}")
        require(arr.shape == (actions.shape[0], width), f"{npz_path}: bad {key} shape {arr.shape}")
        require(np.isfinite(arr).all(), f"{npz_path}: non-finite {key}")
    require(
        np.allclose(np.linalg.norm(target_quat, axis=1), 1.0, atol=1e-3),
        f"{npz_path}: target quaternions are not normalized",
    )
    require(
        np.allclose(np.linalg.norm(actual_ee_quat, axis=1), 1.0, atol=1e-3),
        f"{npz_path}: actual EE quaternions are not normalized",
    )

    return {
        "samples": int(actions.shape[0]),
        "arm_valid": valid_mask[:, :6].mean(axis=0),
        "base_valid": valid_mask[:, 6:].mean(axis=0),
        "max_abs_action": float(np.max(np.abs(actions))),
        "duration_s": float(dt.sum()),
    }


def main() -> int:
    args = parse_args()
    demo_dir = args.demo_dir.resolve()
    manifest_path = demo_dir / "manifest.json"
    require(manifest_path.exists(), f"missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        manifest.get("schema") in {"cinebotrl_ik_demo_v1", "cinebotrl_ik_demo_v2", "cinebotrl_ik_demo_v2_filtered"},
        "unsupported demo schema",
    )
    items = manifest.get("items", [])
    require(items, "manifest contains no demo items")

    summaries = [validate_item(item, demo_dir) for item in items]
    total_samples = sum(s["samples"] for s in summaries)
    arm_valid = np.stack([s["arm_valid"] for s in summaries], axis=0)
    base_valid = np.stack([s["base_valid"] for s in summaries], axis=0)
    duration_s = sum(s["duration_s"] for s in summaries)

    print(f"Demo dir:        {demo_dir}")
    print(f"Logs:            {len(items)}")
    print(f"Action samples:  {total_samples}")
    print(f"Duration:        {duration_s:.1f}s")
    print(f"Max |action|:    {max(s['max_abs_action'] for s in summaries):.3f}")
    print("Arm valid mean:  " + " ".join(f"{v:.3f}" for v in arm_valid.mean(axis=0)))
    print("Base valid mean: " + " ".join(f"{v:.3f}" for v in base_valid.mean(axis=0)))

    if args.strict_arm and np.any(arm_valid < 1.0):
        raise SystemExit("strict arm validation failed: some arm/gimbal labels are outside the current RL envelope")
    if np.any(base_valid < 1.0):
        raise SystemExit("base validation failed: some base labels are outside normalized command limits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
