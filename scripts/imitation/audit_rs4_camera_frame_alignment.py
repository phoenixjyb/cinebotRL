#!/usr/bin/env python3
"""Test whether one fixed transform maps GIK camera attitude to URDF cam_link."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_gik_obs_dataset import (
    chain_to_link,
    fk_ee_from_q,
    normalize_quat_wxyz,
    parse_urdf,
    quat_conj,
    quat_multiply,
    quat_to_axis_angle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=Path("assets_own/recomoProto2-1190_moveit.urdf"))
    parser.add_argument("--max_sources", type=int, default=32)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def angular_error_deg(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    relative = quat_multiply(quat_conj(predicted), actual)
    return np.rad2deg(np.linalg.norm(quat_to_axis_angle(relative), axis=1))


def resolve_npz(item: dict, manifest: Path) -> Path:
    raw = Path(item["output_npz"])
    if raw.exists():
        return raw
    candidates = [manifest.parent / raw.name, manifest.parent.parent / raw.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(raw)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = manifest.get("items", [])[: args.max_sources]
    if not items:
        raise ValueError("manifest contains no items")

    joints = parse_urdf(args.urdf)
    chain = chain_to_link(joints, "base_root", "cam_link")
    all_right: list[np.ndarray] = []
    all_left: list[np.ndarray] = []
    all_position: list[np.ndarray] = []
    per_source: list[dict[str, object]] = []

    for source_index, item in enumerate(items):
        path = resolve_npz(item, args.manifest)
        with np.load(path, allow_pickle=False) as data:
            q_current = data["q_current"].astype(np.float32)
            camera_quat = normalize_quat_wxyz(
                data[
                    "gimbal_attitude_target_quat_wxyz"
                    if "gimbal_attitude_target_quat_wxyz" in data.files
                    else "target_quat_wxyz"
                ].astype(np.float32)
            )
            target_pos = data["target_pos"].astype(np.float32)
        fk_pos, fk_quat = fk_ee_from_q(q_current, chain)
        fk_quat = normalize_quat_wxyz(fk_quat)

        right_offset = quat_multiply(quat_conj(camera_quat[:1]), fk_quat[:1])
        left_offset = quat_multiply(fk_quat[:1], quat_conj(camera_quat[:1]))
        right_predicted = quat_multiply(camera_quat, np.repeat(right_offset, camera_quat.shape[0], axis=0))
        left_predicted = quat_multiply(np.repeat(left_offset, camera_quat.shape[0], axis=0), camera_quat)
        right_error = angular_error_deg(fk_quat, right_predicted)
        left_error = angular_error_deg(fk_quat, left_predicted)
        position_error = np.linalg.norm(fk_pos - target_pos, axis=1)
        all_right.append(right_error)
        all_left.append(left_error)
        all_position.append(position_error)
        per_source.append(
            {
                "source_index": source_index,
                "file": path.name,
                "rows": int(q_current.shape[0]),
                "right_error_deg": summarize(right_error),
                "left_error_deg": summarize(left_error),
                "fk_target_position_error_m": summarize(position_error),
                "right_offset_wxyz": right_offset[0].astype(float).tolist(),
                "left_offset_wxyz": left_offset[0].astype(float).tolist(),
            }
        )

    report = {
        "schema": "cinebotrl_rs4_camera_frame_alignment_audit_v1",
        "manifest": str(args.manifest),
        "urdf": str(args.urdf),
        "sources": len(per_source),
        "right_multiply_error_deg": summarize(np.concatenate(all_right)),
        "left_multiply_error_deg": summarize(np.concatenate(all_left)),
        "fk_target_position_error_m": summarize(np.concatenate(all_position)),
        "per_source": per_source,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
