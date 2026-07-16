#!/usr/bin/env python3
"""Build an accepted-only corrected camera stage without Isaac/Gym imports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SUMMARY_SCHEMA = "gik_physical_split_teacher_all79_v1"
EXPECTED_EPISODE_SCHEMA = "cinebotrl_gik_split_teacher_v1"
EXPECTED_LEARNED_CONTRACT = (
    "base_arm_6 plus separate world DFR gimbal attitude target"
)
EXPECTED_GIMBAL_CONTRACT = (
    "diagnostic only; DJI/runtime adapter performs attitude IK"
)
PHYSICAL_JOINTS = (
    "joint6_arm_yaw",
    "joint5_arm_pitch",
    "joint4_elbow_pitch",
    "joint3_gimbal_yaw",
    "joint2_gimbal_roll",
    "joint1_gimbal_pitch",
)


@dataclass(frozen=True)
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def scalar(data: np.lib.npyio.NpzFile, key: str) -> object:
    require(key in data.files, f"missing {key}")
    value = np.asarray(data[key])
    require(value.size == 1, f"{key} must be scalar")
    return value.reshape(-1)[0].item()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_vector(value: str | None, default: tuple[float, ...]) -> np.ndarray:
    return np.asarray(
        default if value is None else [float(item) for item in value.split()],
        dtype=np.float64,
    )


def transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    value[:3, 3] = xyz
    return value


def parse_chain(urdf: Path, target_link: str = "cam_link") -> tuple[Joint, ...]:
    root = ET.parse(urdf).getroot()
    by_child: dict[str, Joint] = {}
    for element in root.findall("joint"):
        origin = element.find("origin")
        axis = element.find("axis")
        joint = Joint(
            name=element.attrib["name"],
            joint_type=element.attrib["type"],
            parent=element.find("parent").attrib["link"],
            child=element.find("child").attrib["link"],
            origin=transform(
                parse_vector(
                    origin.attrib.get("xyz") if origin is not None else None,
                    (0.0, 0.0, 0.0),
                ),
                parse_vector(
                    origin.attrib.get("rpy") if origin is not None else None,
                    (0.0, 0.0, 0.0),
                ),
            ),
            axis=parse_vector(
                axis.attrib.get("xyz") if axis is not None else None,
                (0.0, 0.0, 1.0),
            ),
        )
        by_child[joint.child] = joint
    reverse = []
    link = target_link
    while link != "base_root":
        require(link in by_child, f"cannot trace {target_link} from {link}")
        reverse.append(by_child[link])
        link = by_child[link].parent
    return tuple(reversed(reverse))


def motion(joint: Joint, position: float) -> np.ndarray:
    value = np.eye(4)
    axis = joint.axis / np.linalg.norm(joint.axis)
    if joint.joint_type == "fixed":
        return value
    if joint.joint_type == "prismatic":
        value[:3, 3] = axis * position
        return value
    if joint.joint_type in {"revolute", "continuous"}:
        value[:3, :3] = Rotation.from_rotvec(axis * position).as_matrix()
        return value
    raise ValueError(f"unsupported joint type {joint.joint_type!r}")


def physical_cam_fk(q: np.ndarray, chain: tuple[Joint, ...]) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    require(q.ndim == 2 and q.shape[1] == 9, f"bad physical state {q.shape}")
    result = np.empty((len(q), 3), dtype=np.float64)
    for row_index, row in enumerate(q):
        values = {
            "base_joint_vx": row[0],
            "base_joint_vy": row[1],
            "base_joint_wz": row[2],
            **dict(zip(PHYSICAL_JOINTS, row[3:], strict=True)),
        }
        value = np.eye(4)
        for joint in chain:
            value = value @ joint.origin @ motion(joint, float(values.get(joint.name, 0.0)))
        result[row_index] = value[:3, 3]
    return result


def normalize_quaternions(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    require(values.ndim == 2 and values.shape[1] == 4, "bad quaternion shape")
    require(np.isfinite(values).all() and bool(np.all(norms > 1e-12)), "bad quaternion")
    values = values / norms
    return values * np.where(values[:, :1] < 0.0, -1.0, 1.0)


def interpolate_quaternions(
    values: np.ndarray, source_progress: np.ndarray, output_progress: np.ndarray
) -> np.ndarray:
    values = normalize_quaternions(values)
    for index in range(1, len(values)):
        if float(np.dot(values[index - 1], values[index])) < 0.0:
            values[index] *= -1.0
    interpolated = np.stack(
        [
            np.interp(output_progress, source_progress, values[:, column])
            for column in range(4)
        ],
        axis=1,
    )
    return normalize_quaternions(interpolated)


def retime_corrected_teacher(
    q_current: np.ndarray,
    q_next: np.ndarray,
    semantic_wxyz: np.ndarray,
    *,
    duration_s: float,
    retime_dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(duration_s >= 5.0 and retime_dt_s > 0.0, "invalid retiming")
    states = np.concatenate((q_current[:1], q_next), axis=0).astype(np.float64)
    states[:, 2:] = np.unwrap(states[:, 2:], axis=0)
    source_progress = np.linspace(0.0, 1.0, len(states))
    steps = max(1, int(np.ceil(duration_s / retime_dt_s)))
    output_time = np.linspace(0.0, duration_s, steps + 1)
    output_progress = output_time / duration_s
    output_states = np.stack(
        [
            np.interp(output_progress, source_progress, states[:, column])
            for column in range(9)
        ],
        axis=1,
    )
    attitude_progress = np.linspace(1.0 / len(semantic_wxyz), 1.0, len(semantic_wxyz))
    attitude_values = np.concatenate((semantic_wxyz[:1], semantic_wxyz), axis=0)
    attitude_progress = np.concatenate(([0.0], attitude_progress))
    output_attitude = interpolate_quaternions(
        attitude_values, attitude_progress, output_progress[1:]
    )
    return output_states[:-1], output_states[1:], output_attitude


def load_duration_map(manifest: Path) -> dict[int, float]:
    durations: dict[int, float] = {}
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        if not path.is_file():
            path = manifest.parent / path.name
        require(path.is_file(), f"duration source is missing: {line}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        case = int(metadata.get("episode_index", path.name[:4]))
        duration = float(metadata["duration_s"])
        require(duration >= 5.0, f"case {case} duration is below 5 s")
        durations[case] = duration
    return durations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--summary", default="all79_summary.json")
    parser.add_argument("--duration-manifest", type=Path, required=True)
    parser.add_argument("--source-urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retime-dt", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = args.teacher_root / args.summary
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary.get("schema") == EXPECTED_SUMMARY_SCHEMA, "wrong summary schema")
    require(summary.get("complete") is True, "teacher batch is incomplete")
    require(int(summary.get("error_count", -1)) == 0, "teacher batch has errors")
    accepted = [
        item
        for item in summary.get("items", [])
        if item.get("export_valid_for_training") is True
    ]
    require(len(accepted) == int(summary.get("trainable_count", -1)), "accepted count mismatch")
    durations = load_duration_map(args.duration_manifest)
    chain = parse_chain(args.source_urdf)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_lines = []
    records = []

    for item in accepted:
        case = int(item["episode_index"])
        require(case in durations, f"case {case} has no duration")
        episode_dir = args.teacher_root / f"episode_{case:04d}"
        files = sorted(episode_dir.glob("*_split_teacher_v1.npz"))
        require(len(files) == 1, f"case {case} expected one NPZ")
        source = files[0]
        with np.load(source, allow_pickle=False) as data:
            require(scalar(data, "schema") == EXPECTED_EPISODE_SCHEMA, f"case {case} schema")
            require(bool(scalar(data, "valid_for_training")), f"case {case} not accepted")
            require(scalar(data, "quaternion_order") == "wxyz", f"case {case} quaternion")
            require(scalar(data, "learned_contract") == EXPECTED_LEARNED_CONTRACT, f"case {case} labels")
            require(scalar(data, "physical_gimbal_contract") == EXPECTED_GIMBAL_CONTRACT, f"case {case} gimbal")
            q_current = np.asarray(data["q_current_physical_9"], dtype=np.float64)
            q_next = np.asarray(data["q_next_physical_9"], dtype=np.float64)
            semantic = np.asarray(
                data["gimbal_attitude_target_world_dfr_quat_wxyz"],
                dtype=np.float64,
            )
        current, next_q, attitude = retime_corrected_teacher(
            q_current,
            q_next,
            semantic,
            duration_s=durations[case],
            retime_dt_s=args.retime_dt,
        )
        position = physical_cam_fk(next_q, chain)
        poses = [
            {
                "position": position[index].tolist(),
                "orientation": attitude[index, [1, 2, 3, 0]].tolist(),
            }
            for index in range(len(position))
        ]
        filename = f"episode_{case:04d}_split_teacher_v1.json"
        output = args.output_dir / filename
        payload = {
            "poses": poses,
            "metadata": {
                "source": "corrected_physical_split_teacher",
                "source_npz": source.name,
                "source_npz_sha256": sha256(source),
                "scenario": "no_obstacle",
                "quality_status": "accepted",
                "episode_index": case,
                "duration_s": durations[case],
                "waypoint_dt": durations[case] / len(poses),
                "initial_base_pose_xyyaw": current[0, :3].tolist(),
                "initial_arm_joint_pos": current[0, 3:9].tolist(),
                "target_orientation_contract": "semantic_dfr_to_physical_cam_v1",
                "recorded_quaternion_order": "xyzw",
                "observation_ee_frame": "physical_cam_link_fk",
                "riser_use": "semantic_pose_only_no_source_action_labels",
            },
        }
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manifest_lines.append(filename)
        records.append(
            {
                "case": case,
                "file": filename,
                "samples": len(poses),
                "duration_s": durations[case],
                "minimum_camera_height_m": float(np.min(position[:, 2])),
                "maximum_camera_height_m": float(np.max(position[:, 2])),
                "sha256": sha256(output),
            }
        )

    (args.output_dir / "manifest.txt").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    stage_summary = {
        "schema": "cinebotrl_riser_corrected_reference_stage_v1",
        "source_summary": str(summary_path.resolve()),
        "source_summary_sha256": sha256(summary_path),
        "source_action_labels_used": False,
        "physical_gimbal_labels_used_as_actions": False,
        "accepted_case_count": len(records),
        "rejected_case_count": int(summary.get("rejected_count", 0)),
        "retime_dt_requested_s": args.retime_dt,
        "total_samples": sum(item["samples"] for item in records),
        "total_duration_s": sum(item["duration_s"] for item in records),
        "cases": records,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(stage_summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in stage_summary.items() if key != "cases"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
