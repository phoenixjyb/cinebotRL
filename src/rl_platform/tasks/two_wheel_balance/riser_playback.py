"""Validated, self-contained playback plans for the two-wheel riser robot."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .camera_attitude import (
    quaternion_matrix_wxyz,
    rotation_error_vector,
    semantic_dfr_to_physical_cam_quat_wxyz,
)
from .riser_reference import CorrectedRiserReference, RiserKinematicPlan
from .riser_kinematics import UrdfRiserCameraKinematics
from .riser_rs4_reference import plan_rs4_riser_reference


PLAYBACK_SCHEMA = "cinebotrl_two_wheel_riser_playback_v1"
PLAYBACK_POSITION_P95_LIMIT_M = 0.15
PLAYBACK_POSITION_MAX_LIMIT_M = 0.25
PLAYBACK_ATTITUDE_P95_LIMIT_DEG = 5.0
PLAYBACK_ATTITUDE_MAX_LIMIT_DEG = 10.0
PLAYBACK_BASE_LINEAR_LIMIT_MPS = 0.4
PLAYBACK_BASE_LATERAL_LIMIT_MPS = 0.02
PLAYBACK_BASE_YAW_RATE_LIMIT_RAD_S = 0.4
PLAYBACK_PLANNING_BASE_YAW_RATE_RAD_S = 0.25
PLAYBACK_RISER_RATE_LIMIT_MPS = 1.0
PLAYBACK_PROXY_RATE_LIMIT_RAD_S = math.radians(24.0)
PLAYBACK_CAMERA_HEIGHT_MIN_M = 0.60
PLAYBACK_CAMERA_HEIGHT_MAX_M = 1.80


@dataclass(frozen=True)
class RiserPlaybackPlan:
    case: int
    time_s: np.ndarray
    target_position_world_m: np.ndarray
    target_semantic_dfr_quat_wxyz: np.ndarray
    base_xy_yaw: np.ndarray
    riser_q: np.ndarray
    proxy_gimbal_q: np.ndarray
    feedforward_v_wz: np.ndarray
    feedforward_riser_velocity: np.ndarray
    feedforward_proxy_velocity: np.ndarray
    vertical_shift_m: float
    planning_strategy: str
    source_time_s: np.ndarray | None = None

    def validate(self) -> None:
        count = len(self.time_s)
        checks = {
            "case": isinstance(self.case, int) and self.case > 0,
            "time": self.time_s.shape == (count,)
            and count >= 2
            and self.time_s[0] == 0.0
            and bool(np.all(np.diff(self.time_s) > 0.0)),
            "target_position": self.target_position_world_m.shape == (count, 3),
            "target_attitude": self.target_semantic_dfr_quat_wxyz.shape == (count, 4),
            "base": self.base_xy_yaw.shape == (count, 3),
            "riser": self.riser_q.shape == (count,),
            "proxy": self.proxy_gimbal_q.shape == (count, 3),
            "base_feedforward": self.feedforward_v_wz.shape == (count - 1, 2),
            "riser_feedforward": self.feedforward_riser_velocity.shape == (count - 1,),
            "proxy_feedforward": self.feedforward_proxy_velocity.shape == (count - 1, 3),
            "vertical_shift": math.isfinite(self.vertical_shift_m),
            "strategy": self.planning_strategy in {
                "fixed_path",
                "joint_adaptive",
                "preview_0.10m",
                "preview_0.25m",
                "preview_0.50m",
                "preview_0.10m_g1.15",
                "preview_0.10m_g1.50",
                "preview_0.25m_g1.50",
                "preview_0.50m_g1.50",
                "smoothed_preview_0.05m_g2.75",
                "smoothed_preview_0.10m_g2.75",
                "smoothed_preview_0.15m_g2.75",
                "smoothed_preview_0.25m_g2.75",
            },
        }
        if self.source_time_s is not None:
            checks["source_time"] = (
                self.source_time_s.ndim == 1
                and len(self.source_time_s) >= 2
                and self.source_time_s[0] == 0.0
                and bool(np.all(np.diff(self.source_time_s) > 0.0))
                and bool(np.isfinite(self.source_time_s).all())
            )
        arrays = (
            self.time_s,
            self.target_position_world_m,
            self.target_semantic_dfr_quat_wxyz,
            self.base_xy_yaw,
            self.riser_q,
            self.proxy_gimbal_q,
            self.feedforward_v_wz,
            self.feedforward_riser_velocity,
            self.feedforward_proxy_velocity,
        )
        checks["finite"] = all(np.isfinite(item).all() for item in arrays)
        quaternion_norm = np.linalg.norm(self.target_semantic_dfr_quat_wxyz, axis=1)
        checks["unit_quaternion"] = bool(
            np.allclose(quaternion_norm, 1.0, atol=1e-6)
        )
        # A continuous target must never rely on cyclic post-processing to
        # hide a nearly 2*pi servo reversal.
        checks["continuous_proxy_yaw"] = bool(
            np.max(np.abs(np.diff(self.proxy_gimbal_q[:, 2]))) < math.pi
        )
        if not all(checks.values()):
            raise ValueError(f"invalid riser playback plan: {checks}")


@dataclass(frozen=True)
class RiserPlaybackSample:
    base_xy_yaw: np.ndarray
    riser_q: float
    proxy_gimbal_q: np.ndarray
    target_position_world_m: np.ndarray
    target_semantic_dfr_quat_wxyz: np.ndarray
    feedforward_v_mps: float
    feedforward_wz_rad_s: float
    feedforward_riser_velocity_mps: float
    feedforward_proxy_velocity_rad_s: np.ndarray


def phase_scaled_feedforward(
    sample: RiserPlaybackSample, progress_scale: float
) -> tuple[float, float, float, np.ndarray]:
    """Scale trajectory derivatives consistently with governed phase time."""

    if not math.isfinite(progress_scale) or not 0.0 <= progress_scale <= 1.0:
        raise ValueError("progress scale must be finite and in [0, 1]")
    return (
        sample.feedforward_v_mps * progress_scale,
        sample.feedforward_wz_rad_s * progress_scale,
        sample.feedforward_riser_velocity_mps * progress_scale,
        sample.feedforward_proxy_velocity_rad_s * progress_scale,
    )


def riser_playback_kinematic_metrics(
    plan: RiserPlaybackPlan,
    kinematics: UrdfRiserCameraKinematics,
) -> dict[str, float]:
    """Recompute physical-camera and raw-command metrics from an exported plan."""

    plan.validate()
    count = len(plan.time_s)
    achieved_position = np.empty((count, 3), dtype=np.float64)
    attitude_error = np.empty(count, dtype=np.float64)
    for index in range(count):
        transform = kinematics.world_transform(
            plan.base_xy_yaw[index],
            float(plan.riser_q[index]),
            plan.proxy_gimbal_q[index],
        )
        achieved_position[index] = transform[:3, 3]
        physical_target = quaternion_matrix_wxyz(
            semantic_dfr_to_physical_cam_quat_wxyz(
                plan.target_semantic_dfr_quat_wxyz[index]
            )
        )
        attitude_error[index] = np.linalg.norm(
            rotation_error_vector(transform[:3, :3], physical_target)
        )
    dt = np.diff(plan.time_s)
    delta_xy = np.diff(plan.base_xy_yaw[:, :2], axis=0)
    yaw = np.unwrap(plan.base_xy_yaw[:, 2])
    midpoint_yaw = 0.5 * (yaw[:-1] + yaw[1:])
    forward = (
        np.cos(midpoint_yaw) * delta_xy[:, 0]
        + np.sin(midpoint_yaw) * delta_xy[:, 1]
    ) / dt
    lateral = (
        -np.sin(midpoint_yaw) * delta_xy[:, 0]
        + np.cos(midpoint_yaw) * delta_xy[:, 1]
    ) / dt
    position_error = np.linalg.norm(
        achieved_position - plan.target_position_world_m, axis=1
    )
    return {
        "position_error_p95_m": float(np.percentile(position_error, 95)),
        "position_error_max_m": float(np.max(position_error)),
        "attitude_error_p95_deg": math.degrees(
            float(np.percentile(attitude_error, 95))
        ),
        "attitude_error_max_deg": math.degrees(float(np.max(attitude_error))),
        "maximum_abs_base_linear_velocity_mps": float(np.max(np.abs(forward))),
        "maximum_abs_base_lateral_velocity_mps": float(np.max(np.abs(lateral))),
        "maximum_abs_base_yaw_rate_radps": float(
            np.max(np.abs(np.diff(yaw) / dt))
        ),
        "maximum_abs_riser_rate_mps": float(
            np.max(np.abs(np.diff(plan.riser_q) / dt))
        ),
        "maximum_abs_raw_proxy_target_rate_radps": float(
            np.max(np.abs(np.diff(plan.proxy_gimbal_q, axis=0) / dt[:, None]))
        ),
        "maximum_abs_raw_proxy_target_step_rad": float(
            np.max(np.abs(np.diff(plan.proxy_gimbal_q, axis=0)))
        ),
        "minimum_riser_position_m": float(np.min(plan.riser_q)),
        "maximum_riser_position_m": float(np.max(plan.riser_q)),
        "minimum_target_camera_height_m": float(
            np.min(plan.target_position_world_m[:, 2])
        ),
        "maximum_target_camera_height_m": float(
            np.max(plan.target_position_world_m[:, 2])
        ),
    }


def riser_playback_kinematic_gate(
    metrics: dict[str, float],
    kinematics: UrdfRiserCameraKinematics,
) -> dict[str, bool]:
    epsilon = 1e-9
    return {
        "position_p95_bounded": metrics["position_error_p95_m"]
        <= PLAYBACK_POSITION_P95_LIMIT_M + epsilon,
        "position_max_bounded": metrics["position_error_max_m"]
        <= PLAYBACK_POSITION_MAX_LIMIT_M + epsilon,
        "attitude_p95_bounded": metrics["attitude_error_p95_deg"]
        <= PLAYBACK_ATTITUDE_P95_LIMIT_DEG + epsilon,
        "attitude_max_bounded": metrics["attitude_error_max_deg"]
        <= PLAYBACK_ATTITUDE_MAX_LIMIT_DEG + epsilon,
        "base_linear_velocity_bounded": metrics[
            "maximum_abs_base_linear_velocity_mps"
        ]
        <= PLAYBACK_BASE_LINEAR_LIMIT_MPS + epsilon,
        "base_lateral_velocity_bounded": metrics[
            "maximum_abs_base_lateral_velocity_mps"
        ]
        <= PLAYBACK_BASE_LATERAL_LIMIT_MPS + epsilon,
        "base_yaw_rate_bounded": metrics["maximum_abs_base_yaw_rate_radps"]
        <= PLAYBACK_BASE_YAW_RATE_LIMIT_RAD_S + epsilon,
        "riser_rate_bounded": metrics["maximum_abs_riser_rate_mps"]
        <= PLAYBACK_RISER_RATE_LIMIT_MPS + epsilon,
        "raw_proxy_target_rate_bounded": metrics[
            "maximum_abs_raw_proxy_target_rate_radps"
        ]
        <= PLAYBACK_PROXY_RATE_LIMIT_RAD_S + epsilon,
        "riser_lower_bound": metrics["minimum_riser_position_m"]
        >= kinematics.riser_lower - epsilon,
        "riser_upper_bound": metrics["maximum_riser_position_m"]
        <= kinematics.riser_upper + epsilon,
        "target_camera_height_lower_bound": metrics[
            "minimum_target_camera_height_m"
        ]
        >= PLAYBACK_CAMERA_HEIGHT_MIN_M - epsilon,
        "target_camera_height_upper_bound": metrics[
            "maximum_target_camera_height_m"
        ]
        <= PLAYBACK_CAMERA_HEIGHT_MAX_M + epsilon,
    }


def playback_plan_from_kinematic_plan(
    reference: CorrectedRiserReference,
    plan: RiserKinematicPlan,
) -> RiserPlaybackPlan:
    if reference.case <= 0 or plan.time_s.shape != reference.time_s.shape:
        raise ValueError("reference and plan do not match")
    dt = np.diff(plan.time_s)
    yaw = np.unwrap(plan.base_xy_yaw[:, 2])
    midpoint_yaw = 0.5 * (yaw[:-1] + yaw[1:])
    delta_xy = np.diff(plan.base_xy_yaw[:, :2], axis=0)
    forward_velocity = (
        np.cos(midpoint_yaw) * delta_xy[:, 0]
        + np.sin(midpoint_yaw) * delta_xy[:, 1]
    ) / dt
    yaw_rate = np.diff(yaw) / dt
    result = RiserPlaybackPlan(
        case=reference.case,
        time_s=plan.time_s.copy(),
        target_position_world_m=plan.targets_m.copy(),
        target_semantic_dfr_quat_wxyz=reference.semantic_dfr_quat_wxyz.copy(),
        base_xy_yaw=np.column_stack((plan.base_xy_yaw[:, :2], yaw)),
        riser_q=plan.riser_q.copy(),
        proxy_gimbal_q=plan.gimbal_q.copy(),
        feedforward_v_wz=np.column_stack((forward_velocity, yaw_rate)),
        feedforward_riser_velocity=np.diff(plan.riser_q) / dt,
        feedforward_proxy_velocity=np.diff(plan.gimbal_q, axis=0) / dt[:, None],
        vertical_shift_m=plan.vertical_shift_m,
        planning_strategy=plan.planning_strategy,
        source_time_s=reference.time_s.copy(),
    )
    result.validate()
    return result


def build_riser_playback_plan(
    reference: CorrectedRiserReference,
    kinematics: UrdfRiserCameraKinematics,
    *,
    minimum_camera_height_m: float = 0.6,
    maximum_base_yaw_rate_rad_s: float = PLAYBACK_PLANNING_BASE_YAW_RATE_RAD_S,
) -> RiserPlaybackPlan:
    if not (
        math.isfinite(maximum_base_yaw_rate_rad_s)
        and 0.0 < maximum_base_yaw_rate_rad_s <= PLAYBACK_BASE_YAW_RATE_LIMIT_RAD_S
    ):
        raise ValueError("planning base yaw rate must be within the playback limit")
    shift = max(
        0.0,
        minimum_camera_height_m - float(np.min(reference.positions_m[:, 2])),
    )
    plan = plan_rs4_riser_reference(
        reference,
        kinematics,
        vertical_shift_m=shift,
        maximum_base_yaw_rate_rad_s=maximum_base_yaw_rate_rad_s,
    )
    playback = playback_plan_from_kinematic_plan(reference, plan)
    metrics = riser_playback_kinematic_metrics(playback, kinematics)
    checks = riser_playback_kinematic_gate(metrics, kinematics)
    if not all(checks.values()):
        raise ValueError(
            f"case {reference.case} failed playback export gate: {checks}; {metrics}"
        )
    return playback


def save_riser_playback_plan(path: Path, plan: RiserPlaybackPlan) -> None:
    plan.validate()
    metadata = {
        "schema": PLAYBACK_SCHEMA,
        "case": plan.case,
        "vertical_shift_m": plan.vertical_shift_m,
        "planning_strategy": plan.planning_strategy,
        "target_attitude_contract": "semantic_dfr_to_physical_cam_v1",
        "proxy_joint_order": ["pitch", "roll", "continuous_yaw"],
        "training_started": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        time_s=plan.time_s,
        execution_time_s=plan.time_s,
        target_position_world_m=plan.target_position_world_m,
        target_semantic_dfr_quat_wxyz=plan.target_semantic_dfr_quat_wxyz,
        base_xy_yaw=plan.base_xy_yaw,
        riser_q=plan.riser_q,
        proxy_gimbal_q=plan.proxy_gimbal_q,
        feedforward_v_wz=plan.feedforward_v_wz,
        feedforward_riser_velocity=plan.feedforward_riser_velocity,
        feedforward_proxy_velocity=plan.feedforward_proxy_velocity,
        source_time_s=(
            plan.time_s if plan.source_time_s is None else plan.source_time_s
        ),
    )


def load_riser_playback_plan(path: Path) -> RiserPlaybackPlan:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if metadata.get("schema") != PLAYBACK_SCHEMA:
            raise ValueError(f"unexpected playback schema in {path}")
        execution_time_s = np.asarray(
            (
                data["execution_time_s"]
                if "execution_time_s" in data.files
                else data["time_s"]
            ),
            dtype=np.float64,
        )
        if "time_s" in data.files and not np.array_equal(
            np.asarray(data["time_s"], dtype=np.float64), execution_time_s
        ):
            raise ValueError(f"ambiguous execution time aliases in {path}")
        source_time_s = np.asarray(
            (
                data["source_time_s"]
                if "source_time_s" in data.files
                else execution_time_s
            ),
            dtype=np.float64,
        )
        plan = RiserPlaybackPlan(
            case=int(metadata["case"]),
            time_s=execution_time_s,
            target_position_world_m=np.asarray(
                data["target_position_world_m"], dtype=np.float64
            ),
            target_semantic_dfr_quat_wxyz=np.asarray(
                data["target_semantic_dfr_quat_wxyz"], dtype=np.float64
            ),
            base_xy_yaw=np.asarray(data["base_xy_yaw"], dtype=np.float64),
            riser_q=np.asarray(data["riser_q"], dtype=np.float64),
            proxy_gimbal_q=np.asarray(data["proxy_gimbal_q"], dtype=np.float64),
            feedforward_v_wz=np.asarray(data["feedforward_v_wz"], dtype=np.float64),
            feedforward_riser_velocity=np.asarray(
                data["feedforward_riser_velocity"], dtype=np.float64
            ),
            feedforward_proxy_velocity=np.asarray(
                data["feedforward_proxy_velocity"], dtype=np.float64
            ),
            vertical_shift_m=float(metadata["vertical_shift_m"]),
            planning_strategy=str(metadata["planning_strategy"]),
            source_time_s=source_time_s,
        )
    plan.validate()
    return plan


def interpolate_riser_playback_plan(
    plan: RiserPlaybackPlan,
    elapsed_s: float,
) -> RiserPlaybackSample:
    if not math.isfinite(elapsed_s):
        raise ValueError("elapsed time must be finite")
    elapsed = float(np.clip(elapsed_s, plan.time_s[0], plan.time_s[-1]))
    upper = int(np.searchsorted(plan.time_s, elapsed, side="right"))
    upper = min(max(upper, 1), len(plan.time_s) - 1)
    lower = upper - 1
    dt = float(plan.time_s[upper] - plan.time_s[lower])
    alpha = float(np.clip((elapsed - plan.time_s[lower]) / dt, 0.0, 1.0))

    def blend(values: np.ndarray) -> np.ndarray:
        return (1.0 - alpha) * values[lower] + alpha * values[upper]

    q0 = plan.target_semantic_dfr_quat_wxyz[lower]
    q1 = plan.target_semantic_dfr_quat_wxyz[upper]
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    attitude = (1.0 - alpha) * q0 + alpha * q1
    attitude /= np.linalg.norm(attitude)
    return RiserPlaybackSample(
        base_xy_yaw=blend(plan.base_xy_yaw),
        riser_q=float(blend(plan.riser_q)),
        proxy_gimbal_q=blend(plan.proxy_gimbal_q),
        target_position_world_m=blend(plan.target_position_world_m),
        target_semantic_dfr_quat_wxyz=attitude,
        feedforward_v_mps=float(plan.feedforward_v_wz[lower, 0]),
        feedforward_wz_rad_s=float(plan.feedforward_v_wz[lower, 1]),
        feedforward_riser_velocity_mps=float(
            plan.feedforward_riser_velocity[lower]
        ),
        feedforward_proxy_velocity_rad_s=plan.feedforward_proxy_velocity[
            lower
        ].copy(),
    )
