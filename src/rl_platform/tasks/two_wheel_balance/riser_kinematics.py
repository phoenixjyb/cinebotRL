"""Pure NumPy kinematics for the two-wheel camera-riser robot.

The external target is the semantic DFR camera pose.  The physical gimbal
angles are solved by an internal adapter and are never learned actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .camera_attitude import (
    AttitudeIkResult,
    PHYSICAL_GIMBAL_JOINTS,
    quaternion_matrix_wxyz,
    rotation_error_vector,
    semantic_dfr_to_physical_cam_quat_wxyz,
)
from .whole_body_kinematics import _axis_rotation, _rpy_matrix, _transform


RISER_JOINT = "riser_joint"
RISER_CAMERA_CHAIN_JOINTS = (RISER_JOINT,) + PHYSICAL_GIMBAL_JOINTS


@dataclass(frozen=True)
class _ChainJoint:
    name: str
    joint_type: str
    origin: np.ndarray
    axis: np.ndarray


@dataclass(frozen=True)
class RiserPoseSolution:
    base_xy_yaw_riser: np.ndarray
    position_error_m: float
    reachable: bool


class UrdfRiserCameraKinematics:
    """FK and bounded inverse kinematics for ``base -> riser -> cam_link``."""

    def __init__(self, urdf: Path) -> None:
        root = ET.parse(urdf).getroot()
        by_child: dict[str, ET.Element] = {}
        limits: dict[str, tuple[float, float]] = {}
        for joint in root.findall("joint"):
            child = joint.find("child").attrib["link"]
            if child in by_child:
                raise ValueError(f"multiple parent joints for {child}")
            by_child[child] = joint
            if joint.attrib["name"] in RISER_CAMERA_CHAIN_JOINTS:
                limit = joint.find("limit")
                if joint.attrib["type"] == "continuous":
                    # Continuous proxy coordinates still use one principal
                    # branch for the legacy numerical IK seed search.
                    limits[joint.attrib["name"]] = (-math.pi, math.pi)
                else:
                    limits[joint.attrib["name"]] = (
                        float(limit.attrib["lower"]),
                        float(limit.attrib["upper"]),
                    )

        reverse_chain: list[_ChainJoint] = []
        link = "cam_link"
        while link != "base_link":
            joint = by_child.get(link)
            if joint is None:
                raise ValueError(f"cannot trace cam_link to base_link from {link}")
            origin = joint.find("origin")
            axis_element = joint.find("axis")
            reverse_chain.append(
                _ChainJoint(
                    name=joint.attrib["name"],
                    joint_type=joint.attrib["type"],
                    origin=_transform(
                        np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" "),
                        _rpy_matrix(
                            np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
                        ),
                    ),
                    axis=(
                        np.fromstring(axis_element.attrib["xyz"], sep=" ")
                        if axis_element is not None
                        else np.array([0.0, 0.0, 1.0])
                    ),
                )
            )
            link = joint.find("parent").attrib["link"]

        self.chain = tuple(reversed(reverse_chain))
        movable = tuple(item.name for item in self.chain if item.joint_type != "fixed")
        if movable != RISER_CAMERA_CHAIN_JOINTS:
            raise ValueError(f"unexpected movable chain to cam_link: {movable}")
        if set(limits) != set(RISER_CAMERA_CHAIN_JOINTS):
            raise ValueError(f"missing camera-chain limits: {set(RISER_CAMERA_CHAIN_JOINTS) - set(limits)}")

        self.riser_lower, self.riser_upper = limits[RISER_JOINT]
        self.gimbal_lower = np.array(
            [limits[name][0] for name in PHYSICAL_GIMBAL_JOINTS], dtype=np.float64
        )
        self.gimbal_upper = np.array(
            [limits[name][1] for name in PHYSICAL_GIMBAL_JOINTS], dtype=np.float64
        )
        joint_types = {item.name: item.joint_type for item in self.chain}
        self.gimbal_continuous = np.array(
            [joint_types[name] == "continuous" for name in PHYSICAL_GIMBAL_JOINTS],
            dtype=bool,
        )

    def normalized_gimbal_limit_margin(self, gimbal_q: np.ndarray) -> float:
        q = np.asarray(gimbal_q, dtype=np.float64)
        if q.shape != (3,) or not np.isfinite(q).all():
            raise ValueError("gimbal_q must be finite shape (3,)")
        span = self.gimbal_upper - self.gimbal_lower
        return float(
            np.min(np.minimum(q - self.gimbal_lower, self.gimbal_upper - q) / span)
        )

    def _joint_map(self, riser_q: float, gimbal_q: np.ndarray) -> dict[str, float]:
        gimbal = np.asarray(gimbal_q, dtype=np.float64)
        if not math.isfinite(riser_q) or gimbal.shape != (3,) or not np.isfinite(gimbal).all():
            raise ValueError("riser_q and gimbal_q must be finite with gimbal shape (3,)")
        return {
            RISER_JOINT: float(riser_q),
            **dict(zip(PHYSICAL_GIMBAL_JOINTS, gimbal, strict=True)),
        }

    def relative_transform(self, riser_q: float, gimbal_q: np.ndarray) -> np.ndarray:
        values = self._joint_map(riser_q, gimbal_q)
        result = np.eye(4)
        for joint in self.chain:
            result = result @ joint.origin
            if joint.joint_type in {"revolute", "continuous"}:
                result = result @ _transform(
                    np.zeros(3), _axis_rotation(joint.axis, values[joint.name])
                )
            elif joint.joint_type == "prismatic":
                result = result @ _transform(
                    joint.axis / np.linalg.norm(joint.axis) * values[joint.name], np.eye(3)
                )
            elif joint.joint_type != "fixed":
                raise ValueError(f"unsupported joint type {joint.joint_type!r}")
        return result

    def world_transform(
        self,
        base_xy_yaw: np.ndarray,
        riser_q: float,
        gimbal_q: np.ndarray,
        *,
        root_z_m: float = 0.0,
    ) -> np.ndarray:
        base = np.asarray(base_xy_yaw, dtype=np.float64)
        if base.shape != (3,) or not np.isfinite(base).all() or not math.isfinite(root_z_m):
            raise ValueError("base_xy_yaw must be finite shape (3,) and root_z_m finite")
        root = _transform(
            np.array([base[0], base[1], root_z_m]),
            _rpy_matrix(np.array([0.0, 0.0, base[2]])),
        )
        return root @ self.relative_transform(riser_q, gimbal_q)

    def world_rotation(
        self,
        root_quat_wxyz: np.ndarray,
        riser_q: float,
        gimbal_q: np.ndarray,
    ) -> np.ndarray:
        return quaternion_matrix_wxyz(root_quat_wxyz) @ self.relative_transform(
            riser_q, gimbal_q
        )[:3, :3]

    def solve_semantic_attitude(
        self,
        root_quat_wxyz: np.ndarray,
        riser_q: float,
        semantic_dfr_quat_wxyz: np.ndarray,
        seed_gimbal_q: np.ndarray,
        *,
        maximum_iterations: int = 30,
        damping: float = 0.03,
        maximum_step_rad: float = 0.15,
        tolerance_rad: float = math.radians(0.1),
    ) -> AttitudeIkResult:
        target = quaternion_matrix_wxyz(
            semantic_dfr_to_physical_cam_quat_wxyz(semantic_dfr_quat_wxyz)
        )
        seed = np.asarray(seed_gimbal_q, dtype=np.float64)
        lower = np.where(self.gimbal_continuous, -np.inf, self.gimbal_lower)
        upper = np.where(self.gimbal_continuous, np.inf, self.gimbal_upper)
        q = np.clip(seed, lower, upper)
        if q.shape != (3,) or not np.isfinite(q).all():
            raise ValueError("seed_gimbal_q must be finite shape (3,)")
        iterations = 0
        for iterations in range(1, maximum_iterations + 1):
            current = self.world_rotation(root_quat_wxyz, riser_q, q)
            residual = rotation_error_vector(current, target)
            if np.linalg.norm(residual) <= tolerance_rad:
                break
            jacobian = np.empty((3, 3), dtype=np.float64)
            epsilon = 1e-5
            for index in range(3):
                shifted = q.copy()
                shifted[index] += epsilon
                jacobian[:, index] = rotation_error_vector(
                    current, self.world_rotation(root_quat_wxyz, riser_q, shifted)
                ) / epsilon
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * damping * np.eye(3), residual
            )
            q = np.clip(
                q + np.clip(delta, -maximum_step_rad, maximum_step_rad),
                lower,
                upper,
            )
        error = float(
            np.linalg.norm(
                rotation_error_vector(
                    self.world_rotation(root_quat_wxyz, riser_q, q), target
                )
            )
        )
        return AttitudeIkResult(q, error, iterations, error <= tolerance_rad)

    def solve_semantic_attitude_robust(
        self,
        root_quat_wxyz: np.ndarray,
        riser_q: float,
        semantic_dfr_quat_wxyz: np.ndarray,
        seed_gimbal_q: np.ndarray,
        *,
        tolerance_rad: float = math.radians(0.1),
    ) -> AttitudeIkResult:
        """Preserve the current branch, then recover from a bad seed if needed."""

        raw_seed = np.asarray(seed_gimbal_q, dtype=np.float64)
        if raw_seed.shape != (3,) or not np.isfinite(raw_seed).all():
            raise ValueError("seed_gimbal_q must be finite shape (3,)")
        seed = np.where(
            self.gimbal_continuous,
            raw_seed,
            np.clip(raw_seed, self.gimbal_lower, self.gimbal_upper),
        )
        primary = self.solve_semantic_attitude(
            root_quat_wxyz,
            riser_q,
            semantic_dfr_quat_wxyz,
            seed,
            maximum_iterations=50,
            tolerance_rad=tolerance_rad,
        )
        if primary.converged:
            return primary

        center = 0.5 * (self.gimbal_lower + self.gimbal_upper)
        axes = [
            (
                np.array([seed[index] - math.pi, seed[index], seed[index] + math.pi])
                if self.gimbal_continuous[index]
                else np.unique(
                    np.clip(
                        np.array(
                            [
                                self.gimbal_lower[index] + 0.05,
                                center[index],
                                0.0,
                                self.gimbal_upper[index] - 0.05,
                            ]
                        ),
                        self.gimbal_lower[index],
                        self.gimbal_upper[index],
                    )
                )
            )
            for index in range(3)
        ]
        candidates = [primary]
        for yaw in axes[0]:
            for roll in axes[1]:
                for pitch in axes[2]:
                    candidate_seed = np.array([yaw, roll, pitch])
                    candidate = self.solve_semantic_attitude(
                        root_quat_wxyz,
                        riser_q,
                        semantic_dfr_quat_wxyz,
                        candidate_seed,
                        maximum_iterations=50,
                        tolerance_rad=tolerance_rad,
                    )
                    candidates.append(candidate)
        converged = [item for item in candidates if item.converged]
        if converged:
            return min(
                converged,
                key=lambda item: float(np.linalg.norm(item.gimbal_q - seed)),
            )
        return min(candidates, key=lambda item: item.orientation_error_rad)

    def solve_position(
        self,
        target_world_m: np.ndarray,
        base_yaw_rad: float,
        gimbal_q: np.ndarray,
        *,
        root_z_m: float = 0.0,
        tolerance_m: float = 1e-8,
    ) -> RiserPoseSolution:
        """Solve base XY and riser height for a fixed base yaw and gimbal pose."""

        target = np.asarray(target_world_m, dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("target_world_m must be finite shape (3,)")
        if not math.isfinite(base_yaw_rad) or tolerance_m <= 0.0:
            raise ValueError("base yaw and tolerance must be valid")

        at_zero = self.relative_transform(0.0, gimbal_q)[:3, 3]
        at_one = self.relative_transform(1.0, gimbal_q)[:3, 3]
        direction = at_one - at_zero
        if abs(direction[2]) <= 1e-9:
            raise ValueError("riser does not change camera height")
        riser_q = (target[2] - root_z_m - at_zero[2]) / direction[2]
        clipped = float(np.clip(riser_q, self.riser_lower, self.riser_upper))

        relative = self.relative_transform(clipped, gimbal_q)[:3, 3]
        yaw_rotation = _rpy_matrix(np.array([0.0, 0.0, base_yaw_rad]))
        rotated = yaw_rotation @ relative
        base_xy = target[:2] - rotated[:2]
        state = np.array([base_xy[0], base_xy[1], base_yaw_rad, clipped])
        achieved = self.world_transform(
            state[:3], state[3], gimbal_q, root_z_m=root_z_m
        )[:3, 3]
        error = float(np.linalg.norm(achieved - target))
        return RiserPoseSolution(state, error, error <= tolerance_m)
