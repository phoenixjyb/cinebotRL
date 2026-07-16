"""Semantic DFR to physical-camera adapter for the two-wheel robot.

The public command is a world-frame semantic DFR attitude.  Physical DJI
gimbal joint angles are an internal simulator detail and are never policy
actions or teacher labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .whole_body_kinematics import _axis_rotation, _rpy_matrix, _transform


ARM_JOINTS = ("joint6_arm_yaw", "joint5_arm_pitch", "joint4_elbow_pitch")
PHYSICAL_GIMBAL_JOINTS = (
    "joint3_gimbal_yaw",
    "joint2_gimbal_roll",
    "joint1_gimbal_pitch",
)
PHYSICAL_CAMERA_CHAIN_JOINTS = ARM_JOINTS + PHYSICAL_GIMBAL_JOINTS


def normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError(f"expected one finite wxyz quaternion, got {value.shape}")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("zero-length quaternion")
    return value / norm


def quaternion_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion_wxyz(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def matrix_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError(f"expected finite rotation shape (3,3), got {value.shape}")
    trace = float(np.trace(value))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [0.25 * scale, (value[2, 1] - value[1, 2]) / scale,
             (value[0, 2] - value[2, 0]) / scale, (value[1, 0] - value[0, 1]) / scale]
        )
    else:
        index = int(np.argmax(np.diag(value)))
        if index == 0:
            scale = math.sqrt(max(1.0 + value[0, 0] - value[1, 1] - value[2, 2], 0.0)) * 2.0
            quat = np.array([(value[2, 1] - value[1, 2]) / scale, 0.25 * scale,
                             (value[0, 1] + value[1, 0]) / scale, (value[0, 2] + value[2, 0]) / scale])
        elif index == 1:
            scale = math.sqrt(max(1.0 + value[1, 1] - value[0, 0] - value[2, 2], 0.0)) * 2.0
            quat = np.array([(value[0, 2] - value[2, 0]) / scale,
                             (value[0, 1] + value[1, 0]) / scale, 0.25 * scale,
                             (value[1, 2] + value[2, 1]) / scale])
        else:
            scale = math.sqrt(max(1.0 + value[2, 2] - value[0, 0] - value[1, 1], 0.0)) * 2.0
            quat = np.array([(value[1, 0] - value[0, 1]) / scale,
                             (value[0, 2] + value[2, 0]) / scale,
                             (value[1, 2] + value[2, 1]) / scale, 0.25 * scale])
    quat = normalize_quaternion_wxyz(quat)
    return -quat if quat[0] < 0.0 else quat


def rotation_error_vector(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return shortest world-frame rotation vector from current to target."""

    relative = np.asarray(target) @ np.asarray(current).T
    quat = matrix_quaternion_wxyz(relative)
    vector_norm = float(np.linalg.norm(quat[1:]))
    if vector_norm <= 1e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(vector_norm, float(quat[0]))
    return quat[1:] * (angle / vector_norm)


def semantic_dfr_to_physical_cam_quat_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Apply ``R_world_cam = R_world_DFR * Rz(+pi/2)`` in wxyz order."""

    w, x, y, z = normalize_quaternion_wxyz(quaternion)
    half_sqrt = 2.0**-0.5
    return normalize_quaternion_wxyz(
        half_sqrt * np.array([w - z, x + y, y - x, w + z])
    )


def physical_cam_to_semantic_dfr_quat_wxyz(quaternion: np.ndarray) -> np.ndarray:
    """Inverse of :func:`semantic_dfr_to_physical_cam_quat_wxyz`."""

    cam_rotation = quaternion_matrix_wxyz(quaternion)
    dfr_rotation = cam_rotation @ _rpy_matrix(np.array([0.0, 0.0, -math.pi / 2.0]))
    return matrix_quaternion_wxyz(dfr_rotation)


@dataclass(frozen=True)
class _TreeJoint:
    name: str
    joint_type: str
    child_link: str
    origin: np.ndarray
    axis: np.ndarray
    child_mass_kg: float
    child_com: np.ndarray


@dataclass(frozen=True)
class AttitudeIkResult:
    gimbal_q: np.ndarray
    orientation_error_rad: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class AttitudeFeedbackResult:
    gimbal_q: np.ndarray
    correction_q: np.ndarray
    orientation_error_rad: float


class UrdfPhysicalCameraKinematics:
    """FK, bounded attitude IK, and gravity effort for the physical camera chain."""

    def __init__(self, urdf: Path) -> None:
        root = ET.parse(urdf).getroot()
        inertials: dict[str, tuple[float, np.ndarray]] = {}
        for link in root.findall("link"):
            inertial = link.find("inertial")
            if inertial is None:
                inertials[link.attrib["name"]] = (0.0, np.eye(4))
                continue
            origin = inertial.find("origin")
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
            inertials[link.attrib["name"]] = (
                float(inertial.find("mass").attrib["value"]),
                _transform(xyz, _rpy_matrix(rpy)),
            )

        by_child = {}
        tree_by_parent: dict[str, list[_TreeJoint]] = {}
        limits = {}
        for joint in root.findall("joint"):
            name = joint.attrib["name"]
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            by_child[child] = joint
            origin = joint.find("origin")
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
            axis_element = joint.find("axis")
            axis = (
                np.fromstring(axis_element.attrib["xyz"], sep=" ")
                if axis_element is not None
                else np.array([0.0, 0.0, 1.0])
            )
            tree_by_parent.setdefault(parent, []).append(
                _TreeJoint(
                    name=name,
                    joint_type=joint.attrib["type"],
                    child_link=child,
                    origin=_transform(xyz, _rpy_matrix(rpy)),
                    axis=axis,
                    child_mass_kg=inertials[child][0],
                    child_com=inertials[child][1],
                )
            )
            if name in PHYSICAL_CAMERA_CHAIN_JOINTS:
                limit = joint.find("limit")
                limits[name] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
        self._tree_by_parent = {key: tuple(value) for key, value in tree_by_parent.items()}

        reverse_chain = []
        link = "cam_link"
        while link != "base_link":
            joint = by_child.get(link)
            if joint is None:
                raise ValueError(f"cannot trace cam_link to base_link from {link}")
            parent = joint.find("parent").attrib["link"]
            reverse_chain.append(next(item for item in self._tree_by_parent[parent]
                                      if item.child_link == link))
            link = parent
        self.chain = tuple(reversed(reverse_chain))
        movable = tuple(item.name for item in self.chain if item.joint_type != "fixed")
        if movable != PHYSICAL_CAMERA_CHAIN_JOINTS:
            raise ValueError(f"unexpected movable chain to cam_link: {movable}")
        self.gimbal_lower = np.array([limits[name][0] for name in PHYSICAL_GIMBAL_JOINTS])
        self.gimbal_upper = np.array([limits[name][1] for name in PHYSICAL_GIMBAL_JOINTS])

    @staticmethod
    def _joint_map(arm_q: np.ndarray, gimbal_q: np.ndarray) -> dict[str, float]:
        arm = np.asarray(arm_q, dtype=np.float64)
        gimbal = np.asarray(gimbal_q, dtype=np.float64)
        if arm.shape != (3,) or gimbal.shape != (3,) or not np.isfinite(np.r_[arm, gimbal]).all():
            raise ValueError("arm_q and gimbal_q must each be finite shape (3,)")
        return dict(zip(PHYSICAL_CAMERA_CHAIN_JOINTS, np.r_[arm, gimbal], strict=True))

    def relative_transform(self, arm_q: np.ndarray, gimbal_q: np.ndarray) -> np.ndarray:
        joint_values = self._joint_map(arm_q, gimbal_q)
        value = np.eye(4)
        for joint in self.chain:
            value = value @ joint.origin
            if joint.joint_type != "fixed":
                value = value @ _transform(
                    np.zeros(3), _axis_rotation(joint.axis, joint_values[joint.name])
                )
        return value

    def world_rotation(
        self, root_quat_wxyz: np.ndarray, arm_q: np.ndarray, gimbal_q: np.ndarray
    ) -> np.ndarray:
        return quaternion_matrix_wxyz(root_quat_wxyz) @ self.relative_transform(arm_q, gimbal_q)[:3, :3]

    def solve_semantic_attitude(
        self,
        root_quat_wxyz: np.ndarray,
        arm_q: np.ndarray,
        semantic_dfr_quat_wxyz: np.ndarray,
        seed_gimbal_q: np.ndarray,
        *,
        maximum_iterations: int = 30,
        damping: float = 0.03,
        maximum_step_rad: float = 0.15,
        tolerance_rad: float = math.radians(0.1),
        gimbal_lower_bound: np.ndarray | None = None,
        gimbal_upper_bound: np.ndarray | None = None,
    ) -> AttitudeIkResult:
        target_rotation = quaternion_matrix_wxyz(
            semantic_dfr_to_physical_cam_quat_wxyz(semantic_dfr_quat_wxyz)
        )
        lower = (
            self.gimbal_lower
            if gimbal_lower_bound is None
            else np.asarray(gimbal_lower_bound, dtype=np.float64)
        )
        upper = (
            self.gimbal_upper
            if gimbal_upper_bound is None
            else np.asarray(gimbal_upper_bound, dtype=np.float64)
        )
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or not np.isfinite(np.r_[lower, upper]).all()
            or np.any(lower < self.gimbal_lower)
            or np.any(upper > self.gimbal_upper)
            or np.any(lower >= upper)
        ):
            raise ValueError("invalid local gimbal bounds")
        q = np.clip(np.asarray(seed_gimbal_q, dtype=np.float64), lower, upper)
        iterations = 0
        for iterations in range(1, maximum_iterations + 1):
            current = self.world_rotation(root_quat_wxyz, arm_q, q)
            residual = rotation_error_vector(current, target_rotation)
            if np.linalg.norm(residual) <= tolerance_rad:
                break
            jacobian = np.empty((3, 3))
            epsilon = 1e-5
            for index in range(3):
                shifted = q.copy()
                shifted[index] += epsilon
                shifted_rotation = self.world_rotation(root_quat_wxyz, arm_q, shifted)
                jacobian[:, index] = rotation_error_vector(current, shifted_rotation) / epsilon
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * damping * np.eye(3), residual
            )
            q = np.clip(
                q + np.clip(delta, -maximum_step_rad, maximum_step_rad),
                lower,
                upper,
            )
        error = float(np.linalg.norm(rotation_error_vector(
            self.world_rotation(root_quat_wxyz, arm_q, q), target_rotation
        )))
        return AttitudeIkResult(q, error, iterations, error <= tolerance_rad)

    def solve_semantic_attitude_near_branch(
        self,
        root_quat_wxyz: np.ndarray,
        arm_q: np.ndarray,
        semantic_dfr_quat_wxyz: np.ndarray,
        nominal_gimbal_q: np.ndarray,
        seed_gimbal_q: np.ndarray,
        *,
        maximum_joint_offset_rad: float = math.radians(15.0),
        maximum_iterations: int = 50,
        tolerance_rad: float = math.radians(0.1),
    ) -> AttitudeIkResult:
        """Solve locally around a nominal Euler branch to prevent branch jumps."""

        nominal = np.asarray(nominal_gimbal_q, dtype=np.float64)
        if (
            nominal.shape != (3,)
            or not np.isfinite(nominal).all()
            or maximum_joint_offset_rad <= 0.0
            or not math.isfinite(maximum_joint_offset_rad)
        ):
            raise ValueError("invalid nominal gimbal target or branch offset")
        lower = np.maximum(
            self.gimbal_lower, nominal - maximum_joint_offset_rad
        )
        upper = np.minimum(
            self.gimbal_upper, nominal + maximum_joint_offset_rad
        )
        return self.solve_semantic_attitude(
            root_quat_wxyz,
            arm_q,
            semantic_dfr_quat_wxyz,
            seed_gimbal_q,
            maximum_iterations=maximum_iterations,
            tolerance_rad=tolerance_rad,
            gimbal_lower_bound=lower,
            gimbal_upper_bound=upper,
        )

    def bounded_attitude_feedback_target(
        self,
        root_quat_wxyz: np.ndarray,
        arm_q: np.ndarray,
        actual_gimbal_q: np.ndarray,
        semantic_dfr_quat_wxyz: np.ndarray,
        nominal_gimbal_q: np.ndarray,
        previous_correction_q: np.ndarray,
        dt: float,
        *,
        gain: float = 0.7,
        damping: float = 0.05,
        maximum_correction_rad: float = math.radians(15.0),
        time_constant_s: float = 0.10,
    ) -> AttitudeFeedbackResult:
        """Apply bounded differential camera feedback around a nominal branch."""

        actual = np.asarray(actual_gimbal_q, dtype=np.float64)
        nominal = np.asarray(nominal_gimbal_q, dtype=np.float64)
        previous = np.asarray(previous_correction_q, dtype=np.float64)
        if any(
            value.shape != (3,) or not np.isfinite(value).all()
            for value in (actual, nominal, previous)
        ):
            raise ValueError("gimbal feedback vectors must be finite shape (3,)")
        if not (
            dt > 0.0
            and 0.0 <= gain <= 1.0
            and damping > 0.0
            and maximum_correction_rad > 0.0
            and time_constant_s >= 0.0
            and all(
                math.isfinite(value)
                for value in (
                    dt,
                    gain,
                    damping,
                    maximum_correction_rad,
                    time_constant_s,
                )
            )
        ):
            raise ValueError("invalid camera-attitude feedback configuration")

        target_rotation = quaternion_matrix_wxyz(
            semantic_dfr_to_physical_cam_quat_wxyz(
                semantic_dfr_quat_wxyz
            )
        )
        current_rotation = self.world_rotation(root_quat_wxyz, arm_q, actual)
        residual = rotation_error_vector(current_rotation, target_rotation)
        jacobian = np.empty((3, 3), dtype=np.float64)
        epsilon = 1e-5
        for index in range(3):
            shifted = actual.copy()
            shifted[index] += epsilon
            shifted_rotation = self.world_rotation(
                root_quat_wxyz, arm_q, shifted
            )
            jacobian[:, index] = (
                rotation_error_vector(current_rotation, shifted_rotation)
                / epsilon
            )
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping * damping * np.eye(3),
            residual,
        )
        requested_correction = np.clip(
            gain * (actual + delta - nominal),
            -maximum_correction_rad,
            maximum_correction_rad,
        )
        alpha = (
            1.0
            if time_constant_s == 0.0
            else 1.0 - math.exp(-dt / time_constant_s)
        )
        correction = previous + alpha * (requested_correction - previous)
        target = np.clip(
            nominal + correction, self.gimbal_lower, self.gimbal_upper
        )
        correction = target - nominal
        return AttitudeFeedbackResult(
            target,
            correction,
            float(np.linalg.norm(residual)),
        )

    def solve_semantic_attitude_continuous(
        self,
        root_quat_wxyz: np.ndarray,
        arm_q: np.ndarray,
        semantic_dfr_quat_wxyz: np.ndarray,
        seed_gimbal_q: np.ndarray,
        *,
        maximum_iterations: int = 100,
        tolerance_rad: float = math.radians(0.1),
    ) -> AttitudeIkResult:
        """Solve from the previous command, then search alternate Euler branches.

        The DJI chain can represent one camera attitude with multiple joint
        triples. A local DLS solve can become trapped at a limit even when a
        different branch is feasible, so the bounded grid is used only after
        the continuity seed fails.
        """

        seed = np.clip(
            np.asarray(seed_gimbal_q, dtype=np.float64),
            self.gimbal_lower,
            self.gimbal_upper,
        )
        primary = self.solve_semantic_attitude(
            root_quat_wxyz,
            arm_q,
            semantic_dfr_quat_wxyz,
            seed,
            maximum_iterations=maximum_iterations,
            tolerance_rad=tolerance_rad,
        )
        if primary.converged:
            return primary

        levels = tuple(
            (lower, 0.5 * (lower + upper), upper)
            for lower, upper in zip(
                self.gimbal_lower, self.gimbal_upper, strict=True
            )
        )
        results = [primary]
        for yaw in levels[0]:
            for roll in levels[1]:
                for pitch in levels[2]:
                    candidate_seed = np.array([yaw, roll, pitch])
                    result = self.solve_semantic_attitude(
                        root_quat_wxyz,
                        arm_q,
                        semantic_dfr_quat_wxyz,
                        candidate_seed,
                        maximum_iterations=maximum_iterations,
                        tolerance_rad=tolerance_rad,
                    )
                    results.append(result)

        converged = [result for result in results if result.converged]
        if converged:
            return min(
                converged,
                key=lambda result: (
                    float(np.linalg.norm(result.gimbal_q - seed)),
                    result.orientation_error_rad,
                ),
            )
        return min(results, key=lambda result: result.orientation_error_rad)

    def gimbal_gravitational_effort_nm(
        self,
        root_quat_wxyz: np.ndarray,
        arm_q: np.ndarray,
        gimbal_q: np.ndarray,
        *,
        gravity_mps2: float = 9.81,
        epsilon: float = 1e-5,
    ) -> np.ndarray:
        root_rotation = quaternion_matrix_wxyz(root_quat_wxyz)

        def potential_energy(test_gimbal_q: np.ndarray) -> float:
            values = self._joint_map(arm_q, test_gimbal_q)

            def subtree(link: str, parent_transform: np.ndarray) -> float:
                energy = 0.0
                for joint in self._tree_by_parent.get(link, ()):
                    child_transform = parent_transform @ joint.origin
                    if joint.joint_type != "fixed":
                        child_transform = child_transform @ _transform(
                            np.zeros(3), _axis_rotation(joint.axis, values.get(joint.name, 0.0))
                        )
                    if joint.child_mass_kg > 0.0:
                        z = float((child_transform @ joint.child_com)[2, 3])
                        energy += joint.child_mass_kg * gravity_mps2 * z
                    energy += subtree(joint.child_link, child_transform)
                return energy

            root_transform = _transform(np.zeros(3), root_rotation)
            return subtree("base_link", root_transform)

        q = np.asarray(gimbal_q, dtype=np.float64)
        effort = np.empty(3)
        for index in range(3):
            plus, minus = q.copy(), q.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            effort[index] = (potential_energy(plus) - potential_energy(minus)) / (2.0 * epsilon)
        return effort
