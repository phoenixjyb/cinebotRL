import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from rl_platform.tasks.two_wheel_balance.riser_reference import CorrectedRiserReference
from rl_platform.tasks.two_wheel_balance.riser_rs4_attitude import (
    bounded_path_yaw_schedule,
    compose_semantic_dfr_rotation,
    proxy_joint_to_rs4_command_order,
    proxy_joint_rates_rad_s,
    resolve_rs4_position_command,
    resolve_rs4_position_command_sequence,
    rs4_command_to_proxy_joint_order,
    unwrap_proxy_joint_yaw,
)


def test_rs4_position_mapping_roundtrips_semantic_rotation() -> None:
    basis = Rotation.from_euler("ZYX", [-1.0, 0.4, -0.2]).as_matrix()
    expected_command = np.array([0.25, -0.35, 0.55])
    target = compose_semantic_dfr_rotation(basis, expected_command)
    command, feasible = resolve_rs4_position_command(
        basis, target, np.zeros(3)
    )
    achieved = compose_semantic_dfr_rotation(basis, command)
    assert feasible
    np.testing.assert_allclose(achieved, target, atol=1e-10)


def test_rs4_position_mapping_is_yaw_from_x_roll_from_y_pitch_from_z() -> None:
    command = np.array([0.2, -0.3, 0.4])
    expected = (
        Rotation.from_euler("z", command[2]).as_matrix()
        @ Rotation.from_euler("y", command[1]).as_matrix()
        @ Rotation.from_euler("x", command[0]).as_matrix()
    )
    np.testing.assert_allclose(
        compose_semantic_dfr_rotation(np.eye(3), command), expected, atol=1e-12
    )
    proxy = rs4_command_to_proxy_joint_order(command)
    np.testing.assert_allclose(proxy, [command[2], command[1], command[0]])
    np.testing.assert_allclose(proxy_joint_to_rs4_command_order(proxy), command)


def test_proxy_joint_rate_treats_yaw_boundary_as_shortest_move() -> None:
    proxy = np.deg2rad(
        np.array([[0.0, 0.0, 179.0], [0.0, 0.0, -179.0]])
    )
    rate = proxy_joint_rates_rad_s(proxy, np.array([0.0, 0.1]))
    np.testing.assert_allclose(np.rad2deg(rate[0]), [0.0, 0.0, 20.0], atol=1e-10)


def test_proxy_yaw_targets_are_unwrapped_for_continuous_servo() -> None:
    wrapped = np.deg2rad(
        np.array([[1.0, 2.0, 179.0], [1.5, 2.5, -179.0]])
    )
    continuous = unwrap_proxy_joint_yaw(wrapped)
    np.testing.assert_allclose(
        np.rad2deg(continuous[:, 2]), [179.0, 181.0], atol=1e-10
    )
    np.testing.assert_allclose(
        np.rad2deg(continuous[:, :2]), np.rad2deg(wrapped[:, :2])
    )


def test_bounded_path_yaw_respects_rate_limit() -> None:
    time_s = np.linspace(0.0, 1.0, 11)
    positions = np.column_stack(
        (np.zeros(11), np.linspace(0.0, 1.0, 11), np.ones(11))
    )
    reference = CorrectedRiserReference(
        case=1,
        path=Path("synthetic"),
        positions_m=positions,
        semantic_dfr_quat_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (11, 1)),
        time_s=time_s,
        initial_base_yaw_rad=0.0,
        metadata={},
    )
    yaw = bounded_path_yaw_schedule(reference, maximum_yaw_rate_rad_s=0.4)
    assert np.max(np.abs(np.diff(yaw) / np.diff(time_s))) <= 0.4 + 1e-12
    assert math.isclose(abs(yaw[-1]), 0.4)


def test_sequence_solver_avoids_equivalent_euler_branch_jump() -> None:
    time_s = np.array([0.0, 0.1, 0.2])
    basis = np.tile(np.eye(3), (3, 1, 1))
    target = Rotation.from_euler(
        "ZYX",
        np.deg2rad(
            [
                [179.0, 89.0, 5.0],
                [180.0, 89.5, 5.0],
                [-179.0, 89.0, 5.0],
            ]
        ),
    ).as_matrix()
    command, feasible = resolve_rs4_position_command_sequence(
        basis, target, time_s
    )
    assert np.all(feasible)
    achieved = np.stack(
        [compose_semantic_dfr_rotation(basis[i], command[i]) for i in range(3)]
    )
    np.testing.assert_allclose(achieved, target, atol=1e-9)
    delta = np.diff(command, axis=0)
    delta[:, 0] = (delta[:, 0] + math.pi) % (2.0 * math.pi) - math.pi
    assert np.max(np.abs(np.rad2deg(delta / 0.1))) < 30.0
