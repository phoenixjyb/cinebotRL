from pathlib import Path

import numpy as np

from rl_platform.tasks.two_wheel_balance.whole_body_kinematics import (
    UrdfPositionKinematics,
    integrate_unicycle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF = (
    PROJECT_ROOT
    / "assets_own/recomoProto2_two_wheel_whole_body"
    / "recomoProto2_two_wheel_whole_body.urdf"
)


def test_whole_body_ee1_tool_kinematics_is_finite() -> None:
    kinematics = UrdfPositionKinematics(URDF)
    q = np.array([0.0, 0.0, 0.0, 0.0, np.pi / 2.0, 3.0 * np.pi / 4.0])
    position = kinematics.position(q)
    assert position.shape == (3,)
    assert np.isfinite(position).all()
    np.testing.assert_allclose(kinematics.arm_lower, [-3.1416, -1.57, -2.3562])
    np.testing.assert_allclose(kinematics.arm_upper, [3.1416, 1.5707963268, 2.3562])


def test_unicycle_has_no_lateral_body_action() -> None:
    straight = integrate_unicycle(np.array([0.0, 0.0, 0.0]), 1.0, 0.0, 0.1)
    np.testing.assert_allclose(straight, [0.1, 0.0, 0.0])
    turning = integrate_unicycle(np.array([0.0, 0.0, 0.0]), 1.0, 1.0, 0.1)
    np.testing.assert_allclose(
        turning,
        [np.sin(0.1), 1.0 - np.cos(0.1), 0.1],
    )


def test_gravity_effort_is_finite_and_yaw_axis_is_unloaded() -> None:
    kinematics = UrdfPositionKinematics(URDF)
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.7, 2.1])
    effort = kinematics.gravitational_effort_nm(q)
    assert effort.shape == (3,)
    assert np.isfinite(effort).all()
    assert abs(effort[0]) < 1e-6
    # This includes the physical arm/camera subtree beyond semantic ee1_tool.
    np.testing.assert_allclose(effort[1:], [7.74618326, -18.91914978], atol=1e-5)


def test_analytic_gravity_matches_finite_difference_oracle() -> None:
    kinematics = UrdfPositionKinematics(URDF)
    generator = np.random.default_rng(20260716)
    for _ in range(12):
        q = np.concatenate(
            (
                generator.uniform(-1.0, 1.0, size=2),
                generator.uniform(-np.pi, np.pi, size=1),
                generator.uniform(kinematics.arm_lower, kinematics.arm_upper),
            )
        )
        analytic = kinematics.gravitational_effort_nm(q)
        finite_difference = kinematics.gravitational_effort_nm(
            q, finite_difference=True
        )
        np.testing.assert_allclose(analytic, finite_difference, atol=2e-7)


def test_tilted_root_gravity_matches_finite_difference_oracle() -> None:
    kinematics = UrdfPositionKinematics(URDF)
    q = np.array([0.2, -0.1, 0.4, -0.2, 0.75, -0.35])
    root_roll_pitch = np.radians(np.array([2.0, 9.0]))
    analytic = kinematics.gravitational_effort_nm(
        q, root_roll_pitch_rad=root_roll_pitch
    )
    finite_difference = kinematics.gravitational_effort_nm(
        q,
        root_roll_pitch_rad=root_roll_pitch,
        finite_difference=True,
    )
    upright = kinematics.gravitational_effort_nm(q)

    np.testing.assert_allclose(analytic, finite_difference, atol=2e-7)
    assert np.max(np.abs(analytic - upright)) > 1.0


def test_complete_tree_com_produces_finite_posture_dependent_balance_pitch() -> None:
    kinematics = UrdfPositionKinematics(URDF)
    home = np.array([0.0, 0.0, 0.0, 0.0, np.pi / 2.0, 3.0 * np.pi / 4.0])
    extended = np.array([0.0, 0.0, 0.0, -0.2, 0.6, -0.5])

    home_com = kinematics.center_of_mass_root_m(home)
    extended_com = kinematics.center_of_mass_root_m(extended)
    home_pitch = kinematics.equilibrium_pitch_rad(home, 0.1016)
    extended_pitch = kinematics.equilibrium_pitch_rad(extended, 0.1016)

    assert np.isfinite(home_com).all()
    assert np.isfinite(extended_com).all()
    assert np.isfinite([home_pitch, extended_pitch]).all()
    assert np.linalg.norm(home_com - extended_com) > 0.01
    assert abs(home_pitch - extended_pitch) > np.radians(1.0)
