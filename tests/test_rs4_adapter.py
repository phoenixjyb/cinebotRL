"""Tests for the pure RS4 attitude-rate adapter."""

from __future__ import annotations

import numpy as np
import torch

try:
    import pytest
except ImportError:  # Allows direct execution in the minimal Isaac venv.
    pytest = None

from rl_platform.tasks.mobile_mm.rs4_adapter import (
    Rs4RateAdapterConfig,
    clamp_policy_rate_delta,
    integrate_policy_attitude_deg,
    local_gimbal_order_to_policy_order,
    local_gimbal_to_rs4_axis_order,
    normalized_policy_rates_to_deg_s,
    policy_order_to_local_gimbal_order,
    policy_rates_to_rs4_command_deg_s,
    quaternion_feedforward_policy_rates_deg_s,
    quaternion_residual_policy_rates_deg_s,
    quaternion_residual_policy_rates_rad_s_torch,
    quaternion_world_error_rotvec_rad_torch,
    quaternion_tracking_policy_rates_deg_s,
    slew_limit_policy_rate_sequence_deg_s,
)


def test_normalized_policy_rates_scale_and_mask_roll_by_default():
    config = Rs4RateAdapterConfig(
        max_yaw_rate_deg_s=100.0,
        max_pitch_rate_deg_s=50.0,
        max_roll_rate_deg_s=25.0,
        enable_roll=False,
    )
    rates = normalized_policy_rates_to_deg_s(np.array([0.5, -1.0, 1.0]), config)
    np.testing.assert_allclose(rates, np.array([50.0, -50.0, 0.0], dtype=np.float32))


def test_normalized_policy_rates_can_enable_roll():
    config = Rs4RateAdapterConfig(
        max_yaw_rate_deg_s=100.0,
        max_pitch_rate_deg_s=50.0,
        max_roll_rate_deg_s=25.0,
        enable_roll=True,
    )
    rates = normalized_policy_rates_to_deg_s(np.array([0.5, -1.0, 1.0]), config)
    np.testing.assert_allclose(rates, np.array([50.0, -50.0, 25.0], dtype=np.float32))


def test_policy_to_local_gimbal_order_roundtrip():
    policy_rates = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    local_rates = policy_order_to_local_gimbal_order(policy_rates)
    np.testing.assert_allclose(local_rates, np.array([30.0, 20.0, 10.0], dtype=np.float32))
    np.testing.assert_allclose(local_gimbal_order_to_policy_order(local_rates), policy_rates)


def test_default_rs4_axis_mapping_matches_documented_order():
    # Local gimbal order is [roll, pitch, yaw].  Default map [2,0,1] gives
    # RS4 command order [yaw, roll, pitch].
    local_rates = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    rs4_rates = local_gimbal_to_rs4_axis_order(local_rates)
    np.testing.assert_allclose(rs4_rates, np.array([3.0, 1.0, 2.0], dtype=np.float32))


def test_full_policy_to_rs4_mapping_with_roll_disabled():
    config = Rs4RateAdapterConfig(
        max_yaw_rate_deg_s=90.0,
        max_pitch_rate_deg_s=60.0,
        max_roll_rate_deg_s=30.0,
        enable_roll=False,
    )
    # Policy [yaw, pitch, roll] normalized -> [45, -60, 0] deg/s.
    # Local [roll, pitch, yaw] -> [0, -60, 45].
    # RS4 [yaw, roll, pitch] -> [45, 0, -60].
    rs4_rates = policy_rates_to_rs4_command_deg_s(np.array([0.5, -1.0, 1.0]), config)
    np.testing.assert_allclose(rs4_rates, np.array([45.0, 0.0, -60.0], dtype=np.float32))


def test_clamp_policy_rate_delta_respects_acceleration_and_roll_mask():
    config = Rs4RateAdapterConfig(
        max_yaw_accel_deg_s2=100.0,
        max_pitch_accel_deg_s2=50.0,
        max_roll_accel_deg_s2=25.0,
        enable_roll=False,
    )
    previous = np.array([0.0, 0.0, 10.0], dtype=np.float32)
    desired = np.array([100.0, -100.0, -100.0], dtype=np.float32)
    limited = clamp_policy_rate_delta(desired, previous, dt_s=0.1, config=config)
    np.testing.assert_allclose(limited, np.array([10.0, -5.0, 0.0], dtype=np.float32))


def test_integrate_policy_attitude_wraps_yaw_only():
    attitude = np.array([179.0, 10.0, -5.0], dtype=np.float32)
    rates = np.array([20.0, -10.0, 5.0], dtype=np.float32)
    out = integrate_policy_attitude_deg(attitude, rates, dt_s=0.1)
    np.testing.assert_allclose(out, np.array([-179.0, 9.0, -4.5], dtype=np.float32))


def test_quaternion_residual_is_bounded_near_euler_pitch_singularity():
    # Same near-vertical camera attitude with a small local yaw correction.
    pitch = np.deg2rad(-89.999)
    current = np.array([[np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]])
    yaw_delta = np.deg2rad(5.0)
    local_delta = np.array([[np.cos(yaw_delta / 2), 0.0, 0.0, np.sin(yaw_delta / 2)]])

    def multiply(lhs, rhs):
        w1, x1, y1, z1 = lhs[0]
        w2, x2, y2, z2 = rhs[0]
        return np.array([[
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]])

    target = multiply(current, local_delta)
    rates, residual = quaternion_residual_policy_rates_deg_s(
        current,
        target,
        response_horizon_s=0.5,
        config=Rs4RateAdapterConfig(enable_roll=True),
    )
    np.testing.assert_allclose(residual, np.array([[5.0, 0.0, 0.0]]), atol=1e-3)
    np.testing.assert_allclose(rates, np.array([[10.0, 0.0, 0.0]]), atol=1e-3)


def test_rate_sequence_matches_runtime_acceleration_limit():
    config = Rs4RateAdapterConfig(
        max_yaw_accel_deg_s2=100.0,
        max_pitch_accel_deg_s2=50.0,
        enable_roll=False,
    )
    desired = np.array([[90.0, -90.0, 0.0]] * 3, dtype=np.float32)
    limited = slew_limit_policy_rate_sequence_deg_s(desired, 0.1, config)
    np.testing.assert_allclose(
        limited,
        np.array([[10.0, -5.0, 0.0], [20.0, -10.0, 0.0], [30.0, -15.0, 0.0]]),
    )


def test_torch_quaternion_residual_matches_numpy_adapter():
    yaw_delta = np.deg2rad(12.0)
    current = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    target = np.array(
        [[np.cos(yaw_delta / 2), 0.0, 0.0, np.sin(yaw_delta / 2)]],
        dtype=np.float32,
    )
    config = Rs4RateAdapterConfig(enable_roll=True)
    expected_rates_deg, expected_residual_deg = quaternion_residual_policy_rates_deg_s(
        current,
        target,
        response_horizon_s=0.4,
        config=config,
    )
    rates_rad, residual_rad = quaternion_residual_policy_rates_rad_s_torch(
        torch.from_numpy(current),
        torch.from_numpy(target),
        response_horizon_s=0.4,
        max_policy_rates_rad_s=torch.from_numpy(np.deg2rad(config.max_policy_order_rates)),
        enable_roll=True,
    )
    np.testing.assert_allclose(np.rad2deg(rates_rad.numpy()), expected_rates_deg, atol=1e-4)
    np.testing.assert_allclose(np.rad2deg(residual_rad.numpy()), expected_residual_deg, atol=1e-4)


def test_world_error_rotvec_respects_rotated_camera_frame():
    half = np.deg2rad(45.0)
    current = torch.tensor([[np.cos(half), 0.0, np.sin(half), 0.0]], dtype=torch.float32)
    local_yaw = np.deg2rad(10.0)
    delta = torch.tensor(
        [[np.cos(local_yaw / 2), 0.0, 0.0, np.sin(local_yaw / 2)]],
        dtype=torch.float32,
    )
    w1, x1, y1, z1 = current.unbind(dim=-1)
    w2, x2, y2, z2 = delta.unbind(dim=-1)
    target = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )
    world_error = quaternion_world_error_rotvec_rad_torch(current, target)
    np.testing.assert_allclose(
        world_error.numpy(),
        np.array([[local_yaw, 0.0, 0.0]], dtype=np.float32),
        atol=1e-5,
    )


def test_quaternion_feedforward_preserves_small_target_motion_at_vertical_pitch():
    pitch = np.deg2rad(-90.0)
    base = np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0])
    quats = []
    for yaw_deg in (0.0, 1.0, 2.0):
        yaw = np.deg2rad(yaw_deg)
        delta = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        w1, x1, y1, z1 = base
        w2, x2, y2, z2 = delta
        quats.append([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])
    rates = quaternion_feedforward_policy_rates_deg_s(np.asarray(quats), 0.1)
    np.testing.assert_allclose(rates[:, 0], np.array([10.0, 10.0, 10.0]), atol=1e-3)
    np.testing.assert_allclose(rates[:, 1:], 0.0, atol=1e-3)


def test_tracking_rate_combines_feedforward_and_feedback():
    target = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [np.cos(np.deg2rad(1.0) / 2), 0.0, 0.0, np.sin(np.deg2rad(1.0) / 2)],
    ])
    current = np.array([[1.0, 0.0, 0.0, 0.0]] * 2)
    desired, feedforward, residual = quaternion_tracking_policy_rates_deg_s(
        current,
        target,
        dt_s=0.1,
        response_horizon_s=0.5,
    )
    np.testing.assert_allclose(feedforward[:, 0], np.array([10.0, 10.0]), atol=1e-3)
    np.testing.assert_allclose(residual[:, 0], np.array([0.0, 1.0]), atol=1e-3)
    np.testing.assert_allclose(desired[:, 0], np.array([10.0, 12.0]), atol=1e-3)


def test_adapter_validates_shape_and_dt():
    if pytest is None:
        return
    with pytest.raises(ValueError):
        normalized_policy_rates_to_deg_s(np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        clamp_policy_rate_delta(np.zeros(3), np.zeros(3), dt_s=0.0)
    with pytest.raises(ValueError):
        Rs4RateAdapterConfig(rs4_axis_map_from_gimbal=(0, 0, 1))


if __name__ == "__main__":
    test_normalized_policy_rates_scale_and_mask_roll_by_default()
    test_normalized_policy_rates_can_enable_roll()
    test_policy_to_local_gimbal_order_roundtrip()
    test_default_rs4_axis_mapping_matches_documented_order()
    test_full_policy_to_rs4_mapping_with_roll_disabled()
    test_clamp_policy_rate_delta_respects_acceleration_and_roll_mask()
    test_integrate_policy_attitude_wraps_yaw_only()
    test_quaternion_residual_is_bounded_near_euler_pitch_singularity()
    test_rate_sequence_matches_runtime_acceleration_limit()
    test_torch_quaternion_residual_matches_numpy_adapter()
    test_world_error_rotvec_respects_rotated_camera_frame()
    test_quaternion_feedforward_preserves_small_target_motion_at_vertical_pitch()
    test_tracking_rate_combines_feedforward_and_feedback()
    # Manual equivalent for pytest.raises checks.
    for fn in (
        lambda: normalized_policy_rates_to_deg_s(np.array([1.0, 2.0])),
        lambda: clamp_policy_rate_delta(np.zeros(3), np.zeros(3), dt_s=0.0),
        lambda: Rs4RateAdapterConfig(rs4_axis_map_from_gimbal=(0, 0, 1)),
    ):
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    print("rs4_adapter assertions passed")
