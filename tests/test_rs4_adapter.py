"""Tests for the pure RS4 attitude-rate adapter."""

from __future__ import annotations

import numpy as np

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
