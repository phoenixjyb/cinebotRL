import math

import numpy as np
import pytest

from rl_platform.tasks.two_wheel_balance.riser_control import (
    RiserLimits,
    RiserMotorThermalMonitor,
    QuinticRiserMove,
    balance_progress_scale,
    required_stopping_distance,
    safe_riser_velocity_bounds,
    safe_velocity_for_stopping_distance,
)


def test_motor_thermal_monitor_accepts_continuous_force_and_cools() -> None:
    monitor = RiserMotorThermalMonitor()
    for _ in range(60_000):
        monitor.step(monitor.continuous_force_n, 0.005)
    assert monitor.maximum_thermal_load < 1.0
    hot_load = monitor.thermal_load
    for _ in range(6_000):
        monitor.step(0.0, 0.005)
    assert monitor.thermal_load < hot_load
    assert monitor.passed


def test_motor_thermal_monitor_rejects_sustained_peak_or_peak_violation() -> None:
    monitor = RiserMotorThermalMonitor()
    for _ in range(1_000):
        monitor.step(monitor.peak_force_n, 0.005)
    assert monitor.maximum_thermal_load > 1.0
    assert not monitor.passed

    peak_violation = RiserMotorThermalMonitor()
    peak_violation.step(peak_violation.peak_force_n + 1.0, 0.005)
    assert peak_violation.peak_force_violation_count == 1
    assert not peak_violation.passed


def test_motor_thermal_monitor_rejects_invalid_parameters_and_samples() -> None:
    with pytest.raises(ValueError, match="thermal monitor"):
        RiserMotorThermalMonitor(continuous_force_n=0.0)
    monitor = RiserMotorThermalMonitor()
    with pytest.raises(ValueError, match="positive dt"):
        monitor.step(1.0, 0.0)


def test_riser_reference_completes_move_with_all_limits() -> None:
    limits = RiserLimits()
    move = QuinticRiserMove.for_peak_velocity(0.0, 1.2, 1.0, limits)
    samples = [move.sample(value) for value in np.linspace(0.0, move.duration_s, 2001)]
    assert samples[0].position_m == 0.0
    assert samples[-1].position_m == 1.2
    assert samples[0].velocity_mps == samples[-1].velocity_mps == 0.0
    assert samples[0].acceleration_mps2 == samples[-1].acceleration_mps2 == 0.0
    velocities = np.array([item.velocity_mps for item in samples])
    accelerations = np.array([item.acceleration_mps2 for item in samples])
    jerks = np.array([item.jerk_mps3 for item in samples])
    assert np.max(np.abs(velocities)) <= limits.maximum_velocity_mps + 1e-12
    assert np.max(np.abs(accelerations)) <= limits.maximum_acceleration_mps2 + 1e-12
    assert np.max(np.abs(jerks)) <= limits.maximum_jerk_mps3 + 1e-9


def test_velocity_scale_reaches_requested_stage_speed() -> None:
    dt = 0.005
    for speed in (0.1, 0.25, 0.5, 1.0):
        move = QuinticRiserMove.for_peak_velocity(0.0, 1.2, speed)
        samples = [move.sample(t) for t in np.arange(0.0, move.duration_s + dt, dt)]
        samples.append(move.sample(move.duration_s))
        peak = max(abs(state.velocity_mps) for state in samples)
        assert samples[-1].position_m == 1.2
        assert abs(peak - speed) < 1e-9


def test_balance_progress_governor_is_monotonic_and_bounded() -> None:
    samples = [
        balance_progress_scale(math.radians(degrees))
        for degrees in np.linspace(0.0, 10.0, 101)
    ]
    assert samples[0] == 1.0
    assert samples[-1] == 0.0
    assert all(0.0 <= value <= 1.0 for value in samples)
    assert all(a >= b for a, b in zip(samples, samples[1:]))


def test_invalid_riser_contract_is_rejected() -> None:
    try:
        RiserLimits(lower_m=1.0, upper_m=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid travel range was accepted")


def test_stopping_envelope_inverts_delay_and_deceleration_model() -> None:
    distance = required_stopping_distance(1.0, 5.0, 0.02)
    assert distance == pytest.approx(0.12)
    assert safe_velocity_for_stopping_distance(distance, 5.0, 0.02) == pytest.approx(
        1.0
    )
    assert safe_velocity_for_stopping_distance(0.05, 5.0, 0.02) == pytest.approx(
        0.614142842854285
    )
    assert safe_velocity_for_stopping_distance(0.02, 5.0, 0.02) == pytest.approx(
        0.358257569495584
    )


def test_buffered_mechanical_stroke_allows_full_speed_at_software_limits() -> None:
    for position in (0.0, 1.2):
        lower, upper = safe_riser_velocity_bounds(
            position,
            hard_lower_m=-0.15,
            hard_upper_m=1.35,
            maximum_velocity_mps=1.0,
            maximum_deceleration_mps2=5.0,
            response_delay_s=0.02,
            hard_margin_m=0.03,
        )
        assert lower == pytest.approx(-1.0)
        assert upper == pytest.approx(1.0)


def test_unbuffered_stroke_requires_endpoint_velocity_governor() -> None:
    lower, upper = safe_riser_velocity_bounds(
        0.0,
        hard_lower_m=0.0,
        hard_upper_m=1.2,
        maximum_velocity_mps=1.0,
        maximum_deceleration_mps2=5.0,
        response_delay_s=0.02,
        hard_margin_m=0.03,
    )
    assert lower == 0.0
    assert upper == pytest.approx(1.0)

    lower, upper = safe_riser_velocity_bounds(
        1.15,
        hard_lower_m=0.0,
        hard_upper_m=1.2,
        maximum_velocity_mps=1.0,
        maximum_deceleration_mps2=5.0,
        response_delay_s=0.02,
        hard_margin_m=0.03,
    )
    assert lower == pytest.approx(-1.0)
    assert upper == pytest.approx(0.358257569495584)
