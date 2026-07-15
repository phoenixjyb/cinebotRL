import math

import numpy as np

from rl_platform.tasks.two_wheel_balance.riser_control import (
    RiserLimits,
    QuinticRiserMove,
    balance_progress_scale,
)


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
