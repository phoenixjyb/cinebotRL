import numpy as np

from scripts.two_wheel_balance.build_riser_corrected_stage import (
    interpolate_quaternions,
    retime_corrected_teacher,
)


def test_retime_corrected_teacher_preserves_endpoints_and_duration_count() -> None:
    current = np.zeros((2, 9), dtype=np.float64)
    next_q = np.zeros((2, 9), dtype=np.float64)
    next_q[0, 0] = 1.0
    next_q[1, 0] = 2.0
    semantic = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    output_current, output_next, attitude = retime_corrected_teacher(
        current,
        next_q,
        semantic,
        duration_s=5.0,
        retime_dt_s=0.05,
    )
    assert output_current.shape == output_next.shape == (100, 9)
    assert attitude.shape == (100, 4)
    np.testing.assert_allclose(output_current[0], current[0])
    np.testing.assert_allclose(output_next[-1], next_q[-1])
    np.testing.assert_allclose(np.linalg.norm(attitude, axis=1), 1.0)


def test_quaternion_interpolation_uses_shortest_sign_consistent_path() -> None:
    values = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
    result = interpolate_quaternions(values, np.array([0.0, 1.0]), np.linspace(0.0, 1.0, 5))
    np.testing.assert_allclose(result, np.tile([1.0, 0.0, 0.0, 0.0], (5, 1)))
