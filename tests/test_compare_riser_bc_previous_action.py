import numpy as np

from scripts.two_wheel_balance.compare_riser_bc_previous_action import (
    comparison_checks,
    error_metrics,
)


def _policy_metrics(teacher_mse, recursive_mse):
    return {
        "teacher_previous_action": {
            "mse_per_action": list(teacher_mse),
            "aggregate_mse": float(np.mean(teacher_mse)),
            "prediction_abs_max_per_action": [0.2, 0.3, 0.1],
        },
        "recursive_previous_action": {
            "mse_per_action": list(recursive_mse),
            "aggregate_mse": float(np.mean(recursive_mse)),
            "prediction_abs_max_per_action": [0.3, 0.4, 0.1],
        },
    }


def test_error_metrics_reports_per_channel_and_bounds() -> None:
    target = np.zeros((2, 3), dtype=np.float32)
    prediction = np.array([[1.0, 0.5, 0.0], [0.0, -0.5, 0.2]], dtype=np.float32)
    metrics = error_metrics(target, prediction)
    np.testing.assert_allclose(metrics["mse_per_action"], [0.5, 0.25, 0.02])
    np.testing.assert_allclose(metrics["prediction_abs_max_per_action"], [1.0, 0.5, 0.2])


def test_comparison_requires_recursive_gain_without_channel_regression() -> None:
    original = _policy_metrics([0.010, 0.010, 0.010], [0.50, 0.40, 0.30])
    masked = _policy_metrics([0.20, 0.20, 0.20], [0.20, 0.20, 0.20])
    candidate = _policy_metrics([0.10, 0.10, 0.10], [0.18, 0.18, 0.18])
    assert all(comparison_checks(original, masked, candidate).values())

    candidate["recursive_previous_action"]["mse_per_action"] = [0.10, 0.10, 0.23]
    candidate["recursive_previous_action"]["aggregate_mse"] = 0.143
    checks = comparison_checks(original, masked, candidate)
    assert checks["recursive_aggregate_beats_masked_by_one_percent"]
    assert not checks["recursive_channels_within_ten_percent_of_masked"]


def test_comparison_rejects_action_bound_overflow() -> None:
    original = _policy_metrics([0.01] * 3, [0.50] * 3)
    masked = _policy_metrics([0.20] * 3, [0.20] * 3)
    candidate = _policy_metrics([0.10] * 3, [0.10] * 3)
    candidate["recursive_previous_action"]["prediction_abs_max_per_action"][1] = 1.01
    assert not comparison_checks(original, masked, candidate)[
        "candidate_predictions_within_action_bounds"
    ]
