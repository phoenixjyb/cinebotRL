import math

import pytest

from scripts.two_wheel_balance import (
    audit_case78_camera_correction_cap_candidate as module,
)


def trace_item(*, lever_error_x: float, position_error_x: float) -> dict:
    correction = module.clipped_correction(
        [lever_error_x, 0.0], module.CURRENT_CAP_M
    )
    return {
        "camera_lever_arm_error_xy_m": [lever_error_x, 0.0],
        "camera_lever_arm_correction_xy_m": correction,
        "camera_position_error_xyz_m": [position_error_x, 0.0, 0.0],
        "position_error_m": abs(position_error_x),
    }


def test_larger_bounded_cap_reduces_consistent_camera_error() -> None:
    projection = module.project_trace(
        [trace_item(lever_error_x=0.20, position_error_x=0.20)],
        candidate_cap_m=0.10,
    )

    assert projection["current_position_max_m"] == pytest.approx(0.20)
    assert projection["projected_position_max_m"] == pytest.approx(0.15)
    assert projection["current_correction_reconstruction_mismatch_max_m"] == 0.0


def test_projection_preserves_vertical_error() -> None:
    item = trace_item(lever_error_x=0.20, position_error_x=0.20)
    item["camera_position_error_xyz_m"] = [0.20, 0.0, 0.03]
    item["position_error_m"] = math.hypot(0.20, 0.03)

    projection = module.project_trace([item], candidate_cap_m=0.10)

    assert projection["projected_position_max_m"] == pytest.approx(
        math.hypot(0.15, 0.03)
    )


def test_projection_rejects_non_reconstructing_recorded_correction() -> None:
    item = trace_item(lever_error_x=0.20, position_error_x=0.20)
    item["camera_lever_arm_correction_xy_m"] = [0.0, 0.0]

    projection = module.project_trace([item], candidate_cap_m=0.10)

    assert projection["current_correction_reconstruction_mismatch_max_m"] == pytest.approx(
        0.05
    )


@pytest.mark.parametrize("candidate", [0.05, 0.04, math.nan])
def test_projection_rejects_non_increasing_or_nonfinite_cap(candidate: float) -> None:
    with pytest.raises(ValueError):
        module.project_trace(
            [trace_item(lever_error_x=0.20, position_error_x=0.20)],
            candidate_cap_m=candidate,
        )
