import json
import math
from pathlib import Path

import numpy as np
import pytest

from rl_platform.tasks.two_wheel_balance.riser_reference import (
    CorrectedRiserReference,
    bidirectional_path_heading,
    discover_corrected_riser_stage,
    load_corrected_riser_reference,
    plan_rate_metrics,
    retarget_bounded_unicycle_pose,
)
from rl_platform.tasks.two_wheel_balance.camera_attitude import (
    matrix_quaternion_wxyz,
    physical_cam_to_semantic_dfr_quat_wxyz,
)
from rl_platform.tasks.two_wheel_balance.riser_kinematics import (
    UrdfRiserCameraKinematics,
)
from rl_platform.tasks.two_wheel_balance.riser_rs4_attitude import (
    proxy_joint_rates_rad_s,
)
from rl_platform.tasks.two_wheel_balance.riser_rs4_reference import (
    plan_rs4_riser_reference,
)


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets_own/recomoProto2_two_wheel_riser/recomoProto2_two_wheel_riser.urdf"


def _write_case(path: Path, case: int, *, source: str = "corrected_physical_split_teacher") -> Path:
    payload = {
        "metadata": {
            "source": source,
            "source_npz": f"case_{case}.npz",
            "scenario": "no_obstacle",
            "quality_status": "accepted",
            "episode_index": case,
            "duration_s": 0.1,
            "waypoint_dt": 0.1,
            "initial_base_pose_xyyaw": [0.0, 0.0, 0.2],
            "target_orientation_contract": "semantic_dfr_to_physical_cam_v1",
            "recorded_quaternion_order": "xyzw",
            "observation_ee_frame": "physical_cam_link_fk",
        },
        "poses": [
            {"position": [0.0, 0.0, 0.9], "orientation": [0.0, 0.0, 0.0, 1.0]},
            {"position": [0.1, 0.0, 1.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        ],
    }
    result = path / f"episode_{case:04d}_split_teacher_v1.json"
    result.write_text(json.dumps(payload), encoding="utf-8")
    return result


def test_corrected_stage_loader_enforces_frame_contract(tmp_path: Path) -> None:
    first = _write_case(tmp_path, 1)
    _write_case(tmp_path, 2)
    reference = load_corrected_riser_reference(first)
    assert reference.case == 1
    np.testing.assert_allclose(reference.semantic_dfr_quat_wxyz[0], [1.0, 0.0, 0.0, 0.0])
    assert set(discover_corrected_riser_stage(tmp_path, expected_count=2)) == {1, 2}


def test_corrected_stage_loader_rejects_old_or_ambiguous_source(tmp_path: Path) -> None:
    path = _write_case(tmp_path, 1, source="legacy_virtual_teacher")
    with pytest.raises(ValueError, match="wrong source"):
        load_corrected_riser_reference(path)


def test_corrected_stage_loader_accepts_explicit_nonuniform_time(tmp_path: Path) -> None:
    path = _write_case(tmp_path, 1)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["poses"].append(
        {"position": [0.2, 0.0, 1.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
    )
    payload["time_s"] = [0.0, 0.2, 0.3]
    payload["metadata"]["duration_s"] = 0.3
    payload["metadata"]["timing_contract"] = "explicit_time_s_v1"
    path.write_text(json.dumps(payload), encoding="utf-8")

    reference = load_corrected_riser_reference(path)
    np.testing.assert_allclose(reference.time_s, [0.0, 0.2, 0.3])


def test_bidirectional_heading_uses_reverse_motion_without_pi_turn() -> None:
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0], [0.0, 0.0]])
    yaw = bidirectional_path_heading(xy, 0.1)
    assert np.max(np.abs(yaw)) < 1e-12


def test_bounded_unicycle_retarget_tracks_feasible_straight_camera_path() -> None:
    kinematics = UrdfRiserCameraKinematics(URDF)
    time_s = np.linspace(0.0, 0.5, 6)
    base = np.column_stack((0.2 * time_s, np.zeros(6), np.zeros(6)))
    gimbal = np.zeros(3)
    transforms = np.stack(
        [kinematics.world_transform(item, 0.3, gimbal) for item in base]
    )
    semantic = np.stack(
        [
            physical_cam_to_semantic_dfr_quat_wxyz(
                matrix_quaternion_wxyz(transform[:3, :3])
            )
            for transform in transforms
        ]
    )
    reference = CorrectedRiserReference(
        case=1,
        path=Path("synthetic"),
        positions_m=transforms[:, :3, 3],
        semantic_dfr_quat_wxyz=semantic,
        time_s=time_s,
        initial_base_yaw_rad=0.0,
        metadata={},
    )

    plan = retarget_bounded_unicycle_pose(
        reference,
        kinematics,
        maximum_linear_velocity_mps=0.4,
        maximum_base_yaw_rate_radps=0.4,
        maximum_riser_rate_mps=1.0,
        maximum_gimbal_rate_radps=0.5,
        attitude_tolerance_rad=math.radians(2.0),
    )
    metrics = plan_rate_metrics(plan)
    assert metrics["position_error_max_m"] < 1e-3
    assert metrics["position_error_p95_m"] < 1e-3
    assert metrics["attitude_error_max_deg"] < 0.1
    assert metrics["attitude_ik_converged_ratio"] == 1.0
    assert metrics["maximum_abs_base_lateral_velocity_mps"] < 1e-6


def test_rs4_portfolio_tracks_feasible_full_pose_without_rate_violation() -> None:
    kinematics = UrdfRiserCameraKinematics(URDF)
    time_s = np.linspace(0.0, 0.5, 6)
    base = np.column_stack((0.2 * time_s, np.zeros(6), np.zeros(6)))
    proxy = np.zeros(3)
    transforms = np.stack(
        [kinematics.world_transform(item, 0.3, proxy) for item in base]
    )
    semantic = np.stack(
        [
            physical_cam_to_semantic_dfr_quat_wxyz(
                matrix_quaternion_wxyz(transform[:3, :3])
            )
            for transform in transforms
        ]
    )
    reference = CorrectedRiserReference(
        case=1,
        path=Path("synthetic"),
        positions_m=transforms[:, :3, 3],
        semantic_dfr_quat_wxyz=semantic,
        time_s=time_s,
        initial_base_yaw_rad=0.0,
        metadata={},
    )

    plan = plan_rs4_riser_reference(reference, kinematics)
    metrics = plan_rate_metrics(plan)
    assert plan.planning_strategy in {
        "fixed_path",
        "joint_adaptive",
        "preview_0.10m_g1.15",
        "preview_0.10m_g1.50",
        "preview_0.25m_g1.50",
        "preview_0.50m_g1.50",
    }
    assert metrics["position_error_p95_m"] < 1e-3
    assert metrics["attitude_error_max_deg"] < 0.1
    assert np.max(np.abs(proxy_joint_rates_rad_s(plan.gimbal_q, time_s))) <= math.radians(24.0) + 1e-9
