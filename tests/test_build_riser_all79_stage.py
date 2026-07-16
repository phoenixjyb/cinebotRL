import numpy as np

from scripts.two_wheel_balance.build_riser_all79_stage import (
    exported_anchor_metrics,
)
from rl_platform.tasks.two_wheel_balance.all79_reference import (
    FullReference,
    SparseTeacher,
)


def test_exported_anchor_metrics_excludes_matlab_initialization_sample(tmp_path) -> None:
    full = FullReference(
        case=34,
        path=tmp_path / "0034_case.json",
        positions_m=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]),
        attitudes_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]] * 3),
        time_s=np.array([0.0, 1.0, 2.0]),
        metadata={},
    )
    teacher = SparseTeacher(
        case=34,
        path=tmp_path / "teacher.npz",
        base_arm_q=np.zeros((3, 6)),
        time_s=np.array([0.0, 1.0, 2.0]),
        dfr_attitudes_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]] * 2),
        action_order=(),
    )
    rotated_initialization = np.array([np.cos(0.2), 0.0, 0.0, np.sin(0.2)])
    semantic_quat = np.vstack(
        (rotated_initialization, teacher.dfr_attitudes_wxyz)
    )
    poses = np.tile(np.eye(4), (3, 1, 1))
    poses[:, :3, 3] = full.positions_m
    metrics = exported_anchor_metrics(
        full,
        teacher,
        {
            "semantic_poses": poses,
            "physical_targets": poses.copy(),
            "semantic_quat": semantic_quat,
            "q_path": teacher.base_arm_q.copy(),
            "time_s": teacher.time_s.copy(),
        },
    )
    assert metrics["first_exported_full_index"] == 1
    assert metrics["full_anchor_attitude_max_error_deg"] < 1e-10
    assert metrics["source_attitude_export_max_error_deg"] < 1e-10
