from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts/two_wheel_balance"))

from assemble_corrected_retarget_gate79 import (  # noqa: E402
    CANDIDATE_SCHEMA,
    validate_candidate,
)


def write_candidate(path: Path, **overrides) -> None:
    values = {
        "schema": np.asarray(CANDIDATE_SCHEMA),
        "trajectory_integrity_contract": np.asarray("exact_source_v1"),
        "source_trajectory_integrity_passed": np.bool_(True),
        "source_teacher_quality_passed": np.bool_(True),
        "valid_for_candidate_training": np.bool_(True),
        "case": np.int32(73),
        "runtime_approved": np.bool_(False),
        "training_started": np.bool_(False),
        "position_target_link": np.asarray("ee1_tool"),
        "attitude_target_contract": np.asarray(
            "world_semantic_DFR_quaternion_wxyz_option_B"
        ),
        "physical_gimbal_joint_labels_included": np.bool_(False),
        "time_s": np.array([0.0, 0.1, 0.2]),
        "semantic_start_index": np.int32(1),
        "target_position_world_m": np.zeros((3, 3)),
        "target_attitude_world_dfr_quat_wxyz": np.tile(
            np.array([1.0, 0.0, 0.0, 0.0]), (3, 1)
        ),
        "base_arm_q": np.zeros((3, 6)),
        "control_v_wz_darm": np.zeros((2, 5)),
    }
    values.update(overrides)
    np.savez_compressed(path, **values)


def test_gate79_candidate_contract_accepts_semantic_schema_v3(tmp_path: Path) -> None:
    path = tmp_path / "case_0073.npz"
    write_candidate(path)
    validate_candidate(path, 73)


def test_gate79_candidate_contract_rejects_physical_gimbal_labels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "case_0073.npz"
    write_candidate(
        path,
        physical_gimbal_joint_labels_included=np.bool_(True),
        physical_gimbal_q=np.zeros((3, 3)),
    )
    with np.testing.assert_raises(ValueError):
        validate_candidate(path, 73)


def test_gate79_candidate_contract_rejects_wrong_action_shape(
    tmp_path: Path,
) -> None:
    path = tmp_path / "case_0073.npz"
    write_candidate(path, control_v_wz_darm=np.zeros((2, 8)))
    with np.testing.assert_raises(ValueError):
        validate_candidate(path, 73)
