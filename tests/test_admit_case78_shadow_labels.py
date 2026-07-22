import numpy as np
import pytest

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (
    OBSERVATION_INDEX,
    OBSERVATION_NAMES,
)
from scripts.two_wheel_balance.admit_case78_shadow_labels import build_admission


def _inputs():
    count = 4
    scales = np.asarray([0.35, 0.4, 0.1])
    observations = np.zeros((count, len(OBSERVATION_NAMES)), dtype=np.float32)
    observations[:, OBSERVATION_INDEX["feedforward_vx_m_s"]] = 0.1
    observations[:, OBSERVATION_INDEX["feedforward_wz_rad_s"]] = -0.1
    observations[:, OBSERVATION_INDEX["riser_position_m"]] = 1.0
    raw = np.asarray(
        [[0.1, 0.05, 0.01], [0.2, -0.1, 0.02], [-0.1, 0.0, -0.01], [0.0, 0.1, 0.0]],
        dtype=np.float32,
    )
    teacher = np.column_stack(
        (0.1 + raw[:, 0], -0.1 + raw[:, 1], 1.0 + raw[:, 2])
    )
    trace = {
        "observations": observations,
        "shadow_teacher_raw_residual_commands": raw,
        "shadow_teacher_normalized_residual_actions": raw / scales,
        "applied_residual_actions": np.zeros((count, 3), dtype=np.float32),
        "shadow_teacher_high_level_commands": teacher,
        "elapsed_time_s": np.arange(count, dtype=np.float64) * 0.005,
        "phase_time_s": np.linspace(0.0, 2.0, count),
        "case_ids": np.full(count, 78, dtype=np.int16),
    }
    metadata = {
        "visited_state_source": "deterministic_controller",
        "shadow_teacher_applied_to_commands": False,
        "trace_only": True,
        "valid_for_training": False,
        "camera_observation_frame": "physical_cam_link_fk",
        "target_attitude_contract": "semantic_dfr_to_physical_cam_v1",
    }
    final = {
        "passed": True,
        "physical_quality_passed": True,
        "shadow_trace_passed": True,
        "candidate_scale_overflow_passed": True,
        "labels_applied_to_commands": False,
        "dataset_created": False,
        "valid_for_training": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "source_duration_s": 1.0,
        "execution_duration_s": 2.0,
    }
    gate = {
        "cases": [78],
        "passed": True,
        "position_observation_link": "physical_cam_link_fk",
        "target_attitude_contract": "semantic_dfr_to_physical_cam_v1",
        "results": [{
            "case": 78,
            "dynamic_quality_passed": True,
            "thermal_admission_passed": True,
            "controller_evidence_passed": True,
            "termination": None,
        }],
    }
    runtime = {
        "case": 78,
        "shadow_measurement_authorized": True,
        "dataset_creation_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
    }
    scale = {
        "teacher40_action_contract_retained": True,
        "teacher40_candidate_scale": scales.tolist(),
        "action_clipping_permitted": False,
    }
    split = {
        "split_admitted": True,
        "admitted_split_cases": {
            "validation": [8, 16, 22, 32, 78],
            "holdout": [3, 5, 13, 19, 24],
        },
        "holdout_opened": False,
    }
    return final, gate, runtime, scale, split, metadata, trace


def test_admits_deterministic_zero_overflow_shadow_series() -> None:
    result = build_admission(
        *_inputs(), expected_row_count=4, source_duration_s=1.0, execution_duration_s=2.0
    )
    assert result["label_admission_passed"]
    assert result["raw_teacher_conversion_authorized"]
    assert result["offline_dataset_rebuild_authorized"]
    assert all(type(value) is bool for value in result["input_contract_checks"].values())
    assert not result["bc_authorized"]
    assert not result["ppo_authorized"]


def test_rejects_applied_action_or_nonsemantic_camera_contract() -> None:
    values = list(_inputs())
    values[-1]["applied_residual_actions"][1, 0] = 0.01
    values[1]["position_observation_link"] = "virtual_frame"
    with pytest.raises(ValueError, match="admission failed"):
        build_admission(
            *values, expected_row_count=4, source_duration_s=1.0, execution_duration_s=2.0
        )
