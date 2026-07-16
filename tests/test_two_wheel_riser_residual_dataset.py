import numpy as np

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (
    ACTION_SCALES,
    OBSERVATION_NAMES,
    apply_residual_action,
    build_executed_observation,
    build_residual_action,
    load_case_dataset,
    save_case_dataset,
)


def test_residual_action_reconstructs_bounded_teacher_command() -> None:
    action = build_residual_action(
        feedforward_vx_m_s=0.10,
        feedforward_wz_rad_s=-0.10,
        commanded_vx_m_s=0.14,
        commanded_wz_rad_s=0.02,
        actual_riser_position_m=0.40,
        target_riser_position_m=0.43,
    )
    np.testing.assert_allclose(action, [0.2, 0.3, 0.3], atol=1e-6)
    np.testing.assert_allclose(
        apply_residual_action(0.10, -0.10, 0.40, action),
        [0.14, 0.02, 0.43],
        atol=1e-6,
    )
    assert ACTION_SCALES.tolist() == [0.2, 0.4, 0.1]


def test_executed_observation_uses_body_frame_errors() -> None:
    observation = build_executed_observation(
        lqr_state=np.arange(6, dtype=float),
        actual_base_xy_yaw=np.array([1.0, 2.0, np.pi / 2]),
        target_base_xy_yaw=np.array([1.0, 3.0, np.pi / 2 + 0.1]),
        actual_camera_position_world_m=np.array([1.0, 2.0, 1.0]),
        target_camera_position_world_m=np.array([1.0, 3.0, 1.2]),
        actual_camera_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        target_camera_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        riser_position_m=0.4,
        riser_velocity_m_s=0.1,
        riser_target_m=0.45,
        feedforward_vx_m_s=0.2,
        feedforward_wz_rad_s=0.1,
        feedforward_riser_velocity_m_s=0.3,
        phase_fraction=0.25,
        progress_scale=0.8,
        previous_residual_action=np.zeros(3),
    )
    assert observation.shape == (len(OBSERVATION_NAMES),)
    np.testing.assert_allclose(observation[6:9], [1.0, 0.0, 0.1], atol=1e-6)
    np.testing.assert_allclose(observation[9:12], [1.0, 0.0, 0.2], atol=1e-6)


def test_case_dataset_round_trip_and_rejects_mixed_cases(tmp_path) -> None:
    count = 3
    payload = {
        "observations": np.zeros((count, len(OBSERVATION_NAMES)), dtype=np.float32),
        "actions": np.zeros((count, 3), dtype=np.float32),
        "case_ids": np.full(count, 18, dtype=np.int16),
        "elapsed_time_s": np.arange(count, dtype=np.float64) * 0.005,
        "phase_time_s": np.arange(count, dtype=np.float64) * 0.004,
        "baseline_wheel_actions": np.zeros((count, 2), dtype=np.float32),
        "teacher_commands": np.zeros((count, 3), dtype=np.float32),
    }
    path = tmp_path / "case_0018.npz"
    save_case_dataset(path, 18, payload)
    metadata, restored = load_case_dataset(path)
    assert metadata["case"] == 18
    np.testing.assert_array_equal(restored["case_ids"], payload["case_ids"])

    payload["case_ids"][2] = 19
    try:
        save_case_dataset(path, 18, payload)
    except ValueError as error:
        assert "mixes trajectories" in str(error)
    else:
        raise AssertionError("mixed-case dataset was accepted")
