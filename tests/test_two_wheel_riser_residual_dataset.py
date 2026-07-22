import json

import numpy as np
import pytest

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (
    ACTION_SCALES,
    BASE_OBSERVATION_NAMES,
    LOOKAHEAD_CHANNEL_NAMES,
    LOOKAHEAD_HORIZONS_S,
    OBSERVATION_INDEX,
    OBSERVATION_NAMES,
    PREVIOUS_ACTION_INDICES,
    apply_residual_action,
    build_executed_observation,
    build_raw_residual_command,
    build_residual_action,
    load_case_dataset,
    load_policy_trace,
    load_shadow_teacher_trace,
    load_raw_teacher_case,
    normalize_raw_teacher_payload,
    save_case_dataset,
    save_policy_trace,
    save_shadow_teacher_trace,
    save_raw_teacher_case,
    normalize_residual_command,
    residual_action_envelope_passed,
)


def _policy_trace_payload(count: int = 3, case: int = 4) -> dict[str, np.ndarray]:
    return {
        "observations": np.zeros((count, len(OBSERVATION_NAMES)), dtype=np.float32),
        "applied_residual_actions": np.zeros((count, 3), dtype=np.float32),
        "final_high_level_commands": np.zeros((count, 3), dtype=np.float32),
        "baseline_wheel_actions": np.zeros((count, 2), dtype=np.float32),
        "case_ids": np.full(count, case, dtype=np.int16),
        "elapsed_time_s": np.arange(count, dtype=np.float64) * 0.005,
        "phase_time_s": np.arange(count, dtype=np.float64) * 0.004,
        "post_step_position_error_m": np.zeros(count, dtype=np.float32),
        "post_step_attitude_error_deg": np.zeros(count, dtype=np.float32),
        "post_step_base_xy_yaw": np.zeros((count, 3), dtype=np.float32),
        "post_step_camera_position_world_m": np.zeros((count, 3), dtype=np.float32),
        "post_step_pitch_deg": np.zeros(count, dtype=np.float32),
        "post_step_riser_position_m": np.zeros(count, dtype=np.float32),
        "post_step_proxy_position_rad": np.zeros((count, 3), dtype=np.float32),
    }


def test_policy_trace_round_trip_is_diagnostic_only(tmp_path) -> None:
    path = tmp_path / "case_0004_policy_trace_v1.npz"
    payload = _policy_trace_payload()
    save_policy_trace(path, 4, payload)
    metadata, restored = load_policy_trace(path)
    assert metadata["trace_only"] is True
    assert metadata["teacher_labels_present"] is False
    assert metadata["residual_dataset_present"] is False
    assert metadata["valid_for_training"] is False
    assert metadata["bc_authorized"] is False
    assert metadata["dagger_authorized"] is False
    assert metadata["ppo_authorized"] is False
    assert metadata["sample_alignment_contract"] == (
        "pre_action_observation_and_command_to_post_step_outcome_v1"
    )
    assert "actions" not in restored
    np.testing.assert_array_equal(restored["case_ids"], payload["case_ids"])


def test_policy_trace_rejects_misaligned_rows_and_mixed_cases(tmp_path) -> None:
    payload = _policy_trace_payload()
    payload["post_step_position_error_m"] = np.zeros(2, dtype=np.float32)
    with pytest.raises(ValueError, match="row counts"):
        save_policy_trace(tmp_path / "bad_rows.npz", 4, payload)
    payload = _policy_trace_payload()
    payload["case_ids"][-1] = 5
    with pytest.raises(ValueError, match="mixes trajectories"):
        save_policy_trace(tmp_path / "mixed.npz", 4, payload)


def test_shadow_teacher_trace_round_trip_is_unapplied_and_not_trainable(
    tmp_path,
) -> None:
    payload = _policy_trace_payload()
    payload["shadow_teacher_raw_residual_commands"] = np.zeros(
        (3, 3), dtype=np.float32
    )
    payload["shadow_teacher_normalized_residual_actions"] = np.zeros(
        (3, 3), dtype=np.float32
    )
    payload["shadow_teacher_high_level_commands"] = np.zeros(
        (3, 3), dtype=np.float32
    )
    path = tmp_path / "case_0004_shadow_teacher_trace_v1.npz"
    save_shadow_teacher_trace(path, 4, payload)
    metadata, restored = load_shadow_teacher_trace(path)
    assert metadata["shadow_teacher_labels_present"] is True
    assert metadata["shadow_teacher_applied_to_commands"] is False
    assert metadata["shadow_teacher_labels_admitted_for_training"] is False
    assert metadata["valid_for_training"] is False
    assert metadata["dagger_authorized"] is False
    assert "actions" not in restored


def test_shadow_teacher_trace_rejects_command_reconstruction_mismatch(
    tmp_path,
) -> None:
    payload = _policy_trace_payload()
    payload["shadow_teacher_raw_residual_commands"] = np.zeros((3, 3))
    payload["shadow_teacher_normalized_residual_actions"] = np.zeros((3, 3))
    payload["shadow_teacher_high_level_commands"] = np.ones((3, 3))
    with pytest.raises(ValueError, match="command reconstruction"):
        save_shadow_teacher_trace(tmp_path / "bad.npz", 4, payload)


def test_shadow_teacher_trace_uses_runtime_policy_action_scales(tmp_path) -> None:
    payload = _policy_trace_payload()
    payload["observations"][:, OBSERVATION_INDEX["feedforward_vx_m_s"]] = 0.1
    payload["observations"][:, OBSERVATION_INDEX["feedforward_wz_rad_s"]] = -0.2
    payload["observations"][:, OBSERVATION_INDEX["riser_position_m"]] = 0.8
    normalized = np.tile([0.2, -0.1, 0.3], (3, 1))
    scales = np.array([0.35, 0.4, 0.1])
    payload["shadow_teacher_normalized_residual_actions"] = normalized
    payload["shadow_teacher_raw_residual_commands"] = normalized * scales
    payload["shadow_teacher_high_level_commands"] = np.column_stack(
        (
            np.full(3, 0.1) + normalized[:, 0] * scales[0],
            np.full(3, -0.2) + normalized[:, 1] * scales[1],
            np.full(3, 0.8) + normalized[:, 2] * scales[2],
        )
    )
    path = tmp_path / "scaled.npz"
    save_shadow_teacher_trace(path, 4, payload, action_scales=scales)
    metadata, _ = load_shadow_teacher_trace(path)
    assert metadata["action_scales"] == [0.35, 0.4, 0.1]


def test_shadow_teacher_trace_marks_deterministic_controller_states(tmp_path) -> None:
    payload = _policy_trace_payload()
    payload["shadow_teacher_raw_residual_commands"] = np.zeros((3, 3))
    payload["shadow_teacher_normalized_residual_actions"] = np.zeros((3, 3))
    payload["shadow_teacher_high_level_commands"] = np.zeros((3, 3))
    path = tmp_path / "deterministic_shadow.npz"
    save_shadow_teacher_trace(
        path,
        4,
        payload,
        visited_state_source="deterministic_controller",
    )
    metadata, _ = load_shadow_teacher_trace(path)
    assert metadata["visited_state_source"] == "deterministic_controller"
    assert metadata["shadow_teacher_computed_before_policy_overwrite"] is False
    assert metadata["shadow_label_computed_before_command_application"] is True
    assert metadata["shadow_teacher_applied_to_commands"] is False


def test_shadow_teacher_trace_rejects_unknown_visited_state_source(tmp_path) -> None:
    payload = _policy_trace_payload()
    payload["shadow_teacher_raw_residual_commands"] = np.zeros((3, 3))
    payload["shadow_teacher_normalized_residual_actions"] = np.zeros((3, 3))
    payload["shadow_teacher_high_level_commands"] = np.zeros((3, 3))
    with pytest.raises(ValueError, match="visited-state source"):
        save_shadow_teacher_trace(
            tmp_path / "bad_source.npz",
            4,
            payload,
            visited_state_source="zero_policy_feedforward",
        )


def test_shadow_teacher_loader_infers_legacy_policy_visited_source(tmp_path) -> None:
    payload = _policy_trace_payload()
    payload["shadow_teacher_raw_residual_commands"] = np.zeros((3, 3))
    payload["shadow_teacher_normalized_residual_actions"] = np.zeros((3, 3))
    payload["shadow_teacher_high_level_commands"] = np.zeros((3, 3))
    path = tmp_path / "legacy_shadow.npz"
    save_shadow_teacher_trace(path, 4, payload)
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata.pop("visited_state_source")
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **arrays)
    restored_metadata, _ = load_shadow_teacher_trace(path)
    assert restored_metadata["visited_state_source"] == "learned_policy"


def test_residual_action_reconstructs_bounded_teacher_command() -> None:
    action = build_residual_action(
        feedforward_vx_m_s=0.10,
        feedforward_wz_rad_s=-0.10,
        commanded_vx_m_s=0.14,
        commanded_wz_rad_s=0.02,
        actual_riser_position_m=0.40,
        target_riser_position_m=0.43,
    )
    np.testing.assert_allclose(action, [0.04 / 0.3, 0.3, 0.3], atol=1e-6)
    np.testing.assert_allclose(
        apply_residual_action(0.10, -0.10, 0.40, action),
        [0.14, 0.02, 0.43],
        atol=1e-6,
    )
    assert ACTION_SCALES.tolist() == [0.3, 0.4, 0.1]


def test_residual_action_clips_final_base_and_riser_commands() -> None:
    command = apply_residual_action(
        0.35,
        -0.30,
        1.18,
        np.array([1.0, -1.0, 1.0]),
    )
    np.testing.assert_allclose(command, [0.4, -0.4, 1.2])


def test_residual_action_uses_explicit_frozen_scales() -> None:
    command = apply_residual_action(
        0.10,
        -0.10,
        0.40,
        np.array([0.5, 0.25, -0.2]),
        action_scales=np.array([0.35, 0.40, 0.10]),
    )
    np.testing.assert_allclose(command, [0.275, 0.0, 0.38])


def test_residual_teacher_action_rejects_scale_clipping() -> None:
    with pytest.raises(ValueError, match="scale is too small"):
        build_residual_action(
            feedforward_vx_m_s=0.1,
            feedforward_wz_rad_s=0.0,
            commanded_vx_m_s=-0.21,
            commanded_wz_rad_s=0.0,
            actual_riser_position_m=0.4,
            target_riser_position_m=0.4,
        )


def test_raw_residual_overflow_remains_diagnostic_without_clipping() -> None:
    raw = build_raw_residual_command(
        feedforward_vx_m_s=0.1,
        feedforward_wz_rad_s=0.0,
        commanded_vx_m_s=0.400601935,
        commanded_wz_rad_s=0.079273308,
        actual_riser_position_m=0.4,
        target_riser_position_m=0.40356378,
    )
    normalized = normalize_residual_command(raw)
    np.testing.assert_allclose(normalized, [1.00200645, 0.19818327, 0.0356378])
    assert not residual_action_envelope_passed(normalized)
    np.testing.assert_allclose(
        [0.400601935, 0.079273308, 0.40356378],
        [
            0.1 + raw[0],
            raw[1],
            0.4 + raw[2],
        ],
    )


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
        lookahead_base_xy_yaw=np.array(
            [
                [1.0, 3.5, np.pi / 2 + 0.2],
                [1.0, 4.0, np.pi / 2 + 0.3],
                [1.0, 5.0, np.pi / 2 + 0.4],
            ]
        ),
        lookahead_camera_position_world_m=np.array(
            [
                [1.0, 3.5, 1.3],
                [1.0, 4.0, 1.4],
                [1.0, 5.0, 1.5],
            ]
        ),
        lookahead_camera_quat_wxyz=np.tile(
            np.array([1.0, 0.0, 0.0, 0.0]), (3, 1)
        ),
        lookahead_riser_target_m=np.array([0.50, 0.55, 0.60]),
        lookahead_feedforward_v_wz_riser=np.array(
            [
                [0.25, 0.11, 0.31],
                [0.30, 0.12, 0.32],
                [0.35, 0.13, 0.33],
            ]
        ),
    )
    assert observation.shape == (len(OBSERVATION_NAMES),)
    np.testing.assert_allclose(observation[6:9], [1.0, 0.0, 0.1], atol=1e-6)
    np.testing.assert_allclose(observation[9:12], [1.0, 0.0, 0.2], atol=1e-6)
    assert len(LOOKAHEAD_HORIZONS_S) == 3
    assert len(LOOKAHEAD_CHANNEL_NAMES) == 13
    assert len(OBSERVATION_NAMES) == 65
    start = len(BASE_OBSERVATION_NAMES)
    np.testing.assert_allclose(
        observation[start : start + len(LOOKAHEAD_CHANNEL_NAMES)],
        [1.5, 0.0, 0.2, 1.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.1, 0.25, 0.11, 0.31],
        atol=1e-6,
    )


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
    assert metadata["schema"] == "cinebotrl_two_wheel_riser_executed_residual_v2"
    assert metadata["lookahead_horizons_s"] == [0.25, 0.5, 1.0]
    np.testing.assert_array_equal(restored["case_ids"], payload["case_ids"])

    payload["case_ids"][2] = 19
    try:
        save_case_dataset(path, 18, payload)
    except ValueError as error:
        assert "mixes trajectories" in str(error)
    else:
        raise AssertionError("mixed-case dataset was accepted")


def test_raw_teacher_capture_is_scale_independent_and_not_trainable(tmp_path) -> None:
    count = 4
    payload = {
        "observations": np.zeros(
            (count, len(OBSERVATION_NAMES)), dtype=np.float32
        ),
        "raw_residual_commands": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.31, -0.2, 0.01],
                [0.32, 0.1, -0.02],
                [0.1, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "case_ids": np.full(count, 10, dtype=np.int16),
        "elapsed_time_s": np.arange(count, dtype=np.float64) * 0.005,
        "phase_time_s": np.arange(count, dtype=np.float64) * 0.004,
        "baseline_wheel_actions": np.zeros((count, 2), dtype=np.float32),
        "teacher_commands": np.zeros((count, 3), dtype=np.float32),
    }
    path = tmp_path / "case_0010_executed_raw_teacher_v1.npz"
    save_raw_teacher_case(path, 10, payload)
    metadata, restored = load_raw_teacher_case(path)
    assert metadata["action_scale_frozen"] is False
    assert metadata["raw_residual_applied_to_commands"] is False
    assert metadata["valid_for_training"] is False
    assert "action_scales" not in metadata
    np.testing.assert_array_equal(
        restored["raw_residual_commands"], payload["raw_residual_commands"]
    )


def test_raw_teacher_capture_rejects_nonzero_previous_action_placeholder(
    tmp_path,
) -> None:
    count = 2
    observations = np.zeros((count, len(OBSERVATION_NAMES)), dtype=np.float32)
    observations[1, 23] = 0.1
    payload = {
        "observations": observations,
        "raw_residual_commands": np.zeros((count, 3), dtype=np.float32),
        "case_ids": np.full(count, 2, dtype=np.int16),
        "elapsed_time_s": np.arange(count, dtype=np.float64) * 0.005,
        "phase_time_s": np.arange(count, dtype=np.float64) * 0.004,
        "baseline_wheel_actions": np.zeros((count, 2), dtype=np.float32),
        "teacher_commands": np.zeros((count, 3), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="previous-action placeholders"):
        save_raw_teacher_case(tmp_path / "bad.npz", 2, payload)


def test_normalize_raw_teacher_rebuilds_previous_actions() -> None:
    count = 3
    observations = np.zeros((count, len(OBSERVATION_NAMES)), dtype=np.float32)
    observations[:, OBSERVATION_INDEX["feedforward_vx_m_s"]] = 0.1
    observations[:, OBSERVATION_INDEX["feedforward_wz_rad_s"]] = -0.2
    observations[:, OBSERVATION_INDEX["riser_position_m"]] = 1.0
    raw = np.array(
        [[0.1, 0.2, 0.01], [-0.2, 0.1, 0.02], [0.05, -0.1, -0.01]],
        dtype=np.float32,
    )
    scales = np.array([0.4, 0.4, 0.1], dtype=np.float64)
    payload = {
        "observations": observations,
        "raw_residual_commands": raw,
        "case_ids": np.full(count, 2, dtype=np.int16),
        "elapsed_time_s": np.arange(count, dtype=np.float64) * 0.005,
        "phase_time_s": np.arange(count, dtype=np.float64) * 0.004,
        "baseline_wheel_actions": np.zeros((count, 2), dtype=np.float32),
        "teacher_commands": np.column_stack(
            (0.1 + raw[:, 0], -0.2 + raw[:, 1], 1.0 + raw[:, 2])
        ),
    }
    normalized = normalize_raw_teacher_payload(payload, scales)
    expected_actions = raw / scales
    np.testing.assert_allclose(normalized["actions"], expected_actions, atol=1e-7)
    previous = normalized["observations"][:, PREVIOUS_ACTION_INDICES]
    np.testing.assert_array_equal(previous[0], np.zeros(3))
    np.testing.assert_allclose(previous[1:], expected_actions[:-1], atol=1e-7)


def test_normalize_raw_teacher_rejects_scale_without_margin() -> None:
    count = 2
    payload = {
        "observations": np.zeros((count, len(OBSERVATION_NAMES)), dtype=np.float32),
        "raw_residual_commands": np.array([[0.4, 0.0, 0.0]] * count),
        "case_ids": np.full(count, 2, dtype=np.int16),
        "elapsed_time_s": np.arange(count, dtype=np.float64) * 0.005,
        "phase_time_s": np.arange(count, dtype=np.float64) * 0.004,
        "baseline_wheel_actions": np.zeros((count, 2), dtype=np.float32),
        "teacher_commands": np.array([[0.4, 0.0, 0.0]] * count),
    }
    with pytest.raises(ValueError, match="unclipped margin"):
        normalize_raw_teacher_payload(payload, np.array([0.4, 0.4, 0.1]))
