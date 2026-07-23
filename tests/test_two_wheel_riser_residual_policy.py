import pytest


torch = pytest.importorskip("torch")

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    ACTION_NAMES,
    BASE_OBSERVATION_NAMES,
    LOOKAHEAD_CHANNEL_NAMES,
    LOOKAHEAD_HORIZONS_S,
    OBSERVATION_NAMES,
    PREVIOUS_ACTION_INDICES,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_policy import (  # noqa: E402
    MODEL_BASED_RESIDUAL_SAFETY_PROJECTION,
    MODEL_BASED_ZERO_INITIALIZED_RESIDUAL_POLICY_ARCHITECTURE,
    POLICY_ARCHITECTURE,
    ModelBasedResidualSafetyProjection,
    RiserResidualPolicy,
    initialize_model_based_residual_from_planner_imitation,
)


def test_residual_policy_is_dimensioned_and_bounded() -> None:
    assert len(OBSERVATION_NAMES) == 65
    assert len(ACTION_NAMES) == 3
    policy = RiserResidualPolicy(
        torch.zeros(len(OBSERVATION_NAMES)),
        torch.ones(len(OBSERVATION_NAMES)),
        state_hidden_sizes=(32, 16),
        lookahead_hidden_sizes=(16, 8),
        fusion_hidden_sizes=(32, 16),
    )
    output = policy(torch.randn(7, len(OBSERVATION_NAMES)))
    assert output.shape == (7, len(ACTION_NAMES))
    assert torch.isfinite(output).all()
    assert torch.max(torch.abs(output)).item() <= 1.0
    assert POLICY_ARCHITECTURE == "state_shared_lookahead_fusion_v1"
    assert policy.state_observation_count == len(BASE_OBSERVATION_NAMES)
    assert policy.lookahead_count == len(LOOKAHEAD_HORIZONS_S)
    assert policy.lookahead_channel_count == len(LOOKAHEAD_CHANNEL_NAMES)


def test_residual_policy_torchscript_matches_eager_shared_lookahead() -> None:
    policy = RiserResidualPolicy(
        torch.zeros(len(OBSERVATION_NAMES)),
        torch.ones(len(OBSERVATION_NAMES)),
        state_hidden_sizes=(16,),
        lookahead_hidden_sizes=(8,),
        fusion_hidden_sizes=(16,),
    ).eval()
    scripted = torch.jit.script(policy)
    observations = torch.randn(4, len(OBSERVATION_NAMES))
    with torch.inference_mode():
        torch.testing.assert_close(scripted(observations), policy(observations))


def test_residual_policy_rejects_bad_normalization() -> None:
    with pytest.raises(ValueError, match="dimension"):
        RiserResidualPolicy(torch.zeros(2), torch.ones(2))
    with pytest.raises(ValueError, match="positive"):
        RiserResidualPolicy(
            torch.zeros(len(OBSERVATION_NAMES)),
            torch.zeros(len(OBSERVATION_NAMES)),
        )


def test_masked_previous_actions_are_invariant_and_scriptable() -> None:
    policy = RiserResidualPolicy(
        torch.zeros(len(OBSERVATION_NAMES)),
        torch.ones(len(OBSERVATION_NAMES)),
        state_hidden_sizes=(16,),
        lookahead_hidden_sizes=(8,),
        fusion_hidden_sizes=(16,),
        masked_observation_indices=PREVIOUS_ACTION_INDICES,
    ).eval()
    first = torch.randn(4, len(OBSERVATION_NAMES))
    second = first.clone()
    second[:, PREVIOUS_ACTION_INDICES] = torch.randn(4, 3) * 100.0
    scripted = torch.jit.script(policy)
    with torch.inference_mode():
        torch.testing.assert_close(policy(first), policy(second))
        torch.testing.assert_close(scripted(first), scripted(second))


def test_residual_policy_rejects_bad_mask_index() -> None:
    with pytest.raises(ValueError, match="out of range"):
        RiserResidualPolicy(
            torch.zeros(len(OBSERVATION_NAMES)),
            torch.ones(len(OBSERVATION_NAMES)),
            masked_observation_indices=(len(OBSERVATION_NAMES),),
        )


def test_previous_action_gain_attenuates_normalized_inputs_and_is_scriptable() -> None:
    policy = RiserResidualPolicy(
        torch.zeros(len(OBSERVATION_NAMES)),
        torch.ones(len(OBSERVATION_NAMES)),
        state_hidden_sizes=(16,),
        lookahead_hidden_sizes=(8,),
        fusion_hidden_sizes=(16,),
        previous_action_observation_gain=0.1,
    ).eval()
    assert policy.observation_mask[list(PREVIOUS_ACTION_INDICES)].tolist() == pytest.approx(
        [0.1, 0.1, 0.1]
    )
    scripted = torch.jit.script(policy)
    observations = torch.randn(4, len(OBSERVATION_NAMES))
    with torch.inference_mode():
        torch.testing.assert_close(scripted(observations), policy(observations))


def test_previous_action_gain_rejects_out_of_range_value() -> None:
    with pytest.raises(ValueError, match="gain"):
        RiserResidualPolicy(
            torch.zeros(len(OBSERVATION_NAMES)),
            torch.ones(len(OBSERVATION_NAMES)),
            previous_action_observation_gain=1.01,
        )


def test_previous_action_channel_gains_are_applied_independently() -> None:
    policy = RiserResidualPolicy(
        torch.zeros(len(OBSERVATION_NAMES)),
        torch.ones(len(OBSERVATION_NAMES)),
        state_hidden_sizes=(16,),
        lookahead_hidden_sizes=(8,),
        fusion_hidden_sizes=(16,),
        previous_action_observation_gain=(0.1, 0.0, 0.1),
    )
    assert policy.observation_mask[list(PREVIOUS_ACTION_INDICES)].tolist() == pytest.approx(
        [0.1, 0.0, 0.1]
    )


def test_zero_initialized_model_based_residual_is_exactly_null_and_scriptable() -> None:
    policy = RiserResidualPolicy(
        torch.zeros(len(OBSERVATION_NAMES)),
        torch.ones(len(OBSERVATION_NAMES)),
        state_hidden_sizes=(16,),
        lookahead_hidden_sizes=(8,),
        fusion_hidden_sizes=(16,),
        zero_initialize_action_head=True,
    ).eval()
    observations = torch.randn(32, len(OBSERVATION_NAMES))
    scripted = torch.jit.script(policy)
    with torch.inference_mode():
        torch.testing.assert_close(policy(observations), torch.zeros(32, 3))
        torch.testing.assert_close(scripted(observations), torch.zeros(32, 3))
    assert (
        MODEL_BASED_ZERO_INITIALIZED_RESIDUAL_POLICY_ARCHITECTURE
        == "model_based_shared_encoder_zero_initialized_residual_v1"
    )


def test_planner_imitation_encoder_transfer_resets_action_head() -> None:
    source = RiserResidualPolicy(
        torch.zeros(len(OBSERVATION_NAMES)),
        torch.ones(len(OBSERVATION_NAMES)),
        state_hidden_sizes=(16,),
        lookahead_hidden_sizes=(8,),
        fusion_hidden_sizes=(16,),
    )
    target = RiserResidualPolicy(
        torch.zeros(len(OBSERVATION_NAMES)),
        torch.ones(len(OBSERVATION_NAMES)),
        state_hidden_sizes=(16,),
        lookahead_hidden_sizes=(8,),
        fusion_hidden_sizes=(16,),
        zero_initialize_action_head=True,
    )
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.add_(torch.randn_like(parameter))
    initialize_model_based_residual_from_planner_imitation(target, source)
    for name, value in target.state_dict().items():
        if name.startswith("action_head."):
            torch.testing.assert_close(value, torch.zeros_like(value))
        else:
            torch.testing.assert_close(value, source.state_dict()[name])


def test_planner_imitation_encoder_transfer_rejects_shape_drift() -> None:
    source = RiserResidualPolicy(
        torch.zeros(len(OBSERVATION_NAMES)),
        torch.ones(len(OBSERVATION_NAMES)),
        state_hidden_sizes=(16,),
        lookahead_hidden_sizes=(8,),
        fusion_hidden_sizes=(16,),
    )
    target = RiserResidualPolicy(
        torch.zeros(len(OBSERVATION_NAMES)),
        torch.ones(len(OBSERVATION_NAMES)),
        state_hidden_sizes=(32,),
        lookahead_hidden_sizes=(8,),
        fusion_hidden_sizes=(16,),
        zero_initialize_action_head=True,
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        initialize_model_based_residual_from_planner_imitation(target, source)


def test_model_based_residual_safety_projection_is_scriptable() -> None:
    projection = ModelBasedResidualSafetyProjection().eval()
    model_commands = torch.tensor(
        [[0.38, -0.38, 1.19], [0.10, 0.20, 0.70]]
    )
    requested_actions = torch.tensor(
        [[1.0, -1.0, 1.0], [0.4, -0.6, 0.5]]
    )
    expected = projection(model_commands, requested_actions)
    scripted = torch.jit.script(projection)
    actual = scripted(model_commands, requested_actions)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value)
    torch.testing.assert_close(
        actual[0],
        torch.tensor([[0.40, -0.40, 1.20], [0.12, 0.17, 0.71]]),
    )
    torch.testing.assert_close(
        actual[1],
        torch.tensor([[0.4, -0.4, 0.5], [0.4, -0.6, 0.5]]),
    )
    assert actual[2].tolist() == [
        [True, True, True],
        [False, False, False],
    ]
    assert (
        MODEL_BASED_RESIDUAL_SAFETY_PROJECTION
        == "model_based_residual_safety_projection_v1"
    )


def test_model_based_residual_safety_projection_is_differentiable() -> None:
    projection = ModelBasedResidualSafetyProjection()
    model_commands = torch.tensor([[0.0, 0.0, 0.6]])
    requested_actions = torch.tensor(
        [[0.2, -0.3, 0.4]], requires_grad=True
    )
    _, effective_actions, _ = projection(model_commands, requested_actions)
    effective_actions.square().sum().backward()
    assert requested_actions.grad is not None
    assert torch.isfinite(requested_actions.grad).all()
    assert torch.count_nonzero(requested_actions.grad).item() == 3


def test_model_based_residual_safety_projection_rejects_invalid_contract() -> None:
    with pytest.raises(ValueError, match="dimension"):
        ModelBasedResidualSafetyProjection(action_scales=(0.05, 0.05))
    with pytest.raises(ValueError, match="limits"):
        ModelBasedResidualSafetyProjection(action_scales=(0.05, 0.0, 0.02))
    projection = ModelBasedResidualSafetyProjection()
    with pytest.raises(ValueError, match="shape"):
        projection(torch.zeros(3), torch.zeros(3))
    with pytest.raises(ValueError, match="non-finite"):
        projection(
            torch.zeros(1, 3),
            torch.tensor([[0.0, float("nan"), 0.0]]),
        )
