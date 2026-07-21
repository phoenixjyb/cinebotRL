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
    POLICY_ARCHITECTURE,
    RiserResidualPolicy,
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
