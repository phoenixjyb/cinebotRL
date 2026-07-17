import pytest


torch = pytest.importorskip("torch")

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    ACTION_NAMES,
    OBSERVATION_NAMES,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_policy import (  # noqa: E402
    RiserResidualPolicy,
)


def test_residual_policy_is_dimensioned_and_bounded() -> None:
    assert len(OBSERVATION_NAMES) == 65
    assert len(ACTION_NAMES) == 3
    policy = RiserResidualPolicy(
        torch.zeros(len(OBSERVATION_NAMES)),
        torch.ones(len(OBSERVATION_NAMES)),
        hidden_sizes=(32, 16),
    )
    output = policy(torch.randn(7, len(OBSERVATION_NAMES)))
    assert output.shape == (7, len(ACTION_NAMES))
    assert torch.isfinite(output).all()
    assert torch.max(torch.abs(output)).item() <= 1.0


def test_residual_policy_rejects_bad_normalization() -> None:
    with pytest.raises(ValueError, match="dimension"):
        RiserResidualPolicy(torch.zeros(2), torch.ones(2))
    with pytest.raises(ValueError, match="positive"):
        RiserResidualPolicy(
            torch.zeros(len(OBSERVATION_NAMES)),
            torch.zeros(len(OBSERVATION_NAMES)),
        )
