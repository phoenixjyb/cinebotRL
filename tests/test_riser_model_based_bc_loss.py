import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl_platform.tasks.two_wheel_balance.riser_model_based_bc_loss import (  # noqa: E402
    MODEL_BASED_PROJECTED_BC_LOSS,
    REQUESTED_OUTPUT_SLEW_REGULARIZATION,
    ModelBasedProjectedBCLoss,
)

ROOT = Path(__file__).parents[1]
AUDIT_SCRIPT = (
    ROOT / "scripts/two_wheel_balance/" "audit_model_based_corrective_bc_loss.py"
)
SPEC = importlib.util.spec_from_file_location("projected_bc_loss_audit", AUDIT_SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def _loss_inputs(
    model_commands: list[list[float]],
    requested: list[list[float]],
    effective: list[list[float]],
    *,
    previous: list[list[float]] | None = None,
    delta_time_s: list[float] | None = None,
    transition_valid: list[bool] | None = None,
) -> tuple[torch.Tensor, ...]:
    count = len(requested)
    return (
        torch.tensor(model_commands, dtype=torch.float32),
        torch.tensor(requested, dtype=torch.float32),
        torch.tensor(effective, dtype=torch.float32),
        torch.tensor(previous or requested, dtype=torch.float32),
        torch.tensor(delta_time_s or [0.1] * count, dtype=torch.float32),
        torch.tensor(transition_valid or [False] * count, dtype=torch.bool),
        torch.ones(count, dtype=torch.float32),
    )


def test_projected_loss_uses_effective_target_not_requested_teacher() -> None:
    loss = ModelBasedProjectedBCLoss(slew_regularization_weight=1.0)
    inputs = _loss_inputs(
        [[0.39, 0.0, 0.5], [0.39, 0.0, 0.5]],
        [[0.8, 0.2, 0.1], [0.8, 0.2, 0.1]],
        [[0.2, 0.2, 0.1], [0.2, 0.2, 0.1]],
    )
    total, pointwise, slew, projected, clipped = loss(*inputs)
    assert float(total) == pytest.approx(0.0, abs=1e-12)
    assert float(pointwise) == pytest.approx(0.0, abs=1e-12)
    assert float(slew) == pytest.approx(0.0, abs=1e-12)
    torch.testing.assert_close(
        projected,
        torch.tensor([[0.2, 0.2, 0.1], [0.2, 0.2, 0.1]]),
    )
    assert clipped[:, 0].tolist() == [True, True]
    naive_loss = torch.mean(torch.square(inputs[1] - inputs[2]))
    assert float(naive_loss) > 0.0


def test_requested_slew_is_independent_of_effective_projection_jumps() -> None:
    loss = ModelBasedProjectedBCLoss(slew_regularization_weight=1.0)
    inputs = _loss_inputs(
        [[0.4, 0.0, 0.5], [0.3, 0.0, 0.5]],
        [[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        previous=[[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]],
        delta_time_s=[0.01, 0.01],
        transition_valid=[False, True],
    )
    total, pointwise, slew, projected, _ = loss(*inputs)
    assert float(total) == pytest.approx(0.0, abs=1e-12)
    assert float(pointwise) == pytest.approx(0.0, abs=1e-12)
    assert float(slew) == pytest.approx(0.0, abs=1e-12)
    effective_rate = (
        torch.abs(projected[1] - projected[0]) * torch.tensor([0.05, 0.05, 0.02]) / 0.01
    )
    assert float(effective_rate[0]) > 0.1


def test_requested_chatter_is_penalized_but_case_boundary_is_not() -> None:
    loss = ModelBasedProjectedBCLoss(slew_regularization_weight=1.0)
    common = dict(
        model_commands=[[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
        requested=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        effective=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        previous=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        delta_time_s=[0.01, 0.01],
    )
    valid = _loss_inputs(**common, transition_valid=[False, True])
    _, _, valid_slew, _, _ = loss(*valid)
    assert float(valid_slew) > 0.0
    boundary = _loss_inputs(**common, transition_valid=[False, False])
    _, _, boundary_slew, _, _ = loss(*boundary)
    assert float(boundary_slew) == pytest.approx(0.0, abs=1e-12)


def test_projected_loss_is_differentiable_and_scriptable() -> None:
    loss = ModelBasedProjectedBCLoss()
    requested = torch.zeros((2, 3), requires_grad=True)
    inputs = list(
        _loss_inputs(
            [[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.2, -0.1, 0.1], [0.2, -0.1, 0.1]],
        )
    )
    inputs[1] = requested
    total, _, _, _, _ = loss(*inputs)
    total.backward()
    assert requested.grad is not None
    assert torch.isfinite(requested.grad).all()
    assert torch.count_nonzero(requested.grad) == 6

    scripted = torch.jit.script(loss)
    eager = loss(*inputs)
    compiled = scripted(*inputs)
    for eager_value, compiled_value in zip(eager, compiled, strict=True):
        torch.testing.assert_close(eager_value, compiled_value)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda values: values.__setitem__(4, torch.zeros(2)), "timing"),
        (
            lambda values: values.__setitem__(5, torch.ones(2, dtype=torch.float32)),
            "boolean",
        ),
        (
            lambda values: values.__setitem__(6, torch.tensor([1.0, -1.0])),
            "weights",
        ),
        (
            lambda values: values[1].__setitem__((0, 0), torch.tensor(1.1)),
            "bounds",
        ),
    ],
)
def test_projected_loss_rejects_invalid_contract(mutator, message) -> None:
    loss = ModelBasedProjectedBCLoss()
    inputs = list(
        _loss_inputs(
            [[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
    )
    mutator(inputs)
    with pytest.raises(ValueError, match=message):
        loss(*inputs)


def test_projected_loss_rejects_zero_weight_valid_transitions() -> None:
    loss = ModelBasedProjectedBCLoss()
    inputs = list(
        _loss_inputs(
            [[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            transition_valid=[False, True],
        )
    )
    inputs[6] = torch.tensor([1.0, 0.0])
    with pytest.raises(ValueError, match="zero weight"):
        loss(*inputs)


def test_real_case30_audit_proves_loss_semantics_without_training() -> None:
    result = AUDIT.audit(AUDIT.DEFAULT_DATASET, AUDIT.DEFAULT_PROFILE)
    assert result["passed"] is True
    assert result["loss_contract"] == MODEL_BASED_PROJECTED_BC_LOSS
    assert (
        result["requested_slew_regularization_contract"]
        == REQUESTED_OUTPUT_SLEW_REGULARIZATION
    )
    assert result["metrics"]["projected_pointwise_loss"] <= 1e-9
    assert result["metrics"]["requested_slew_violation_count_per_channel"] == [
        0,
        0,
        0,
    ]
    assert result["metrics"]["effective_slew_violation_count_per_channel"] == [
        30,
        49,
        8,
    ]
    assert result["metrics"][
        "effective_unclipped_slew_violation_count_per_channel"
    ] == [0, 0, 0]
    assert result["valid_for_training"] is False
    assert result["bc_authorized"] is False
    assert result["ppo_authorized"] is False
    assert result["training_started"] is False


def test_audit_cli_writes_lf_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    command = [
        sys.executable,
        str(AUDIT_SCRIPT),
        "--output",
        str(output),
    ]
    first = subprocess.run(command, check=False, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert output.read_bytes().endswith(b"\n")
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is True
    second = subprocess.run(command, check=False, capture_output=True, text=True)
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr
