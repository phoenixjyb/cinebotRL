import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/two_wheel_balance/smoke_riser_case2_natural_error_pair.py"
)
SPEC = importlib.util.spec_from_file_location("case2_projection_adapter", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_adapter_observes_effective_projection_without_changing_result() -> None:
    expected = np.array([-0.4, 0.204, 0.801], dtype=np.float64)
    calls = 0

    def original(*args, **kwargs):
        nonlocal calls
        calls += 1
        return expected

    telemetry = MODULE.ProjectionTelemetry(enabled=True)
    observed = telemetry.wrap(original)
    result = observed(
        -0.4,
        0.2,
        0.8,
        np.array([-0.2, 0.08, 0.05]),
        action_scales=np.array([0.05, 0.05, 0.02]),
    )
    assert calls == 1
    assert result is expected
    summary = telemetry.summary()
    assert summary["sample_count"] == 1
    assert summary["command_clipped_sample_count"] == [1, 0, 0]
    np.testing.assert_allclose(
        summary["effective_normalized_action_abs_max"],
        [0.0, 0.08, 0.05],
    )
    assert summary["observer_modified_commands"] is False
    assert summary["labels_captured"] is False
    assert summary["dataset_created"] is False
    assert summary["training_started"] is False


def test_disabled_adapter_returns_original_result_without_samples() -> None:
    expected = np.array([0.1, -0.2, 0.7])
    telemetry = MODULE.ProjectionTelemetry(enabled=False)
    result = telemetry.wrap(lambda *args, **kwargs: expected)(
        0.1,
        -0.2,
        0.7,
        np.zeros(3),
        action_scales=np.array([0.05, 0.05, 0.02]),
    )
    assert result is expected
    assert telemetry.summary()["sample_count"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        {"normalized_action": np.zeros(2)},
        {"model_based_command": np.array([0.0, np.nan, 0.0])},
        {"action_scales": np.array([0.05, 0.0, 0.02])},
    ],
)
def test_projection_telemetry_rejects_invalid_inputs(mutation) -> None:
    arguments = {
        "model_based_command": np.zeros(3),
        "normalized_action": np.zeros(3),
        "action_scales": np.array([0.05, 0.05, 0.02]),
        "effective_command": np.zeros(3),
    }
    arguments.update(mutation)
    with pytest.raises(ValueError, match="case2 projection telemetry"):
        MODULE.ProjectionTelemetry(enabled=True).record(**arguments)


def test_inject_projection_telemetry_requires_one_result(tmp_path) -> None:
    output = tmp_path / "result.json"
    output.write_text(json.dumps({"results": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        MODULE.inject_projection_telemetry(
            output, MODULE.ProjectionTelemetry(enabled=False).summary()
        )


def test_inject_projection_telemetry_is_non_training(tmp_path) -> None:
    output = tmp_path / "result.json"
    output.write_text(
        json.dumps({"results": [{"case": 2}]}), encoding="utf-8"
    )
    telemetry = MODULE.ProjectionTelemetry(enabled=True).summary()
    MODULE.inject_projection_telemetry(output, telemetry)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["results"][0][
        "corrective_teacher_projection_telemetry"
    ] == telemetry
    adapter = payload["case2_projection_telemetry_adapter"]
    assert adapter["observer_modified_commands"] is False
    assert adapter["label_capture_started"] is False
    assert adapter["dataset_creation_started"] is False
    assert adapter["training_started"] is False
