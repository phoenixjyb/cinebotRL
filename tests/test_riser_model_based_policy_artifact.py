from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from scripts.two_wheel_balance.train_riser_residual_bc import (  # noqa: E402
    build_projection_aware_residual_policy,
)
from rl_platform.tasks.two_wheel_balance.riser_model_based_policy_artifact import (  # noqa: E402
    MODEL_BASED_RESIDUAL_POLICY_PARAMETER_COUNT,
    _external_inspection,
    _windows_path,
    inspect_model_based_residual_torchscript,
    model_based_residual_torchscript_valid,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    OBSERVATION_NAMES,
)


def _write_valid_policy(path: Path) -> None:
    policy = build_projection_aware_residual_policy(
        torch.zeros(len(OBSERVATION_NAMES)),
        torch.ones(len(OBSERVATION_NAMES)),
    ).eval()
    torch.jit.script(policy).save(str(path))


def test_inspector_accepts_exact_model_based_residual_policy(tmp_path: Path) -> None:
    path = tmp_path / "policy.pt"
    _write_valid_policy(path)
    report = inspect_model_based_residual_torchscript(path)
    assert report == {
        "observation_dimension": 65,
        "base_observation_dimension": 26,
        "lookahead_horizon_count": 3,
        "lookahead_channel_count": 13,
        "action_dimension": 3,
        "parameter_count": MODEL_BASED_RESIDUAL_POLICY_PARAMETER_COUNT,
        "smoke_batch_size": 2,
        "output_abs_max": 0.0,
        "device": "cpu",
        "passed": True,
    }
    assert model_based_residual_torchscript_valid(path) is True


def test_external_inspector_uses_windows_paths_and_validates_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy = tmp_path / "policy.pt"
    policy.write_bytes(b"hash-bound-placeholder")
    receipt = {
        "observation_dimension": 65,
        "base_observation_dimension": 26,
        "lookahead_horizon_count": 3,
        "lookahead_channel_count": 13,
        "action_dimension": 3,
        "parameter_count": MODEL_BASED_RESIDUAL_POLICY_PARAMETER_COUNT,
        "smoke_batch_size": 2,
        "output_abs_max": 0.4,
        "device": "cpu",
        "passed": True,
    }
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(stdout=__import__("json").dumps(receipt))

    monkeypatch.setattr(
        "rl_platform.tasks.two_wheel_balance."
        "riser_model_based_policy_artifact.subprocess.run",
        fake_run,
    )
    assert _external_inspection(policy, "/mnt/g/python.exe") == receipt
    assert observed["command"][0] == "/mnt/g/python.exe"
    assert observed["command"][1] == "-c"
    assert observed["kwargs"]["timeout"] == 120
    assert "RISER_POLICY_INSPECTOR_PYTHON" not in observed["kwargs"]["env"]
    assert _windows_path(Path("/mnt/g/example/policy.pt")) == (
        "G:\\example\\policy.pt"
    )


def test_external_inspector_rejects_incomplete_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    policy = tmp_path / "policy.pt"
    policy.write_bytes(b"hash-bound-placeholder")
    monkeypatch.setattr(
        "rl_platform.tasks.two_wheel_balance."
        "riser_model_based_policy_artifact.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout='{"passed": true, "device": "cpu"}'
        ),
    )
    with pytest.raises(ValueError, match="receipt is invalid"):
        _external_inspection(policy, "/mnt/g/python.exe")


@pytest.mark.parametrize("artifact", ["missing", "corrupt", "wrong_network"])
def test_inspector_rejects_nonconforming_artifact(
    tmp_path: Path,
    artifact: str,
) -> None:
    path = tmp_path / "policy.pt"
    if artifact == "corrupt":
        path.write_bytes(b"not-a-torchscript")
    elif artifact == "wrong_network":
        torch.jit.script(torch.nn.Linear(65, 3)).save(str(path))
    assert model_based_residual_torchscript_valid(path) is False
    with pytest.raises(ValueError, match="model-based residual TorchScript"):
        inspect_model_based_residual_torchscript(path)
