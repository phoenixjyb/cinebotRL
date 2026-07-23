"""CPU-only structural validation for learned model-based residual policies."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

from .riser_residual_dataset import (
    ACTION_NAMES,
    BASE_OBSERVATION_NAMES,
    LOOKAHEAD_CHANNEL_NAMES,
    LOOKAHEAD_HORIZONS_S,
    OBSERVATION_NAMES,
)


MODEL_BASED_RESIDUAL_POLICY_PARAMETER_COUNT = 142019


def _inspection_report_valid(report: object) -> bool:
    return (
        isinstance(report, dict)
        and set(report)
        == {
            "observation_dimension",
            "base_observation_dimension",
            "lookahead_horizon_count",
            "lookahead_channel_count",
            "action_dimension",
            "parameter_count",
            "smoke_batch_size",
            "output_abs_max",
            "device",
            "passed",
        }
        and report.get("observation_dimension") == len(OBSERVATION_NAMES)
        and report.get("base_observation_dimension")
        == len(BASE_OBSERVATION_NAMES)
        and report.get("lookahead_horizon_count") == len(LOOKAHEAD_HORIZONS_S)
        and report.get("lookahead_channel_count")
        == len(LOOKAHEAD_CHANNEL_NAMES)
        and report.get("action_dimension") == len(ACTION_NAMES)
        and report.get("parameter_count")
        == MODEL_BASED_RESIDUAL_POLICY_PARAMETER_COUNT
        and report.get("smoke_batch_size") == 2
        and isinstance(report.get("output_abs_max"), (int, float))
        and not isinstance(report.get("output_abs_max"), bool)
        and 0.0 <= float(report["output_abs_max"]) <= 1.0 + 1e-6
        and report.get("device") == "cpu"
        and report.get("passed") is True
    )


def _windows_path(path: Path) -> str:
    resolved = path.resolve().as_posix()
    parts = resolved.split("/")
    if len(parts) >= 4 and parts[1] == "mnt" and len(parts[2]) == 1:
        return f"{parts[2].upper()}:\\" + "\\".join(parts[3:])
    return str(path.resolve())


def _external_inspection(path: Path, python_executable: str) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[4]
    source_root = project_root / "src"
    source = (
        "import json,sys;"
        f"sys.path.insert(0,{_windows_path(source_root)!r});"
        "from pathlib import Path;"
        "from rl_platform.tasks.two_wheel_balance."
        "riser_model_based_policy_artifact import "
        "inspect_model_based_residual_torchscript;"
        f"print(json.dumps(inspect_model_based_residual_torchscript("
        f"Path({_windows_path(path)!r})),sort_keys=True))"
    )
    environment = os.environ.copy()
    environment.pop("RISER_POLICY_INSPECTOR_PYTHON", None)
    try:
        result = subprocess.run(
            [python_executable, "-c", source],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
            timeout=120,
        )
        report = json.loads(result.stdout)
    except (
        OSError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
    ) as error:
        raise ValueError(
            "external model-based residual TorchScript inspection failed"
        ) from error
    if not _inspection_report_valid(report):
        raise ValueError("external policy inspection receipt is invalid")
    return report


def inspect_model_based_residual_torchscript(path: Path) -> dict[str, Any]:
    """Load and exercise a residual policy without initializing CUDA or Isaac."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError("model-based residual TorchScript is missing")
    try:
        import torch
    except ModuleNotFoundError as error:
        external_python = os.environ.get("RISER_POLICY_INSPECTOR_PYTHON")
        if not external_python:
            raise ValueError(
                "PyTorch is unavailable and no external inspector is configured"
            ) from error
        return _external_inspection(resolved, external_python)
    try:
        policy = torch.jit.load(str(resolved), map_location="cpu").eval()
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("model-based residual TorchScript cannot be loaded") from error

    expected_observations = len(OBSERVATION_NAMES)
    expected_actions = len(ACTION_NAMES)
    expected_state = len(BASE_OBSERVATION_NAMES)
    expected_horizons = len(LOOKAHEAD_HORIZONS_S)
    expected_channels = len(LOOKAHEAD_CHANNEL_NAMES)
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    try:
        checks = {
            "observation_mean": tuple(policy.observation_mean.shape)
            == (expected_observations,),
            "observation_std": tuple(policy.observation_std.shape)
            == (expected_observations,),
            "observation_mask": tuple(policy.observation_mask.shape)
            == (expected_observations,),
            "normalization_finite": bool(
                torch.isfinite(policy.observation_mean).all()
                and torch.isfinite(policy.observation_std).all()
                and torch.isfinite(policy.observation_mask).all()
            ),
            "normalization_positive": bool(torch.all(policy.observation_std > 0.0)),
            "state_observation_count": (
                policy.state_observation_count == expected_state
            ),
            "lookahead_count": policy.lookahead_count == expected_horizons,
            "lookahead_channel_count": (
                policy.lookahead_channel_count == expected_channels
            ),
            "parameter_count": parameter_count
            == MODEL_BASED_RESIDUAL_POLICY_PARAMETER_COUNT,
        }
    except (AttributeError, TypeError) as error:
        raise ValueError(
            "model-based residual TorchScript structure is incomplete"
        ) from error
    if not all(checks.values()):
        raise ValueError(
            f"model-based residual TorchScript structure mismatch: {checks}"
        )

    inputs = torch.linspace(
        -1.0,
        1.0,
        steps=2 * expected_observations,
        dtype=torch.float32,
    ).reshape(2, expected_observations)
    try:
        with torch.no_grad():
            outputs = policy(inputs)
    except (RuntimeError, ValueError) as error:
        raise ValueError("model-based residual TorchScript smoke failed") from error
    if (
        not isinstance(outputs, torch.Tensor)
        or tuple(outputs.shape) != (2, expected_actions)
        or not bool(torch.isfinite(outputs).all())
        or float(torch.max(torch.abs(outputs)).item()) > 1.0 + 1e-6
    ):
        raise ValueError("model-based residual TorchScript output contract mismatch")
    return {
        "observation_dimension": expected_observations,
        "base_observation_dimension": expected_state,
        "lookahead_horizon_count": expected_horizons,
        "lookahead_channel_count": expected_channels,
        "action_dimension": expected_actions,
        "parameter_count": parameter_count,
        "smoke_batch_size": int(outputs.shape[0]),
        "output_abs_max": float(torch.max(torch.abs(outputs)).item()),
        "device": "cpu",
        "passed": True,
    }


def model_based_residual_torchscript_valid(path: Path) -> bool:
    try:
        inspect_model_based_residual_torchscript(path)
    except (AttributeError, TypeError, ValueError):
        return False
    return True
