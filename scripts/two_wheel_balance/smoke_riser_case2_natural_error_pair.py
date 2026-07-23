#!/usr/bin/env python3
"""Observe case-2 residual projection without changing playback commands."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys
from typing import Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance import (  # noqa: E402
    riser_residual_dataset,
)


PLAYBACK = (
    PROJECT_ROOT / "scripts/two_wheel_balance/smoke_riser_reference_playback.py"
)
TELEMETRY_SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_projection_telemetry_v1"
)
ProjectionFunction = Callable[..., np.ndarray]


class ProjectionTelemetry:
    """Measure a projection call while preserving its exact return value."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.sample_count = 0
        self.requested_abs_max = np.zeros(3, dtype=np.float64)
        self.effective_abs_max = np.zeros(3, dtype=np.float64)
        self.effective_normalized_abs_max = np.zeros(3, dtype=np.float64)
        self.requested_effective_delta_abs_max = np.zeros(3, dtype=np.float64)
        self.command_clipped_sample_count = np.zeros(3, dtype=np.int64)
        self.any_command_clipped_sample_count = 0

    def wrap(self, original: ProjectionFunction) -> ProjectionFunction:
        def observed(
            model_vx_m_s: float,
            model_wz_rad_s: float,
            model_riser_target_m: float,
            action: np.ndarray,
            **kwargs: object,
        ) -> np.ndarray:
            result = original(
                model_vx_m_s,
                model_wz_rad_s,
                model_riser_target_m,
                action,
                **kwargs,
            )
            if self.enabled:
                self.record(
                    model_based_command=np.asarray(
                        [
                            model_vx_m_s,
                            model_wz_rad_s,
                            model_riser_target_m,
                        ],
                        dtype=np.float64,
                    ),
                    normalized_action=np.asarray(action, dtype=np.float64),
                    action_scales=np.asarray(
                        kwargs.get(
                            "action_scales",
                            riser_residual_dataset
                            .MODEL_BASED_POLICY_RESIDUAL_SCALES,
                        ),
                        dtype=np.float64,
                    ),
                    effective_command=np.asarray(result, dtype=np.float64),
                )
            return result

        return observed

    def record(
        self,
        *,
        model_based_command: np.ndarray,
        normalized_action: np.ndarray,
        action_scales: np.ndarray,
        effective_command: np.ndarray,
    ) -> None:
        values = (
            model_based_command,
            normalized_action,
            action_scales,
            effective_command,
        )
        if any(value.shape != (3,) for value in values):
            raise ValueError("case2 projection telemetry dimension mismatch")
        if not all(np.isfinite(value).all() for value in values):
            raise ValueError("case2 projection telemetry is non-finite")
        if np.any(action_scales <= 0.0):
            raise ValueError("case2 projection telemetry scales are invalid")

        requested_residual = action_scales * normalized_action
        effective_residual = effective_command - model_based_command
        delta = effective_residual - requested_residual
        clipped = np.abs(delta) > 2e-7
        self.sample_count += 1
        self.requested_abs_max = np.maximum(
            self.requested_abs_max, np.abs(requested_residual)
        )
        self.effective_abs_max = np.maximum(
            self.effective_abs_max, np.abs(effective_residual)
        )
        self.effective_normalized_abs_max = np.maximum(
            self.effective_normalized_abs_max,
            np.abs(effective_residual / action_scales),
        )
        self.requested_effective_delta_abs_max = np.maximum(
            self.requested_effective_delta_abs_max, np.abs(delta)
        )
        self.command_clipped_sample_count += clipped
        self.any_command_clipped_sample_count += int(np.any(clipped))

    def summary(self) -> dict[str, object]:
        return {
            "schema": TELEMETRY_SCHEMA,
            "enabled": self.enabled,
            "sample_count": self.sample_count,
            "requested_residual_abs_max": self.requested_abs_max.tolist(),
            "effective_residual_abs_max": self.effective_abs_max.tolist(),
            "effective_normalized_action_abs_max": (
                self.effective_normalized_abs_max.tolist()
            ),
            "requested_effective_delta_abs_max": (
                self.requested_effective_delta_abs_max.tolist()
            ),
            "command_clipped_sample_count": (
                self.command_clipped_sample_count.tolist()
            ),
            "any_command_clipped_sample_count": (
                self.any_command_clipped_sample_count
            ),
            "observer_modified_commands": False,
            "applied_to_commands": False,
            "labels_captured": False,
            "dataset_created": False,
            "training_started": False,
        }


def _argument_value(argv: list[str], name: str) -> str:
    for index, value in enumerate(argv):
        if value == name:
            if index + 1 >= len(argv):
                raise ValueError(f"{name} is missing its value")
            return argv[index + 1]
        prefix = f"{name}="
        if value.startswith(prefix):
            return value[len(prefix) :]
    raise ValueError(f"{name} is required")


def inject_projection_telemetry(
    output: Path, telemetry: dict[str, object]
) -> None:
    payload = json.loads(output.read_text(encoding="utf-8"))
    results = payload.get("results")
    if (
        not isinstance(results, list)
        or len(results) != 1
        or not isinstance(results[0], dict)
    ):
        raise ValueError("case2 adapter requires exactly one playback result")
    results[0]["corrective_teacher_projection_telemetry"] = telemetry
    payload["case2_projection_telemetry_adapter"] = {
        "schema": (
            "cinebotrl_two_wheel_riser_case2_projection_telemetry_adapter_v1"
        ),
        "shared_playback_path": str(PLAYBACK.resolve()),
        "observer_modified_commands": False,
        "label_capture_started": False,
        "dataset_creation_started": False,
        "training_started": False,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    output = Path(_argument_value(sys.argv[1:], "--output"))
    enabled = "--corrective-teacher-profile" in sys.argv[1:] or any(
        value.startswith("--corrective-teacher-profile=")
        for value in sys.argv[1:]
    )
    telemetry = ProjectionTelemetry(enabled=enabled)
    original = riser_residual_dataset.apply_model_based_policy_residual
    riser_residual_dataset.apply_model_based_policy_residual = telemetry.wrap(
        original
    )
    exit_code = 0
    try:
        runpy.run_path(str(PLAYBACK), run_name="__main__")
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1
    finally:
        riser_residual_dataset.apply_model_based_policy_residual = original

    if output.is_file():
        inject_projection_telemetry(output, telemetry.summary())
    elif exit_code == 0:
        raise FileNotFoundError("case2 playback did not write its output")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
