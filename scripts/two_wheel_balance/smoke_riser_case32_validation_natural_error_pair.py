#!/usr/bin/env python3
"""Observe case-32 effective residual projection without changing commands."""

from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATH = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/smoke_riser_case2_natural_error_pair.py"
)
PLAYBACK = (
    PROJECT_ROOT / "scripts/two_wheel_balance/smoke_riser_reference_playback.py"
)


def _load_engine():
    name = "_cinebotrl_case32_projection_telemetry_engine"
    spec = importlib.util.spec_from_file_location(name, ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load projection telemetry engine: {ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ENGINE = _load_engine()
ProjectionTelemetry = ENGINE.ProjectionTelemetry


def _argument_value(argv: list[str], name: str) -> str:
    return ENGINE._argument_value(argv, name)


def inject_projection_telemetry(
    output: Path, telemetry: dict[str, object]
) -> None:
    payload = json.loads(output.read_text(encoding="utf-8"))
    results = payload.get("results")
    if (
        not isinstance(results, list)
        or len(results) != 1
        or not isinstance(results[0], dict)
        or results[0].get("case") != 32
    ):
        raise ValueError("case32 adapter requires exactly one case-32 result")
    results[0]["corrective_teacher_projection_telemetry"] = telemetry
    payload["case32_validation_projection_telemetry_adapter"] = {
        "schema": (
            "cinebotrl_two_wheel_riser_case32_validation_"
            "projection_telemetry_adapter_v1"
        ),
        "shared_playback_path": str(PLAYBACK.resolve()),
        "telemetry_engine_path": str(ENGINE_PATH.resolve()),
        "observer_modified_commands": False,
        "label_capture_started": False,
        "dataset_creation_started": False,
        "teacher_admission_opened": False,
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
    original = ENGINE.riser_residual_dataset.apply_model_based_policy_residual
    ENGINE.riser_residual_dataset.apply_model_based_policy_residual = (
        telemetry.wrap(original)
    )
    exit_code = 0
    try:
        runpy.run_path(str(PLAYBACK), run_name="__main__")
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1
    finally:
        ENGINE.riser_residual_dataset.apply_model_based_policy_residual = (
            original
        )

    if output.is_file():
        inject_projection_telemetry(output, telemetry.summary())
    elif exit_code == 0:
        raise FileNotFoundError("case32 playback did not write its output")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
