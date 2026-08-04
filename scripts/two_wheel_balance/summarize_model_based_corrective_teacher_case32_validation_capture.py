#!/usr/bin/env python3
"""Seal one authorized case-32 natural-error validation capture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.two_wheel_balance.summarize_model_based_corrective_teacher_case8_validation_capture import (
    summarize as summarize_shared,
)
from scripts.two_wheel_balance.validate_model_based_corrective_teacher_case32_validation_capture import (
    NAMESPACE,
)

CAPTURE_NAME = "case_0032_corrective_teacher_capture_v2.npz"


def summarize(
    root: Path,
    admission_path: Path,
    *,
    runtime_commit: str,
    playback_exit_code: int,
    gpu_release_passed: bool,
) -> dict[str, object]:
    result = summarize_shared(
        root,
        admission_path,
        runtime_commit=runtime_commit,
        playback_exit_code=playback_exit_code,
        gpu_release_passed=gpu_release_passed,
        expected_case=32,
        expected_namespace=NAMESPACE,
        capture_name=CAPTURE_NAME,
        plan_identity_name="case32_plan",
        expected_perturbation_rows=0,
    )
    result["external_wrench_used"] = False
    result["conversion_authorized"] = False
    result["dataset_creation_authorized"] = False
    result["bc_authorized"] = False
    result["ppo_authorized"] = False
    result["training_started"] = False
    result["valid_for_training"] = False
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--playback-exit-code", type=int, required=True)
    parser.add_argument(
        "--gpu-release-passed",
        type=int,
        choices=(0, 1),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.root,
        args.admission,
        runtime_commit=args.runtime_commit,
        playback_exit_code=args.playback_exit_code,
        gpu_release_passed=bool(args.gpu_release_passed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
