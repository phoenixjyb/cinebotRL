#!/usr/bin/env python3
"""Build a CPU-only proposal for a paired corrective-teacher canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_corrective_teacher import (  # noqa: E402
    CORRECTIVE_TARGET_ADMISSION_CONTRACT,
    CorrectiveTeacherConfig,
)


HOLDOUT_CASES = [3, 5, 13, 19, 24]


def _identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing proposal input: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_proposal(
    case8: dict[str, object],
    case78: dict[str, object],
    case30: dict[str, object],
    split_summary: dict[str, object],
) -> dict[str, object]:
    split_cases = split_summary.get("split_cases", {})
    checks = {
        "case8_zero_residual_preserved": case8.get("case") == 8
        and case8.get("zero_residual_preservation_passed") is True,
        "case78_zero_residual_preserved": case78.get("case") == 78
        and case78.get("zero_residual_preservation_passed") is True,
        "case30_prior_measurement_safe": case30.get("case") == 30
        and case30.get("physical_quality_passed") is True,
        "old_shadow_labels_not_promoted": case30.get(
            "dagger_dataset_proposal_supported"
        )
        is False
        and case30.get("dataset_created") is False
        and case30.get("valid_for_training") is False,
        "case30_is_training_split": 30 in split_cases.get("train", []),
        "validation_cases_excluded": 8 in split_cases.get("validation", [])
        and 78 in split_cases.get("validation", []),
        "holdout_unchanged": split_cases.get("holdout") == HOLDOUT_CASES,
        "source_dataset_not_rewritten": split_summary.get(
            "base_dataset_rewrite_performed"
        )
        is False,
        "learning_closed": case8.get("training_started") is False
        and case78.get("training_started") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"corrective-teacher proposal inputs failed: {checks}")
    config = CorrectiveTeacherConfig()
    return {
        "schema": "cinebotrl_two_wheel_riser_corrective_teacher_proposal_v1",
        "candidate_case": 30,
        "candidate_split": "train",
        "teacher": config.as_dict(),
        "paired_admission_contract": CORRECTIVE_TARGET_ADMISSION_CONTRACT,
        "experiment": {
            "baseline": "complete_model_based_planner_plus_exact_zero_residual",
            "candidate": "complete_model_based_planner_plus_corrective_teacher",
            "same_plan_seed_physics_and_perturbation_required": True,
            "baseline_must_run_first": True,
            "candidate_may_run_only_after_baseline_passes": True,
            "label_capture_during_pair": False,
            "minimum_position_p95_improvement_m": 0.003,
            "minimum_position_p95_relative_improvement": 0.02,
        },
        "input_contract_checks": checks,
        "holdout_cases": HOLDOUT_CASES,
        "validation_cases_opened": [],
        "runtime_authorized": False,
        "authorization_token_issued": False,
        "namespace_created": False,
        "label_capture_authorized": False,
        "dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case8-final", type=Path, required=True)
    parser.add_argument("--case78-final", type=Path, required=True)
    parser.add_argument("--case30-final", type=Path, required=True)
    parser.add_argument("--split-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite proposal: {args.output}")
    inputs = {
        "case8_final": args.case8_final,
        "case78_final": args.case78_final,
        "case30_final": args.case30_final,
        "split_summary": args.split_summary,
    }
    result = build_proposal(
        _load(args.case8_final),
        _load(args.case78_final),
        _load(args.case30_final),
        _load(args.split_summary),
    )
    result["inputs"] = {name: _identity(path) for name, path in inputs.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
