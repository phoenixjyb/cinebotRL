#!/usr/bin/env python3
"""Preflight or promote an admitted corrective corpus to the BC input schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_training_dataset import (  # noqa: E402
    build_training_dataset,
    save_training_dataset,
)

DEFAULT_LOSS_MODULE = (
    PROJECT_ROOT / "src/rl_platform/tasks/two_wheel_balance/"
    "riser_model_based_bc_loss.py"
)
DEFAULT_LOSS_AUDIT = (
    PROJECT_ROOT / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_model_based_corrective_bc_loss_v1/summary.json"
)
DEFAULT_PROMOTION_MODULE = (
    PROJECT_ROOT / "src/rl_platform/tasks/two_wheel_balance/"
    "riser_model_based_corrective_training_dataset.py"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loss-module", type=Path, default=DEFAULT_LOSS_MODULE)
    parser.add_argument("--loss-audit-summary", type=Path, default=DEFAULT_LOSS_AUDIT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"projection-aware training output already exists: {args.output}"
        )
    metadata, payload = build_training_dataset(
        args.corpus,
        args.admission,
        loss_module_path=args.loss_module,
        loss_audit_summary_path=args.loss_audit_summary,
        promotion_module_path=DEFAULT_PROMOTION_MODULE,
        promotion_script_path=Path(__file__),
    )
    if args.execute:
        save_training_dataset(args.output, metadata, payload)
    result = {
        "schema": (
            "cinebotrl_two_wheel_riser_model_based_corrective_"
            "training_promotion_result_v1"
        ),
        "passed": True,
        "execute_requested": args.execute,
        "output_created": args.execute,
        "dataset_schema": metadata["schema"],
        "row_count": metadata["row_count"],
        "case_count": metadata["case_count"],
        "split_cases": metadata["split_cases"],
        "reserved_holdout_cases": metadata["reserved_holdout_cases"],
        "holdout_rows_present": False,
        "loss_contract": metadata["loss_contract"],
        "transition_contract": metadata["transition_contract"],
        "valid_for_projection_aware_bc_input": True,
        "valid_for_training": True,
        "bc_authorized": False,
        "ppo_authorized": False,
        "learned_rollout_authorized": False,
        "training_started": False,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
