#!/usr/bin/env python3
"""Preflight or build a case-disjoint model-based corrective corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_corpus import (  # noqa: E402
    build_corpus,
    save_corpus,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    metadata, payload = build_corpus(args.manifest)
    if args.output.exists():
        raise FileExistsError(f"corrective corpus already exists: {args.output}")
    if args.execute:
        save_corpus(args.output, metadata, payload)
    result = {
        "schema": "cinebotrl_two_wheel_riser_model_based_corrective_corpus_result_v1",
        "passed": True,
        "execute_requested": args.execute,
        "output_created": args.execute,
        "case_count": metadata["case_count"],
        "row_count": metadata["row_count"],
        "split_cases": metadata["split_cases"],
        "reserved_holdout_cases": metadata["reserved_holdout_cases"],
        "holdout_rows_present": False,
        "valid_for_bc_admission_review": True,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
