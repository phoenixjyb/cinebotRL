#!/usr/bin/env python3
"""Validate or convert one admitted corrective capture into a closed case dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_dataset import (  # noqa: E402
    convert_admitted_capture,
    save_case_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--final-status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write the converted case dataset after the fail-closed preflight.",
    )
    args = parser.parse_args()

    metadata, payload = convert_admitted_capture(args.capture, args.final_status)
    if args.output.exists():
        raise FileExistsError(f"conversion output already exists: {args.output}")
    if args.execute:
        save_case_dataset(args.output, metadata, payload)
    result = {
        "schema": "cinebotrl_two_wheel_riser_corrective_conversion_result_v1",
        "passed": True,
        "execute_requested": args.execute,
        "output_created": args.execute,
        "output": str(args.output.resolve()),
        "case": metadata["case"],
        "split": metadata["split"],
        "sample_count": metadata["sample_count"],
        "source_capture_sha256": metadata["source_capture_sha256"],
        "source_final_status_sha256": metadata["source_final_status_sha256"],
        "training_target_contract": metadata["training_target_contract"],
        "previous_action_contract": metadata["previous_action_contract"],
        "requested_actions_used_as_training_targets": False,
        "effective_actions_used_as_training_targets": True,
        "valid_for_case_merge": True,
        "merged_dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
