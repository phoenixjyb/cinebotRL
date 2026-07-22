#!/usr/bin/env python3
"""Convert an admitted case-78 shadow series to the raw-teacher schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    load_raw_teacher_case,
    load_shadow_teacher_trace,
    save_raw_teacher_case,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-admission", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.output.with_suffix(".summary.json")
    if args.output.exists() or summary_path.exists():
        raise ValueError("refusing to overwrite case-78 raw-teacher output")
    admission = json.loads(args.label_admission.read_text(encoding="utf-8"))
    trace_sha = sha256_file(args.trace)
    checks = {
        "label_admitted": admission.get("label_admission_passed") is True,
        "conversion_authorized": admission.get("raw_teacher_conversion_authorized")
        is True,
        "case_and_split": admission.get("case") == 78
        and admission.get("split") == "validation",
        "trace_identity": admission.get("inputs", {}).get("trace", {}).get("sha256")
        == trace_sha,
        "learning_closed": admission.get("bc_authorized") is False
        and admission.get("ppo_authorized") is False
        and admission.get("training_started") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"case-78 raw-teacher conversion rejected: {checks}")
    _, trace = load_shadow_teacher_trace(args.trace)
    payload = {
        "observations": np.asarray(trace["observations"]).copy(),
        "raw_residual_commands": np.asarray(
            trace["shadow_teacher_raw_residual_commands"]
        ).copy(),
        "case_ids": np.asarray(trace["case_ids"]).copy(),
        "elapsed_time_s": np.asarray(trace["elapsed_time_s"]).copy(),
        "phase_time_s": np.asarray(trace["phase_time_s"]).copy(),
        "baseline_wheel_actions": np.asarray(
            trace["baseline_wheel_actions"]
        ).copy(),
        "teacher_commands": np.asarray(
            trace["shadow_teacher_high_level_commands"]
        ).copy(),
    }
    save_raw_teacher_case(args.output, 78, payload)
    metadata, restored = load_raw_teacher_case(args.output)
    if len(restored["observations"]) != admission.get("row_count"):
        args.output.unlink(missing_ok=True)
        raise ValueError("case-78 raw-teacher row count mismatch")
    summary = {
        "schema": "cinebotrl_two_wheel_riser_case78_raw_teacher_conversion_v1",
        "case": 78,
        "split": "validation",
        "row_count": len(restored["observations"]),
        "raw_teacher_schema": metadata["schema"],
        "raw_teacher": str(args.output.resolve()),
        "raw_teacher_sha256": sha256_file(args.output),
        "source_trace": str(args.trace.resolve()),
        "source_trace_sha256": trace_sha,
        "label_admission": str(args.label_admission.resolve()),
        "label_admission_sha256": sha256_file(args.label_admission),
        "historical_dataset_modified": False,
        "labels_applied_to_commands": False,
        "raw_teacher_conversion_passed": True,
        "offline_dataset_rebuild_authorized": admission.get(
            "offline_dataset_rebuild_authorized"
        )
        is True,
        "dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
        "input_contract_checks": checks,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
