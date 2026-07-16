#!/usr/bin/env python3
"""Add deterministic execution-schedule metadata to an exact-source candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.exact_source_reference import (  # noqa: E402
    validate_exact_source_candidate,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(
    candidate_path: Path,
    result_path: Path,
    summary_path: Path | None = None,
) -> str:
    candidate_path = candidate_path.resolve()
    result_path = result_path.resolve()
    with np.load(candidate_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    if str(arrays.get("schema", "").item()) != "cinebotrl_two_wheel_exact_source_retarget_v1":
        raise ValueError("refusing to seal a non-exact-source candidate")
    execution_time = np.asarray(arrays["execution_time_s"], dtype=np.float64)
    mapping = np.asarray(arrays["source_anchor_execution_index"], dtype=np.int64)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    route = str(result.get("base_acquisition_route", ""))
    route_yaw_deg = float(result.get("base_acquisition_total_yaw_travel_deg", -1.0))
    if route not in {"forward", "reverse", "rotate_in_place"}:
        raise ValueError("result lacks a supported acquisition route")
    if not np.isfinite(route_yaw_deg) or not 0.0 <= route_yaw_deg <= 2.0 * 180.0:
        raise ValueError("result has invalid acquisition yaw travel")
    arrays.update(
        {
            "execution_transition_dt_s": np.diff(execution_time),
            "source_anchor_execution_time_s": execution_time[mapping],
            "source_interval_execution_step_count": np.diff(mapping),
            "source_interval_execution_duration_s": np.diff(execution_time[mapping]),
            "acquisition_route_contract": np.asarray(
                "minimum_total_yaw_forward_or_reverse_v1"
            ),
            "base_acquisition_route": np.asarray(route),
            "base_acquisition_total_yaw_travel_deg": np.float64(route_yaw_deg),
            "execution_schedule_metadata_sealed": np.bool_(True),
        }
    )
    temporary = candidate_path.with_suffix(".sealed.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, candidate_path)
    validate_exact_source_candidate(
        candidate_path,
        require_offline_quality=False,
        require_dynamic_approval=False,
    )
    digest = sha256(candidate_path)
    result["candidate_sha256"] = digest
    result["execution_plan_sha256"] = digest
    result["execution_schedule_metadata_sealed"] = True
    result_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    if summary_path is not None:
        summary_path = summary_path.resolve()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in summary.get("results", []):
            if int(row.get("case", -1)) == int(result.get("case", -2)):
                row["candidate_sha256"] = digest
                row["execution_schedule_metadata_sealed"] = True
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    print(seal(args.candidate, args.result, args.summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
