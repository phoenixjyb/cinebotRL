#!/usr/bin/env python3
"""Monitor shared Windows headroom and stop one playback on hard pressure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.two_wheel_balance.check_windows_shared_resource_admission import (
    RUNTIME_MINIMUM_GPU_FREE_MEMORY_MIB,
    RUNTIME_MINIMUM_WINDOWS_FREE_MEMORY_GIB,
    evaluate_snapshot,
    probe_live_snapshot,
)


SCHEMA = "cinebotrl_windows_shared_resource_monitor_v1"


def _pid_running(pid: int) -> bool:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        fields = stat_path.read_text(encoding="utf-8").split()
    except OSError:
        return False
    return len(fields) > 2 and fields[2] != "Z"


def build_report(
    *,
    monitored_pid: int,
    samples: list[dict[str, Any]],
    termination_requested: bool,
    process_exit_observed: bool,
) -> dict[str, Any]:
    valid_samples = [
        sample for sample in samples if isinstance(sample, dict)
    ]
    passed = bool(
        valid_samples
        and process_exit_observed
        and not termination_requested
        and all(sample.get("passed") is True for sample in valid_samples)
    )
    windows_free = [
        sample.get("observed", {}).get("windows_free_memory_gib")
        for sample in valid_samples
        if isinstance(sample.get("observed"), dict)
    ]
    gpu_free = [
        sample.get("observed", {}).get("gpu_free_memory_mib")
        for sample in valid_samples
        if isinstance(sample.get("observed"), dict)
    ]
    windows_free = [
        float(value)
        for value in windows_free
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    gpu_free = [
        int(value)
        for value in gpu_free
        if isinstance(value, int) and not isinstance(value, bool)
    ]
    return {
        "schema": SCHEMA,
        "monitored_pid": monitored_pid,
        "runtime_thresholds": {
            "minimum_windows_free_memory_gib": (
                RUNTIME_MINIMUM_WINDOWS_FREE_MEMORY_GIB
            ),
            "minimum_gpu_free_memory_mib": (
                RUNTIME_MINIMUM_GPU_FREE_MEMORY_MIB
            ),
        },
        "sample_count": len(valid_samples),
        "minimum_observed_windows_free_memory_gib": (
            min(windows_free) if windows_free else None
        ),
        "minimum_observed_gpu_free_memory_mib": (
            min(gpu_free) if gpu_free else None
        ),
        "termination_requested": termination_requested,
        "process_exit_observed": process_exit_observed,
        "samples": valid_samples,
        "passed": passed,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def monitor(pid: int, output: Path, interval_s: float) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    termination_requested = False
    process_exit_observed = False

    while _pid_running(pid):
        try:
            sample = evaluate_snapshot(
                probe_live_snapshot(),
                phase="runtime",
            )
        except Exception as error:  # The monitor itself must fail closed.
            sample = {
                "schema": "cinebotrl_windows_shared_resource_admission_v2",
                "phase": "runtime",
                "error": f"{type(error).__name__}: {error}",
                "runtime_started": True,
                "authorization_consumed": True,
                "passed": False,
            }
        sample["sample_index"] = len(samples)
        sample["sampled_at_utc"] = datetime.now(timezone.utc).isoformat()
        samples.append(sample)
        if sample.get("passed") is not True:
            termination_requested = True
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            break
        _write_report(
            output,
            build_report(
                monitored_pid=pid,
                samples=samples,
                termination_requested=False,
                process_exit_observed=False,
            ),
        )
        time.sleep(interval_s)

    if not termination_requested:
        process_exit_observed = not _pid_running(pid)
    report = build_report(
        monitored_pid=pid,
        samples=samples,
        termination_requested=termination_requested,
        process_exit_observed=process_exit_observed,
    )
    _write_report(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, default=5.0)
    args = parser.parse_args()
    if args.pid <= 0:
        parser.error("--pid must be positive")
    if args.interval_s <= 0:
        parser.error("--interval-s must be positive")
    report = monitor(args.pid, args.output, args.interval_s)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
