#!/usr/bin/env python3
"""Audit physical gates and raw residual labels before dataset admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.two_wheel_balance.relabel_riser_residual_cases import (  # noqa: E402
    load_source,
    parse_scales,
    raw_residual,
    sha256,
)


TRACKING_PROFILE = "riser_phase_consistent_v2"
PHASE_CONTRACT = "derivatives_scaled_by_progress_v1"
DYNAMIC_METRICS = (
    "position_error_p95_m",
    "position_error_max_m",
    "attitude_error_p95_deg",
    "attitude_error_max_deg",
    "pitch_max_deg",
    "riser_servo_error_p95_m",
    "proxy_servo_error_p95_deg",
)


def recommended_scales(
    raw_abs_max: np.ndarray,
    minimum_scales: np.ndarray,
    quantums: np.ndarray,
    margin: float,
) -> np.ndarray:
    if margin <= 1.0 or np.any(quantums <= 0.0):
        raise ValueError("margin must exceed one and scale quantums must be positive")
    required = raw_abs_max * margin
    quantized = np.ceil(required / quantums - 1e-12) * quantums
    return np.maximum(minimum_scales, quantized)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument(
        "--minimum-scales",
        type=parse_scales,
        default=parse_scales("0.25,0.4,0.1"),
    )
    parser.add_argument(
        "--scale-quantums",
        type=parse_scales,
        default=parse_scales("0.05,0.05,0.05"),
    )
    parser.add_argument("--scale-margin", type=float, default=1.10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
        raise ValueError("source commit must be a full lowercase Git SHA-1")

    gate_paths = sorted(args.gate_dir.glob("case_*.json"))
    case_paths = sorted(args.case_dir.glob("case_*_executed_residual_v1.npz"))
    if len(gate_paths) != args.expected_count or len(case_paths) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} gates/cases, found "
            f"{len(gate_paths)}/{len(case_paths)}"
        )
    gates = {}
    for path in gate_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if len(payload.get("results", [])) != 1:
            raise ValueError(f"invalid gate result count: {path}")
        result = payload["results"][0]
        case = int(result["case"])
        valid = (
            payload.get("passed") is True
            and result.get("passed") is True
            and payload.get("tracking_profile") == TRACKING_PROFILE
            and payload.get("phase_feedforward_contract") == PHASE_CONTRACT
            and result.get("termination") is None
        )
        if not valid or case in gates:
            raise ValueError(f"invalid or duplicate physical gate: {path}")
        gates[case] = (path, result)

    rows = []
    residual_chunks = []
    for path in case_paths:
        metadata, payload = load_source(path)
        case = int(metadata["case"])
        if case not in gates:
            raise ValueError(f"case {case} has no matching physical gate")
        source_scales = np.asarray(metadata["action_scales"], dtype=np.float64)
        residual = raw_residual(payload)
        residual_chunks.append(residual)
        stored_reconstructed = np.column_stack(
            (
                payload["observations"][:, 18]
                + source_scales[0] * payload["actions"][:, 0],
                payload["observations"][:, 19]
                + source_scales[1] * payload["actions"][:, 1],
                payload["observations"][:, 15]
                + source_scales[2] * payload["actions"][:, 2],
            )
        )
        reconstruction_error = float(
            np.max(np.abs(stored_reconstructed - payload["teacher_commands"]))
        )
        clip_counts = np.sum(np.abs(payload["actions"]) >= 1.0 - 1e-6, axis=0)
        gate_path, result = gates[case]
        rows.append(
            {
                "case": case,
                "gate_file": gate_path.name,
                "gate_sha256": sha256(gate_path),
                "source_file": path.name,
                "source_sha256": sha256(path),
                "row_count": len(payload["observations"]),
                "raw_residual_abs_max": np.max(np.abs(residual), axis=0).tolist(),
                "stored_action_clip_counts": clip_counts.tolist(),
                "stored_reconstruction_max_error": reconstruction_error,
                **{metric: result[metric] for metric in DYNAMIC_METRICS},
            }
        )
    rows.sort(key=lambda row: row["case"])
    expected_cases = list(range(1, args.expected_count + 1))
    if [row["case"] for row in rows] != expected_cases:
        raise ValueError("capture cases are not contiguous from one")

    all_residuals = np.concatenate(residual_chunks, axis=0)
    raw_abs_max = np.max(np.abs(all_residuals), axis=0)
    scales = recommended_scales(
        raw_abs_max,
        args.minimum_scales,
        args.scale_quantums,
        args.scale_margin,
    )
    stored_clip_counts = np.sum(
        np.asarray([row["stored_action_clip_counts"] for row in rows]), axis=0
    )
    stored_reconstruction_error = max(
        row["stored_reconstruction_max_error"] for row in rows
    )
    summary = {
        "schema": "cinebotrl_two_wheel_riser_physical_capture_audit_v1",
        "source_commit": args.source_commit,
        "plan_manifest": str(args.plan_manifest.resolve()),
        "plan_manifest_sha256": sha256(args.plan_manifest),
        "case_count": len(rows),
        "total_rows": sum(row["row_count"] for row in rows),
        "total_steps": sum(
            json.loads(
                (args.gate_dir / row["gate_file"]).read_text(encoding="utf-8")
            )["results"][0]["completed_steps"]
            for row in rows
        ),
        "physical_capture_passed": len(rows) == args.expected_count,
        "raw_residual_abs_max": raw_abs_max.tolist(),
        "raw_residual_signed_min": np.min(all_residuals, axis=0).tolist(),
        "raw_residual_signed_max": np.max(all_residuals, axis=0).tolist(),
        "raw_residual_abs_percentiles": {
            str(percentile): np.percentile(
                np.abs(all_residuals), percentile, axis=0
            ).tolist()
            for percentile in (50, 90, 95, 99)
        },
        "source_action_clip_counts": stored_clip_counts.tolist(),
        "source_reconstruction_max_error": stored_reconstruction_error,
        "source_labels_admissible": bool(
            np.all(stored_clip_counts == 0) and stored_reconstruction_error <= 2e-6
        ),
        "relabel_required": bool(
            np.any(stored_clip_counts != 0) or stored_reconstruction_error > 2e-6
        ),
        "recommended_action_scales": scales.tolist(),
        "scale_margin": args.scale_margin,
        "scale_quantums": args.scale_quantums.tolist(),
        "worst_dynamic_metrics": {
            metric: max(row[metric] for row in rows) for metric in DYNAMIC_METRICS
        },
        "rows": rows,
        "training_authorized": False,
        "ppo_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
