#!/usr/bin/env python3
"""Derive a conservative playback wall bound from sealed runtime evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


POLICY_HZ = 200
CANONICAL_SHA256 = {
    "reference_admission": (
        "b01478172476c6e0b598a2102e721ba745e6ecb2293fe5f0f37d416f74e15382"
    ),
    "reference_gate": (
        "5aa269f310399328654c2d37cb235e2b6196cc7edf46785dac8b3c57b3d272cf"
    ),
    "reference_final_status": (
        "fc08b890ec0dfee0a1f0d05505afd806f68c54ad564afc99fe5e73533e3ebfb6"
    ),
    "case78_timeout_final_status": (
        "94fe6c39ae9550802979ddd68fcbf45ef3d1964600db8afa01a69ddd804defb8"
    ),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "mtime_epoch_s": path.stat().st_mtime,
    }


def audit(
    admission_path: Path,
    gate_path: Path,
    final_status_path: Path,
    timeout_final_status_path: Path,
    *,
    maximum_steps: int,
    margin_s: float,
    rounding_s: int,
    enforce_canonical_hashes: bool = False,
) -> dict[str, object]:
    gate = load_json(gate_path)
    final_status = load_json(final_status_path)
    timeout_final = load_json(timeout_final_status_path)
    results = gate.get("results")
    result = (
        results[0]
        if isinstance(results, list)
        and len(results) == 1
        and isinstance(results[0], dict)
        else {}
    )
    completed_steps = int(result.get("completed_steps", 0))
    virtual_step_duration_s = completed_steps / POLICY_HZ
    host_envelope_s = (
        final_status_path.stat().st_mtime - admission_path.stat().st_mtime
    )
    conservative_step_rate_hz = (
        completed_steps / host_envelope_s if host_envelope_s > 0.0 else 0.0
    )
    estimated_maximum_loop_wall_s = (
        maximum_steps / conservative_step_rate_hz
        if conservative_step_rate_hz > 0.0
        else math.inf
    )
    unrounded_bound_s = estimated_maximum_loop_wall_s + margin_s
    proposed_bound_s = (
        math.ceil(unrounded_bound_s / rounding_s) * rounding_s
        if math.isfinite(unrounded_bound_s)
        else None
    )
    evidence = {
        "reference_admission": identity(admission_path),
        "reference_gate": identity(gate_path),
        "reference_final_status": identity(final_status_path),
        "case78_timeout_final_status": identity(timeout_final_status_path),
    }
    checks = {
        "canonical_evidence_hashes": not enforce_canonical_hashes
        or all(
            evidence[name]["sha256"] == expected
            for name, expected in CANONICAL_SHA256.items()
        ),
        "reference_case_passed": result.get("case") == 30
        and result.get("passed") is True
        and final_status.get("passed") is True,
        "reference_step_count_positive": completed_steps > 0,
        "reported_wall_is_virtual_step_time": math.isclose(
            float(result.get("wall_duration_s", math.nan)),
            virtual_step_duration_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        "host_envelope_positive": host_envelope_s > 0.0,
        "case78_timeout_is_exact_failure": timeout_final.get(
            "playback_exit_code"
        )
        == 124
        and timeout_final.get("dynamic_qualification_passed") is False
        and timeout_final.get("case78_validation_admitted") is False,
        "bound_is_finite": proposed_bound_s is not None,
        "bound_exceeds_failed_timeout": proposed_bound_s is not None
        and proposed_bound_s
        > float(timeout_final.get("maximum_wall_duration_s", math.inf)),
    }
    return {
        "schema": "cinebotrl_two_wheel_riser_runtime_wall_bound_audit_v1",
        "reference_case": 30,
        "target_case": 78,
        "policy_hz": POLICY_HZ,
        "reference_completed_steps": completed_steps,
        "reference_reported_wall_duration_s": result.get("wall_duration_s"),
        "reference_virtual_step_duration_s": virtual_step_duration_s,
        "reference_host_filesystem_envelope_s": host_envelope_s,
        "conservative_policy_step_rate_hz": conservative_step_rate_hz,
        "target_maximum_steps": maximum_steps,
        "estimated_maximum_loop_wall_s": estimated_maximum_loop_wall_s,
        "startup_shutdown_and_diagnosis_margin_s": margin_s,
        "rounding_quantum_s": rounding_s,
        "proposed_maximum_wall_duration_s": proposed_bound_s,
        "evidence": evidence,
        "checks": checks,
        "audit_passed": all(checks.values()),
        "runtime_retry_authorized": False,
        "gpu_launch_authorized": False,
        "split_change_authorized": False,
        "dataset_creation_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "valid_for_training": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-admission", type=Path, required=True)
    parser.add_argument("--reference-gate", type=Path, required=True)
    parser.add_argument("--reference-final-status", type=Path, required=True)
    parser.add_argument("--timeout-final-status", type=Path, required=True)
    parser.add_argument("--maximum-steps", type=int, required=True)
    parser.add_argument("--margin-s", type=float, default=900.0)
    parser.add_argument("--rounding-s", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.maximum_steps <= 0 or args.margin_s <= 0.0 or args.rounding_s <= 0:
        parser.error("maximum steps, margin, and rounding quantum must be positive")
    result = audit(
        args.reference_admission,
        args.reference_gate,
        args.reference_final_status,
        args.timeout_final_status,
        maximum_steps=args.maximum_steps,
        margin_s=args.margin_s,
        rounding_s=args.rounding_s,
        enforce_canonical_hashes=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
