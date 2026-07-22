#!/usr/bin/env python3
"""Finalize the model-based zero-residual case-8 preservation canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.two_wheel_balance.validate_model_based_zero_residual_case8_contract import (
        ADMISSION_SCHEMA,
        EXPECTED_CONTROLLER,
        EXPECTED_PLAN,
        EXPECTED_PRESERVATION,
        NAMESPACE,
        ZERO_POLICY_TORCHSCRIPT_SHA256,
    )
except ModuleNotFoundError:  # Direct script execution from the repository root.
    from validate_model_based_zero_residual_case8_contract import (
        ADMISSION_SCHEMA,
        EXPECTED_CONTROLLER,
        EXPECTED_PLAN,
        EXPECTED_PRESERVATION,
        NAMESPACE,
        ZERO_POLICY_TORCHSCRIPT_SHA256,
    )


SCHEMA = "cinebotrl_two_wheel_riser_model_based_zero_residual_case8_final_v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path) if exists else None,
        "exists": exists,
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _single_result(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", [])
    return results[0] if isinstance(results, list) and len(results) == 1 else {}


def summarize(
    *,
    explicit_zero: dict[str, Any],
    zero_checkpoint: dict[str, Any],
    cpu_admission: dict[str, Any],
    explicit_zero_exit_code: int,
    zero_checkpoint_exit_code: int,
    policy_sha256: str,
    dataset_files_present: bool,
) -> dict[str, Any]:
    baseline = _single_result(explicit_zero)
    candidate = _single_result(zero_checkpoint)
    position_metrics = ("position_error_p95_m", "position_error_max_m")
    attitude_metrics = ("attitude_error_p95_deg", "attitude_error_max_deg")
    pitch_metrics = ("pitch_p95_deg", "pitch_max_deg")
    riser_metrics = ("riser_servo_error_p95_m", "riser_servo_error_max_m")
    proxy_metrics = ("proxy_servo_error_p95_deg", "proxy_servo_error_max_deg")

    def maximum_delta(names: tuple[str, ...]) -> float:
        try:
            return max(abs(float(baseline[name]) - float(candidate[name])) for name in names)
        except (KeyError, TypeError, ValueError):
            return float("inf")

    deltas = {
        "position_m": maximum_delta(position_metrics),
        "attitude_deg": maximum_delta(attitude_metrics),
        "pitch_deg": maximum_delta(pitch_metrics),
        "riser_m": maximum_delta(riser_metrics),
        "proxy_deg": maximum_delta(proxy_metrics),
    }
    result_contract = {
        "case": 8,
        "source_duration_s": EXPECTED_PLAN["source_duration_s"],
        "execution_duration_s": EXPECTED_PLAN["execution_duration_s"],
        "policy_command_base": EXPECTED_CONTROLLER["policy_command_base"],
        "policy_residual_contract": EXPECTED_CONTROLLER["policy_residual_contract"],
        "residual_action_scales": EXPECTED_CONTROLLER["residual_action_scales"],
    }

    def result_matches(payload: dict[str, Any], item: dict[str, Any]) -> bool:
        return (
            payload.get("cases") == [8]
            and item.get("case") == 8
            and item.get("source_duration_s") == result_contract["source_duration_s"]
            and item.get("execution_duration_s")
            == result_contract["execution_duration_s"]
            and abs(
                float(item.get("completed_phase_time_s", -1.0))
                - result_contract["execution_duration_s"]
            )
            <= 1e-6
            and payload.get("policy_command_base")
            == result_contract["policy_command_base"]
            and payload.get("policy_residual_contract")
            == result_contract["policy_residual_contract"]
            and payload.get("residual_action_scales")
            == result_contract["residual_action_scales"]
            and payload.get("dynamic_quality_passed") is True
            and payload.get("passed") is True
            and item.get("dynamic_quality_passed") is True
            and item.get("passed") is True
            and item.get("residual_action_abs_max") == [0.0, 0.0, 0.0]
            and item.get("executed_residual_dataset") is None
            and payload.get("raw_teacher_capture_started") is False
            and payload.get("normalized_dataset_capture_started") is False
            and payload.get("policy_trace_started") is False
            and payload.get("shadow_teacher_trace_started") is False
        )

    checks = {
        "cpu_admission_exact": cpu_admission.get("schema") == ADMISSION_SCHEMA
        and cpu_admission.get("passed") is True
        and cpu_admission.get("cpu_contract_ready") is True
        and cpu_admission.get("namespace") == NAMESPACE
        and cpu_admission.get("runtime_authorized") is False
        and cpu_admission.get("gpu_launch_authorized") is False
        and cpu_admission.get("dynamic_canary_authorized") is False
        and cpu_admission.get("dataset_creation_authorized") is False
        and cpu_admission.get("bc_authorized") is False
        and cpu_admission.get("ppo_authorized") is False
        and cpu_admission.get("holdout_opened") is False,
        "rollout_exit_codes_zero": explicit_zero_exit_code == 0
        and zero_checkpoint_exit_code == 0,
        "explicit_zero_source_exact": explicit_zero.get("trajectory_command_source")
        == "model_based_planner_plus_zero_policy_residual"
        and explicit_zero.get("residual_policy") is None,
        "zero_checkpoint_source_exact": zero_checkpoint.get(
            "trajectory_command_source"
        )
        == "model_based_planner_plus_torchscript_residual"
        and bool(zero_checkpoint.get("residual_policy")),
        "explicit_zero_result_passes": result_matches(explicit_zero, baseline),
        "zero_checkpoint_result_passes": result_matches(
            zero_checkpoint, candidate
        ),
        "zero_checkpoint_identity_exact": policy_sha256
        == ZERO_POLICY_TORCHSCRIPT_SHA256,
        "position_metrics_preserved": deltas["position_m"]
        <= EXPECTED_PRESERVATION["maximum_position_metric_delta_m"],
        "attitude_metrics_preserved": deltas["attitude_deg"]
        <= EXPECTED_PRESERVATION["maximum_attitude_metric_delta_deg"],
        "pitch_metrics_preserved": deltas["pitch_deg"]
        <= EXPECTED_PRESERVATION["maximum_pitch_metric_delta_deg"],
        "riser_metrics_preserved": deltas["riser_m"]
        <= EXPECTED_PRESERVATION["maximum_riser_metric_delta_m"],
        "proxy_metrics_preserved": deltas["proxy_deg"]
        <= EXPECTED_PRESERVATION["maximum_proxy_metric_delta_deg"],
        "dataset_absent": not dataset_files_present,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = all(checks.values())
    return {
        "schema": SCHEMA,
        "namespace": NAMESPACE,
        "case": 8,
        "split": "validation",
        "checks": checks,
        "metric_absolute_deltas": deltas,
        "explicit_zero_exit_code": explicit_zero_exit_code,
        "zero_checkpoint_exit_code": zero_checkpoint_exit_code,
        "zero_checkpoint_sha256": policy_sha256,
        "zero_residual_preservation_passed": passed,
        "case78_authorized": False,
        "runtime_authorized": False,
        "dataset_creation_authorized": False,
        "training_authorized": False,
        "training_started": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "holdout_opened": False,
        "valid_for_training": False,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-admission", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--explicit-zero-exit-code", type=int, required=True)
    parser.add_argument("--zero-checkpoint-exit-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite final status: {args.output}")
    explicit_path = args.root / "explicit_zero/case_0008.json"
    checkpoint_path = args.root / "zero_checkpoint/case_0008.json"
    payload = summarize(
        explicit_zero=load_json(explicit_path) if explicit_path.is_file() else {},
        zero_checkpoint=(
            load_json(checkpoint_path) if checkpoint_path.is_file() else {}
        ),
        cpu_admission=load_json(args.cpu_admission),
        explicit_zero_exit_code=args.explicit_zero_exit_code,
        zero_checkpoint_exit_code=args.zero_checkpoint_exit_code,
        policy_sha256=sha256_file(args.policy),
        dataset_files_present=any(args.root.rglob("*.npz")),
    )
    payload["explicit_zero"] = identity(explicit_path)
    payload["zero_checkpoint"] = identity(checkpoint_path)
    payload["cpu_admission"] = identity(args.cpu_admission)
    payload["policy"] = identity(args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
