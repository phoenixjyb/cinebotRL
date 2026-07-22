#!/usr/bin/env python3
"""Audit the failed teacher-41 case-78 learned rollout without running Isaac."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Iterable


EXPECTED_HASHES = {
    "learned": "570546d39b4267d20c6c203a1eb1a3a04a7544cb2887270c9d99a6b952ad9c41",
    "final": "c14467a93622a887b861b7a8628ec1ff13b9cdf0d1a2380336b8ee9963172f1b",
    "teacher": "ad0dc3ee618819ec808ac4d0318bda711dc2cba38dd041119a1f78584e97e459",
    "policy_report": "b7915caddea9467847430a247924eae2e856ad486da06135e1b8f543c42b891a",
    "playback": "ffe45cd5747f6e628caebafbc405d589f34df764df944ddc2b36a2efd0926b1d",
    "residual_contract": "98b2e0bcd5341a58fb34ef829c1934543f9bb0ac81d8a9c085ca8cf7b6d153f2",
}
ACTION_SCALES = (0.35, 0.4)
POSITION_P95_GATE_M = 0.15


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def identity(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"missing case-78 audit input: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _single_result(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", [])
    return results[0] if isinstance(results, list) and len(results) == 1 else {}


def _inferred_action(row: dict[str, Any]) -> tuple[float, float]:
    return (
        (row["vx_reference_mps"] - row["phase_feedforward_v_mps"])
        / ACTION_SCALES[0],
        (row["wz_reference_rad_s"] - row["phase_feedforward_wz_rad_s"])
        / ACTION_SCALES[1],
    )


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return statistics.mean(rows) if rows else 0.0


def _nearest_index(times: list[float], value: float) -> int:
    index = bisect.bisect_left(times, value)
    candidates = (max(0, index - 1), min(len(times) - 1, index))
    return min(candidates, key=lambda candidate: abs(times[candidate] - value))


def _contiguous_groups(indices: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    for index in indices:
        if not groups or index > groups[-1][-1] + 1:
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


def audit(
    learned_payload: dict[str, Any],
    teacher_payload: dict[str, Any],
    final: dict[str, Any],
    policy_report: dict[str, Any],
    *,
    playback_source: str,
    residual_contract_source: str,
) -> dict[str, Any]:
    learned = _single_result(learned_payload)
    teacher = _single_result(teacher_payload)
    learned_trace = learned.get("trace", [])
    teacher_trace = teacher.get("trace", [])
    teacher_times = [row["phase_time_s"] for row in teacher_trace]
    learned_checks = learned.get("checks", {})
    false_dynamic_checks = [name for name, passed in learned_checks.items() if not passed]
    high_indices = [
        index
        for index, row in enumerate(learned_trace)
        if row.get("position_error_m", 0.0) > POSITION_P95_GATE_M
    ]
    groups = _contiguous_groups(high_indices)

    paired: list[tuple[tuple[float, float], tuple[float, float]]] = []
    paired_high: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for index, learned_row in enumerate(learned_trace):
        teacher_row = teacher_trace[
            _nearest_index(teacher_times, learned_row["phase_time_s"])
        ]
        pair = (_inferred_action(learned_row), _inferred_action(teacher_row))
        paired.append(pair)
        if index in high_indices:
            paired_high.append(pair)

    def action_summary(
        rows: list[tuple[tuple[float, float], tuple[float, float]]]
    ) -> dict[str, Any]:
        channels = []
        for channel, name in enumerate(("vx", "wz")):
            learned_values = [row[0][channel] for row in rows]
            teacher_values = [row[1][channel] for row in rows]
            differences = [
                learned_value - teacher_value
                for learned_value, teacher_value in zip(
                    learned_values, teacher_values, strict=True
                )
            ]
            channels.append(
                {
                    "channel": name,
                    "learned_mean": _mean(learned_values),
                    "teacher_mean": _mean(teacher_values),
                    "signed_bias": _mean(differences),
                    "mean_absolute_error": _mean(abs(value) for value in differences),
                    "sign_mismatch_ratio": _mean(
                        (learned_value > 0.0) != (teacher_value > 0.0)
                        for learned_value, teacher_value in zip(
                            learned_values, teacher_values, strict=True
                        )
                    ),
                }
            )
        return {"sample_count": len(rows), "channels": channels}

    interval_rows = []
    for group in groups:
        pairs = [paired[index] for index in group]
        interval_rows.append(
            {
                "trace_index_start": group[0],
                "trace_index_end": group[-1],
                "phase_time_start_s": learned_trace[group[0]]["phase_time_s"],
                "phase_time_end_s": learned_trace[group[-1]]["phase_time_s"],
                "learned_peak_position_error_m": max(
                    learned_trace[index]["position_error_m"] for index in group
                ),
                "learned_progress_scale_mean": _mean(
                    learned_trace[index]["progress_scale"] for index in group
                ),
                "teacher_progress_scale_mean": _mean(
                    teacher_trace[
                        _nearest_index(
                            teacher_times, learned_trace[index]["phase_time_s"]
                        )
                    ]["progress_scale"]
                    for index in group
                ),
                "action_delta": action_summary(pairs),
            }
        )

    architecture_checks = {
        "teacher_labels_are_planner_minus_phase_feedforward": (
            "commanded_vx_m_s - feedforward_vx_m_s" in residual_contract_source
            and "commanded_wz_rad_s - feedforward_wz_rad_s"
            in residual_contract_source
        ),
        "learned_action_is_applied_over_phase_feedforward": (
            "apply_residual_action(\n                phase_feedforward_v_mps,"
            in playback_source
        ),
        "deterministic_tracking_command_is_computed_before_policy": (
            playback_source.index("raw_residual_command = build_raw_residual_command(")
            < playback_source.index("policy_output = residual_policy(")
        ),
    }
    checks = {
        "case78_completed_reference": learned.get("completed_phase_time_s")
        == learned.get("execution_duration_s"),
        "case78_only_failed_dynamic_check_is_position_p95": false_dynamic_checks
        == ["position_p95_bounded"],
        "case78_position_p95_failed": learned.get("position_error_p95_m", 0.0)
        > POSITION_P95_GATE_M,
        "case78_position_max_passed": learned.get("position_error_max_m", 1.0)
        <= 0.25,
        "case78_no_termination_or_saturation": learned.get("termination") is None
        and learned.get("action_saturation_ratio") == 0.0
        and learned.get("riser_saturation_ratio") == 0.0
        and learned.get("proxy_saturation_ratio") == 0.0,
        "case78_final_fail_closed": final.get("passed") is False
        and final.get("remaining_validation_cases_authorized") is False
        and final.get("dataset_created") is False
        and final.get("bc_authorized") is False
        and final.get("ppo_authorized") is False,
        "teacher_reference_passed": teacher_payload.get("passed") is True
        and teacher.get("passed") is True
        and teacher.get("position_error_p95_m", 1.0) <= POSITION_P95_GATE_M,
        "policy_is_masked_offline_bc": policy_report.get("training_method")
        == "offline_behavior_cloning"
        and policy_report.get("policy_architecture")
        == "state_shared_lookahead_fusion_previous_action_masked_v1"
        and policy_report.get("masked_observation_indices") == [23, 24, 25],
        "architecture_evidence_complete": all(architecture_checks.values()),
        "high_error_intervals_observed": bool(interval_rows),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"case-78 failure audit input contract failed: {checks}")

    return {
        "schema": "cinebotrl_two_wheel_riser_initial_teacher41_case78_failure_audit_v1",
        "case": 78,
        "checks": checks,
        "failed_dynamic_gate": "position_p95_bounded",
        "metrics": {
            "learned_position_p95_m": learned["position_error_p95_m"],
            "teacher_position_p95_m": teacher["position_error_p95_m"],
            "position_p95_gate_m": POSITION_P95_GATE_M,
            "learned_position_max_m": learned["position_error_max_m"],
            "teacher_position_max_m": teacher["position_error_max_m"],
            "learned_completed_steps": learned["completed_steps"],
            "teacher_completed_steps": teacher["completed_steps"],
            "learned_total_simulated_duration_s": learned["total_simulated_duration_s"],
            "teacher_total_simulated_duration_s": teacher["total_simulated_duration_s"],
            "learned_progress_scale_mean": learned["progress_scale_mean"],
            "teacher_progress_scale_mean": teacher["progress_scale_mean"],
            "learned_progress_hold_step_count": learned["progress_hold_step_count"],
            "teacher_progress_hold_step_count": teacher["progress_hold_step_count"],
        },
        "high_error_trace_intervals": interval_rows,
        "inferred_action_comparison": {
            "all_trace_samples": action_summary(paired),
            "high_error_trace_samples": action_summary(paired_high),
            "inference_contract": (
                "normalized_action=(applied_command-phase_feedforward)/action_scale"
            ),
        },
        "architecture_audit": {
            "checks": architecture_checks,
            "current_bc_target": "deterministic_planner_command_minus_phase_feedforward",
            "current_learned_deployment": "phase_feedforward_plus_bc_prediction",
            "required_final_contract": "model_based_planner_command_plus_bounded_policy_residual",
            "required_contract_satisfied": False,
            "checkpoint_classification": "planner_imitation_bc_initialization_only",
        },
        "diagnosis": (
            "Small BC command errors reduce progress through several camera-error "
            "transients, increasing their time-weighted occupancy until position p95 "
            "fails. The current deployment also substitutes BC reconstruction for the "
            "model-based planner command instead of adding a bounded correction above it."
        ),
        "decision": {
            "case78_dynamic_admitted": False,
            "case16_22_32_tranche_authorized": False,
            "threshold_relaxation_authorized": False,
            "zero_baseline_rerun_required": False,
            "capture_authorized": False,
            "bc_retraining_authorized": False,
            "ppo_authorized": False,
            "next_task": (
                "define_and_test_model_based_plus_zero_initialized_residual_contract_cpu_only"
            ),
        },
        "passed": True,
        "valid_for_training": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in EXPECTED_HASHES:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite case-78 failure audit: {args.output}")
    paths = {name: getattr(args, name) for name in EXPECTED_HASHES}
    identities = {name: identity(path) for name, path in paths.items()}
    if any(
        identities[name]["sha256"] != expected
        for name, expected in EXPECTED_HASHES.items()
    ):
        raise ValueError("case-78 failure audit input hash mismatch")
    result = audit(
        load_json(paths["learned"]),
        load_json(paths["teacher"]),
        load_json(paths["final"]),
        load_json(paths["policy_report"]),
        playback_source=paths["playback"].read_text(encoding="utf-8"),
        residual_contract_source=paths["residual_contract"].read_text(
            encoding="utf-8"
        ),
    )
    result["inputs"] = identities
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
