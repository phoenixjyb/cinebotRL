#!/usr/bin/env python3
"""Seal the case-78 deterministic dynamic qualification result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


NAMESPACE = "20260721_case78_dynamic_qualification_v1_exclusive"
SOURCE_DURATION_S = 135.487646
EXECUTION_DURATION_S = 192.29956737098348
MAXIMUM_WALL_DURATION_S = 900.0


def identity(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def summarize(
    root: Path,
    *,
    runtime_commit: str,
    playback_exit_code: int,
    wall_started_epoch_s: float,
    wall_finished_epoch_s: float,
    gpu_release_passed: bool,
) -> dict[str, object]:
    admission_path = root / "admission.json"
    gate_path = root / "gates/case_0078.json"
    log_path = root / "logs/playback.log"
    admission = load_json(admission_path)
    gate = load_json(gate_path)
    results = gate.get("results")
    result = (
        results[0]
        if isinstance(results, list)
        and len(results) == 1
        and isinstance(results[0], dict)
        else {}
    )
    observed_wall_s = max(0.0, wall_finished_epoch_s - wall_started_epoch_s)
    admission_checks = {
        "runtime_authorized": admission.get("runtime_authorized") is True,
        "dynamic_authorized": admission.get(
            "dynamic_qualification_authorized"
        )
        is True,
        "runtime_commit": admission.get("runtime_commit") == runtime_commit,
        "case": admission.get("case") == 78,
        "current_split": admission.get("current_split") == "unused",
        "namespace": admission.get("namespace") == NAMESPACE,
        "split_closed": admission.get("split_change_authorized") is False,
        "learning_closed": admission.get("dataset_creation_authorized") is False
        and admission.get("dagger_authorized") is False
        and admission.get("bc_authorized") is False
        and admission.get("ppo_authorized") is False,
    }
    gate_contract_checks = {
        "single_case": gate.get("cases") == [78] and bool(result),
        "result_case": result.get("case") == 78,
        "deterministic_teacher": gate.get("trajectory_command_source")
        == "deterministic_teacher",
        "no_residual_policy": gate.get("residual_policy") is None,
        "source_clock": result.get("source_duration_s") == SOURCE_DURATION_S,
        "execution_clock": result.get("execution_duration_s")
        == EXECUTION_DURATION_S,
        "complete_reference": result.get("completed_phase_time_s")
        == EXECUTION_DURATION_S,
        "wall_clock_ordered": wall_finished_epoch_s >= wall_started_epoch_s,
        "wall_timeout_bounded": observed_wall_s <= MAXIMUM_WALL_DURATION_S,
        "gpu_released": gpu_release_passed,
    }
    physical_checks = {
        "dynamic_quality": result.get("dynamic_quality_passed") is True,
        "thermal_admission": result.get("thermal_admission_passed") is True,
        "controller_evidence": result.get("controller_evidence_passed") is True,
        "no_termination": result.get("termination") is None,
        "gate_passed": gate.get("passed") is True
        and result.get("passed") is True,
    }
    no_data_checks = {
        "runtime_no_dataset": "executed_residual_dataset" in result
        and result["executed_residual_dataset"] is None,
        "runtime_no_raw_teacher": "executed_raw_teacher_capture" in result
        and result["executed_raw_teacher_capture"] is None,
        "runtime_no_policy_trace": "executed_policy_trace" in result
        and result["executed_policy_trace"] is None,
        "runtime_no_shadow_trace": "executed_shadow_teacher_trace" in result
        and result["executed_shadow_teacher_trace"] is None,
        "top_level_no_capture": gate.get("raw_teacher_capture_started") is False
        and gate.get("normalized_dataset_capture_started") is False
        and gate.get("policy_trace_started") is False
        and gate.get("shadow_teacher_trace_started") is False,
        "top_level_training_closed": gate.get("training_started") is False
        and gate.get("dagger_authorized") is False
        and gate.get("ppo_authorized") is False,
    }
    physical_passed = all(physical_checks.values())
    dynamic_passed = physical_passed and all(gate_contract_checks.values())
    passed = (
        playback_exit_code == 0
        and all(admission_checks.values())
        and all(gate_contract_checks.values())
        and dynamic_passed
        and all(no_data_checks.values())
    )
    return {
        "schema": "cinebotrl_two_wheel_riser_case78_dynamic_final_v1",
        "runtime_commit": runtime_commit,
        "case": 78,
        "current_split": "unused",
        "namespace": NAMESPACE,
        "playback_exit_code": playback_exit_code,
        "wall_started_epoch_s": wall_started_epoch_s,
        "wall_finished_epoch_s": wall_finished_epoch_s,
        "observed_wall_duration_s": observed_wall_s,
        "maximum_wall_duration_s": MAXIMUM_WALL_DURATION_S,
        "source_duration_s": result.get("source_duration_s"),
        "execution_duration_s": result.get("execution_duration_s"),
        "admission_checks": admission_checks,
        "gate_contract_checks": gate_contract_checks,
        "physical_checks": physical_checks,
        "no_data_checks": no_data_checks,
        "physical_quality_passed": physical_passed,
        "dynamic_qualification_passed": dynamic_passed,
        "residual_label_envelope_passed": result.get(
            "residual_label_envelope_passed"
        ),
        "admission": identity(admission_path),
        "gate": identity(gate_path),
        "playback_log": identity(log_path),
        "runtime_authorized": all(admission_checks.values()),
        "runtime_started": gate_path.is_file() or playback_exit_code != 99,
        "split_changed": False,
        "case78_validation_admitted": False,
        "dataset_created": False,
        "valid_for_training": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--playback-exit-code", type=int, required=True)
    parser.add_argument("--wall-started-epoch-s", type=float, required=True)
    parser.add_argument("--wall-finished-epoch-s", type=float, required=True)
    parser.add_argument("--gpu-release-passed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.root,
        runtime_commit=args.runtime_commit,
        playback_exit_code=args.playback_exit_code,
        wall_started_epoch_s=args.wall_started_epoch_s,
        wall_finished_epoch_s=args.wall_finished_epoch_s,
        gpu_release_passed=bool(args.gpu_release_passed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
