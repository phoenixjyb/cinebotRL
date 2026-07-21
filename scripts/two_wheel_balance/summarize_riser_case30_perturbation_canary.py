#!/usr/bin/env python3
"""Seal independent physical, perturbation, and label outcomes for case 30."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


NAMESPACE = "20260721_case30_perturbation_measurement_v1_exclusive"
EXPECTED_ACTION_SCALES = [0.35, 0.4, 0.1]
EXPECTED_PERTURBATION_PROFILE = {
    "schema": "cinebotrl_two_wheel_riser_deterministic_wrench_pulse_v1",
    "case": 30,
    "start_phase_time_s": 15.666592937559889,
    "duration_steps": 20,
    "force_body_x_n": 20.0,
    "application_height_m": 0.5,
}


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
    diagnosis_exit_code: int,
) -> dict[str, object]:
    admission_path = root / "admission.json"
    gate_path = root / "learned/case_0030.json"
    trace_path = root / "traces/case_0030_shadow_teacher_trace_v1.npz"
    diagnosis_path = root / "diagnosis/shadow_teacher_gap.json"
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
    diagnosis = load_json(diagnosis_path)
    pulse = result.get("deterministic_wrench_perturbation", {})
    admission_checks = {
        "runtime_authorized": admission.get("runtime_authorized") is True,
        "measurement_authorized": admission.get("measurement_authorized")
        is True,
        "runtime_commit": admission.get("runtime_commit") == runtime_commit,
        "case": admission.get("case") == 30,
        "split": admission.get("split") == "train",
        "namespace": admission.get("namespace") == NAMESPACE,
        "dataset_closed": admission.get("dataset_creation_authorized") is False,
        "training_closed": admission.get("dagger_authorized") is False
        and admission.get("bc_authorized") is False
        and admission.get("ppo_authorized") is False,
    }
    source_duration_s = result.get("source_duration_s")
    execution_duration_s = result.get("execution_duration_s")
    gate_contract_checks = {
        "single_case_result": bool(result),
        "case": result.get("case") == 30,
        "top_level_cases": gate.get("cases") == [30],
        "source_clock_present": isinstance(source_duration_s, (int, float))
        and source_duration_s > 0.0,
        "execution_clock_present": isinstance(execution_duration_s, (int, float))
        and execution_duration_s > 0.0,
        "clocks_are_distinct": isinstance(source_duration_s, (int, float))
        and isinstance(execution_duration_s, (int, float))
        and abs(source_duration_s - execution_duration_s) > 1e-9,
        "shadow_labels_unapplied": gate.get("shadow_teacher_labels_applied")
        is False,
        "shadow_labels_unadmitted": gate.get(
            "shadow_teacher_labels_admitted_for_training"
        )
        is False,
        "dagger_closed": gate.get("dagger_authorized") is False,
    }
    physical_checks = {
        "dynamic_quality": result.get("dynamic_quality_passed") is True,
        "thermal_admission": result.get("thermal_admission_passed") is True,
        "controller_evidence": result.get("controller_evidence_passed") is True,
        "no_termination": result.get("termination") is None,
    }
    perturbation_checks = {
        "contract_passed": result.get("perturbation_contract_passed") is True,
        "enabled": pulse.get("enabled") is True,
        "triggered": pulse.get("triggered") is True,
        "exact_duration": pulse.get("active_step_count")
        == pulse.get("expected_active_step_count")
        == 20,
        "profile_exact": pulse.get("profile") == EXPECTED_PERTURBATION_PROFILE,
        "released": pulse.get("released_after_pulse") is True,
        "planner_unmodified": result.get("perturbation_applied_to_planner_commands")
        is False,
        "policy_unmodified": result.get("perturbation_applied_to_policy_actions")
        is False,
    }
    trace_checks = {
        "present": trace_path.is_file(),
        "trace_only": False,
        "not_trainable": False,
        "row_count": False,
    }
    if trace_path.is_file():
        with np.load(trace_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            trace_checks.update({
                "schema": metadata.get("schema")
                == "cinebotrl_two_wheel_riser_shadow_teacher_trace_v1",
                "case": metadata.get("case") == 30,
                "trace_only": metadata.get("trace_only") is True,
                "not_trainable": metadata.get("valid_for_training") is False,
                "teacher_unapplied": metadata.get(
                    "shadow_teacher_applied_to_commands"
                )
                is False,
                "labels_unadmitted": metadata.get(
                    "shadow_teacher_labels_admitted_for_training"
                )
                is False,
                "action_scales": metadata.get("action_scales")
                == EXPECTED_ACTION_SCALES,
                "row_count": len(data["observations"])
                == result.get("completed_steps"),
            })
    label_checks = {
        "diagnosis_exit_zero": diagnosis_exit_code == 0,
        "diagnosis_present": diagnosis_path.is_file(),
        "case": diagnosis.get("case") == 30,
        "sample_count": diagnosis.get("sample_count")
        == result.get("completed_steps"),
        "input_contract": all(
            diagnosis.get("input_contract_checks", {}).values()
        )
        and bool(diagnosis.get("input_contract_checks")),
        "no_dataset": diagnosis.get("dataset_created") is False,
        "training_closed": diagnosis.get("dagger_authorized") is False
        and diagnosis.get("bc_authorized") is False
        and diagnosis.get("ppo_authorized") is False,
    }
    no_data_checks = {
        "runtime_no_dataset": "executed_residual_dataset" in result
        and result["executed_residual_dataset"] is None,
        "runtime_no_raw_teacher": "executed_raw_teacher_capture" in result
        and result["executed_raw_teacher_capture"] is None,
        "runtime_no_policy_trace": "executed_policy_trace" in result
        and result["executed_policy_trace"] is None,
        "shadow_trace_only": result.get("executed_shadow_teacher_trace")
        == str(trace_path.resolve()) if trace_path.is_file() else False,
    }
    physical_passed = all(physical_checks.values())
    perturbation_passed = all(perturbation_checks.values())
    label_measurement_completed = all(label_checks.values())
    measurement_passed = (
        playback_exit_code == 0
        and gate.get("passed") is True
        and all(admission_checks.values())
        and all(gate_contract_checks.values())
        and physical_passed
        and perturbation_passed
        and all(trace_checks.values())
        and label_measurement_completed
        and all(no_data_checks.values())
    )
    return {
        "schema": "cinebotrl_two_wheel_riser_case30_perturbation_final_v1",
        "runtime_commit": runtime_commit,
        "case": 30,
        "split": "train",
        "playback_exit_code": playback_exit_code,
        "diagnosis_exit_code": diagnosis_exit_code,
        "source_duration_s": source_duration_s,
        "execution_duration_s": execution_duration_s,
        "admission_checks": admission_checks,
        "gate_contract_checks": gate_contract_checks,
        "physical_checks": physical_checks,
        "perturbation_checks": perturbation_checks,
        "trace_checks": trace_checks,
        "label_checks": label_checks,
        "no_data_checks": no_data_checks,
        "physical_quality_passed": physical_passed,
        "perturbation_contract_passed": perturbation_passed,
        "label_measurement_completed": label_measurement_completed,
        "material_shadow_shift_by_channel": diagnosis.get(
            "material_shadow_shift_by_channel"
        ),
        "dagger_dataset_proposal_supported": diagnosis.get(
            "dagger_dataset_proposal_supported"
        ),
        "admission": identity(admission_path),
        "gate": identity(gate_path),
        "trace": identity(trace_path),
        "diagnosis": identity(diagnosis_path),
        "playback_log": identity(log_path),
        "runtime_authorized": all(admission_checks.values()),
        "runtime_started": gate_path.is_file() or playback_exit_code != 99,
        "dataset_created": False,
        "valid_for_training": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "passed": measurement_passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime-commit", required=True)
    parser.add_argument("--playback-exit-code", type=int, required=True)
    parser.add_argument("--diagnosis-exit-code", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        args.root,
        runtime_commit=args.runtime_commit,
        playback_exit_code=args.playback_exit_code,
        diagnosis_exit_code=args.diagnosis_exit_code,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
