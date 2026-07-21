import json
from pathlib import Path

import numpy as np

from scripts.two_wheel_balance.summarize_riser_case30_perturbation_canary import summarize


def test_summary_keeps_three_outcomes_independent(tmp_path: Path) -> None:
    (tmp_path / "learned").mkdir()
    (tmp_path / "traces").mkdir()
    (tmp_path / "diagnosis").mkdir()
    (tmp_path / "logs").mkdir()
    runtime_commit = "a" * 40
    (tmp_path / "admission.json").write_text(json.dumps({
        "runtime_authorized": True,
        "measurement_authorized": True,
        "runtime_commit": runtime_commit,
        "case": 30,
        "split": "train",
        "namespace": "20260721_case30_perturbation_measurement_v2_exclusive",
        "dataset_creation_authorized": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
    }))
    result = {
        "case": 30,
        "completed_steps": 2,
        "source_duration_s": 4.0,
        "execution_duration_s": 20.0,
        "dynamic_quality_passed": True,
        "thermal_admission_passed": True,
        "controller_evidence_passed": True,
        "termination": None,
        "perturbation_contract_passed": True,
        "deterministic_wrench_perturbation": {
            "enabled": True,
            "triggered": True,
            "active_step_count": 20,
            "expected_active_step_count": 20,
            "released_after_pulse": True,
            "profile": {
                "schema": (
                    "cinebotrl_two_wheel_riser_deterministic_wrench_pulse_v1"
                ),
                "case": 30,
                "start_phase_time_s": 15.666592937559889,
                "duration_steps": 20,
                "force_body_x_n": 20.0,
                "application_height_m": 0.5,
            },
        },
        "perturbation_applied_to_planner_commands": False,
        "perturbation_applied_to_policy_actions": False,
        "executed_residual_dataset": None,
        "executed_raw_teacher_capture": None,
        "executed_policy_trace": None,
        "executed_shadow_teacher_trace": str(
            (tmp_path / "traces/case_0030_shadow_teacher_trace_v1.npz").resolve()
        ),
    }
    (tmp_path / "learned/case_0030.json").write_text(
        json.dumps({
            "passed": True,
            "cases": [30],
            "shadow_teacher_labels_applied": False,
            "shadow_teacher_labels_admitted_for_training": False,
            "dagger_authorized": False,
            "results": [result],
        })
    )
    metadata = {
        "schema": "cinebotrl_two_wheel_riser_shadow_teacher_trace_v1",
        "case": 30,
        "trace_only": True,
        "valid_for_training": False,
        "shadow_teacher_applied_to_commands": False,
        "shadow_teacher_labels_admitted_for_training": False,
        "action_scales": [0.35, 0.4, 0.1],
    }
    np.savez_compressed(
        tmp_path / "traces/case_0030_shadow_teacher_trace_v1.npz",
        metadata_json=np.array(json.dumps(metadata)),
        observations=np.zeros((2, 3)),
    )
    (tmp_path / "diagnosis/shadow_teacher_gap.json").write_text(json.dumps({
        "case": 30,
        "sample_count": 2,
        "input_contract_checks": {"schema": True, "trace_only": True},
        "dataset_created": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "material_shadow_shift_by_channel": [True, False, False],
        "dagger_dataset_proposal_supported": False,
    }))
    (tmp_path / "logs/playback.log").write_text("ok")
    final = summarize(
        tmp_path,
        runtime_commit=runtime_commit,
        playback_exit_code=0,
        diagnosis_exit_code=0,
    )
    assert final["physical_quality_passed"]
    assert final["perturbation_contract_passed"]
    assert final["label_measurement_completed"]
    assert final["material_shadow_shift_by_channel"] == [True, False, False]
    assert final["dagger_dataset_proposal_supported"] is False
    assert final["passed"]
    assert final["valid_for_training"] is False


def test_physical_failure_does_not_change_label_result(tmp_path: Path) -> None:
    (tmp_path / "learned").mkdir(parents=True)
    (tmp_path / "diagnosis").mkdir()
    (tmp_path / "learned/case_0030.json").write_text(json.dumps({
        "passed": False,
        "results": [{
            "dynamic_quality_passed": False,
            "thermal_admission_passed": True,
            "controller_evidence_passed": True,
            "termination": None,
            "perturbation_contract_passed": False,
        }],
    }))
    (tmp_path / "diagnosis/shadow_teacher_gap.json").write_text(json.dumps({
        "case": 30,
        "dataset_created": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "material_shadow_shift_by_channel": [True, True, False],
    }))
    final = summarize(
        tmp_path,
        runtime_commit="b" * 40,
        playback_exit_code=2,
        diagnosis_exit_code=0,
    )
    assert not final["physical_quality_passed"]
    assert final["material_shadow_shift_by_channel"] == [True, True, False]
    assert not final["passed"]


def test_missing_dataset_fields_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "learned").mkdir(parents=True)
    (tmp_path / "learned/case_0030.json").write_text(json.dumps({
        "passed": True,
        "results": [{
            "case": 30,
            "dynamic_quality_passed": True,
            "thermal_admission_passed": True,
            "controller_evidence_passed": True,
            "termination": None,
        }],
    }))
    final = summarize(
        tmp_path,
        runtime_commit="c" * 40,
        playback_exit_code=0,
        diagnosis_exit_code=0,
    )
    assert not final["no_data_checks"]["runtime_no_dataset"]
    assert not final["no_data_checks"]["runtime_no_raw_teacher"]
    assert not final["no_data_checks"]["runtime_no_policy_trace"]
    assert not final["passed"]
