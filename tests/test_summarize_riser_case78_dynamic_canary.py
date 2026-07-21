import json
from pathlib import Path

from scripts.two_wheel_balance.summarize_riser_case78_dynamic_canary import (
    summarize,
)


def _write_healthy_artifacts(root: Path, runtime_commit: str) -> None:
    (root / "gates").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "admission.json").write_text(json.dumps({
        "runtime_authorized": True,
        "dynamic_qualification_authorized": True,
        "runtime_commit": runtime_commit,
        "case": 78,
        "current_split": "unused",
        "namespace": "20260721_case78_dynamic_qualification_v1_exclusive",
        "split_change_authorized": False,
        "dataset_creation_authorized": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
    }))
    result = {
        "case": 78,
        "source_duration_s": 135.487646,
        "execution_duration_s": 192.29956737098348,
        "completed_phase_time_s": 192.29956737098348,
        "dynamic_quality_passed": True,
        "thermal_admission_passed": True,
        "controller_evidence_passed": True,
        "termination": None,
        "executed_residual_dataset": None,
        "executed_raw_teacher_capture": None,
        "executed_policy_trace": None,
        "executed_shadow_teacher_trace": None,
        "residual_label_envelope_passed": True,
        "passed": True,
    }
    (root / "gates/case_0078.json").write_text(json.dumps({
        "cases": [78],
        "trajectory_command_source": "deterministic_teacher",
        "residual_policy": None,
        "raw_teacher_capture_started": False,
        "normalized_dataset_capture_started": False,
        "policy_trace_started": False,
        "shadow_teacher_trace_started": False,
        "training_started": False,
        "dagger_authorized": False,
        "ppo_authorized": False,
        "results": [result],
        "passed": True,
    }))
    (root / "logs/playback.log").write_text("ok")


def test_summary_passes_without_applying_split(tmp_path: Path) -> None:
    runtime_commit = "a" * 40
    _write_healthy_artifacts(tmp_path, runtime_commit)
    final = summarize(
        tmp_path,
        runtime_commit=runtime_commit,
        playback_exit_code=0,
        wall_started_epoch_s=100.0,
        wall_finished_epoch_s=500.0,
        gpu_release_passed=True,
    )
    assert final["physical_quality_passed"]
    assert final["dynamic_qualification_passed"]
    assert final["passed"]
    assert not final["split_changed"]
    assert not final["case78_validation_admitted"]
    assert not final["dataset_created"]


def test_missing_dataset_fields_fail_closed(tmp_path: Path) -> None:
    runtime_commit = "b" * 40
    _write_healthy_artifacts(tmp_path, runtime_commit)
    gate_path = tmp_path / "gates/case_0078.json"
    gate = json.loads(gate_path.read_text())
    del gate["results"][0]["executed_residual_dataset"]
    gate_path.write_text(json.dumps(gate))
    final = summarize(
        tmp_path,
        runtime_commit=runtime_commit,
        playback_exit_code=0,
        wall_started_epoch_s=100.0,
        wall_finished_epoch_s=500.0,
        gpu_release_passed=True,
    )
    assert not final["no_data_checks"]["runtime_no_dataset"]
    assert not final["passed"]


def test_timeout_or_gpu_leak_fails_independently(tmp_path: Path) -> None:
    runtime_commit = "c" * 40
    _write_healthy_artifacts(tmp_path, runtime_commit)
    final = summarize(
        tmp_path,
        runtime_commit=runtime_commit,
        playback_exit_code=0,
        wall_started_epoch_s=100.0,
        wall_finished_epoch_s=1001.0,
        gpu_release_passed=False,
    )
    assert final["physical_quality_passed"]
    assert not final["dynamic_qualification_passed"]
    assert not final["gate_contract_checks"]["wall_timeout_bounded"]
    assert not final["gate_contract_checks"]["gpu_released"]
    assert not final["passed"]
