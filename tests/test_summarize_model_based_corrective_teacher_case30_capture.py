import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from rl_platform.tasks.two_wheel_balance.riser_corrective_capture import (
    CORRECTIVE_CAPTURE_ADMISSION_SCHEMA,
    save_corrective_capture,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (
    MODEL_BASED_POLICY_RESIDUAL_SCALES,
    OBSERVATION_NAMES,
)


SCRIPT = Path(__file__).parents[1] / "scripts/two_wheel_balance/summarize_model_based_corrective_teacher_case30_capture.py"
SPEC = importlib.util.spec_from_file_location("corrective_capture_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

PLAN_SHA = "a" * 64
PROFILE_SHA = "b" * 64
PAIR_SHA = "c" * 64
COMMIT = "d" * 40


def _fixture(tmp_path: Path, *, perturbation_rows: int = 20, source_end: float = 18.0):
    root = tmp_path / MODULE.NAMESPACE
    (root / "capture").mkdir(parents=True)
    contract = {
        "namespace": MODULE.NAMESPACE,
        "reviewed_parent_commit": "e" * 40,
        "case": 30,
        "split": "train",
        "identities": {
            "case30_plan": {"sha256": PLAN_SHA},
            "corrective_profile": {"sha256": PROFILE_SHA},
            "paired_final_status": {"sha256": PAIR_SHA},
        },
    }
    contract_path = root / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    admission = {
        "schema": CORRECTIVE_CAPTURE_ADMISSION_SCHEMA,
        "passed": True,
        "reviewed_parent_commit": "e" * 40,
        "runtime_commit": COMMIT,
        "case": 30,
        "split": "train",
        "runtime_authorized": True,
        "label_capture_authorized": True,
        "corrective_target_admission_passed": True,
        "dataset_creation_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "plan_sha256": PLAN_SHA,
        "corrective_profile_sha256": PROFILE_SHA,
        "paired_final_status_sha256": PAIR_SHA,
    }
    admission_path = root / "admission.json"
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    count = 30
    normalized = np.zeros((count, 3), dtype=np.float32)
    normalized[:, 0] = np.linspace(0.0, 0.4, count)
    residual = normalized * MODEL_BASED_POLICY_RESIDUAL_SCALES
    model = np.tile([0.1, 0.0, 0.5], (count, 1))
    perturbation = np.zeros(count, dtype=bool)
    perturbation[:perturbation_rows] = True
    capture_path = root / "capture" / MODULE.CAPTURE_NAME
    save_corrective_capture(
        capture_path,
        {
            "observations": np.zeros((count, len(OBSERVATION_NAMES)), dtype=np.float32),
            "model_based_commands": model,
            "corrective_residual_commands": residual,
            "corrective_normalized_actions": normalized,
            "final_high_level_commands": model + residual,
            "case_ids": np.full(count, 30),
            "elapsed_time_s": np.arange(count) / 200.0,
            "execution_time_s": np.linspace(0.0, 29.0, count),
            "source_time_s": np.linspace(0.0, source_end, count),
            "initialization_mask": np.zeros(count, dtype=bool),
            "amplitude_limited": np.zeros((count, 3), dtype=bool),
            "slew_limited": np.ones((count, 3), dtype=bool),
            "perturbation_active": perturbation,
            "sample_plan_sha256": np.full(count, PLAN_SHA),
            "sample_runtime_commit": np.full(count, COMMIT),
        },
        source_duration_s=18.0,
        execution_duration_s=29.0,
        plan_sha256=PLAN_SHA,
        runtime_commit=COMMIT,
        corrective_profile_sha256=PROFILE_SHA,
        paired_final_status_sha256=PAIR_SHA,
    )
    result = {
        "case": 30,
        "passed": True,
        "completed_steps": count,
        "source_duration_s": 18.0,
        "execution_duration_s": 29.0,
        "dynamic_quality_passed": True,
        "thermal_admission_passed": True,
        "controller_evidence_passed": True,
        "perturbation_contract_passed": True,
        "corrective_teacher_labels_captured": True,
        "executed_corrective_teacher_capture": str(capture_path.resolve()),
        "corrective_teacher_telemetry": {"sample_count": count},
    }
    gate = {
        "cases": [30],
        "passed": True,
        "training_started": False,
        "ppo_authorized": False,
        "trajectory_command_source": "model_based_planner_plus_corrective_teacher",
        "corrective_teacher_capture_started": True,
        "corrective_teacher_label_capture_authorized": True,
        "raw_teacher_capture_started": False,
        "normalized_dataset_capture_started": False,
        "policy_trace_started": False,
        "shadow_teacher_trace_started": False,
        "results": [result],
    }
    (root / "case_0030.json").write_text(json.dumps(gate), encoding="utf-8")
    (root / "runtime_heartbeat.json").write_text(
        json.dumps({"case": 30, "completed_steps": count}), encoding="utf-8"
    )
    return root, admission_path


def test_finalizer_admits_only_archive_conversion_not_training(tmp_path) -> None:
    root, admission = _fixture(tmp_path)
    result = MODULE.summarize(
        root,
        admission,
        runtime_commit=COMMIT,
        playback_exit_code=0,
        gpu_release_passed=True,
    )
    assert result["passed"] is True
    assert result["capture_admitted_for_dataset_conversion"] is True
    assert result["capture_metrics"]["perturbation_active_rows"] == 20
    assert result["valid_for_training"] is False
    assert result["bc_authorized"] is False


def test_finalizer_rejects_wrong_perturbation_row_count(tmp_path) -> None:
    root, admission = _fixture(tmp_path, perturbation_rows=19)
    result = MODULE.summarize(
        root, admission, runtime_commit=COMMIT, playback_exit_code=0, gpu_release_passed=True
    )
    assert result["archive_checks"]["perturbation_rows"] is False
    assert result["passed"] is False


def test_finalizer_rejects_incomplete_source_clock(tmp_path) -> None:
    root, admission = _fixture(tmp_path, source_end=17.5)
    result = MODULE.summarize(
        root, admission, runtime_commit=COMMIT, playback_exit_code=0, gpu_release_passed=True
    )
    assert result["archive_checks"]["source_clock"] is False
    assert result["passed"] is False


def test_finalizer_rejects_gate_or_gpu_failure(tmp_path) -> None:
    root, admission = _fixture(tmp_path)
    result = MODULE.summarize(
        root, admission, runtime_commit=COMMIT, playback_exit_code=2, gpu_release_passed=False
    )
    assert result["gate_checks"]["exit_zero"] is False
    assert result["gate_checks"]["gpu_released"] is False
    assert result["passed"] is False


def test_finalizer_writes_fail_closed_result_when_gate_is_missing(tmp_path) -> None:
    root, admission = _fixture(tmp_path)
    (root / "case_0030.json").unlink()
    result = MODULE.summarize(
        root, admission, runtime_commit=COMMIT, playback_exit_code=1, gpu_release_passed=True
    )
    assert result["gate_checks"]["single_case"] is False
    assert result["archive_checks"]["source_clock"] is False
    assert result["passed"] is False
