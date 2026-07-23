import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rl_platform.tasks.two_wheel_balance.riser_corrective_capture import (
    CORRECTIVE_CAPTURE_ADMISSION_SCHEMA,
    save_corrective_capture,
)
from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_dataset import (
    convert_admitted_capture,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (
    MODEL_BASED_POLICY_RESIDUAL_SCALES,
    OBSERVATION_NAMES,
    PREVIOUS_ACTION_INDICES,
)


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/two_wheel_balance/summarize_model_based_corrective_teacher_case23_capture.py"
SPEC = importlib.util.spec_from_file_location("case23_capture_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

PLAN_SHA = "a" * 64
PROFILE_SHA = "b" * 64
PAIR_SHA = "c" * 64
COMMIT = "d" * 40


def _real_case23_fixture(tmp_path: Path, *, capture_case: int = 23):
    root = tmp_path / MODULE.NAMESPACE
    (root / "capture").mkdir(parents=True)
    contract = {
        "namespace": MODULE.NAMESPACE,
        "reviewed_parent_commit": "e" * 40,
        "case": 23,
        "split": "train",
        "identities": {
            "case23_plan": {"sha256": PLAN_SHA},
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
        "case": 23,
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
    requested = np.zeros((count, 3), dtype=np.float32)
    requested[:, 0] = np.linspace(0.0, 0.4, count)
    effective = requested.copy()
    effective[5:8, 0] -= 0.02
    requested_residual = requested * MODEL_BASED_POLICY_RESIDUAL_SCALES
    effective_residual = effective * MODEL_BASED_POLICY_RESIDUAL_SCALES
    delta = effective_residual - requested_residual
    model = np.tile([0.1, 0.0, 0.5], (count, 1))
    capture_path = root / "capture" / MODULE.CAPTURE_NAME
    save_corrective_capture(
        capture_path,
        {
            "observations": np.zeros(
                (count, len(OBSERVATION_NAMES)), dtype=np.float32
            ),
            "model_based_commands": model,
            "requested_corrective_residual_commands": requested_residual,
            "requested_corrective_normalized_actions": requested,
            "effective_corrective_residual_commands": effective_residual,
            "effective_corrective_normalized_actions": effective,
            "requested_vs_effective_residual_delta": delta,
            "command_clipped": np.abs(delta) > 2e-7,
            "final_high_level_commands": model + effective_residual,
            "case_ids": np.full(count, capture_case),
            "elapsed_time_s": np.arange(count) / 200.0,
            "execution_time_s": np.linspace(0.0, 29.0, count),
            "source_time_s": np.linspace(0.0, 18.0, count),
            "initialization_mask": np.zeros(count, dtype=bool),
            "amplitude_limited": np.zeros((count, 3), dtype=bool),
            "slew_limited": np.ones((count, 3), dtype=bool),
            "perturbation_active": np.arange(count) < 20,
            "sample_plan_sha256": np.full(count, PLAN_SHA),
            "sample_runtime_commit": np.full(count, COMMIT),
        },
        source_duration_s=18.0,
        execution_duration_s=29.0,
        plan_sha256=PLAN_SHA,
        runtime_commit=COMMIT,
        corrective_profile_sha256=PROFILE_SHA,
        paired_final_status_sha256=PAIR_SHA,
        case=capture_case,
        split="train",
    )
    result = {
        "case": 23,
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
        "cases": [23],
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
    (root / "case_0023.json").write_text(json.dumps(gate), encoding="utf-8")
    (root / "runtime_heartbeat.json").write_text(
        json.dumps(
            {"case": 23, "completed_steps": count, "capture_outputs_enabled": True}
        ),
        encoding="utf-8",
    )
    return root, admission_path, capture_path


def test_case23_finalizer_passes_exact_archive_contract(monkeypatch, tmp_path) -> None:
    observed = {}

    def fake_summarize(root, admission_path, **kwargs):
        observed.update(kwargs)
        return {"passed": True}

    monkeypatch.setattr(MODULE, "summarize_capture", fake_summarize)
    result = MODULE.summarize(
        tmp_path,
        tmp_path / "admission.json",
        runtime_commit="a" * 40,
        playback_exit_code=0,
        gpu_release_passed=True,
    )
    assert result["passed"] is True
    assert observed["expected_case"] == 23
    assert observed["expected_namespace"] == MODULE.NAMESPACE
    assert observed["capture_name"] == MODULE.CAPTURE_NAME
    assert observed["plan_identity_name"] == "case23_plan"
    assert observed["runtime_commit"] == "a" * 40
    assert observed["playback_exit_code"] == 0
    assert observed["gpu_release_passed"] is True


def test_real_case23_archive_finalizes_and_converts_only_on_explicit_route(
    tmp_path,
) -> None:
    root, admission, capture = _real_case23_fixture(tmp_path)
    result = MODULE.summarize(
        root,
        admission,
        runtime_commit=COMMIT,
        playback_exit_code=0,
        gpu_release_passed=True,
    )
    assert result["passed"] is True
    assert result["case"] == 23
    assert result["capture_admitted_for_dataset_conversion"] is True
    assert result["valid_for_training"] is False

    final_status = root / "final_status.json"
    final_status.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="unreviewed case"):
        convert_admitted_capture(capture, final_status)
    metadata, payload = convert_admitted_capture(
        capture,
        final_status,
        expected_case=23,
        expected_split="train",
    )
    assert metadata["case"] == 23
    assert metadata["requested_actions_used_as_training_targets"] is False
    assert metadata["effective_actions_used_as_training_targets"] is True
    assert metadata["valid_for_training"] is False
    np.testing.assert_allclose(
        payload["observations"][1:, PREVIOUS_ACTION_INDICES],
        payload["actions"][:-1],
    )


def test_case23_finalizer_rejects_case30_labeled_capture(tmp_path) -> None:
    root, admission, _ = _real_case23_fixture(tmp_path, capture_case=30)
    result = MODULE.summarize(
        root,
        admission,
        runtime_commit=COMMIT,
        playback_exit_code=0,
        gpu_release_passed=True,
    )
    assert result["passed"] is False
    assert result["archive_checks"]["loaded"] is False
    assert "unreviewed case" in result["capture_error"]
