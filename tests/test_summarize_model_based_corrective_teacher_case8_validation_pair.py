import hashlib
import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/two_wheel_balance/"
    "summarize_model_based_corrective_teacher_case8_validation_pair.py"
)
SPEC = importlib.util.spec_from_file_location("case8_validation_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


CORRECTIVE_SHA = "c" * 64
PERTURBATION_SHA = "d" * 64


def _gate(*, candidate: bool, position_p95: float = 0.120):
    perturbation = {
        "enabled": True,
        "profile": {
            "case": 8,
            "force_body_x_n": 18.0,
            "duration_steps": 20,
        },
        "trigger_step": 100,
        "active_step_count": 20,
        "expected_active_step_count": 20,
        "released_after_pulse": True,
        "triggered": True,
    }
    result = {
        "case": 8,
        "source_duration_s": 12.940941,
        "execution_duration_s": 18.1173174,
        "dynamic_quality_passed": True,
        "position_error_p95_m": position_p95 if candidate else 0.131254,
        "position_error_max_m": 0.140 if candidate else 0.143331,
        "attitude_error_max_deg": 0.45,
        "pitch_max_deg": 6.2,
        "riser_servo_error_max_m": 0.012,
        "action_saturation_ratio": 0.0,
        "residual_action_abs_max": (
            [0.30, 0.16, 0.05] if candidate else [0.0, 0.0, 0.0]
        ),
        "corrective_teacher_telemetry": {
            "normalized_action_abs_max": [0.30, 0.16, 0.05]
        },
        "corrective_teacher_labels_captured": False,
        "deterministic_wrench_perturbation": perturbation,
        "perturbation_contract_passed": True,
        "executed_residual_dataset": None,
        "executed_raw_teacher_capture": None,
        "executed_policy_trace": None,
        "executed_shadow_teacher_trace": None,
        "executed_corrective_teacher_capture": None,
    }
    return {
        "cases": [8],
        "passed": True,
        "trajectory_command_source": MODULE.EXPECTED_SOURCES[
            "candidate" if candidate else "baseline"
        ],
        "policy_command_base": "model_based_planner",
        "residual_action_scales": MODULE.EXPECTED_SCALES,
        "corrective_teacher_enabled": candidate,
        "corrective_teacher_profile": (
            {"sha256": CORRECTIVE_SHA} if candidate else None
        ),
        "deterministic_wrench_profile": {"sha256": PERTURBATION_SHA},
        "corrective_teacher_capture_started": False,
        "corrective_teacher_label_capture_authorized": False,
        "raw_teacher_capture_started": False,
        "normalized_dataset_capture_started": False,
        "policy_trace_started": False,
        "shadow_teacher_trace_started": False,
        "results": [result],
    }


def _fixture(tmp_path: Path, *, candidate_p95: float = 0.120):
    root = tmp_path / MODULE.NAMESPACE
    (root / "baseline").mkdir(parents=True)
    (root / "candidate").mkdir()
    contract = {
        "namespace": MODULE.NAMESPACE,
        "case": 8,
        "split": "validation",
        "controller_arguments": {"reset_seed": 20260724},
        "identities": {
            "case8_plan": {"sha256": "a" * 64},
            "corrective_profile": {"sha256": CORRECTIVE_SHA},
            "perturbation_profile": {"sha256": PERTURBATION_SHA},
        },
    }
    contract_path = root / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    admission = {
        "runtime_commit": "b" * 40,
        "passed": True,
        "runtime_authorized": True,
        "gpu_launch_authorized": True,
        "label_capture_authorized": False,
        "dataset_creation_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "contract_sha256": hashlib.sha256(
            contract_path.read_bytes()
        ).hexdigest(),
    }
    admission_path = root / "admission.json"
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    resource_admission = {
        "schema": "cinebotrl_windows_shared_resource_admission_v2",
        "phase": "launch",
        "thresholds": {
            "minimum_windows_free_memory_gib": 5.0,
            "minimum_gpu_free_memory_mib": 9216,
            "cad_coexistence_allowed": True,
        },
        "observed": {
            "windows_free_memory_gib": 12.0,
            "gpu_free_memory_mib": 12000,
        },
        "checks": {
            "windows_memory_probe_valid": True,
            "windows_free_memory_sufficient": True,
            "cad_process_probe_valid": True,
            "cad_coexistence_allowed": True,
            "gpu_memory_probe_valid": True,
            "gpu_free_memory_sufficient": True,
        },
        "passed": True,
    }
    (root / "resource_admission.json").write_text(
        json.dumps(resource_admission), encoding="utf-8"
    )
    resource_monitor = {
        "schema": "cinebotrl_windows_shared_resource_monitor_v1",
        "runtime_thresholds": {
            "minimum_windows_free_memory_gib": 1.5,
            "minimum_gpu_free_memory_mib": 2048,
        },
        "sample_count": 2,
        "minimum_observed_windows_free_memory_gib": 7.0,
        "minimum_observed_gpu_free_memory_mib": 9000,
        "termination_requested": False,
        "process_exit_observed": True,
        "passed": True,
    }
    (root / "baseline/case_0008.json").write_text(
        json.dumps(_gate(candidate=False)), encoding="utf-8"
    )
    (root / "candidate/case_0008.json").write_text(
        json.dumps(_gate(candidate=True, position_p95=candidate_p95)),
        encoding="utf-8",
    )
    for name in ("baseline", "candidate"):
        (root / name / "runtime_heartbeat.json").write_text(
            json.dumps({"case": 8, "completed_steps": 100}),
            encoding="utf-8",
        )
        (root / name / "resource_monitor.json").write_text(
            json.dumps(resource_monitor), encoding="utf-8"
        )
    return root, admission_path


def _summarize(root: Path, admission: Path, **kwargs):
    return MODULE.summarize(
        root,
        admission,
        runtime_commit="b" * 40,
        baseline_exit_code=kwargs.get("baseline_exit_code", 0),
        candidate_exit_code=kwargs.get("candidate_exit_code", 0),
        gpu_release_passed=kwargs.get("gpu_release_passed", True),
    )


def test_summary_accepts_closed_validation_pair(tmp_path) -> None:
    root, admission = _fixture(tmp_path)
    result = _summarize(root, admission)
    assert result["passed"] is True
    assert result["case"] == 8
    assert result["split"] == "validation"
    assert result["dynamic_pair_completed"] is True
    assert result["validation_pair_passed"] is True
    assert result["teacher_admission_opened"] is False
    assert result["label_capture_authorized"] is False
    assert result["dataset_created"] is False
    assert result["training_started"] is False


def test_summary_rejects_insufficient_improvement(tmp_path) -> None:
    root, admission = _fixture(tmp_path, candidate_p95=0.130)
    result = _summarize(root, admission)
    assert result["validation_pair_passed"] is False
    assert result["passed"] is False


def test_summary_rejects_capture_or_gpu_release_drift(tmp_path) -> None:
    root, admission = _fixture(tmp_path)
    payload_path = root / "candidate/case_0008.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["corrective_teacher_capture_started"] = True
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    result = _summarize(root, admission, gpu_release_passed=False)
    assert result["rollout_checks"]["candidate_capture_closed"] is False
    assert result["rollout_checks"]["gpu_released"] is False
    assert result["passed"] is False


def test_summary_rejects_missing_or_failed_resource_evidence(
    tmp_path: Path,
) -> None:
    root, admission = _fixture(tmp_path)
    (root / "baseline/resource_monitor.json").unlink()
    candidate_monitor = root / "candidate/resource_monitor.json"
    payload = json.loads(candidate_monitor.read_text(encoding="utf-8"))
    payload["minimum_observed_windows_free_memory_gib"] = 1.0
    candidate_monitor.write_text(json.dumps(payload), encoding="utf-8")
    result = _summarize(root, admission)
    assert result["rollout_checks"]["baseline_resource_monitor"] is False
    assert result["rollout_checks"]["candidate_resource_monitor"] is False
    assert result["passed"] is False


def test_summary_rejects_wrong_split(tmp_path) -> None:
    root, admission = _fixture(tmp_path)
    contract_path = root / "contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["split"] = "train"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    result = _summarize(root, admission)
    assert result["contract_checks"]["case_split"] is False
    assert result["passed"] is False
