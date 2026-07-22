import hashlib
import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/two_wheel_balance/summarize_model_based_corrective_teacher_case23_pair.py"
)
SPEC = importlib.util.spec_from_file_location("case23_pair_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _gate(*, candidate: bool, position_p95: float = 0.004) -> dict[str, object]:
    perturbation = {
        "enabled": True,
        "profile": {"case": 23, "force_body_x_n": 20.0, "duration_steps": 20},
        "trigger_step": 100,
        "active_step_count": 20,
        "expected_active_step_count": 20,
        "released_after_pulse": True,
        "triggered": True,
    }
    result = {
        "case": 23,
        "source_duration_s": 9.929694,
        "execution_duration_s": 9.929693999999989,
        "dynamic_quality_passed": True,
        "position_error_p95_m": position_p95 if candidate else 0.0107,
        "position_error_max_m": 0.025 if candidate else 0.030,
        "attitude_error_max_deg": 0.22,
        "pitch_max_deg": 7.0,
        "riser_servo_error_max_m": 0.014,
        "action_saturation_ratio": 0.0,
        "residual_action_abs_max": [0.5, 0.4, 0.3] if candidate else [0.0, 0.0, 0.0],
        "corrective_teacher_telemetry": {
            "normalized_action_abs_max": [0.5, 0.4, 0.3]
        },
        "corrective_teacher_labels_captured": False,
        "deterministic_wrench_perturbation": perturbation,
        "perturbation_contract_passed": True,
        "executed_residual_dataset": None,
        "executed_raw_teacher_capture": None,
        "executed_policy_trace": None,
        "executed_shadow_teacher_trace": None,
    }
    return {
        "cases": [23],
        "passed": True,
        "trajectory_command_source": MODULE.EXPECTED_SOURCES[
            "candidate" if candidate else "baseline"
        ],
        "policy_command_base": "model_based_planner",
        "residual_action_scales": MODULE.EXPECTED_SCALES,
        "corrective_teacher_enabled": candidate,
        "corrective_teacher_profile": {"sha256": "c" * 64} if candidate else None,
        "raw_teacher_capture_started": False,
        "normalized_dataset_capture_started": False,
        "policy_trace_started": False,
        "shadow_teacher_trace_started": False,
        "results": [result],
    }


def _fixture(tmp_path: Path, *, candidate_p95: float = 0.004):
    root = tmp_path / MODULE.NAMESPACE
    (root / "baseline").mkdir(parents=True)
    (root / "candidate").mkdir()
    contract = {
        "namespace": MODULE.NAMESPACE,
        "case": 23,
        "split": "train",
        "controller_arguments": {"reset_seed": 20260739},
        "identities": {"case23_plan": {"sha256": "a" * 64}},
    }
    contract_path = root / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    admission = {
        "runtime_commit": "b" * 40,
        "passed": True,
        "runtime_authorized": True,
        "label_capture_authorized": False,
        "dataset_creation_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    }
    admission_path = root / "admission.json"
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    (root / "baseline/case_0023.json").write_text(
        json.dumps(_gate(candidate=False)), encoding="utf-8"
    )
    (root / "candidate/case_0023.json").write_text(
        json.dumps(_gate(candidate=True, position_p95=candidate_p95)),
        encoding="utf-8",
    )
    for name in ("baseline", "candidate"):
        (root / name / "runtime_heartbeat.json").write_text(
            json.dumps({"case": 23, "completed_steps": 100}), encoding="utf-8"
        )
    return root, admission_path


def test_summary_admits_only_safe_measurable_pair_improvement(tmp_path) -> None:
    root, admission = _fixture(tmp_path)
    result = MODULE.summarize(
        root,
        admission,
        runtime_commit="b" * 40,
        baseline_exit_code=0,
        candidate_exit_code=0,
        gpu_release_passed=True,
    )
    assert result["passed"] is True
    assert result["corrective_target_admission_passed"] is True
    assert result["label_capture_authorized"] is False
    assert result["valid_for_training"] is False


def test_summary_rejects_weak_improvement(tmp_path) -> None:
    root, admission = _fixture(tmp_path, candidate_p95=0.0106)
    result = MODULE.summarize(
        root,
        admission,
        runtime_commit="b" * 40,
        baseline_exit_code=0,
        candidate_exit_code=0,
        gpu_release_passed=True,
    )
    assert result["corrective_target_admission_passed"] is False
    assert result["passed"] is False


def test_summary_rejects_any_capture_path(tmp_path) -> None:
    root, admission = _fixture(tmp_path)
    path = root / "candidate/case_0023.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["policy_trace_started"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = MODULE.summarize(
        root,
        admission,
        runtime_commit="b" * 40,
        baseline_exit_code=0,
        candidate_exit_code=0,
        gpu_release_passed=True,
    )
    assert result["rollout_checks"]["candidate_capture_closed"] is False
    assert result["passed"] is False
