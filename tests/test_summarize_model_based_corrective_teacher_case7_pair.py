import hashlib
import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/two_wheel_balance/summarize_model_based_corrective_teacher_case7_pair.py"
)
SPEC = importlib.util.spec_from_file_location("case7_pair_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


CORRECTIVE_SHA = "c" * 64
PERTURBATION_SHA = "d" * 64


def _gate(
    *,
    candidate: bool,
    position_p95: float = 0.120,
    source_duration_s: float = 12.940941,
) -> dict[str, object]:
    perturbation = {
        "enabled": True,
        "profile": {
            "case": 7,
            "force_body_x_n": 20.0,
            "duration_steps": 20,
        },
        "trigger_step": 100,
        "active_step_count": 20,
        "expected_active_step_count": 20,
        "released_after_pulse": True,
        "triggered": True,
    }
    result = {
        "case": 7,
        "source_duration_s": source_duration_s,
        "execution_duration_s": 18.1173174,
        "dynamic_quality_passed": True,
        "position_error_p95_m": position_p95 if candidate else 0.130904,
        "position_error_max_m": 0.140 if candidate else 0.142948,
        "attitude_error_max_deg": 0.45,
        "pitch_max_deg": 6.3,
        "riser_servo_error_max_m": 0.012,
        "action_saturation_ratio": 0.0,
        "residual_action_abs_max": (
            [0.38, 0.21, 0.07] if candidate else [0.0, 0.0, 0.0]
        ),
        "corrective_teacher_telemetry": {
            "normalized_action_abs_max": [0.38, 0.21, 0.07]
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
        "cases": [7],
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
        "case": 7,
        "split": "train",
        "controller_arguments": {"reset_seed": 20260723},
        "identities": {
            "case7_plan": {"sha256": "a" * 64},
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
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    }
    admission_path = root / "admission.json"
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    (root / "baseline/case_0007.json").write_text(
        json.dumps(_gate(candidate=False)), encoding="utf-8"
    )
    (root / "candidate/case_0007.json").write_text(
        json.dumps(_gate(candidate=True, position_p95=candidate_p95)),
        encoding="utf-8",
    )
    for name in ("baseline", "candidate"):
        (root / name / "runtime_heartbeat.json").write_text(
            json.dumps({"case": 7, "completed_steps": 100}),
            encoding="utf-8",
        )
    return root, admission_path


def _summarize(root: Path, admission: Path, **overrides):
    arguments = {
        "runtime_commit": "b" * 40,
        "baseline_exit_code": 0,
        "candidate_exit_code": 0,
        "gpu_release_passed": True,
    }
    arguments.update(overrides)
    return MODULE.summarize(root, admission, **arguments)


def test_summary_admits_only_safe_measurable_pair_improvement(tmp_path) -> None:
    root, admission = _fixture(tmp_path)
    result = _summarize(root, admission)
    assert result["passed"] is True
    assert result["corrective_target_admission_passed"] is True
    assert result["dynamic_pair_completed"] is True
    assert result["label_capture_authorized"] is False
    assert result["valid_for_training"] is False


def test_summary_rejects_weak_improvement(tmp_path) -> None:
    root, admission = _fixture(tmp_path, candidate_p95=0.129)
    result = _summarize(root, admission)
    assert result["corrective_target_admission_passed"] is False
    assert result["passed"] is False


def test_summary_rejects_mismatched_source_clock(tmp_path) -> None:
    root, admission = _fixture(tmp_path)
    path = root / "candidate/case_0007.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"][0]["source_duration_s"] = 16.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = _summarize(root, admission)
    assert (
        result["paired_admission"]["checks"]["same_case_plan_seed_and_clocks"]
        is False
    )
    assert result["passed"] is False


def test_summary_rejects_wrong_corrective_profile(tmp_path) -> None:
    root, admission = _fixture(tmp_path)
    path = root / "candidate/case_0007.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["corrective_teacher_profile"]["sha256"] = "e" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = _summarize(root, admission)
    assert result["rollout_checks"]["candidate_profile_bound"] is False
    assert result["passed"] is False


def test_summary_rejects_any_capture_path(tmp_path) -> None:
    root, admission = _fixture(tmp_path)
    path = root / "candidate/case_0007.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["corrective_teacher_capture_started"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = _summarize(root, admission)
    assert result["rollout_checks"]["candidate_capture_closed"] is False
    assert result["passed"] is False


def test_summary_rejects_failed_gpu_release(tmp_path) -> None:
    root, admission = _fixture(tmp_path)
    result = _summarize(root, admission, gpu_release_passed=False)
    assert result["rollout_checks"]["gpu_released"] is False
    assert result["passed"] is False


def test_summary_rejects_non_authorized_admission(tmp_path) -> None:
    root, admission = _fixture(tmp_path)
    payload = json.loads(admission.read_text(encoding="utf-8"))
    payload["runtime_authorized"] = False
    admission.write_text(json.dumps(payload), encoding="utf-8")
    result = _summarize(root, admission)
    assert result["contract_checks"]["runtime_authorized"] is False
    assert result["passed"] is False
