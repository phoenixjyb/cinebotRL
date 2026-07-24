import hashlib
import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/two_wheel_balance/"
    "summarize_model_based_corrective_teacher_case2_natural_error_pair.py"
)
SPEC = importlib.util.spec_from_file_location(
    "case2_natural_error_pair_summary", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


CORRECTIVE_SHA = "c" * 64
REAL_REJECTED_PAIR = (
    Path(__file__).parents[1]
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_case2_natural_error_pair_execution_v1"
)


def _gate(
    *,
    candidate: bool,
    position_p95: float = 0.130,
    source_duration_s: float = 9.439314,
) -> dict[str, object]:
    projection = {
        "schema": (
            "cinebotrl_two_wheel_riser_corrective_projection_telemetry_v1"
        ),
        "enabled": candidate,
        "sample_count": 100 if candidate else 0,
        "requested_residual_abs_max": (
            [0.010, 0.004, 0.0006] if candidate else [0.0, 0.0, 0.0]
        ),
        "effective_residual_abs_max": (
            [0.008, 0.003, 0.0006] if candidate else [0.0, 0.0, 0.0]
        ),
        "effective_normalized_action_abs_max": (
            [0.16, 0.06, 0.03] if candidate else [0.0, 0.0, 0.0]
        ),
        "requested_effective_delta_abs_max": (
            [0.002, 0.001, 0.0] if candidate else [0.0, 0.0, 0.0]
        ),
        "command_clipped_sample_count": (
            [10, 5, 0] if candidate else [0, 0, 0]
        ),
        "any_command_clipped_sample_count": 15 if candidate else 0,
        "applied_to_commands": False,
        "labels_captured": False,
        "dataset_created": False,
        "training_started": False,
    }
    perturbation = {
        "enabled": False,
        "profile": None,
        "trigger_step": None,
        "active_step_count": 0,
        "expected_active_step_count": 0,
        "released_after_pulse": False,
        "triggered": False,
    }
    result = {
        "case": 2,
        "source_duration_s": source_duration_s,
        "execution_duration_s": 18.241927671727208,
        "dynamic_quality_passed": True,
        "position_error_p95_m": position_p95 if candidate else 0.139830,
        "position_error_max_m": 0.151 if candidate else 0.153514,
        "attitude_error_max_deg": 0.17,
        "pitch_max_deg": 6.5,
        "riser_servo_error_max_m": 0.012,
        "action_saturation_ratio": 0.0,
        "residual_action_abs_max": (
            [0.20, 0.08, 0.03] if candidate else [0.0, 0.0, 0.0]
        ),
        "completed_steps": 100,
        "requested_policy_residual_action_abs_max": (
            [0.20, 0.08, 0.03] if candidate else [0.0, 0.0, 0.0]
        ),
        "effective_policy_residual_action_abs_max": (
            [0.16, 0.06, 0.03] if candidate else [0.0, 0.0, 0.0]
        ),
        "policy_residual_projection_delta_abs_max": (
            [0.04, 0.02, 0.0] if candidate else [0.0, 0.0, 0.0]
        ),
        "policy_residual_projection_sample_count": 15 if candidate else 0,
        "corrective_teacher_telemetry": {
            "normalized_action_abs_max": [0.20, 0.08, 0.03]
        },
        "corrective_teacher_projection_telemetry": projection,
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
        "cases": [2],
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
        "deterministic_wrench_profile": None,
        "corrective_teacher_capture_started": False,
        "corrective_teacher_label_capture_authorized": False,
        "raw_teacher_capture_started": False,
        "normalized_dataset_capture_started": False,
        "policy_trace_started": False,
        "shadow_teacher_trace_started": False,
        "results": [result],
    }


def _fixture(tmp_path: Path, *, candidate_p95: float = 0.130):
    root = tmp_path / MODULE.NAMESPACE
    (root / "baseline").mkdir(parents=True)
    (root / "candidate").mkdir()
    contract = {
        "namespace": MODULE.NAMESPACE,
        "case": 2,
        "split": "train",
        "controller_arguments": {"reset_seed": 20260718},
        "identities": {
            "case2_plan": {"sha256": "a" * 64},
            "corrective_profile": {"sha256": CORRECTIVE_SHA},
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
    (root / "baseline/case_0002.json").write_text(
        json.dumps(_gate(candidate=False)), encoding="utf-8"
    )
    (root / "candidate/case_0002.json").write_text(
        json.dumps(_gate(candidate=True, position_p95=candidate_p95)),
        encoding="utf-8",
    )
    for name in ("baseline", "candidate"):
        (root / name / "runtime_heartbeat.json").write_text(
            json.dumps({"case": 2, "completed_steps": 100}),
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


def test_summary_admits_safe_natural_error_improvement(tmp_path) -> None:
    root, admission = _fixture(tmp_path)
    result = _summarize(root, admission)
    assert result["passed"] is True
    assert result["corrective_target_admission_passed"] is True
    assert result["dynamic_pair_completed"] is True
    assert result["external_wrench_used"] is False
    assert result["effective_projection_telemetry_required"] is True
    assert result["label_capture_authorized"] is False
    assert result["valid_for_training"] is False


def test_summary_rejects_weak_improvement(tmp_path) -> None:
    root, admission = _fixture(tmp_path, candidate_p95=0.138)
    result = _summarize(root, admission)
    assert result["corrective_target_admission_passed"] is False
    assert result["passed"] is False


def test_summary_ignores_missing_unreliable_adapter_telemetry(tmp_path) -> None:
    root, admission = _fixture(tmp_path)
    path = root / "candidate/case_0002.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"][0]["corrective_teacher_projection_telemetry"] = {}
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = _summarize(root, admission)
    assert result["rollout_checks"]["candidate_projection_measured"] is True
    assert result["passed"] is True


def test_summary_rejects_missing_runtime_projection_aggregate(tmp_path) -> None:
    root, admission = _fixture(tmp_path)
    path = root / "candidate/case_0002.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["results"][0][
        "effective_policy_residual_action_abs_max"
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = _summarize(root, admission)
    assert result["rollout_checks"]["candidate_projection_measured"] is False
    assert result["passed"] is False


def test_summary_rejects_any_external_wrench(tmp_path) -> None:
    root, admission = _fixture(tmp_path)
    path = root / "candidate/case_0002.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["deterministic_wrench_profile"] = {"sha256": "d" * 64}
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = _summarize(root, admission)
    assert result["rollout_checks"]["external_wrench_absent"] is False
    assert result["passed"] is False


def test_summary_rejects_mismatched_source_clock(tmp_path) -> None:
    root, admission = _fixture(tmp_path)
    path = root / "candidate/case_0002.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"][0]["source_duration_s"] = 10.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = _summarize(root, admission)
    assert (
        result["paired_admission"]["checks"]["same_case_plan_seed_and_clocks"]
        is False
    )
    assert result["passed"] is False


def test_summary_rejects_capture_or_failed_gpu_release(tmp_path) -> None:
    root, admission = _fixture(tmp_path)
    path = root / "candidate/case_0002.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["corrective_teacher_capture_started"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = _summarize(
        root, admission, gpu_release_passed=False
    )
    assert result["rollout_checks"]["candidate_capture_closed"] is False
    assert result["rollout_checks"]["gpu_released"] is False
    assert result["passed"] is False


def test_real_case2_archive_is_reclassified_as_weak_improvement() -> None:
    result = MODULE.summarize(
        REAL_REJECTED_PAIR,
        REAL_REJECTED_PAIR / "admission.json",
        runtime_commit="9363f2818688653c2c6db60699caba496a0c8d3a",
        baseline_exit_code=0,
        candidate_exit_code=0,
        gpu_release_passed=True,
    )
    assert result["rollout_checks"]["candidate_projection_measured"] is True
    assert result["dynamic_pair_completed"] is True
    assert result["corrective_target_admission_passed"] is False
    assert result["paired_admission"]["checks"][
        "minimum_position_p95_improvement"
    ] is False
    assert (
        result["paired_admission"]["position_p95_absolute_improvement_m"]
        < 0.003
    )
    assert (
        result["paired_admission"]["position_p95_relative_improvement"]
        < 0.02
    )
    assert result["passed"] is False
    assert result["label_capture_authorized"] is False
    assert result["training_started"] is False
