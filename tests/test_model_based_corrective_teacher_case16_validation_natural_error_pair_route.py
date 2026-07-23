import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
SCRIPT_DIR = ROOT / "scripts/two_wheel_balance"
VALIDATOR = (
    SCRIPT_DIR
    / "validate_model_based_corrective_teacher_case16_validation_natural_error_pair.py"
)
BUILDER = (
    SCRIPT_DIR
    / "build_model_based_corrective_teacher_case16_validation_natural_error_pair_contract.py"
)
WRAPPER = (
    SCRIPT_DIR
    / "run_model_based_corrective_teacher_case16_validation_natural_error_pair.sh"
)
ADAPTER = (
    SCRIPT_DIR / "smoke_riser_case16_validation_natural_error_pair.py"
)
FINALIZER = (
    SCRIPT_DIR
    / "summarize_model_based_corrective_teacher_case16_validation_natural_error_pair.py"
)


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR_MODULE = _module(VALIDATOR, "case16_validation_pair_validator")
ADAPTER_MODULE = _module(ADAPTER, "case16_validation_projection_adapter")
FINALIZER_MODULE = _module(FINALIZER, "case16_validation_pair_finalizer")
CONTRACT = ROOT / VALIDATOR_MODULE.CONTRACT_RELATIVE_PATH


def _run(*args: str, **kwargs):
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=kwargs.get("check", True),
        env=kwargs.get("env"),
    )


def test_adapter_observes_projection_without_changing_commands() -> None:
    expected = np.array([0.4, -0.204, 0.751], dtype=np.float64)
    telemetry = ADAPTER_MODULE.ProjectionTelemetry(enabled=True)
    observed = telemetry.wrap(lambda *args, **kwargs: expected)
    result = observed(
        0.4,
        -0.2,
        0.75,
        np.array([0.1, -0.08, 0.05]),
        action_scales=np.array([0.05, 0.05, 0.02]),
    )
    assert result is expected
    summary = telemetry.summary()
    assert summary["sample_count"] == 1
    assert summary["command_clipped_sample_count"] == [1, 0, 0]
    np.testing.assert_allclose(
        summary["effective_normalized_action_abs_max"],
        [0.0, 0.08, 0.05],
    )
    assert summary["observer_modified_commands"] is False
    assert summary["labels_captured"] is False
    assert summary["dataset_created"] is False


def test_adapter_injection_is_case16_validation_only(tmp_path: Path) -> None:
    output = tmp_path / "case16.json"
    output.write_text(
        json.dumps({"results": [{"case": 16}]}), encoding="utf-8"
    )
    telemetry = ADAPTER_MODULE.ProjectionTelemetry(enabled=True).summary()
    ADAPTER_MODULE.inject_projection_telemetry(output, telemetry)
    payload = json.loads(output.read_text(encoding="utf-8"))
    adapter = payload["case16_validation_projection_telemetry_adapter"]
    assert adapter["observer_modified_commands"] is False
    assert adapter["label_capture_started"] is False
    assert adapter["dataset_creation_started"] is False
    assert adapter["teacher_admission_opened"] is False
    assert adapter["training_started"] is False
    output.write_text(
        json.dumps({"results": [{"case": 2}]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="case-16"):
        ADAPTER_MODULE.inject_projection_telemetry(output, telemetry)


def test_contract_is_hash_bound_validation_only_and_tokenless() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema"] == VALIDATOR_MODULE.SCHEMA
    assert contract["reviewed_parent_commit"] == VALIDATOR_MODULE.REVIEWED_PARENT
    assert contract["case"] == 16
    assert contract["split"] == "validation"
    assert contract["selected_validation_cases"] == [8, 16]
    assert set(contract["identities"]) == VALIDATOR_MODULE.REQUIRED_IDENTITIES
    assert len(contract["identities"]) == 24
    for identity in contract["identities"].values():
        path = ROOT / identity["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == identity["sha256"]
        blob = VALIDATOR_MODULE._git(
            ROOT, "hash-object", str(path)
        ).stdout.strip()
        assert blob == identity["git_blob_sha1"]
    assert contract["cpu_preflight_ready"] is True
    assert contract["runtime_route_contract_ready"] is True
    assert contract["execution_route_complete"] is True
    assert contract["runtime_authorization_token_sha256"] == ""
    for field in (
        "runtime_authorized",
        "gpu_launch_authorized",
        "authorization_token_issued",
        "teacher_admission_authorized",
        "label_capture_authorized",
        "dataset_creation_authorized",
        "bc_authorized",
        "ppo_authorized",
        "training_started",
        "valid_for_training",
    ):
        assert contract[field] is False


def test_contract_builder_regenerates_exact_bytes(tmp_path: Path) -> None:
    output = tmp_path / "contract.json"
    result = _run(
        sys.executable,
        str(BUILDER),
        "--output",
        str(output),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == CONTRACT.read_bytes()


def test_wrapper_execute_rejects_before_python_or_isaac() -> None:
    result = _run("bash", str(WRAPPER), "--execute", check=False)
    assert result.returncode == 4
    payload = json.loads(result.stderr)
    assert payload["reason"] == "runtime_authorization_not_issued"
    assert payload["python_started"] is False
    assert payload["isaac_started"] is False
    assert payload["runtime_started"] is False


def test_wrapper_rejects_conflicting_environment_before_python() -> None:
    env = dict(os.environ)
    env["RISER_CORRECTIVE_CASE16_VALIDATION_PERTURBATION"] = "/tmp/forged"
    result = _run(
        "bash", str(WRAPPER), "--preflight", env=env, check=False
    )
    assert result.returncode == 7
    assert "conflicting_environment_override" in result.stderr


def test_wrapper_has_no_wrench_capture_or_dataset_route() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    empty_token = source.index('readonly AUTHORIZATION_SHA256=""')
    execute_reject = source.index('reject "runtime_authorization_not_issued" 4')
    validator = source.index('python3 "$VALIDATOR"')
    playback = source.index("timeout --signal=TERM --kill-after=30s 600")
    assert empty_token < execute_reject < validator < playback
    assert "--cases 16" in source
    assert "case_0016.json" in source
    assert "case16_validation_natural_error_profile_v1.json" in source
    assert "--corrective-teacher-profile" in source
    assert "--deterministic-wrench-profile" not in source
    assert "--dataset-dir" not in source
    assert "--raw-teacher-dir" not in source
    assert "--policy-trace-dir" not in source
    assert "--shadow-teacher-trace-dir" not in source
    assert "--corrective-teacher-capture-dir" not in source


def test_validator_rejects_alternate_contract_or_authorization(
    tmp_path: Path,
) -> None:
    alternate = tmp_path / "contract.json"
    alternate.write_bytes(CONTRACT.read_bytes())
    result = VALIDATOR_MODULE.validate(
        alternate,
        ROOT,
        namespace=VALIDATOR_MODULE.NAMESPACE,
    )
    assert result["checks"]["canonical_contract_path"] is False
    assert result["passed"] is False
    token = tmp_path / "token"
    token.write_text("not authorized\n", encoding="utf-8")
    result = VALIDATOR_MODULE.validate(
        CONTRACT,
        ROOT,
        namespace=VALIDATOR_MODULE.NAMESPACE,
        authorization_file=token,
    )
    assert result["checks"]["authorization_file_absent"] is False
    assert result["runtime_authorized"] is False
    assert result["passed"] is False


CORRECTIVE_SHA = "c" * 64


def _gate(
    *,
    candidate: bool,
    position_p95: float = 0.070,
    source_duration_s: float = 17.548706,
) -> dict[str, object]:
    projection = {
        "schema": (
            "cinebotrl_two_wheel_riser_corrective_projection_telemetry_v1"
        ),
        "enabled": candidate,
        "sample_count": 200 if candidate else 0,
        "requested_residual_abs_max": (
            [0.0042, 0.0070, 0.0010] if candidate else [0.0, 0.0, 0.0]
        ),
        "effective_residual_abs_max": (
            [0.0030, 0.0050, 0.0010] if candidate else [0.0, 0.0, 0.0]
        ),
        "effective_normalized_action_abs_max": (
            [0.06, 0.10, 0.05] if candidate else [0.0, 0.0, 0.0]
        ),
        "requested_effective_delta_abs_max": (
            [0.0012, 0.0020, 0.0] if candidate else [0.0, 0.0, 0.0]
        ),
        "command_clipped_sample_count": (
            [100, 30, 0] if candidate else [0, 0, 0]
        ),
        "any_command_clipped_sample_count": 120 if candidate else 0,
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
        "case": 16,
        "source_duration_s": source_duration_s,
        "execution_duration_s": 26.028629743189363,
        "dynamic_quality_passed": True,
        "position_error_p95_m": position_p95 if candidate else 0.080600,
        "position_error_max_m": 0.079 if candidate else 0.081492,
        "attitude_error_max_deg": 0.20,
        "pitch_max_deg": 6.1,
        "riser_servo_error_max_m": 0.012,
        "action_saturation_ratio": 0.0,
        "residual_action_abs_max": (
            [0.06, 0.10, 0.05] if candidate else [0.0, 0.0, 0.0]
        ),
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
        "cases": [16],
        "passed": True,
        "trajectory_command_source": FINALIZER_MODULE.EXPECTED_SOURCES[
            "candidate" if candidate else "baseline"
        ],
        "policy_command_base": "model_based_planner",
        "residual_action_scales": FINALIZER_MODULE.EXPECTED_SCALES,
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


def _fixture(tmp_path: Path, *, candidate_p95: float = 0.070):
    root = tmp_path / FINALIZER_MODULE.NAMESPACE
    (root / "baseline").mkdir(parents=True)
    (root / "candidate").mkdir()
    contract = {
        "namespace": FINALIZER_MODULE.NAMESPACE,
        "case": 16,
        "split": "validation",
        "controller_arguments": {"reset_seed": 20260732},
        "identities": {
            "case16_plan": {"sha256": "a" * 64},
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
        "teacher_admission_authorized": False,
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
    (root / "baseline/case_0016.json").write_text(
        json.dumps(_gate(candidate=False)), encoding="utf-8"
    )
    (root / "candidate/case_0016.json").write_text(
        json.dumps(_gate(candidate=True, position_p95=candidate_p95)),
        encoding="utf-8",
    )
    for name in ("baseline", "candidate"):
        (root / name / "runtime_heartbeat.json").write_text(
            json.dumps({"case": 16, "completed_steps": 100}),
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
    return FINALIZER_MODULE.summarize(root, admission, **arguments)


def test_finalizer_accepts_closed_held_out_natural_error_pair(
    tmp_path: Path,
) -> None:
    root, admission = _fixture(tmp_path)
    result = _summarize(root, admission)
    assert result["passed"] is True
    assert result["validation_pair_passed"] is True
    assert result["dynamic_pair_completed"] is True
    assert result["external_wrench_used"] is False
    assert result["effective_projection_telemetry_required"] is True
    assert result["teacher_admission_opened"] is False
    assert result["label_capture_authorized"] is False
    assert result["valid_for_training"] is False


def test_finalizer_rejects_weak_improvement_or_clock_drift(
    tmp_path: Path,
) -> None:
    root, admission = _fixture(tmp_path, candidate_p95=0.079)
    result = _summarize(root, admission)
    assert result["validation_pair_passed"] is False
    candidate_path = root / "candidate/case_0016.json"
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    payload["results"][0]["source_duration_s"] = 18.0
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")
    result = _summarize(root, admission)
    assert result["paired_admission"]["checks"][
        "same_case_plan_seed_and_clocks"
    ] is False
    assert result["passed"] is False


def test_finalizer_rejects_wrench_capture_or_missing_projection(
    tmp_path: Path,
) -> None:
    root, admission = _fixture(tmp_path)
    candidate_path = root / "candidate/case_0016.json"
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    payload["deterministic_wrench_profile"] = {"sha256": "d" * 64}
    payload["corrective_teacher_capture_started"] = True
    payload["results"][0]["corrective_teacher_projection_telemetry"] = {}
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")
    result = _summarize(root, admission, gpu_release_passed=False)
    assert result["rollout_checks"]["external_wrench_absent"] is False
    assert result["rollout_checks"]["candidate_capture_closed"] is False
    assert result["rollout_checks"]["candidate_projection_measured"] is False
    assert result["rollout_checks"]["gpu_released"] is False
    assert result["passed"] is False
