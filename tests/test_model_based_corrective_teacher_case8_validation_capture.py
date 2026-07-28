import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).parents[1]
VALIDATOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "validate_model_based_corrective_teacher_case8_validation_capture.py"
)
BUILDER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "build_model_based_corrective_teacher_case8_validation_capture_contract.py"
)
WRAPPER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "run_model_based_corrective_teacher_case8_validation_capture.sh"
)
FINALIZER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "summarize_model_based_corrective_teacher_case8_validation_capture.py"
)


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _module(VALIDATOR, "case8_validation_capture_validator")
FINALIZER_MODULE = _module(FINALIZER, "case8_validation_capture_finalizer")
CONTRACT = ROOT / MODULE.CONTRACT_RELATIVE_PATH


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()


def _resource_admission() -> dict[str, object]:
    return {
        "schema": "cinebotrl_windows_shared_resource_admission_v2",
        "phase": "launch",
        "thresholds": {
            "minimum_windows_free_memory_gib": 5.0,
            "minimum_gpu_free_memory_mib": 9_216,
            "launch_minimum_windows_free_memory_gib": 5.0,
            "launch_minimum_gpu_free_memory_mib": 9_216,
            "runtime_minimum_windows_free_memory_gib": 1.5,
            "runtime_minimum_gpu_free_memory_mib": 2_048,
            "cad_coexistence_allowed": True,
            "cad_process_names": [
                "creoparametric.exe",
                "parametric.exe",
                "proe.exe",
                "sldworks.exe",
                "ugraf.exe",
                "xtop.exe",
            ],
        },
        "observed": {
            "windows_total_memory_gib": 32.0,
            "windows_free_memory_gib": 6.0,
            "cad_processes": [],
            "gpu_count": 1,
            "gpu_total_memory_mib": 24_467,
            "gpu_used_memory_mib": 13_662,
            "gpu_free_memory_mib": 10_500,
            "gpu_unaccounted_memory_mib": 305,
        },
        "checks": {
            "windows_memory_probe_valid": True,
            "windows_free_memory_sufficient": True,
            "cad_process_probe_valid": True,
            "cad_coexistence_allowed": True,
            "gpu_memory_probe_valid": True,
            "gpu_free_memory_sufficient": True,
        },
        "runtime_started": False,
        "authorization_consumed": False,
        "passed": True,
    }


def _resource_monitor() -> dict[str, object]:
    sample = {
        "schema": "cinebotrl_windows_shared_resource_admission_v2",
        "phase": "runtime",
        "observed": {
            "windows_free_memory_gib": 2.5,
            "gpu_free_memory_mib": 3_500,
        },
        "runtime_started": True,
        "authorization_consumed": True,
        "passed": True,
    }
    return {
        "schema": "cinebotrl_windows_shared_resource_monitor_v1",
        "monitored_pid": 123,
        "runtime_thresholds": {
            "minimum_windows_free_memory_gib": 1.5,
            "minimum_gpu_free_memory_mib": 2_048,
        },
        "sample_count": 1,
        "minimum_observed_windows_free_memory_gib": 2.5,
        "minimum_observed_gpu_free_memory_mib": 3_500,
        "termination_requested": False,
        "process_exit_observed": True,
        "samples": [sample],
        "passed": True,
    }


def _write_resources(root: Path) -> None:
    (root / "resource_admission.json").write_text(
        json.dumps(_resource_admission()),
        encoding="utf-8",
    )
    (root / "resource_monitor.json").write_text(
        json.dumps(_resource_monitor()),
        encoding="utf-8",
    )


def test_contract_is_validation_only_tokenless_and_hash_bound() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema"] == MODULE.SCHEMA
    assert contract["reviewed_parent_commit"] == MODULE.REVIEWED_PARENT
    assert contract["namespace"] == MODULE.NAMESPACE
    assert contract["case"] == 8
    assert contract["split"] == "validation"
    assert contract["validation_cases_opened"] == [8]
    assert contract["holdout_opened"] is False
    assert contract["execution_contract"] == MODULE.EXPECTED_EXECUTION
    assert set(contract["identities"]) == MODULE.REQUIRED_IDENTITIES
    for identity in contract["identities"].values():
        path = ROOT / identity["path"]
        assert identity["sha256"] == _sha256(path)
        assert identity["git_blob_sha1"] == _git_blob(path)
    assert contract["runtime_authorization_token_sha256"] == ""
    for field in (
        "runtime_authorized",
        "gpu_launch_authorized",
        "authorization_token_issued",
        "label_capture_authorized",
        "dataset_creation_authorized",
        "conversion_authorized",
        "bc_authorized",
        "ppo_authorized",
        "training_started",
        "valid_for_training",
    ):
        assert contract[field] is False


def test_contract_builder_regenerates_exact_bytes(tmp_path: Path) -> None:
    output = tmp_path / "contract.json"
    result = subprocess.run(
        [sys.executable, str(BUILDER), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == CONTRACT.read_bytes()


def test_wrapper_is_one_validation_capture_without_training() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "--cases 8" in source
    assert "--corrective-teacher-capture-split validation" in source
    assert "--corrective-teacher-capture-dir" in source
    assert "case_0008.json" in source
    assert "case8_validation_capture_v1_coexistence" in source
    assert (
        'AUTHORIZATION_SHA256="${'
        'RISER_CORRECTIVE_CASE8_VALIDATION_CAPTURE_AUTHORIZATION_SHA256:-}"'
    ) in source
    assert re.search(r'AUTHORIZATION_SHA256="[0-9a-f]{64}"', source) is None
    assert "--dataset-dir" not in source
    assert "--checkpoint-output" not in source
    assert "--training-metadata" not in source
    resource_guard = source.index('python3 "$RESOURCE_GUARD" --phase launch')
    namespace_creation = source.index('mkdir -p "$OUTPUT/capture"')
    token_consumption = source.index('rm -f "$AUTHORIZATION_FILE"')
    playback = source.index("setsid timeout --signal=TERM")
    assert resource_guard < namespace_creation < token_consumption < playback
    assert source.count('python3 "$RESOURCE_MONITOR"') == 1


def test_wrapper_execute_without_token_rejects_before_runtime() -> None:
    result = subprocess.run(
        ["bash", str(WRAPPER), "--execute"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 4
    payload = json.loads(result.stderr)
    assert payload["reason"] == "runtime_authorization_not_issued"
    assert payload["runtime_started"] is False
    assert payload["label_capture_started"] is False


def test_authorization_hash_must_be_external_and_exact(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_text("one case-8 validation capture\n", encoding="utf-8")
    token.chmod(0o600)
    token_hash = _sha256(token)
    checks = MODULE._authorization_checks(
        token,
        token_hash,
        ROOT,
        CONTRACT.read_text(encoding="utf-8"),
    )
    if os.name == "nt":
        assert checks["authorization_mode_0600"] is False
        checks.pop("authorization_mode_0600")
    assert all(checks.values())
    embedded = MODULE._authorization_checks(
        token,
        token_hash,
        ROOT,
        CONTRACT.read_text(encoding="utf-8") + token_hash,
    )
    assert embedded["authorization_hash_is_out_of_band"] is False
    wrong = MODULE._authorization_checks(
        token,
        "0" * 64,
        ROOT,
        CONTRACT.read_text(encoding="utf-8"),
    )
    assert wrong["authorization_hash_matches"] is False


def test_finalizer_uses_case8_validation_archive(monkeypatch, tmp_path) -> None:
    observed = {}

    def fake_summary(root, admission_path, **kwargs):
        observed.update(kwargs)
        return {
            "passed": True,
            "capture_admitted_for_dataset_conversion": True,
        }

    monkeypatch.setattr(FINALIZER_MODULE, "summarize_capture", fake_summary)
    _write_resources(tmp_path)
    (tmp_path / "admission.json").write_text(
        json.dumps(
            {
                "case": 8,
                "split": "validation",
                "corrective_target_admission_passed": True,
                "dataset_creation_authorized": False,
                "bc_authorized": False,
                "ppo_authorized": False,
                "training_started": False,
            }
        ),
        encoding="utf-8",
    )
    result = FINALIZER_MODULE.summarize(
        tmp_path,
        tmp_path / "admission.json",
        runtime_commit="a" * 40,
        playback_exit_code=0,
        gpu_release_passed=True,
    )
    assert observed["expected_case"] == 8
    assert observed["expected_split"] == "validation"
    assert observed["expected_namespace"] == MODULE.NAMESPACE
    assert observed["capture_name"] == (
        "case_0008_corrective_teacher_capture_v2.npz"
    )
    assert observed["plan_identity_name"] == "case8_plan"
    assert result["passed"] is True
    assert result["capture_admitted_for_dataset_conversion"] is True
    assert result["conversion_authorized"] is False
    assert result["training_started"] is False


def test_finalizer_fails_closed_without_resource_evidence(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        FINALIZER_MODULE,
        "summarize_capture",
        lambda *args, **kwargs: {
            "passed": True,
            "capture_admitted_for_dataset_conversion": True,
        },
    )
    result = FINALIZER_MODULE.summarize(
        tmp_path,
        tmp_path / "admission.json",
        runtime_commit="a" * 40,
        playback_exit_code=0,
        gpu_release_passed=True,
    )
    assert result["shared_windows_resource_admission_passed"] is False
    assert result["shared_windows_resource_monitor_passed"] is False
    assert result["capture_admitted_for_dataset_conversion"] is False
    assert result["passed"] is False
