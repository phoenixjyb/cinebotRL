import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1]
VALIDATOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "validate_model_based_corrective_teacher_case8_validation_pair.py"
)
BUILDER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "build_model_based_corrective_teacher_case8_validation_pair_contract.py"
)
WRAPPER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "run_model_based_corrective_teacher_case8_validation_pair.sh"
)
SPEC = importlib.util.spec_from_file_location(
    "case8_validation_pair_validator", VALIDATOR
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
CONTRACT = ROOT / MODULE.CONTRACT_RELATIVE_PATH


def _run(*args, **kwargs):
    return subprocess.run(
        args, capture_output=True, text=True, check=kwargs.get("check", True)
    )


def test_contract_is_hash_bound_validation_only_and_tokenless() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema"] == MODULE.SCHEMA
    assert contract["reviewed_parent_commit"] == MODULE.REVIEWED_PARENT
    assert contract["case"] == 8
    assert contract["split"] == "validation"
    assert contract["selected_validation_cases"] == [8, 16]
    assert set(contract["identities"]) == MODULE.REQUIRED_IDENTITIES
    for identity in contract["identities"].values():
        path = ROOT / identity["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == identity["sha256"]
        blob = MODULE._git(ROOT, "hash-object", str(path)).stdout.strip()
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
    result = subprocess.run(
        ["bash", str(WRAPPER), "--preflight"],
        env={"RISER_ROOT": "/tmp/forged"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 7
    assert "conflicting_environment_override:RISER_ROOT" in result.stderr


def test_wrapper_contains_complete_validation_route_without_capture() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    empty_token = source.index('readonly AUTHORIZATION_SHA256=""')
    execute_reject = source.index('reject "runtime_authorization_not_issued" 4')
    validator = source.index('python3 "$VALIDATOR"')
    playback = source.index("timeout --signal=TERM --kill-after=30s 600")
    assert empty_token < execute_reject < validator < playback
    assert "--cases 8" in source
    assert "case_0008.json" in source
    assert "case8_validation_profile_v1.json" in source
    assert "--corrective-teacher-profile" in source
    assert "--dataset-dir" not in source
    assert "--raw-teacher-dir" not in source
    assert "--policy-trace-dir" not in source
    assert "--shadow-teacher-trace-dir" not in source
    assert "--corrective-teacher-capture-dir" not in source


def test_validator_rejects_alternate_contract_path(tmp_path: Path) -> None:
    alternate = tmp_path / "contract.json"
    alternate.write_bytes(CONTRACT.read_bytes())
    result = MODULE.validate(
        alternate,
        ROOT,
        namespace=MODULE.NAMESPACE,
    )
    assert result["checks"]["canonical_contract_path"] is False
    assert result["passed"] is False


def test_validator_rejects_any_authorization_file(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_text("not authorized\n", encoding="utf-8")
    result = MODULE.validate(
        CONTRACT,
        ROOT,
        namespace=MODULE.NAMESPACE,
        authorization_file=token,
    )
    assert result["checks"]["authorization_file_absent"] is False
    assert result["runtime_authorized"] is False
    assert result["passed"] is False
