import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).parents[1]
VALIDATOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "validate_model_based_corrective_teacher_case7_capture.py"
)
WRAPPER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "run_model_based_corrective_teacher_case7_capture.sh"
)
CONTRACT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case7_capture_contract_v1.json"
)
FINALIZER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "summarize_model_based_corrective_teacher_case7_capture.py"
)
RESOURCE_GUARD = (
    ROOT
    / "scripts/two_wheel_balance/check_windows_shared_resource_admission.py"
)


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _module(VALIDATOR, "case7_capture_validator")
FINALIZER_MODULE = _module(FINALIZER, "case7_capture_finalizer")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()


def _projection_payload(*, candidate_sample_count=6597) -> dict[str, object]:
    return {
        "schema": MODULE.PROJECTION_SCHEMA,
        "case": 7,
        "baseline": {
            "enabled": False,
            "passed": True,
            "projection_affected_sample_count": 0,
        },
        "candidate": {
            "enabled": True,
            "passed": True,
            "sample_count": candidate_sample_count,
            "observer_modified_commands": False,
            "applied_to_commands": False,
        },
        "labels_captured": False,
        "dataset_created": False,
        "training_started": False,
        "valid_for_training": False,
        "passed": True,
    }


def _mock_base_result(projection_path: Path) -> dict[str, object]:
    return {
        "schema": "base",
        "passed": True,
        "cpu_contract_ready": True,
        "checks": {},
        "identities": {
            "paired_projection_audit": {
                "path": str(projection_path),
                "passed": True,
            }
        },
    }


def test_contract_is_fresh_no_token_case7_capture_route() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["reviewed_parent_commit"] == MODULE.REVIEWED_PARENT
    assert contract["namespace"] == MODULE.NAMESPACE
    assert contract["route_revision"] == "case7_capture_v1"
    assert contract["case"] == 7
    assert contract["split"] == "train"
    assert contract["execution_contract"] == MODULE.EXPECTED_EXECUTION
    assert contract["paired_projection_evidence"]["passed"] is True
    assert contract["runtime_authorized"] is False
    assert contract["gpu_launch_authorized"] is False
    assert contract["authorization_token_issued"] is False
    assert contract["runtime_authorization_token_sha256"] == ""
    assert contract["label_capture_authorized"] is False
    assert contract["dataset_creation_authorized"] is False
    assert contract["bc_authorized"] is False
    assert contract["ppo_authorized"] is False
    assert contract["training_started"] is False
    assert contract["valid_for_training"] is False


def test_contract_pins_case7_route_and_projection_files() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    paths = {
        "paired_projection_audit": (
            ROOT
            / "docs/03_training/two_wheel_balance/"
            "evidence_20260724_case7_corrective_pair_execution_v1/"
            "projection_audit.json"
        ),
        "contract_validator": VALIDATOR,
        "preflight_wrapper": WRAPPER,
        "shared_windows_resource_guard": RESOURCE_GUARD,
        "capture_finalizer": FINALIZER,
    }
    for identity, path in paths.items():
        row = contract["identities"][identity]
        assert row["path"] == path.relative_to(ROOT).as_posix()
        assert row["sha256"] == _sha256(path)
        assert row["git_blob_sha1"] == _git_blob(path)


def test_validator_forwards_case7_profile_pair_and_plan(
    monkeypatch, tmp_path
) -> None:
    projection = tmp_path / "projection.json"
    projection.write_text(
        json.dumps(_projection_payload()),
        encoding="utf-8",
    )
    observed = {}

    def fake_validate(contract_path, repo, **kwargs):
        observed.update(kwargs)
        return _mock_base_result(projection)

    monkeypatch.setattr(MODULE, "validate_capture", fake_validate)
    monkeypatch.setattr(
        MODULE,
        "drive_profile_checks",
        lambda result: {"active_profile": True},
    )
    result = MODULE.validate(CONTRACT, ROOT, namespace=MODULE.NAMESPACE)
    assert result["passed"] is True
    assert result["paired_projection_checks"][
        "candidate_projection_passed"
    ] is True
    assert observed["expected_case"] == 7
    assert observed["expected_namespace"] == MODULE.NAMESPACE
    assert observed["reviewed_parent"] == MODULE.REVIEWED_PARENT
    assert observed["pair_schema"] == MODULE.PAIR_SCHEMA
    assert observed["plan_identity_name"] == "case7_plan"
    assert (
        observed["expected_profile_maximum_residuals"]
        == MODULE.EXPECTED_PROFILE_MAXIMUM_RESIDUALS
    )


def test_validator_rejects_malformed_projection_sample_count(
    monkeypatch, tmp_path
) -> None:
    projection = tmp_path / "projection.json"
    projection.write_text(
        json.dumps(_projection_payload(candidate_sample_count="6597")),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        MODULE,
        "validate_capture",
        lambda *args, **kwargs: _mock_base_result(projection),
    )
    monkeypatch.setattr(
        MODULE,
        "drive_profile_checks",
        lambda result: {"active_profile": True},
    )
    result = MODULE.validate(CONTRACT, ROOT, namespace=MODULE.NAMESPACE)
    assert result["paired_projection_checks"][
        "candidate_projection_passed"
    ] is False
    assert result["cpu_contract_ready"] is False
    assert result["passed"] is False


def test_wrapper_is_capture_only_and_has_no_committed_token() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "--cases 7" in source
    assert "--corrective-teacher-capture-split train" in source
    assert "--corrective-teacher-capture-dir" in source
    assert "case7_capture_v1_exclusive" in source
    assert "case7_pair_v1_exclusive" not in source
    assert (
        'AUTHORIZATION_SHA256="${'
        'RISER_CORRECTIVE_CASE7_CAPTURE_AUTHORIZATION_SHA256:-}"'
    ) in source
    assert re.search(r'AUTHORIZATION_SHA256="[0-9a-f]{64}"', source) is None
    assert "--dataset-dir" not in source
    assert "--checkpoint-output" not in source
    assert "--training-metadata" not in source
    resource_guard = source.index(
        'python3 "$RESOURCE_GUARD" --output "$RESOURCE_ADMISSION"'
    )
    token_consumption = source.index('rm -f "$AUTHORIZATION_FILE"')
    namespace_creation = source.index('mkdir -p "$OUTPUT/capture"')
    assert resource_guard < namespace_creation < token_consumption
    assert 'cp "$RESOURCE_ADMISSION" "$OUTPUT/resource_admission.json"' in source
    result = subprocess.run(
        ["bash", str(WRAPPER), "--execute"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 4
    assert "runtime_authorization_not_issued" in result.stderr


def test_validator_rejects_wrong_out_of_band_hash(
    monkeypatch, tmp_path
) -> None:
    projection = tmp_path / "projection.json"
    projection.write_text(
        json.dumps(_projection_payload()),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        MODULE,
        "validate_capture",
        lambda *args, **kwargs: _mock_base_result(projection),
    )
    monkeypatch.setattr(
        MODULE,
        "drive_profile_checks",
        lambda result: {"active_profile": True},
    )
    token = tmp_path / "token"
    token.write_text("fixture\n", encoding="utf-8")
    token.chmod(0o600)
    result = MODULE.validate(
        CONTRACT,
        ROOT,
        namespace=MODULE.NAMESPACE,
        authorization_file=token,
        authorization_sha256="0" * 64,
    )
    assert result["authorization_checks"]["authorization_hash_matches"] is False
    assert result["runtime_authorized"] is False
    assert result["label_capture_authorized"] is False
    assert result["passed"] is False


def test_finalizer_seals_only_case7_capture(monkeypatch, tmp_path) -> None:
    observed = {}

    def fake_summary(root, admission_path, **kwargs):
        observed.update(kwargs)
        return {"passed": False}

    monkeypatch.setattr(FINALIZER_MODULE, "summarize_capture", fake_summary)
    FINALIZER_MODULE.summarize(
        tmp_path,
        tmp_path / "admission.json",
        runtime_commit="a" * 40,
        playback_exit_code=1,
        gpu_release_passed=True,
    )
    assert observed["expected_case"] == 7
    assert observed["expected_namespace"] == MODULE.NAMESPACE
    assert observed["capture_name"] == (
        "case_0007_corrective_teacher_capture_v2.npz"
    )
    assert observed["plan_identity_name"] == "case7_plan"
