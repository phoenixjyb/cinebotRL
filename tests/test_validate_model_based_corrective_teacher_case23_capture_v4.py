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
    "validate_model_based_corrective_teacher_case23_capture_v4.py"
)
WRAPPER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "run_model_based_corrective_teacher_case23_capture_v4.sh"
)
CONTRACT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case23_capture_contract_v4.json"
)
PLAYBACK = ROOT / "scripts/two_wheel_balance/smoke_riser_reference_playback.py"
FINALIZER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "summarize_model_based_corrective_teacher_case23_capture_v4.py"
)
V3_MANIFEST = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_case23_corrective_capture_v3_rejected_save_route/"
    "manifest.json"
)


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _module(VALIDATOR, "case23_capture_v4_validator")
FINALIZER_MODULE = _module(FINALIZER, "case23_capture_v4_finalizer")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()


def test_v4_contract_is_fresh_no_token_route_after_consumed_v3() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["reviewed_parent_commit"] == MODULE.REVIEWED_PARENT
    assert contract["namespace"] == MODULE.NAMESPACE
    assert contract["route_revision"] == "case23_capture_v4"
    assert contract["case"] == 23
    assert contract["split"] == "train"
    assert contract["execution_contract"] == MODULE.EXPECTED_EXECUTION
    assert contract["supersedes"]["namespace"].endswith(
        "case23_capture_v3_exclusive"
    )
    assert contract["supersedes"]["retry_authorized"] is False
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


def test_v4_contract_pins_repaired_runtime_and_route_files() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    paths = {
        "playback": PLAYBACK,
        "contract_validator": VALIDATOR,
        "preflight_wrapper": WRAPPER,
        "capture_finalizer": FINALIZER,
    }
    for identity, path in paths.items():
        row = contract["identities"][identity]
        assert row["path"] == path.relative_to(ROOT).as_posix()
        assert row["sha256"] == _sha256(path)
        assert row["git_blob_sha1"] == _git_blob(path)

    supersedes = contract["supersedes"]
    assert supersedes["rejection_manifest_sha256"] == _sha256(V3_MANIFEST)
    assert supersedes["rejection_manifest_git_blob_sha1"] == _git_blob(V3_MANIFEST)


def test_v4_validator_forwards_exact_case_split_and_reviewed_parent(
    monkeypatch,
) -> None:
    observed = {}

    def fake_validate(contract_path, repo, **kwargs):
        observed.update(kwargs)
        return {"schema": "base", "passed": True, "checks": {}}

    monkeypatch.setattr(MODULE, "validate_capture", fake_validate)
    monkeypatch.setattr(
        MODULE,
        "drive_profile_checks",
        lambda result: {"active_profile": True},
    )
    result = MODULE.validate(CONTRACT, ROOT, namespace=MODULE.NAMESPACE)
    assert result["passed"] is True
    assert observed["expected_case"] == 23
    assert observed["expected_namespace"] == MODULE.NAMESPACE
    assert observed["reviewed_parent"] == MODULE.REVIEWED_PARENT
    assert observed["expected_execution"]["case"] == 23
    assert observed["expected_execution"]["split"] == "train"


def test_v4_wrapper_binds_route_and_has_no_committed_token() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "--corrective-teacher-capture-split train" in source
    assert "case23_capture_v4_exclusive" in source
    assert "case23_capture_v3_exclusive" not in source
    assert "summarize_model_based_corrective_teacher_case23_capture_v4.py" in source
    assert (
        'AUTHORIZATION_SHA256="${'
        'RISER_CORRECTIVE_CASE23_CAPTURE_V4_AUTHORIZATION_SHA256:-}"'
    ) in source
    assert re.search(r'AUTHORIZATION_SHA256="[0-9a-f]{64}"', source) is None
    result = subprocess.run(
        ["bash", str(WRAPPER), "--execute"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 4
    assert "runtime_authorization_not_issued" in result.stderr


def test_v4_playback_propagates_admitted_case_and_split_into_save() -> None:
    source = PLAYBACK.read_text(encoding="utf-8")
    save_call = source.split("save_corrective_capture(", 1)[1].split("\n        )", 1)[0]
    assert 'case=int(corrective_capture_admission["case"])' in save_call
    assert "split=args.corrective_teacher_capture_split" in save_call


def test_v4_finalizer_seals_only_the_v4_case23_route(monkeypatch, tmp_path) -> None:
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
    assert observed["expected_case"] == 23
    assert observed["expected_namespace"] == MODULE.NAMESPACE
    assert observed["capture_name"] == (
        "case_0023_corrective_teacher_capture_v2.npz"
    )
    assert observed["plan_identity_name"] == "case23_plan"
