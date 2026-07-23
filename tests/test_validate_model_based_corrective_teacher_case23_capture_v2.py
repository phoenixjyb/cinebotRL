import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).parents[1]
VALIDATOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "validate_model_based_corrective_teacher_case23_capture_v2.py"
)
FINALIZER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "summarize_model_based_corrective_teacher_case23_capture_v2.py"
)
WRAPPER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "run_model_based_corrective_teacher_case23_capture_v2.sh"
)
CONTRACT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case23_capture_contract_v2.json"
)


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VALIDATOR_MODULE = _module(VALIDATOR, "case23_capture_v2_validator")
FINALIZER_MODULE = _module(FINALIZER, "case23_capture_v2_finalizer")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()


def test_v2_contract_is_fresh_route_and_keeps_runtime_closed() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["reviewed_parent_commit"] == VALIDATOR_MODULE.REVIEWED_PARENT
    assert contract["namespace"] == VALIDATOR_MODULE.NAMESPACE
    assert contract["case"] == 23
    assert contract["split"] == "train"
    assert contract["execution_contract"] == VALIDATOR_MODULE.EXPECTED_EXECUTION
    assert contract["runtime_authorized"] is False
    assert contract["gpu_launch_authorized"] is False
    assert contract["authorization_token_issued"] is False
    assert contract["runtime_authorization_mode"] == (
        "out_of_band_sha256_environment_v1"
    )
    assert contract["runtime_authorization_token_sha256"] == ""
    assert contract["label_capture_authorized"] is False
    assert contract["dataset_creation_authorized"] is False
    assert contract["bc_authorized"] is False
    assert contract["ppo_authorized"] is False
    assert contract["training_started"] is False
    assert contract["valid_for_training"] is False


def test_v2_contract_pins_new_runtime_files_by_hash_and_blob() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    paths = {
        "contract_validator": VALIDATOR,
        "preflight_wrapper": WRAPPER,
        "capture_finalizer": FINALIZER,
    }
    for identity, path in paths.items():
        row = contract["identities"][identity]
        assert row["path"] == path.relative_to(ROOT).as_posix()
        assert row["sha256"] == _sha256(path)
        assert row["git_blob_sha1"] == _git_blob(path)


def test_v2_validator_forwards_only_fresh_case23_contract(monkeypatch) -> None:
    observed = {}

    def fake_validate(contract_path, repo, **kwargs):
        observed.update(kwargs)
        return {"schema": "base", "passed": True, "checks": {}}

    monkeypatch.setattr(VALIDATOR_MODULE, "validate_capture", fake_validate)
    monkeypatch.setattr(
        VALIDATOR_MODULE,
        "drive_profile_checks",
        lambda result: {"active_profile": True},
    )
    result = VALIDATOR_MODULE.validate(
        CONTRACT,
        ROOT,
        namespace=VALIDATOR_MODULE.NAMESPACE,
    )
    assert result["passed"] is True
    assert observed["expected_case"] == 23
    assert observed["expected_namespace"] == VALIDATOR_MODULE.NAMESPACE
    assert observed["contract_relative_path"] == (
        VALIDATOR_MODULE.CONTRACT_RELATIVE_PATH
    )
    assert observed["reviewed_parent"] == VALIDATOR_MODULE.REVIEWED_PARENT
    assert observed["plan_identity_name"] == "case23_plan"
    assert observed["required_identities"] == VALIDATOR_MODULE.REQUIRED_IDENTITIES
    assert observed["tracked_identities"] == VALIDATOR_MODULE.TRACKED_IDENTITIES


def test_v2_wrapper_expands_namespace_and_cannot_execute_without_token() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    output_line = next(
        line for line in source.splitlines() if line.startswith("readonly OUTPUT_WIN=")
    )
    assert output_line == (
        "readonly OUTPUT_WIN=\"${WIN_ROOT}\""
        "'\\artifacts\\two_wheel_riser\\'"
        '"${NAMESPACE}"'
    )
    shell = "\n".join(
        [
            r'WIN_ROOT="G:\wSpace\cinebotRL-two-wheel-riser"',
            f'NAMESPACE="{VALIDATOR_MODULE.NAMESPACE}"',
            output_line.replace("readonly ", "", 1),
            "printf '%s' \"$OUTPUT_WIN\"",
        ]
    )
    expanded = subprocess.run(
        ["bash", "-c", shell],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert expanded == (
        r"G:\wSpace\cinebotRL-two-wheel-riser\artifacts\two_wheel_riser"
        rf"\{VALIDATOR_MODULE.NAMESPACE}"
    )
    assert "$NAMESPACE" not in expanded
    assert (
        'AUTHORIZATION_SHA256="${'
        'RISER_CORRECTIVE_CASE23_CAPTURE_V2_AUTHORIZATION_SHA256:-}"'
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


def test_v2_validator_uses_only_exact_out_of_band_token_hash(
    monkeypatch, tmp_path: Path
) -> None:
    def fake_validate(contract_path, repo, **kwargs):
        assert kwargs["authorization_file"] is None
        return {"schema": "base", "passed": True, "checks": {}}

    monkeypatch.setattr(VALIDATOR_MODULE, "validate_capture", fake_validate)
    monkeypatch.setattr(
        VALIDATOR_MODULE,
        "drive_profile_checks",
        lambda result: {"active_profile": True},
    )
    token = ROOT.parent / ".case23-v2-test-authorization-token"
    token.write_bytes(b"runtime-only-random-token-fixture\n")
    token.chmod(0o600)
    if os.name == "nt":
        # Runtime admission executes under WSL; Windows does not expose POSIX
        # chmod bits for this cross-platform unit test.
        monkeypatch.setattr(VALIDATOR_MODULE.stat, "S_IMODE", lambda _: 0o600)
    token_hash = hashlib.sha256(token.read_bytes()).hexdigest()
    try:
        admitted = VALIDATOR_MODULE.validate(
            CONTRACT,
            ROOT,
            namespace=VALIDATOR_MODULE.NAMESPACE,
            authorization_file=token,
            authorization_sha256=token_hash,
        )
        assert admitted["authorization_checks"][
            "authorization_file_outside_repository"
        ] is True
        assert admitted["runtime_authorized"] is True
        assert admitted["label_capture_authorized"] is True
        assert admitted["passed"] is True
        rejected = VALIDATOR_MODULE.validate(
            CONTRACT,
            ROOT,
            namespace=VALIDATOR_MODULE.NAMESPACE,
            authorization_file=token,
            authorization_sha256="0" * 64,
        )
        assert rejected["runtime_authorized"] is False
        assert rejected["label_capture_authorized"] is False
        assert rejected["passed"] is False
    finally:
        token.unlink(missing_ok=True)


def test_v2_validator_rejects_token_inside_repository(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        VALIDATOR_MODULE,
        "validate_capture",
        lambda *args, **kwargs: {
            "schema": "base",
            "passed": True,
            "checks": {},
        },
    )
    monkeypatch.setattr(
        VALIDATOR_MODULE,
        "drive_profile_checks",
        lambda result: {"active_profile": True},
    )
    token = ROOT / ".runtime-secrets" / "case23-v2-test.authorization-token"
    token.parent.mkdir(exist_ok=True)
    token.write_bytes(b"must-not-live-in-worktree\n")
    token.chmod(0o600)
    token_hash = hashlib.sha256(token.read_bytes()).hexdigest()
    try:
        rejected = VALIDATOR_MODULE.validate(
            CONTRACT,
            ROOT,
            namespace=VALIDATOR_MODULE.NAMESPACE,
            authorization_file=token,
            authorization_sha256=token_hash,
        )
        assert rejected["authorization_checks"][
            "authorization_file_outside_repository"
        ] is False
        assert rejected["runtime_authorized"] is False
        assert rejected["passed"] is False
    finally:
        token.unlink(missing_ok=True)


def test_v2_validator_rejects_noncanonical_authorization_hash(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        VALIDATOR_MODULE,
        "validate_capture",
        lambda *args, **kwargs: {
            "schema": "base",
            "passed": True,
            "checks": {},
        },
    )
    monkeypatch.setattr(
        VALIDATOR_MODULE,
        "drive_profile_checks",
        lambda result: {"active_profile": True},
    )
    token = ROOT.parent / ".case23-v2-test-authorization-token"
    token.write_bytes(b"runtime-only-random-token-fixture\n")
    token.chmod(0o600)
    try:
        rejected = VALIDATOR_MODULE.validate(
            CONTRACT,
            ROOT,
            namespace=VALIDATOR_MODULE.NAMESPACE,
            authorization_file=token,
            authorization_sha256=hashlib.sha256(
                token.read_bytes()
            ).hexdigest().upper(),
        )
        assert rejected["authorization_checks"][
            "authorization_hash_is_out_of_band"
        ] is False
        assert rejected["runtime_authorized"] is False
        assert rejected["passed"] is False
    finally:
        token.unlink(missing_ok=True)


def test_v2_wrapper_remains_capture_only() -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    assert "--cases 23" in source
    assert "--policy-command-base model_based_planner" in source
    assert "--corrective-teacher-capture-dir" in source
    assert "--corrective-teacher-capture-admission" in source
    assert "--raw-teacher-capture" not in source
    assert "train_riser_residual_bc" not in "\n".join(
        line for line in source.splitlines() if line.lstrip().startswith('"$PY"')
    )


def test_v2_finalizer_forwards_fresh_namespace(monkeypatch, tmp_path: Path) -> None:
    observed = {}

    def fake_summarize(root, admission_path, **kwargs):
        observed.update(kwargs)
        return {"passed": False}

    monkeypatch.setattr(FINALIZER_MODULE, "summarize_capture", fake_summarize)
    result = FINALIZER_MODULE.summarize(
        tmp_path,
        tmp_path / "admission.json",
        runtime_commit="a" * 40,
        playback_exit_code=2,
        gpu_release_passed=True,
    )
    assert result["passed"] is False
    assert observed["expected_case"] == 23
    assert observed["expected_namespace"] == VALIDATOR_MODULE.NAMESPACE
    assert observed["capture_name"] == "case_0023_corrective_teacher_capture_v2.npz"
    assert observed["plan_identity_name"] == "case23_plan"
