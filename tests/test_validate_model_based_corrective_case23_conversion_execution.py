import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
VALIDATOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "validate_model_based_corrective_case23_conversion_execution.py"
)
FINALIZER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "finalize_model_based_corrective_case23_conversion.py"
)
WRAPPER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "run_model_based_corrective_case23_conversion_v1.sh"
)
CONTRACT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_case23_conversion_execution_contract_v1.json"
)
REVIEW = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260723_case23_corrective_conversion_review_v1/summary.json"
)


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _module(VALIDATOR, "case23_conversion_execution_validator")


def _repository_checks() -> dict[str, bool]:
    return {
        "canonical_contract": True,
        "contract_tracked": True,
        "contract_blob_matches_head": True,
        "head_matches_upstream": True,
        "tracked_worktree_clean": True,
        "reviewed_parent_is_ancestor": True,
    }


def _validate(**kwargs):
    return MODULE.validate(
        CONTRACT,
        ROOT,
        namespace=MODULE.NAMESPACE,
        repository_checks=_repository_checks(),
        git_state={"head": "a" * 40, "upstream": "a" * 40},
        review_result=json.loads(REVIEW.read_text()),
        **kwargs,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload
    ).hexdigest()


def _wsl_executable() -> str:
    return str(
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32/wsl.exe"
    )


def _secure_token(tmp_path: Path) -> Path:
    if os.name != "nt":
        token = tmp_path / "authorization"
        token.write_bytes(b"case23-v4-conversion-test-token\n")
        token.chmod(0o600)
        return token
    wsl_root = "/home/yanbo/.codex_case23_conversion_tests"
    created = subprocess.run(
        [
            _wsl_executable(),
            "sh",
            "-lc",
            f"umask 077; mkdir -p {wsl_root}; mktemp {wsl_root}/token.XXXXXX",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    windows_path = subprocess.run(
        [_wsl_executable(), "wslpath", "-w", created],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    token = Path(windows_path)
    token.write_bytes(b"case23-v4-conversion-test-token\n")
    return token


def _chmod(token: Path, mode: int) -> None:
    if os.name != "nt":
        token.chmod(mode)
        return
    subprocess.run(
        [
            _wsl_executable(),
            "chmod",
            f"{mode:o}",
            MODULE._windows_path_to_wsl(str(token)),
        ],
        check=True,
    )


def test_execution_contract_is_closed_and_pins_route_files() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["schema"] == MODULE.SCHEMA
    assert contract["reviewed_parent_commit"] == MODULE.REVIEWED_PARENT
    assert contract["case"] == 23
    assert contract["split"] == "train"
    assert contract["namespace"] == MODULE.NAMESPACE
    assert contract["conversion_authorized"] is False
    assert contract["authorization_token_issued"] is False
    assert contract["authorization_token_sha256"] == ""
    assert contract["output_created"] is False
    assert contract["merged_dataset_created"] is False
    assert contract["bc_authorized"] is False
    assert contract["ppo_authorized"] is False
    assert contract["training_started"] is False
    assert contract["valid_for_training"] is False
    for name, relative in MODULE.EXPECTED_IDENTITY_PATHS.items():
        path = ROOT / relative
        assert contract["identities"][name]["sha256"] == _sha256(path)
        assert contract["identities"][name]["git_blob_sha1"] == _git_blob(path)


def test_no_token_preflight_is_ready_but_not_authorized() -> None:
    result = _validate()
    assert result["passed"] is True
    assert result["cpu_contract_ready"] is True
    assert result["conversion_authorized"] is False
    assert result["authorization_consumed_before_conversion"] is False
    assert result["output_created"] is False
    assert result["valid_for_case_merge"] is False
    assert result["bc_authorized"] is False
    assert result["ppo_authorized"] is False
    assert result["training_started"] is False


def test_external_mode_0600_token_opens_only_one_conversion(tmp_path) -> None:
    token = _secure_token(tmp_path)
    try:
        result = _validate(
            authorization_file=token,
            authorization_sha256=_sha256(token),
        )
        assert result["passed"] is True
        assert result["conversion_authorized"] is True
        assert result["authorization_consumed_before_conversion"] is True
        assert result["merged_dataset_created"] is False
        assert result["bc_authorized"] is False
        assert result["ppo_authorized"] is False
    finally:
        token.unlink(missing_ok=True)


def test_bad_token_or_repository_check_fails_closed(tmp_path) -> None:
    token = _secure_token(tmp_path)
    try:
        _chmod(token, 0o644)
        assert _validate(
            authorization_file=token,
            authorization_sha256=_sha256(token),
        )["passed"] is False
        _chmod(token, 0o600)
        assert _validate(
            authorization_file=token,
            authorization_sha256="0" * 64,
        )["passed"] is False
    finally:
        token.unlink(missing_ok=True)

    checks = _repository_checks()
    checks["head_matches_upstream"] = False
    result = MODULE.validate(
        CONTRACT,
        ROOT,
        namespace=MODULE.NAMESPACE,
        repository_checks=checks,
        git_state={"head": "a" * 40, "upstream": "b" * 40},
        review_result=json.loads(REVIEW.read_text()),
    )
    assert result["passed"] is False
    assert result["conversion_authorized"] is False


def test_windows_and_wsl_unc_paths_map_to_the_same_wsl_file() -> None:
    assert MODULE._windows_path_to_wsl(
        r"G:\wSpace\cinebotRL-two-wheel-riser"
    ) == "/mnt/g/wSpace/cinebotRL-two-wheel-riser"
    assert MODULE._windows_path_to_wsl(
        r"\\wsl.localhost\Ubuntu\home\yanbo\.codex_authorizations\token"
    ) == "/home/yanbo/.codex_authorizations/token"
    assert MODULE._windows_path_to_wsl(
        r"\\wsl$\Ubuntu\home\yanbo\.codex_authorizations\token"
    ) == "/home/yanbo/.codex_authorizations/token"
    with pytest.raises(ValueError):
        MODULE._windows_path_to_wsl(
            r"\\wsl.localhost\Other\home\yanbo\.codex_authorizations\token"
        )


def test_wrapper_has_no_embedded_token_and_execute_rejects_without_one() -> None:
    source = WRAPPER.read_text()
    assert "case_0023_model_based_corrective_case_dataset_v1.npz" in source
    assert "--expected-case 23" in source
    assert "--expected-split train" in source
    assert "case_0030" not in source
    assert "AUTHORIZATION_SHA256=\"${" in source
    result = subprocess.run(
        ["bash", str(WRAPPER), "--execute"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 4
    assert "conversion_authorization_not_issued" in result.stderr
