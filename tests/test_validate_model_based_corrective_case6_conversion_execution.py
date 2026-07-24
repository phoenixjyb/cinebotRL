import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
VALIDATOR = (
    ROOT
    / "scripts/two_wheel_balance/"
    "validate_model_based_corrective_case6_conversion_execution.py"
)
WRAPPER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "run_model_based_corrective_case6_conversion_v1.sh"
)
CONTRACT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_case6_conversion_execution_contract_v1.json"
)


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _module(VALIDATOR, "case6_conversion_execution_validator")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload
    ).hexdigest()


def _repository_checks() -> dict[str, bool]:
    return {
        "canonical_contract": True,
        "contract_tracked": True,
        "contract_blob_matches_head": True,
        "head_matches_upstream": True,
        "tracked_worktree_clean": True,
        "reviewed_parent_is_ancestor": True,
    }


def _isolated_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract = repo / MODULE.CONTRACT_RELATIVE_PATH
    contract.parent.mkdir(parents=True)
    shutil.copy2(CONTRACT, contract)
    for identity in payload["identities"].values():
        relative = Path(identity["path"])
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return repo, contract


def _validate(
    *,
    contract: Path = CONTRACT,
    repo: Path = ROOT,
    **kwargs,
):
    return MODULE.validate(
        contract,
        repo,
        namespace=MODULE.NAMESPACE,
        repository_checks=_repository_checks(),
        git_state={"head": "a" * 40, "upstream": "a" * 40},
        **kwargs,
    )


def _secure_token(tmp_path: Path) -> Path:
    token = tmp_path / "case6-conversion-authorization"
    token.write_bytes(b"one-case6-cpu-conversion\n")
    token.chmod(0o600)
    return token


def test_execution_contract_is_closed_and_pins_route_files() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["schema"] == MODULE.SCHEMA
    assert contract["reviewed_parent_commit"] == MODULE.REVIEWED_PARENT
    assert contract["case"] == 6
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


def test_no_token_preflight_validates_source_but_stays_closed(
    tmp_path: Path,
) -> None:
    repo, contract = _isolated_repo(tmp_path)
    result = _validate(contract=contract, repo=repo)
    assert result["passed"] is True
    assert result["cpu_contract_ready"] is True
    assert all(result["source_checks"].values())
    assert result["source_metrics"]["sample_count"] == 7933
    assert result["source_metrics"]["clipped_rows"] == [0, 146, 0]
    assert result["conversion_authorized"] is False
    assert result["output_created"] is False
    assert result["valid_for_case_merge"] is False
    assert result["bc_authorized"] is False
    assert result["ppo_authorized"] is False
    assert result["training_started"] is False


def test_external_mode_0600_token_opens_exactly_conversion(tmp_path) -> None:
    repo, contract = _isolated_repo(tmp_path)
    token = _secure_token(tmp_path)
    result = _validate(
        contract=contract,
        repo=repo,
        authorization_file=token,
        authorization_sha256=_sha256(token),
    )
    assert result["passed"] is True
    assert result["conversion_authorized"] is True
    assert result["authorization_consumed_before_conversion"] is True
    assert result["merged_dataset_created"] is False
    assert result["bc_authorized"] is False
    assert result["ppo_authorized"] is False


def test_bad_token_or_repository_check_fails_closed(tmp_path) -> None:
    token = _secure_token(tmp_path)
    token.chmod(0o644)
    assert _validate(
        authorization_file=token,
        authorization_sha256=_sha256(token),
    )["passed"] is False
    token.chmod(0o600)
    assert _validate(
        authorization_file=token,
        authorization_sha256="0" * 64,
    )["passed"] is False

    checks = _repository_checks()
    checks["head_matches_upstream"] = False
    result = MODULE.validate(
        CONTRACT,
        ROOT,
        namespace=MODULE.NAMESPACE,
        repository_checks=checks,
        git_state={"head": "a" * 40, "upstream": "b" * 40},
    )
    assert result["passed"] is False
    assert result["conversion_authorized"] is False


def test_forged_identity_or_namespace_fails_closed(tmp_path) -> None:
    repo, contract = _isolated_repo(tmp_path)
    forged = json.loads(contract.read_text())
    forged["identities"]["source_capture"]["sha256"] = "0" * 64
    contract.write_text(json.dumps(forged))
    assert _validate(contract=contract, repo=repo)["passed"] is False
    assert MODULE.validate(
        CONTRACT,
        ROOT,
        namespace="alternate",
        repository_checks=_repository_checks(),
        git_state={"head": "a" * 40, "upstream": "a" * 40},
    )["passed"] is False


def test_windows_and_wsl_unc_paths_map_to_same_wsl_file() -> None:
    assert MODULE._windows_path_to_wsl(
        r"G:\wSpace\cinebotRL-two-wheel-riser"
    ) == "/mnt/g/wSpace/cinebotRL-two-wheel-riser"
    assert MODULE._windows_path_to_wsl(
        r"\\wsl.localhost\Ubuntu\home\yanbo\.codex_authorizations\token"
    ) == "/home/yanbo/.codex_authorizations/token"
    with pytest.raises(ValueError):
        MODULE._windows_path_to_wsl(
            r"\\wsl.localhost\Other\home\yanbo\.codex_authorizations\token"
        )


def test_wrapper_has_no_embedded_token_and_rejects_execute_without_one() -> None:
    source = WRAPPER.read_text()
    assert "case_0006_model_based_corrective_case_dataset_v1.npz" in source
    assert "--expected-case 6" in source
    assert "--expected-split train" in source
    assert "case_0023" not in source
    assert "AUTHORIZATION_SHA256=\"${" in source
    result = subprocess.run(
        ["bash", str(WRAPPER), "--execute"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 4
    assert "conversion_authorization_not_issued" in result.stderr
