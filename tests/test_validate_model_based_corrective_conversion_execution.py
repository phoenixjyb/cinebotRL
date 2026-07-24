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
    "validate_model_based_corrective_conversion_execution.py"
)
BUILDER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "build_model_based_corrective_conversion_execution_contract.py"
)
WRAPPER = (
    ROOT
    / "scripts/two_wheel_balance/"
    "run_model_based_corrective_conversion_v2.sh"
)
CONTRACT = (
    ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_conversion_execution_contract_v2.json"
)
PROPOSAL_ROOT = (
    ROOT
    / "docs/03_training/two_wheel_balance/"
    "evidence_20260724_generic_corrective_conversion_proposals_v1"
)


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _module(VALIDATOR, "generic_conversion_execution_validator")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload
    ).hexdigest()


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _identity(path: Path, relative: str) -> dict[str, str]:
    return {
        "path": relative,
        "sha256": _sha256(path),
        "git_blob_sha1": _git_blob(path),
    }


def _wsl_executable() -> str:
    return str(
        Path(os.environ.get("WINDIR", r"C:\Windows"))
        / "System32/wsl.exe"
    )


def _secure_token(tmp_path: Path) -> Path:
    if os.name != "nt":
        token = tmp_path / "generic-conversion-authorization"
        token.write_bytes(b"one-generic-cpu-conversion\n")
        token.chmod(0o600)
        return token
    wsl_root = "/home/yanbo/.codex_generic_conversion_tests"
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
    token.write_bytes(b"one-generic-cpu-conversion\n")
    return token


def _contract_payload(repo: Path) -> dict[str, object]:
    return {
        "schema": MODULE.SCHEMA,
        "reviewed_parent_commit": MODULE.REVIEWED_PARENT,
        "implementation_commit": "a" * 40,
        "allowed_splits": list(MODULE.ALLOWED_SPLITS),
        "identities": {
            name: _identity(repo / relative, relative)
            for name, relative in MODULE.CODE_PATHS.items()
        },
        "execution_contract": {
            "namespace_template": (
                "model_based_corrective_case{case:04d}_"
                "conversion_execution_v2_cpu"
            ),
            "dataset_name_template": (
                "case_{case:04d}_model_based_corrective_"
                "case_dataset_v1.npz"
            ),
            "one_use_authorization_required": True,
            "authorization_consumed_before_conversion": True,
            "fresh_namespace_required": True,
            "reopen_output_required": True,
            "proposal_must_be_tracked_and_committed": True,
            "cpu_conversion_only": True,
        },
        "contract_ready": True,
        "conversion_execution_implemented": True,
        "conversion_authorized": False,
        "authorization_token_issued": False,
        "authorization_token_sha256": "",
        "output_created": False,
        "merged_dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
    }


def _proposal_relative(case: int) -> str:
    return (
        "docs/03_training/two_wheel_balance/"
        "evidence_20260724_generic_corrective_conversion_proposals_v1/"
        f"case_{case:04d}.json"
    )


def _isolated_repo(
    tmp_path: Path,
    case: int,
) -> tuple[Path, Path, Path]:
    repo = tmp_path / f"repo_{case}"
    repo.mkdir(parents=True)
    for relative in MODULE.CODE_PATHS.values():
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)

    proposal_source = PROPOSAL_ROOT / f"case_{case:04d}.json"
    proposal_payload = json.loads(proposal_source.read_text())
    proposal = repo / _proposal_relative(case)
    proposal.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(proposal_source, proposal)
    for identity in proposal_payload["identities"].values():
        relative = identity["path"]
        destination = repo / relative
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)

    contract = repo / MODULE.CONTRACT_RELATIVE_PATH
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_text(
        json.dumps(_contract_payload(repo), indent=2) + "\n",
        encoding="utf-8",
    )
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Tests")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fixture")
    return repo, contract, proposal


def _repository_checks() -> dict[str, bool]:
    return {
        "canonical_contract": True,
        "contract_committed": True,
        "proposal_committed": True,
        "head_matches_upstream": True,
        "tracked_worktree_clean": True,
        "reviewed_parent_is_ancestor": True,
        "proposal_commit_is_ancestor": True,
    }


def _validate(
    repo: Path,
    contract: Path,
    proposal: Path,
    **kwargs,
):
    return MODULE.validate(
        contract,
        proposal,
        repo,
        repository_checks=_repository_checks(),
        git_state={"head": "a" * 40, "upstream": "a" * 40},
        proposal_commit_is_ancestor=True,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("case", "sample_count", "clipped_rows"),
    [
        (6, 7933, [0, 146, 0]),
        (23, 3273, [0, 0, 0]),
        (30, 11411, [200, 308, 333]),
    ],
)
def test_one_generic_preflight_accepts_all_existing_proposals(
    tmp_path,
    case,
    sample_count,
    clipped_rows,
) -> None:
    repo, contract, proposal = _isolated_repo(tmp_path, case)
    result = _validate(repo, contract, proposal)
    assert result["passed"] is True, json.dumps(result, indent=2)
    assert result["cpu_contract_ready"] is True
    assert result["case"] == case
    assert result["split"] == "train"
    assert result["namespace"] == MODULE.namespace_for(case)
    assert result["dataset_name"] == MODULE.dataset_name_for(case)
    assert result["source_metrics"]["sample_count"] == sample_count
    assert result["source_metrics"]["clipped_rows"] == clipped_rows
    assert list(result["proposal_identities"]) == list(
        MODULE.PROPOSAL_IDENTITY_NAMES
    )
    assert all(result["contract_checks"].values())
    assert all(result["proposal_checks"].values())
    assert result["conversion_authorized"] is False
    assert result["output_created"] is False
    assert result["valid_for_case_merge"] is False
    assert result["merged_dataset_created"] is False
    assert result["bc_authorized"] is False
    assert result["ppo_authorized"] is False
    assert result["training_started"] is False


def test_external_mode_0600_token_opens_only_one_conversion(tmp_path) -> None:
    repo, contract, proposal = _isolated_repo(tmp_path, 6)
    token = _secure_token(tmp_path)
    try:
        result = _validate(
            repo,
            contract,
            proposal,
            authorization_file=token,
            authorization_sha256=_sha256(token),
        )
        assert result["passed"] is True
        assert result["conversion_authorized"] is True
        assert result["authorization_consumed_before_conversion"] is True
        assert result["output_created"] is False
        assert result["merged_dataset_created"] is False
        assert result["bc_authorized"] is False
        assert result["ppo_authorized"] is False
    finally:
        token.unlink(missing_ok=True)


def test_forged_token_proposal_or_source_fails_closed(tmp_path) -> None:
    repo, contract, proposal = _isolated_repo(tmp_path, 6)
    token = _secure_token(tmp_path)
    try:
        bad_token = _validate(
            repo,
            contract,
            proposal,
            authorization_file=token,
            authorization_sha256="0" * 64,
        )
        assert bad_token["passed"] is False
        assert bad_token["conversion_authorized"] is False
    finally:
        token.unlink(missing_ok=True)

    payload = json.loads(proposal.read_text())
    payload["metrics"]["sample_count"] += 1
    proposal.write_text(json.dumps(payload))
    forged = _validate(repo, contract, proposal)
    assert forged["passed"] is False
    assert forged["contract_checks"]["proposal_file_committed"] is False
    assert forged["conversion_authorized"] is False

    _git(repo, "restore", _proposal_relative(6))
    source_relative = payload["identities"]["source_capture"]["path"]
    with (repo / source_relative).open("ab") as stream:
        stream.write(b"drift")
    drift = _validate(repo, contract, proposal)
    assert drift["passed"] is False
    assert drift["contract_checks"]["proposal_identity_hashes"] is False
    assert drift["conversion_authorized"] is False


def test_stale_namespace_or_wrong_repository_state_fails_closed(
    tmp_path,
) -> None:
    repo, contract, proposal = _isolated_repo(tmp_path, 23)
    namespace = repo / "artifacts/two_wheel_riser" / MODULE.namespace_for(23)
    namespace.mkdir(parents=True)
    stale = _validate(repo, contract, proposal)
    assert stale["passed"] is False
    assert stale["contract_checks"]["fresh_namespace"] is False

    checks = _repository_checks()
    checks["head_matches_upstream"] = False
    result = MODULE.validate(
        contract,
        proposal,
        repo,
        repository_checks=checks,
        git_state={"head": "a" * 40, "upstream": "b" * 40},
        proposal_commit_is_ancestor=True,
    )
    assert result["passed"] is False
    assert result["conversion_authorized"] is False


def test_boolean_case_or_extra_identity_fails_closed(tmp_path) -> None:
    repo, contract, proposal = _isolated_repo(tmp_path / "bool_case", 6)
    payload = json.loads(proposal.read_text())
    payload["case"] = True
    proposal.write_text(json.dumps(payload))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "forge boolean case")
    boolean_case = _validate(repo, contract, proposal)
    assert boolean_case["passed"] is False
    assert boolean_case["case"] == -1
    assert boolean_case["conversion_authorized"] is False

    repo, contract, proposal = _isolated_repo(
        tmp_path / "extra_identity",
        23,
    )
    payload = json.loads(contract.read_text())
    payload["identities"]["unexpected"] = payload["identities"][
        "capture_module"
    ]
    contract.write_text(json.dumps(payload))
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "forge extra identity")
    extra_identity = _validate(repo, contract, proposal)
    assert extra_identity["passed"] is False
    assert extra_identity["contract_checks"]["identity_set"] is False
    assert extra_identity["conversion_authorized"] is False


def test_contract_builder_is_closed_and_output_free(monkeypatch) -> None:
    builder = _module(BUILDER, "generic_conversion_contract_builder")
    monkeypatch.setattr(
        builder,
        "_repository_state",
        lambda repo: {
            "head": "a" * 40,
            "upstream": "a" * 40,
            "checks": {
                "head_matches_upstream": True,
                "tracked_worktree_clean": True,
                "reviewed_parent_is_ancestor": True,
            },
            "passed": True,
        },
    )
    monkeypatch.setattr(
        builder,
        "_identity",
        lambda repo, relative: _identity(ROOT / relative, relative),
    )
    result = builder.build_contract(ROOT)
    assert result["contract_ready"] is True
    assert result["conversion_execution_implemented"] is True
    assert result["conversion_authorized"] is False
    assert result["authorization_token_issued"] is False
    assert result["authorization_token_sha256"] == ""
    assert result["output_created"] is False
    assert result["merged_dataset_created"] is False
    assert result["bc_authorized"] is False
    assert result["ppo_authorized"] is False
    assert result["training_started"] is False


def test_canonical_contract_pins_generic_route_and_stays_closed() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["schema"] == MODULE.SCHEMA
    assert contract["reviewed_parent_commit"] == MODULE.REVIEWED_PARENT
    assert contract["implementation_commit"] == (
        "bf9894e7b0ec48bec06ca7b7848fe8f326257be7"
    )
    assert contract["allowed_splits"] == ["train", "validation"]
    assert set(contract["identities"]) == set(MODULE.CODE_PATHS)
    for name, relative in MODULE.CODE_PATHS.items():
        path = ROOT / relative
        assert contract["identities"][name] == _identity(path, relative)
    assert contract["contract_ready"] is True
    assert contract["conversion_execution_implemented"] is True
    assert contract["conversion_authorized"] is False
    assert contract["authorization_token_issued"] is False
    assert contract["authorization_token_sha256"] == ""
    assert contract["output_created"] is False
    assert contract["merged_dataset_created"] is False
    assert contract["bc_authorized"] is False
    assert contract["ppo_authorized"] is False
    assert contract["training_started"] is False
    assert contract["valid_for_training"] is False


def test_windows_paths_and_wrapper_authorization_boundary() -> None:
    assert MODULE._windows_path_to_wsl(
        r"G:\wSpace\cinebotRL-two-wheel-riser"
    ) == "/mnt/g/wSpace/cinebotRL-two-wheel-riser"
    assert MODULE._windows_path_to_wsl(
        r"\\wsl.localhost\Ubuntu\home\yanbo\authorization"
    ) == "/home/yanbo/authorization"
    source = WRAPPER.read_text()
    assert "--expected-case \"$CASE\"" in source
    assert "--expected-split \"$SPLIT\"" in source
    assert 'rm -f "$AUTHORIZATION_FILE"' in source
    assert source.index('rm -f "$AUTHORIZATION_FILE"') < source.index(
        '"$PY" -X utf8 "$CONVERTER_WIN"'
    )
    assert "case_0006_model_based_corrective_case_dataset_v1.npz" not in source
    assert "case_0023_model_based_corrective_case_dataset_v1.npz" not in source
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "--proposal",
            "docs/proposal.json",
            "--execute",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 4
    assert "conversion_authorization_not_issued" in result.stderr
