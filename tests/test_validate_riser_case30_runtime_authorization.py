import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts.two_wheel_balance import (
    validate_riser_case30_runtime_authorization as validator,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _identity(repo: Path, relative: str) -> dict[str, str]:
    path = repo / relative
    return {
        "path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "git_blob_sha1": _git(repo, "hash-object", str(path)),
    }


@pytest.fixture
def admitted_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    scripts = repo / "scripts/two_wheel_balance"
    scripts.mkdir(parents=True)
    identity_paths = {
        "cpu_contract": validator.CPU_CONTRACT_RELATIVE_PATH,
        "runtime_summarizer": (
            "scripts/two_wheel_balance/"
            "summarize_riser_case30_perturbation_canary.py"
        ),
        "runtime_validator": (
            "scripts/two_wheel_balance/"
            "validate_riser_case30_runtime_authorization.py"
        ),
        "runtime_wrapper": (
            "scripts/two_wheel_balance/"
            "run_riser_case30_perturbation_canary.sh"
        ),
    }
    for name, relative in identity_paths.items():
        path = repo / relative
        path.write_text(f"{name}\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "reviewed CPU state")
    reviewed_commit = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(validator, "REVIEWED_CPU_COMMIT", reviewed_commit)
    monkeypatch.setattr(
        validator, "RUNTIME_IMPLEMENTATION_COMMIT", reviewed_commit
    )

    contract = {
        "schema": validator.SCHEMA,
        "case": 30,
        "split": "train",
        "reviewed_cpu_commit": reviewed_commit,
        "runtime_implementation_commit": reviewed_commit,
        "namespace": validator.NAMESPACE,
        "runtime_authorized": True,
        "gpu_launch_authorized": True,
        "one_case_only": True,
        "maximum_runtime_seconds": 600,
        "dataset_creation_authorized": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "holdout_opened": False,
        "runtime_authorization_token_sha256": hashlib.sha256(
            (validator.AUTHORIZATION + "\n").encode()
        ).hexdigest(),
        "identities": {
            name: _identity(repo, relative)
            for name, relative in identity_paths.items()
        },
    }
    contract_path = repo / validator.CONTRACT_RELATIVE_PATH
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add runtime contract")
    subprocess.run(["git", "init", "--bare", str(remote)], check=True)
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "main")

    cpu_sha = hashlib.sha256(
        (repo / validator.CPU_CONTRACT_RELATIVE_PATH).read_bytes()
    ).hexdigest()
    monkeypatch.setattr(
        validator,
        "validate_cpu_contract",
        lambda *_args, **_kwargs: {
            "cpu_contract_ready": True,
            "runtime_authorized": False,
            "contract_sha256": cpu_sha,
        },
    )
    return repo


def _validate(repo: Path, authorization_file: Path | None = None) -> dict:
    return validator.validate(
        repo / validator.CONTRACT_RELATIVE_PATH,
        repo,
        namespace=validator.NAMESPACE,
        authorization_file=authorization_file,
    )


def test_preflight_is_ready_but_not_runtime_authorized(admitted_repo: Path) -> None:
    result = _validate(admitted_repo)
    assert result["runtime_contract_ready"]
    assert not result["runtime_authorized"]
    assert result["passed"]


def test_exact_mode_0600_token_authorizes(admitted_repo: Path, tmp_path: Path) -> None:
    token = tmp_path / "authorization"
    token.write_text(validator.AUTHORIZATION + "\n")
    os.chmod(token, 0o600)
    result = _validate(admitted_repo, token)
    assert all(result["authorization_checks"].values())
    assert result["runtime_authorized"]
    assert result["passed"]


def test_forged_token_fails_closed(admitted_repo: Path, tmp_path: Path) -> None:
    token = tmp_path / "authorization"
    token.write_text("FORGED\n")
    os.chmod(token, 0o600)
    result = _validate(admitted_repo, token)
    assert not result["authorization_checks"]["authorization_hash_matches"]
    assert not result["runtime_authorized"]
    assert not result["passed"]


def test_alternate_contract_path_fails_closed(
    admitted_repo: Path, tmp_path: Path
) -> None:
    alternate = tmp_path / "alternate.json"
    alternate.write_bytes(
        (admitted_repo / validator.CONTRACT_RELATIVE_PATH).read_bytes()
    )
    result = validator.validate(
        alternate,
        admitted_repo,
        namespace=validator.NAMESPACE,
        authorization_file=None,
    )
    assert not result["checks"]["canonical_contract_path"]
    assert not result["runtime_contract_ready"]


def test_forged_reviewed_parent_fails_closed(admitted_repo: Path) -> None:
    contract_path = admitted_repo / validator.CONTRACT_RELATIVE_PATH
    contract = json.loads(contract_path.read_text())
    contract["reviewed_cpu_commit"] = "f" * 40
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    result = _validate(admitted_repo)
    assert not result["checks"]["reviewed_cpu_commit_matches"]
    assert not result["runtime_contract_ready"]


def test_unpushed_head_fails_closed(admitted_repo: Path) -> None:
    marker = admitted_repo / "tracked.txt"
    marker.write_text("new commit\n")
    _git(admitted_repo, "add", "tracked.txt")
    _git(admitted_repo, "commit", "-m", "unpushed")
    result = _validate(admitted_repo)
    assert not result["checks"]["head_matches_upstream"]
    assert not result["runtime_contract_ready"]
