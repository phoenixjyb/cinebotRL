#!/usr/bin/env python3
"""Validate one-use case-78 deterministic runtime authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

try:
    from .validate_riser_case78_dynamic_contract import (
        validate as validate_cpu_contract,
    )
except ImportError:
    from validate_riser_case78_dynamic_contract import (
        validate as validate_cpu_contract,
    )


SCHEMA = "cinebotrl_two_wheel_riser_case78_runtime_authorization_v1"
REVIEWED_CPU_COMMIT = "dacb00b222cce46289789f12e576764fbc326d2e"
RUNTIME_IMPLEMENTATION_COMMIT = "d3b541dfcf1b8a9f2f099fecb8b4a13dc8e0088e"
NAMESPACE = "20260721_case78_dynamic_qualification_v1_exclusive"
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/case78_runtime_authorization_v1.json"
)
CPU_CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/case78_dynamic_cpu_contract_v1.json"
)
AUTHORIZATION = "AUTHORIZED_CASE78_DETERMINISTIC_CANARY_V1"
REQUIRED_IDENTITIES = {
    "cpu_contract",
    "runtime_summarizer",
    "runtime_validator",
    "runtime_wrapper",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=check,
    )


def identity_row(repo: Path, payload: dict[str, object]) -> dict[str, object]:
    raw_path = Path(str(payload.get("path", "")))
    path = raw_path if raw_path.is_absolute() else repo / raw_path
    exists = path.is_file()
    actual_sha = sha256_file(path) if exists else None
    expected_blob = payload.get("git_blob_sha1")
    actual_blob = None
    committed_blob = None
    if exists and expected_blob is not None:
        result = git(repo, "hash-object", str(path), check=False)
        actual_blob = result.stdout.strip() if result.returncode == 0 else None
        try:
            relative = path.resolve().relative_to(repo)
        except ValueError:
            relative = None
        if relative is not None:
            result = git(
                repo,
                "rev-parse",
                f"HEAD:{relative.as_posix()}",
                check=False,
            )
            committed_blob = (
                result.stdout.strip() if result.returncode == 0 else None
            )
    checks = {
        "file_exists": exists,
        "sha256_matches": actual_sha == payload.get("sha256"),
        "git_blob_matches": expected_blob is None or actual_blob == expected_blob,
        "committed_git_blob_matches": expected_blob is None
        or committed_blob == expected_blob,
    }
    return {
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "git_blob_sha1": actual_blob,
        "committed_git_blob_sha1": committed_blob,
        "checks": checks,
        "passed": all(checks.values()),
    }


def token_checks(path: Path | None, expected_sha: str | None) -> dict[str, bool]:
    if path is None:
        return {
            "authorization_file_present": False,
            "authorization_mode_0600": False,
            "authorization_not_symlink": False,
            "authorization_hash_matches": False,
            "authorization_content_matches": False,
        }
    exists = path.is_file()
    mode = (os.stat(path).st_mode & 0o777) if exists else None
    content = path.read_text(encoding="utf-8").strip() if exists else None
    return {
        "authorization_file_present": exists,
        "authorization_mode_0600": mode == 0o600,
        "authorization_not_symlink": exists and not path.is_symlink(),
        "authorization_hash_matches": exists
        and sha256_file(path) == expected_sha,
        "authorization_content_matches": content == AUTHORIZATION,
    }


def validate(
    contract_path: Path,
    repo: Path,
    *,
    namespace: str,
    authorization_file: Path | None,
) -> dict[str, object]:
    repo = repo.resolve()
    contract_path = contract_path.resolve()
    canonical_path = (repo / CONTRACT_RELATIVE_PATH).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    identities = contract.get("identities", {})
    rows = {
        name: identity_row(repo, payload)
        for name, payload in identities.items()
        if isinstance(payload, dict)
    }
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    upstream = git(repo, "rev-parse", "@{u}").stdout.strip()
    contract_blob = git(
        repo, "hash-object", str(contract_path), check=False
    ).stdout.strip()
    committed_blob = git(
        repo, "rev-parse", f"HEAD:{CONTRACT_RELATIVE_PATH}", check=False
    ).stdout.strip()
    cpu_contract = repo / CPU_CONTRACT_RELATIVE_PATH
    cpu_admission = validate_cpu_contract(
        cpu_contract,
        repo,
        namespace=namespace,
    )
    checks = {
        "schema_matches": contract.get("schema") == SCHEMA,
        "case_is_78_unused": contract.get("case") == 78
        and contract.get("current_split") == "unused",
        "reviewed_cpu_commit_matches": contract.get("reviewed_cpu_commit")
        == REVIEWED_CPU_COMMIT,
        "reviewed_cpu_commit_is_ancestor": git(
            repo,
            "merge-base",
            "--is-ancestor",
            REVIEWED_CPU_COMMIT,
            head,
            check=False,
        ).returncode
        == 0,
        "runtime_implementation_commit_matches": contract.get(
            "runtime_implementation_commit"
        )
        == RUNTIME_IMPLEMENTATION_COMMIT,
        "runtime_implementation_commit_is_ancestor": git(
            repo,
            "merge-base",
            "--is-ancestor",
            RUNTIME_IMPLEMENTATION_COMMIT,
            head,
            check=False,
        ).returncode
        == 0,
        "head_matches_upstream": head == upstream,
        "tracked_worktree_clean": (
            git(repo, "diff", "--quiet", check=False).returncode == 0
            and git(repo, "diff", "--cached", "--quiet", check=False).returncode
            == 0
        ),
        "canonical_contract_path": contract_path == canonical_path,
        "contract_is_tracked": git(
            repo,
            "ls-files",
            "--error-unmatch",
            CONTRACT_RELATIVE_PATH,
            check=False,
        ).returncode
        == 0,
        "contract_blob_matches_head": bool(contract_blob)
        and contract_blob == committed_blob,
        "namespace_matches": contract.get("namespace") == namespace == NAMESPACE,
        "namespace_is_fresh": not (
            repo / "artifacts/two_wheel_riser" / namespace
        ).exists(),
        "cpu_contract_ready": cpu_admission.get("cpu_contract_ready") is True
        and cpu_admission.get("runtime_authorized") is False,
        "cpu_contract_identity": rows.get("cpu_contract", {}).get("passed")
        is True
        and rows["cpu_contract"]["sha256"]
        == cpu_admission.get("contract_sha256"),
        "identity_set_exact": set(identities) == REQUIRED_IDENTITIES,
        "all_identity_hashes_match": bool(rows)
        and len(rows) == len(identities)
        and all(row["passed"] for row in rows.values()),
        "runtime_scope": contract.get("runtime_authorized") is True
        and contract.get("gpu_launch_authorized") is True
        and contract.get("dynamic_qualification_authorized") is True
        and contract.get("one_case_only") is True
        and contract.get("maximum_runtime_seconds") == 900,
        "split_and_learning_closed": contract.get("split_change_authorized")
        is False
        and contract.get("dataset_creation_authorized") is False
        and contract.get("dagger_authorized") is False
        and contract.get("bc_authorized") is False
        and contract.get("ppo_authorized") is False
        and contract.get("holdout_opened") is False,
        "token_hash_pinned": contract.get("runtime_authorization_token_sha256")
        == hashlib.sha256((AUTHORIZATION + "\n").encode()).hexdigest(),
    }
    runtime_contract_ready = all(checks.values())
    authorization_checks = token_checks(
        authorization_file,
        contract.get("runtime_authorization_token_sha256"),
    )
    runtime_authorized = runtime_contract_ready and all(
        authorization_checks.values()
    )
    return {
        "schema": "cinebotrl_two_wheel_riser_case78_runtime_admission_v1",
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "contract_git_blob_sha1": contract_blob,
        "runtime_commit": head,
        "upstream_commit": upstream,
        "case": 78,
        "current_split": "unused",
        "namespace": namespace,
        "cpu_admission": cpu_admission,
        "identities": rows,
        "checks": checks,
        "authorization_checks": authorization_checks,
        "runtime_contract_ready": runtime_contract_ready,
        "runtime_authorized": runtime_authorized,
        "gpu_launch_authorized": runtime_authorized,
        "dynamic_qualification_authorized": runtime_authorized,
        "split_change_authorized": False,
        "dataset_creation_authorized": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "valid_for_training": False,
        "passed": (
            runtime_authorized
            if authorization_file is not None
            else runtime_contract_ready
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        args.contract,
        args.repo_root,
        namespace=args.namespace,
        authorization_file=args.authorization_file,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
