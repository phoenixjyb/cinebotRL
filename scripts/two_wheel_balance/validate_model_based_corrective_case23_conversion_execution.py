#!/usr/bin/env python3
"""Validate the one-shot case-23 v4 CPU conversion execution contract."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.two_wheel_balance.audit_model_based_corrective_case23_conversion_readiness import (  # noqa: E402
    audit_readiness,
)


SCHEMA = "cinebotrl_two_wheel_riser_case23_conversion_execution_contract_v1"
ADMISSION_SCHEMA = (
    "cinebotrl_two_wheel_riser_case23_conversion_execution_admission_v1"
)
REVIEWED_PARENT = "9f104149762bf1a0f0691df0dc7bc6da53e3ff4e"
EXPECTED_WSL_DISTRO = "Ubuntu"
CASE = 23
SPLIT = "train"
NAMESPACE = "20260723_model_based_corrective_case23_conversion_v1_cpu"
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/"
    "model_based_corrective_case23_conversion_execution_contract_v1.json"
)
REVIEW_CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/"
    "model_based_corrective_case23_conversion_review_contract_v1.json"
)
OUTPUT_RELATIVE_PATH = (
    f"artifacts/two_wheel_riser/{NAMESPACE}/"
    "case_0023_model_based_corrective_case_dataset_v1.npz"
)
EXPECTED_IDENTITY_PATHS = {
    "conversion_review": (
        "docs/03_training/two_wheel_balance/"
        "evidence_20260723_case23_corrective_conversion_review_v1/summary.json"
    ),
    "conversion_review_contract": REVIEW_CONTRACT_RELATIVE_PATH,
    "source_capture": (
        "docs/03_training/two_wheel_balance/"
        "evidence_20260723_case23_corrective_capture_v4/"
        "capture/case_0023_corrective_teacher_capture_v2.npz"
    ),
    "source_final_status": (
        "docs/03_training/two_wheel_balance/"
        "evidence_20260723_case23_corrective_capture_v4/final_status.json"
    ),
    "converter_cli": (
        "scripts/two_wheel_balance/convert_model_based_corrective_capture.py"
    ),
    "dataset_module": (
        "src/rl_platform/tasks/two_wheel_balance/"
        "riser_model_based_corrective_dataset.py"
    ),
    "execution_validator": (
        "scripts/two_wheel_balance/"
        "validate_model_based_corrective_case23_conversion_execution.py"
    ),
    "execution_wrapper": (
        "scripts/two_wheel_balance/"
        "run_model_based_corrective_case23_conversion_v1.sh"
    ),
    "conversion_finalizer": (
        "scripts/two_wheel_balance/"
        "finalize_model_based_corrective_case23_conversion.py"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload
    ).hexdigest()


def _windows_path_to_wsl(value: str) -> str:
    if len(value) >= 3 and value[1:3] in (":\\", ":/"):
        return f"/mnt/{value[0].lower()}/{value[3:].replace(chr(92), '/')}"
    normalized = value.replace("/", "\\")
    unc_parts = normalized.split("\\")
    if (
        len(unc_parts) >= 5
        and unc_parts[:2] == ["", ""]
        and unc_parts[2].lower() in {"wsl.localhost", "wsl$"}
        and unc_parts[3].casefold() == EXPECTED_WSL_DISTRO.casefold()
    ):
        return "/" + "/".join(part for part in unc_parts[4:] if part)
    raise ValueError(f"cannot map Windows path into WSL: {value}")


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    if os.name == "nt":
        executable = str(
            Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32/wsl.exe"
        )
        command = [
            executable,
            "git",
            "-C",
            _windows_path_to_wsl(str(repo)),
            *args,
        ]
    else:
        command = ["git", "-C", str(repo), *args]
    return subprocess.run(
        command,
        check=check,
        capture_output=True,
        text=True,
    )


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _identity_rows(
    contract: Mapping[str, object], repo: Path
) -> dict[str, dict[str, object]]:
    identities = contract.get("identities")
    if not isinstance(identities, Mapping):
        return {}
    rows: dict[str, dict[str, object]] = {}
    for name, expected_path in EXPECTED_IDENTITY_PATHS.items():
        identity = identities.get(name)
        if not isinstance(identity, Mapping):
            continue
        relative_path = identity.get("path")
        path = (
            (repo / str(relative_path)).resolve()
            if isinstance(relative_path, str)
            else repo / "__missing__"
        )
        exists = path.is_relative_to(repo) and path.is_file()
        actual_sha = _sha256(path) if exists else None
        actual_blob = _git_blob(path) if exists else None
        checks = {
            "path_exact": relative_path == expected_path,
            "file_exists": exists,
            "sha256_matches": actual_sha == identity.get("sha256"),
            "git_blob_matches": actual_blob == identity.get("git_blob_sha1"),
        }
        rows[name] = {
            "path": str(path),
            "sha256": actual_sha,
            "git_blob_sha1": actual_blob,
            "checks": checks,
            "passed": all(checks.values()),
        }
    return rows


def _repository_checks(
    contract_path: Path, repo: Path
) -> tuple[dict[str, bool], dict[str, str]]:
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    upstream = _git(repo, "rev-parse", "@{upstream}").stdout.strip()
    committed_blob = _git(
        repo,
        "rev-parse",
        f"HEAD:{CONTRACT_RELATIVE_PATH}",
        check=False,
    ).stdout.strip()
    checks = {
        "canonical_contract": contract_path
        == (repo / CONTRACT_RELATIVE_PATH).resolve(),
        "contract_tracked": (
            _git(
                repo,
                "ls-files",
                "--error-unmatch",
                CONTRACT_RELATIVE_PATH,
                check=False,
            ).returncode
            == 0
        ),
        "contract_blob_matches_head": (
            contract_path.is_file() and _git_blob(contract_path) == committed_blob
        ),
        "head_matches_upstream": head == upstream,
        "tracked_worktree_clean": (
            _git(repo, "diff", "--quiet", check=False).returncode == 0
            and _git(repo, "diff", "--cached", "--quiet", check=False).returncode
            == 0
        ),
        "reviewed_parent_is_ancestor": (
            _git(
                repo,
                "merge-base",
                "--is-ancestor",
                REVIEWED_PARENT,
                head,
                check=False,
            ).returncode
            == 0
        ),
    }
    return checks, {"head": head, "upstream": upstream}


def _authorization_checks(
    authorization_file: Path | None,
    authorization_sha256: str | None,
    repo: Path,
    contract_path: Path,
) -> dict[str, bool]:
    present = authorization_file is not None and authorization_file.is_file()
    mode: int | None = None
    is_symlink = False
    if present and os.name == "nt":
        wsl_path = _windows_path_to_wsl(str(authorization_file.resolve()))
        executable = str(
            Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32/wsl.exe"
        )
        mode_result = subprocess.run(
            [executable, "stat", "-c", "%a", wsl_path],
            check=False,
            capture_output=True,
            text=True,
        )
        if mode_result.returncode == 0:
            mode = int(mode_result.stdout.strip(), 8)
        is_symlink = (
            subprocess.run(
                [executable, "test", "-L", wsl_path],
                check=False,
                capture_output=True,
                text=True,
            ).returncode
            == 0
        )
    elif present:
        mode = stat.S_IMODE(authorization_file.stat().st_mode)
        is_symlink = authorization_file.is_symlink()
    actual_hash = _sha256(authorization_file) if present else None
    return {
        "authorization_file_present": present,
        "authorization_mode_0600": mode == 0o600,
        "authorization_not_symlink": present and not is_symlink,
        "authorization_file_outside_repository": present
        and not authorization_file.resolve().is_relative_to(repo),
        "authorization_hash_is_out_of_band": (
            isinstance(authorization_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", authorization_sha256) is not None
            and authorization_sha256
            not in contract_path.read_text(encoding="utf-8")
        ),
        "authorization_hash_matches": (
            present
            and isinstance(authorization_sha256, str)
            and hmac.compare_digest(str(actual_hash), authorization_sha256)
        ),
    }


def validate(
    contract_path: Path,
    repo: Path,
    *,
    namespace: str,
    authorization_file: Path | None = None,
    authorization_sha256: str | None = None,
    repository_checks: Mapping[str, bool] | None = None,
    git_state: Mapping[str, str] | None = None,
    review_result: Mapping[str, object] | None = None,
) -> dict[str, object]:
    repo = repo.resolve()
    contract_path = contract_path.resolve()
    contract = _load_object(contract_path)
    identities = contract.get("identities")
    rows = _identity_rows(contract, repo)
    if repository_checks is None or git_state is None:
        repository_checks, git_state = _repository_checks(contract_path, repo)
    if review_result is None:
        review_result = audit_readiness(
            repo / REVIEW_CONTRACT_RELATIVE_PATH,
            repo,
        )
    committed_review = (
        _load_object(Path(rows["conversion_review"]["path"]))
        if rows.get("conversion_review", {}).get("passed") is True
        else {}
    )
    execution_contract = contract.get("execution_contract")
    contract_checks = {
        "schema": contract.get("schema") == SCHEMA,
        "reviewed_parent": contract.get("reviewed_parent_commit")
        == REVIEWED_PARENT,
        "case_split": contract.get("case") == CASE
        and contract.get("split") == SPLIT,
        "namespace": contract.get("namespace") == namespace == NAMESPACE,
        "identity_set": isinstance(identities, Mapping)
        and set(identities) == set(EXPECTED_IDENTITY_PATHS)
        and set(rows) == set(EXPECTED_IDENTITY_PATHS),
        "identity_hashes": bool(rows)
        and all(row["passed"] is True for row in rows.values()),
        "review_passed_closed": review_result.get("passed") is True
        and review_result.get("conversion_authorized") is False
        and review_result.get("output_created") is False
        and review_result.get("merged_dataset_created") is False
        and review_result.get("bc_authorized") is False
        and review_result.get("ppo_authorized") is False
        and review_result.get("training_started") is False
        and committed_review.get("passed") is True
        and committed_review.get("conversion_authorized") is False
        and committed_review.get("output_created") is False,
        "execution_contract": execution_contract
        == {
            "expected_case": CASE,
            "expected_split": SPLIT,
            "namespace": NAMESPACE,
            "output_relative_path": OUTPUT_RELATIVE_PATH,
            "one_use_authorization_required": True,
            "authorization_consumed_before_conversion": True,
            "fresh_namespace_required": True,
            "reopen_output_required": True,
            "cpu_conversion_only": True,
        },
        "fresh_namespace": not (
            repo / f"artifacts/two_wheel_riser/{NAMESPACE}"
        ).exists(),
        "authorization_state_closed": (
            contract.get("conversion_authorized") is False
            and contract.get("authorization_token_issued") is False
            and contract.get("authorization_token_sha256") == ""
            and contract.get("output_created") is False
            and contract.get("merged_dataset_created") is False
            and contract.get("bc_authorized") is False
            and contract.get("ppo_authorized") is False
            and contract.get("training_started") is False
            and contract.get("valid_for_training") is False
        ),
    }
    cpu_ready = all(repository_checks.values()) and all(contract_checks.values())
    authorization_checks = _authorization_checks(
        authorization_file,
        authorization_sha256,
        repo,
        contract_path,
    )
    conversion_authorized = bool(
        cpu_ready
        and authorization_file is not None
        and all(authorization_checks.values())
    )
    passed = cpu_ready and (
        authorization_file is None or conversion_authorized
    )
    return {
        "schema": ADMISSION_SCHEMA,
        "case": CASE,
        "split": SPLIT,
        "namespace": NAMESPACE,
        "git": dict(git_state),
        "repository_checks": dict(repository_checks),
        "contract_checks": contract_checks,
        "authorization_checks": authorization_checks,
        "identities": rows,
        "review_result_sha256": rows.get("conversion_review", {}).get(
            "sha256"
        ),
        "source_capture_sha256": rows.get("source_capture", {}).get("sha256"),
        "source_final_status_sha256": rows.get(
            "source_final_status", {}
        ).get("sha256"),
        "output_relative_path": OUTPUT_RELATIVE_PATH,
        "cpu_contract_ready": cpu_ready,
        "authorization_consumed_before_conversion": conversion_authorized,
        "conversion_authorized": conversion_authorized,
        "output_created": False,
        "valid_for_case_merge": False,
        "merged_dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        args.contract,
        args.repo_root,
        namespace=args.namespace,
        authorization_file=args.authorization_file,
        authorization_sha256=args.authorization_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
