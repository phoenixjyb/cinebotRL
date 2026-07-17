#!/usr/bin/env python3
"""Validate the CPU-only, non-authorizing case-74 recovery-v4 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


SCHEMA = "cinebotrl_case74_recovery_v4_contract_v1"
REVIEWED_PARENT = "ba8f4e0b44dc15a60d61b8353a208032727ad0ae"
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/case74_recovery_v4_contract_v1.json"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=check,
    )


def resolve_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def identity_row(repo: Path, payload: dict[str, object]) -> dict[str, object]:
    path = resolve_path(repo, str(payload.get("path", "")))
    exists = path.is_file()
    actual_sha = sha256_file(path) if exists else None
    expected_blob = payload.get("git_blob_sha1")
    actual_blob = None
    if exists and expected_blob is not None:
        result = git(repo, "hash-object", str(path), check=False)
        actual_blob = result.stdout.strip() if result.returncode == 0 else None
    checks = {
        "file_exists": exists,
        "sha256_matches": actual_sha == payload.get("sha256"),
        "git_blob_matches": expected_blob is None or actual_blob == expected_blob,
    }
    return {
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "git_blob_sha1": actual_blob,
        "checks": checks,
        "passed": all(checks.values()),
    }


def validate(
    contract_path: Path,
    repo: Path,
    *,
    namespace: str,
) -> dict[str, object]:
    repo = repo.resolve()
    contract_path = contract_path.resolve()
    expected_contract_path = (repo / CONTRACT_RELATIVE_PATH).resolve()
    canonical_contract_path = contract_path == expected_contract_path
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    identities = contract.get("identities", {})
    rows = {
        name: identity_row(repo, value)
        for name, value in identities.items()
        if isinstance(value, dict)
    }
    portfolio_row = rows.get("portfolio_manifest", {})
    plan_row = rows.get("case74_plan", {})
    portfolio = {}
    if portfolio_row.get("passed"):
        portfolio = json.loads(Path(portfolio_row["path"]).read_text(encoding="utf-8"))
    case74_item = next(
        (item for item in portfolio.get("items", []) if item.get("case") == 74),
        None,
    )
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    upstream = git(repo, "rev-parse", "@{u}").stdout.strip()
    contract_tracked = (
        git(repo, "ls-files", "--error-unmatch", CONTRACT_RELATIVE_PATH, check=False).returncode
        == 0
    )
    contract_blob = git(repo, "hash-object", str(contract_path), check=False).stdout.strip()
    committed_contract_blob = git(
        repo, "rev-parse", f"HEAD:{CONTRACT_RELATIVE_PATH}", check=False
    ).stdout.strip()
    parent_is_ancestor = (
        git(
            repo,
            "merge-base",
            "--is-ancestor",
            REVIEWED_PARENT,
            head,
            check=False,
        ).returncode
        == 0
    )
    tracked_clean = (
        git(repo, "diff", "--quiet", check=False).returncode == 0
        and git(repo, "diff", "--cached", "--quiet", check=False).returncode == 0
    )
    namespace_path = repo / "artifacts/two_wheel_riser" / namespace
    checks = {
        "schema_matches": contract.get("schema") == SCHEMA,
        "case_is_74": contract.get("case") == 74,
        "reviewed_parent_matches": contract.get("reviewed_controller_parent_commit")
        == REVIEWED_PARENT,
        "head_matches_upstream": head == upstream,
        "reviewed_parent_is_ancestor": parent_is_ancestor,
        "tracked_worktree_clean": tracked_clean,
        "canonical_contract_path": canonical_contract_path,
        "contract_is_tracked": contract_tracked,
        "contract_blob_matches_head": canonical_contract_path
        and bool(contract_blob)
        and contract_blob == committed_contract_blob,
        "namespace_matches": contract.get("namespace") == namespace,
        "namespace_is_fresh": not namespace_path.exists(),
        "tracking_profile_matches": contract.get("tracking_profile")
        == "riser_recovery_direction_v4",
        "recovery_range_matches": contract.get("recovery_error_range_m")
        == [0.2, 0.4],
        "source_identity_matches": portfolio.get("source_manifest_sha256")
        == contract.get("source_manifest_sha256"),
        "portfolio_case74_is_admitted": 74
        in portfolio.get("kinematic_accepted_cases", []),
        "portfolio_plan_identity_matches": case74_item is not None
        and case74_item.get("plan_sha256") == plan_row.get("sha256"),
        "all_identity_hashes_match": bool(rows)
        and len(rows) == len(identities)
        and all(row["passed"] for row in rows.values()),
        "obsolete_authorization_absent": contract.get("runtime_authorization_token")
        is None,
        "runtime_not_authorized": contract.get("runtime_authorized") is False,
        "gpu_launch_not_authorized": contract.get("gpu_launch_authorized") is False,
        "training_disabled": contract.get("residual_capture_authorized") is False
        and contract.get("bc_authorized") is False
        and contract.get("ppo_authorized") is False,
    }
    identity_passed = all(checks.values())
    return {
        "schema": "cinebotrl_case74_recovery_v4_contract_admission_v1",
        "contract": str(contract_path.resolve()),
        "contract_sha256": sha256_file(contract_path),
        "reviewed_controller_parent_commit": REVIEWED_PARENT,
        "runtime_commit": head,
        "upstream_commit": upstream,
        "contract_git_blob_sha1": contract_blob,
        "case": 74,
        "namespace": namespace,
        "tracking_profile": contract.get("tracking_profile"),
        "recovery_error_range_m": contract.get("recovery_error_range_m"),
        "source_manifest_sha256": contract.get("source_manifest_sha256"),
        "identities": rows,
        "checks": checks,
        "identity_passed": identity_passed,
        "runtime_authorized": False,
        "gate_c_execution_authorized": False,
        "residual_capture_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "passed": identity_passed,
        "valid_for_training": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        args.contract,
        args.repo_root,
        namespace=args.namespace,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
