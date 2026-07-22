#!/usr/bin/env python3
"""Validate one-use case-78 deterministic shadow-label authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

try:
    from .validate_case78_shadow_label_contract import (
        EXPECTED_IMPLEMENTATION,
        EXPECTED_NAMESPACE,
        EXPECTED_REVIEWED_PARENT,
        load_json,
        semantic_checks,
        verify_identity,
    )
    from .validate_riser_case78_dynamic_contract import git, identity_row, sha256_file
except ImportError:
    from validate_case78_shadow_label_contract import (
        EXPECTED_IMPLEMENTATION,
        EXPECTED_NAMESPACE,
        EXPECTED_REVIEWED_PARENT,
        load_json,
        semantic_checks,
        verify_identity,
    )
    from validate_riser_case78_dynamic_contract import git, identity_row, sha256_file


SCHEMA = "cinebotrl_two_wheel_riser_case78_shadow_label_runtime_authorization_v1"
NAMESPACE = EXPECTED_NAMESPACE
RUNTIME_IMPLEMENTATION_COMMIT = "54d171247ab25153b07e9dd286a4ed3db7d25bcc"
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/case78_shadow_label_runtime_authorization_v1.json"
)
CPU_CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/case78_shadow_label_cpu_contract_v1.json"
)
AUTHORIZATION = "AUTHORIZED_CASE78_DETERMINISTIC_SHADOW_LABEL_V1"
REQUIRED_IDENTITIES = {
    "cpu_contract",
    "runtime_summarizer",
    "runtime_validator",
    "runtime_wrapper",
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
    contract = load_json(contract_path)
    identities = contract.get("identities", {})
    rows = {
        name: identity_row(repo, payload)
        for name, payload in identities.items()
        if isinstance(payload, dict)
    }
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    upstream = git(repo, "rev-parse", "@{u}").stdout.strip()
    contract_blob = git(repo, "hash-object", str(contract_path), check=False).stdout.strip()
    committed_blob = git(
        repo, "rev-parse", f"HEAD:{CONTRACT_RELATIVE_PATH}", check=False
    ).stdout.strip()
    cpu_path = repo / CPU_CONTRACT_RELATIVE_PATH
    cpu_contract = load_json(cpu_path)
    cpu_identities = {
        name: verify_identity(repo, payload)
        for name, payload in cpu_contract.get("identities", {}).items()
    }
    cpu_semantics = semantic_checks(
        cpu_contract,
        load_json(Path(cpu_identities["residual_action_audit"]["path"])),
        load_json(Path(cpu_identities["split_admission"]["path"])),
        load_json(Path(cpu_identities["case78_gate"]["path"])),
        load_json(Path(cpu_identities["case78_final_status"]["path"])),
    )
    checks = {
        "schema_matches": contract.get("schema") == SCHEMA,
        "case_and_split": contract.get("case") == 78
        and contract.get("current_split") == "validation",
        "reviewed_cpu_commit": contract.get("reviewed_cpu_commit")
        == "ea126c340ed0ca586fdccd8bc11f9cd9a12efd31",
        "reviewed_cpu_commit_is_ancestor": git(
            repo,
            "merge-base",
            "--is-ancestor",
            "ea126c340ed0ca586fdccd8bc11f9cd9a12efd31",
            head,
            check=False,
        ).returncode
        == 0,
        "runtime_implementation_commit": contract.get(
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
        "cpu_lineage_preserved": cpu_contract.get("reviewed_parent_commit")
        == EXPECTED_REVIEWED_PARENT
        and cpu_contract.get("implementation_commit") == EXPECTED_IMPLEMENTATION
        and all(cpu_semantics.values()),
        "head_matches_upstream": head == upstream,
        "tracked_worktree_clean": git(repo, "diff", "--quiet", check=False).returncode
        == 0
        and git(repo, "diff", "--cached", "--quiet", check=False).returncode == 0,
        "canonical_contract_path": contract_path == canonical_path,
        "contract_blob_matches_head": bool(contract_blob)
        and contract_blob == committed_blob,
        "namespace_matches": contract.get("namespace") == namespace == NAMESPACE,
        "namespace_is_fresh": not (
            repo / "artifacts/two_wheel_riser" / namespace
        ).exists(),
        "identity_set_exact": set(identities) == REQUIRED_IDENTITIES,
        "all_identity_hashes_match": bool(rows)
        and len(rows) == len(identities)
        and all(row["passed"] for row in rows.values()),
        "cpu_contract_identity": rows.get("cpu_contract", {}).get("passed") is True
        and rows["cpu_contract"]["sha256"] == sha256_file(cpu_path),
        "runtime_scope": contract.get("runtime_authorized") is True
        and contract.get("gpu_launch_authorized") is True
        and contract.get("shadow_measurement_authorized") is True
        and contract.get("one_case_only") is True
        and contract.get("maximum_runtime_seconds") == 5400
        and contract.get("heartbeat_interval_policy_steps") == 2000,
        "controller_scope": contract.get("trajectory_command_source")
        == "deterministic_teacher"
        and contract.get("maximum_camera_lever_arm_correction_m") == 0.1
        and contract.get("residual_action_scales") == [0.35, 0.4, 0.1]
        and contract.get("residual_policy") is None
        and contract.get("zero_policy_action") is False,
        "learning_closed": contract.get("label_capture_authorized") is False
        and contract.get("dataset_creation_authorized") is False
        and contract.get("bc_authorized") is False
        and contract.get("ppo_authorized") is False
        and contract.get("holdout_opened") is False,
        "token_hash_pinned": contract.get("runtime_authorization_token_sha256")
        == hashlib.sha256((AUTHORIZATION + "\n").encode()).hexdigest(),
    }
    runtime_contract_ready = all(checks.values())
    authorization_checks = token_checks(
        authorization_file, contract.get("runtime_authorization_token_sha256")
    )
    runtime_authorized = runtime_contract_ready and all(authorization_checks.values())
    return {
        "schema": "cinebotrl_two_wheel_riser_case78_shadow_label_runtime_admission_v1",
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "contract_git_blob_sha1": contract_blob,
        "runtime_commit": head,
        "upstream_commit": upstream,
        "case": 78,
        "current_split": "validation",
        "namespace": namespace,
        "runtime_implementation_commit": RUNTIME_IMPLEMENTATION_COMMIT,
        "identities": rows,
        "checks": checks,
        "authorization_checks": authorization_checks,
        "runtime_contract_ready": runtime_contract_ready,
        "runtime_authorized": runtime_authorized,
        "gpu_launch_authorized": runtime_authorized,
        "shadow_measurement_authorized": runtime_authorized,
        "label_capture_authorized": False,
        "dataset_creation_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "valid_for_training": False,
        "passed": runtime_authorized if authorization_file is not None else runtime_contract_ready,
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
