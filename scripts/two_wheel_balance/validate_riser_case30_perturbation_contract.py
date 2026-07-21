#!/usr/bin/env python3
"""Validate the CPU-only case-30 perturbation measurement contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


SCHEMA = "cinebotrl_two_wheel_riser_case30_perturbation_contract_v1"
REVIEWED_PARENT = "6a8e7a091dc128727bd5e6cded5154eb48c8874c"
NAMESPACE = "20260721_case30_perturbation_measurement_v1_exclusive"
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/case30_perturbation_measurement_contract_v1.json"
)
EXPECTED_PROFILE = {
    "schema": "cinebotrl_two_wheel_riser_deterministic_wrench_pulse_v1",
    "case": 30,
    "start_phase_time_s": 15.666592937559889,
    "duration_steps": 20,
    "force_body_x_n": 20.0,
    "application_height_m": 0.5,
}
EXPECTED_ACTION_SCALES = [0.35, 0.4, 0.1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(
    repo: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
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
    canonical_path = (repo / CONTRACT_RELATIVE_PATH).resolve()
    canonical_contract_path = contract_path == canonical_path
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    identities = contract.get("identities", {})
    rows = {
        name: identity_row(repo, value)
        for name, value in identities.items()
        if isinstance(value, dict)
    }
    proposal = {}
    profile = {}
    if rows.get("proposal", {}).get("passed"):
        proposal = json.loads(Path(rows["proposal"]["path"]).read_text())
    if rows.get("profile", {}).get("passed"):
        profile = json.loads(Path(rows["profile"]["path"]).read_text())

    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    upstream = git(repo, "rev-parse", "@{u}").stdout.strip()
    contract_tracked = (
        git(
            repo,
            "ls-files",
            "--error-unmatch",
            CONTRACT_RELATIVE_PATH,
            check=False,
        ).returncode
        == 0
    )
    contract_blob = git(
        repo, "hash-object", str(contract_path), check=False
    ).stdout.strip()
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
        and git(repo, "diff", "--cached", "--quiet", check=False).returncode
        == 0
    )
    namespace_path = repo / "artifacts/two_wheel_riser" / namespace
    checks = {
        "schema_matches": contract.get("schema") == SCHEMA,
        "case_is_30_train": contract.get("case") == 30
        and contract.get("split") == "train",
        "reviewed_parent_matches": contract.get("reviewed_parent_commit")
        == REVIEWED_PARENT,
        "reviewed_parent_is_ancestor": parent_is_ancestor,
        "head_matches_upstream": head == upstream,
        "tracked_worktree_clean": tracked_clean,
        "canonical_contract_path": canonical_contract_path,
        "contract_is_tracked": contract_tracked,
        "contract_blob_matches_head": canonical_contract_path
        and bool(contract_blob)
        and contract_blob == committed_contract_blob,
        "namespace_matches": contract.get("namespace") == namespace == NAMESPACE,
        "namespace_is_fresh": not namespace_path.exists(),
        "profile_payload_matches": profile == EXPECTED_PROFILE
        and contract.get("profile_payload") == EXPECTED_PROFILE,
        "action_scales_match": contract.get("residual_action_scales")
        == EXPECTED_ACTION_SCALES,
        "proposal_contract": proposal.get("schema")
        == "cinebotrl_two_wheel_riser_case30_perturbation_proposal_v1"
        and proposal.get("decision_status")
        == "cpu_only_profile_not_runtime_authorization"
        and proposal.get("case") == 30
        and proposal.get("split") == "train"
        and proposal.get("profile", {}).get("sha256")
        == rows.get("profile", {}).get("sha256")
        and proposal.get("runtime_authorized") is False
        and proposal.get("dataset_created") is False,
        "all_identity_hashes_match": bool(rows)
        and len(rows) == len(identities)
        and all(row["passed"] for row in rows.values()),
        "cpu_preflight_only": contract.get("cpu_preflight_ready") is True
        and contract.get("runtime_authorized") is False
        and contract.get("gpu_launch_authorized") is False,
        "no_runtime_token": "runtime_authorization_token_sha256" not in contract,
        "training_disabled": contract.get("dataset_creation_authorized") is False
        and contract.get("dagger_authorized") is False
        and contract.get("bc_authorized") is False
        and contract.get("ppo_authorized") is False,
        "holdout_closed": contract.get("holdout_opened") is False,
    }
    passed = all(checks.values())
    return {
        "schema": "cinebotrl_two_wheel_riser_case30_perturbation_admission_v1",
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "contract_git_blob_sha1": contract_blob,
        "reviewed_parent_commit": REVIEWED_PARENT,
        "runtime_commit": head,
        "upstream_commit": upstream,
        "case": 30,
        "split": "train",
        "namespace": namespace,
        "identities": rows,
        "checks": checks,
        "cpu_contract_ready": passed,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "measurement_authorized": False,
        "dataset_creation_authorized": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "valid_for_training": False,
        "passed": passed,
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
