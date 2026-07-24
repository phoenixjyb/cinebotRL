#!/usr/bin/env python3
"""Build the closed generic corrective conversion execution contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from validate_model_based_corrective_conversion_execution import (
    ALLOWED_SPLITS,
    CODE_PATHS,
    REVIEWED_PARENT,
    SCHEMA,
    _git,
    _git_blob,
    _sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _identity(repo: Path, relative: str) -> dict[str, str]:
    path = (repo / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if (
        _git(
            repo,
            "ls-files",
            "--error-unmatch",
            relative,
            check=False,
        ).returncode
        != 0
    ):
        raise ValueError(f"identity is not tracked: {relative}")
    committed_blob = _git(
        repo,
        "rev-parse",
        f"HEAD:{relative}",
    ).stdout.strip()
    actual_blob = _git_blob(path)
    if committed_blob != actual_blob:
        raise ValueError(f"identity differs from HEAD: {relative}")
    return {
        "path": relative,
        "sha256": _sha256(path),
        "git_blob_sha1": actual_blob,
    }


def _repository_state(repo: Path) -> dict[str, Any]:
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    upstream = _git(repo, "rev-parse", "@{upstream}").stdout.strip()
    checks = {
        "head_matches_upstream": head == upstream,
        "tracked_worktree_clean": (
            _git(repo, "diff", "--quiet", check=False).returncode == 0
            and _git(
                repo,
                "diff",
                "--cached",
                "--quiet",
                check=False,
            ).returncode
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
    return {
        "head": head,
        "upstream": upstream,
        "checks": checks,
        "passed": all(checks.values()),
    }


def build_contract(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    repository = _repository_state(repo)
    identities = {
        name: _identity(repo, relative)
        for name, relative in CODE_PATHS.items()
    }
    contract_ready = repository["passed"] and len(identities) == len(
        CODE_PATHS
    )
    return {
        "schema": SCHEMA,
        "reviewed_parent_commit": REVIEWED_PARENT,
        "implementation_commit": repository["head"],
        "allowed_splits": list(ALLOWED_SPLITS),
        "identities": identities,
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
        "contract_ready": contract_ready,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite contract: {args.output}")
    result = build_contract(args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["contract_ready"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
