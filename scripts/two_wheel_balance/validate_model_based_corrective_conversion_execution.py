#!/usr/bin/env python3
"""Validate a generic one-shot corrective case conversion execution."""

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
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_dataset import (  # noqa: E402
    PREVIOUS_ACTION_CONTRACT,
    TRAINING_TARGET_CONTRACT,
    convert_admitted_capture,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    ACTION_NAMES,
    OBSERVATION_NAMES,
    PREVIOUS_ACTION_INDICES,
)


SCHEMA = (
    "cinebotrl_two_wheel_riser_generic_corrective_conversion_execution_"
    "contract_v2"
)
ADMISSION_SCHEMA = (
    "cinebotrl_two_wheel_riser_generic_corrective_conversion_execution_"
    "admission_v2"
)
PROPOSAL_SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_conversion_proposal_v1"
)
REVIEWED_PARENT = "f891f6ed0389006b783b09e1d88a2dfd91d56f8d"
EXPECTED_WSL_DISTRO = "Ubuntu"
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/"
    "model_based_corrective_conversion_execution_contract_v2.json"
)
ALLOWED_SPLITS = ("train", "validation")
PROPOSAL_IDENTITY_NAMES = {
    "source_capture",
    "source_final_status",
    "converter_cli",
    "dataset_module",
    "capture_module",
    "proposal_preparer",
}
CODE_PATHS = {
    "proposal_preparer": (
        "scripts/two_wheel_balance/"
        "prepare_model_based_corrective_conversion_route.py"
    ),
    "converter_cli": (
        "scripts/two_wheel_balance/convert_model_based_corrective_capture.py"
    ),
    "dataset_module": (
        "src/rl_platform/tasks/two_wheel_balance/"
        "riser_model_based_corrective_dataset.py"
    ),
    "capture_module": (
        "src/rl_platform/tasks/two_wheel_balance/riser_corrective_capture.py"
    ),
    "contract_builder": (
        "scripts/two_wheel_balance/"
        "build_model_based_corrective_conversion_execution_contract.py"
    ),
    "execution_validator": (
        "scripts/two_wheel_balance/"
        "validate_model_based_corrective_conversion_execution.py"
    ),
    "execution_wrapper": (
        "scripts/two_wheel_balance/"
        "run_model_based_corrective_conversion_v2.sh"
    ),
    "conversion_finalizer": (
        "scripts/two_wheel_balance/"
        "finalize_model_based_corrective_conversion.py"
    ),
}


def namespace_for(case: int) -> str:
    return (
        f"model_based_corrective_case{case:04d}_"
        "conversion_execution_v2_cpu"
    )


def dataset_name_for(case: int) -> str:
    return f"case_{case:04d}_model_based_corrective_case_dataset_v1.npz"


def output_relative_path_for(case: int) -> str:
    namespace = namespace_for(case)
    return (
        f"artifacts/two_wheel_riser/{namespace}/"
        f"{dataset_name_for(case)}"
    )


def _windows_path_to_wsl(value: str) -> str:
    if len(value) >= 3 and value[1:3] in (":\\", ":/"):
        return f"/mnt/{value[0].lower()}/{value[3:].replace(chr(92), '/')}"
    normalized = value.replace("/", "\\")
    parts = normalized.split("\\")
    if (
        len(parts) >= 5
        and parts[:2] == ["", ""]
        and parts[2].lower() in {"wsl.localhost", "wsl$"}
        and parts[3].casefold() == EXPECTED_WSL_DISTRO.casefold()
    ):
        return "/" + "/".join(part for part in parts[4:] if part)
    raise ValueError(f"cannot map Windows path into WSL: {value}")


def _git(
    repo: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    if os.name == "nt":
        executable = str(
            Path(os.environ.get("WINDIR", r"C:\Windows"))
            / "System32/wsl.exe"
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


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _relative_path(repo: Path, path: Path) -> str | None:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return None


def _committed_identity(repo: Path, path: Path) -> dict[str, Any]:
    relative = _relative_path(repo, path)
    exists = relative is not None and path.is_file()
    tracked = bool(
        relative is not None
        and _git(
            repo,
            "ls-files",
            "--error-unmatch",
            relative,
            check=False,
        ).returncode
        == 0
    )
    committed_blob = (
        _git(
            repo,
            "rev-parse",
            f"HEAD:{relative}",
            check=False,
        ).stdout.strip()
        if tracked
        else ""
    )
    actual_blob = _git_blob(path) if exists else None
    checks = {
        "inside_repository": relative is not None,
        "file_exists": exists,
        "tracked": tracked,
        "committed_blob_matches": bool(
            tracked and actual_blob == committed_blob
        ),
    }
    return {
        "path": relative,
        "sha256": _sha256(path) if exists else None,
        "git_blob_sha1": actual_blob,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _expected_identity_rows(
    repo: Path,
    expected: Mapping[str, str],
    supplied: object,
) -> dict[str, dict[str, Any]]:
    if not isinstance(supplied, Mapping):
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for name, relative in expected.items():
        claimed = supplied.get(name)
        if not isinstance(claimed, Mapping):
            continue
        path = (repo / relative).resolve()
        actual = _committed_identity(repo, path)
        checks = {
            **actual["checks"],
            "path_exact": claimed.get("path") == relative,
            "sha256_matches": claimed.get("sha256") == actual["sha256"],
            "git_blob_matches": (
                claimed.get("git_blob_sha1") == actual["git_blob_sha1"]
            ),
        }
        rows[name] = {
            **actual,
            "checks": checks,
            "passed": all(checks.values()),
        }
    return rows


def _proposal_identity_rows(
    repo: Path,
    supplied: object,
) -> dict[str, dict[str, Any]]:
    if not isinstance(supplied, Mapping):
        return {}
    expected: dict[str, str] = {}
    for name in PROPOSAL_IDENTITY_NAMES:
        claimed = supplied.get(name)
        if isinstance(claimed, Mapping) and isinstance(
            claimed.get("path"), str
        ):
            expected[name] = str(claimed["path"])
    return _expected_identity_rows(repo, expected, supplied)


def _repository_checks(
    contract_path: Path,
    proposal_path: Path,
    repo: Path,
) -> tuple[dict[str, bool], dict[str, str], bool]:
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    upstream = _git(repo, "rev-parse", "@{upstream}").stdout.strip()
    contract_identity = _committed_identity(repo, contract_path)
    proposal_identity = _committed_identity(repo, proposal_path)
    proposal = _load_object(proposal_path)
    proposal_head = proposal.get("repository", {}).get("head")
    proposal_commit_is_ancestor = bool(
        isinstance(proposal_head, str)
        and re.fullmatch(r"[0-9a-f]{40}", proposal_head)
        and _git(
            repo,
            "merge-base",
            "--is-ancestor",
            proposal_head,
            head,
            check=False,
        ).returncode
        == 0
    )
    checks = {
        "canonical_contract": contract_path
        == (repo / CONTRACT_RELATIVE_PATH).resolve(),
        "contract_committed": contract_identity["passed"] is True,
        "proposal_committed": proposal_identity["passed"] is True,
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
        "proposal_commit_is_ancestor": proposal_commit_is_ancestor,
    }
    return checks, {"head": head, "upstream": upstream}, (
        proposal_commit_is_ancestor
    )


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
            Path(os.environ.get("WINDIR", r"C:\Windows"))
            / "System32/wsl.exe"
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


def _review_source(
    capture_path: Path,
    final_status_path: Path,
    *,
    case: int,
    split: str,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any]]:
    metadata, payload = convert_admitted_capture(
        capture_path,
        final_status_path,
        expected_case=case,
        expected_split=split,
    )
    actions = np.asarray(payload["actions"])
    requested = np.asarray(payload["requested_actions_audit"])
    observations = np.asarray(payload["observations"])
    clipped = np.asarray(payload["command_clipped"], dtype=bool)
    elapsed = np.asarray(payload["elapsed_time_s"], dtype=np.float64)
    execution = np.asarray(payload["execution_time_s"], dtype=np.float64)
    source = np.asarray(payload["source_time_s"], dtype=np.float64)
    checks = {
        "case_split": metadata["case"] == case
        and metadata["split"] == split,
        "shape": observations.shape
        == (metadata["sample_count"], len(OBSERVATION_NAMES))
        and actions.shape
        == (metadata["sample_count"], len(ACTION_NAMES)),
        "effective_only": metadata[
            "effective_actions_used_as_training_targets"
        ]
        is True
        and metadata["requested_actions_used_as_training_targets"] is False,
        "previous_action_rebuilt": (
            np.allclose(
                observations[0, PREVIOUS_ACTION_INDICES],
                0.0,
                rtol=0.0,
                atol=1e-12,
            )
            and np.array_equal(
                observations[1:, PREVIOUS_ACTION_INDICES],
                actions[:-1],
            )
        ),
        "clipping_audited": np.array_equal(
            clipped,
            np.abs(
                np.asarray(
                    payload["requested_vs_effective_residual_delta"]
                )
            )
            > 2e-7,
        ),
        "three_clocks_preserved": len(elapsed)
        == len(execution)
        == len(source)
        == metadata["sample_count"],
        "learning_closed": all(
            metadata[name] is False
            for name in (
                "merged_dataset_created",
                "bc_authorized",
                "ppo_authorized",
                "training_started",
                "valid_for_training",
            )
        ),
    }
    metrics = {
        "sample_count": int(metadata["sample_count"]),
        "observation_shape": list(observations.shape),
        "action_shape": list(actions.shape),
        "action_abs_max": np.max(np.abs(actions), axis=0).tolist(),
        "requested_action_abs_max": np.max(
            np.abs(requested), axis=0
        ).tolist(),
        "clipped_rows": np.count_nonzero(clipped, axis=0).tolist(),
        "elapsed_clock_end_s": float(elapsed[-1]),
        "execution_clock_end_s": float(execution[-1]),
        "source_clock_end_s": float(source[-1]),
    }
    return checks, metrics, metadata


def _proposal_checks(
    proposal: Mapping[str, Any],
    *,
    case: int,
    split: str,
    source_metrics: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    proposal_commit_is_ancestor: bool,
) -> dict[str, bool]:
    repository = proposal.get("repository")
    proposal_checks = proposal.get("proposal_checks")
    dataset_contract = proposal.get("dataset_contract")
    metrics = proposal.get("metrics")
    closed = (
        "authorization_token_issued",
        "conversion_authorized",
        "output_created",
        "merged_dataset_created",
        "bc_authorized",
        "ppo_authorized",
        "training_started",
        "valid_for_training",
    )
    return {
        "schema": proposal.get("schema") == PROPOSAL_SCHEMA,
        "case_split": proposal.get("case") == case
        and proposal.get("split") == split,
        "proposal_passed": proposal.get("passed") is True
        and proposal.get("proposal_ready") is True,
        "proposal_commit_is_ancestor": proposal_commit_is_ancestor,
        "proposal_repository_was_clean": isinstance(repository, Mapping)
        and repository.get("passed") is True
        and repository.get("head") == repository.get("upstream"),
        "proposal_checks": isinstance(proposal_checks, Mapping)
        and bool(proposal_checks)
        and all(value is True for value in proposal_checks.values()),
        "metrics_exact": isinstance(metrics, Mapping)
        and dict(metrics) == dict(source_metrics),
        "dataset_contract": isinstance(dataset_contract, Mapping)
        and dataset_contract.get("training_target_contract")
        == TRAINING_TARGET_CONTRACT
        and dataset_contract.get("previous_action_contract")
        == PREVIOUS_ACTION_CONTRACT
        and dataset_contract.get("expected_output_name")
        == dataset_name_for(case)
        and dataset_contract.get("source_capture_sha256")
        == source_metadata.get("source_capture_sha256")
        and dataset_contract.get("source_final_status_sha256")
        == source_metadata.get("source_final_status_sha256")
        and dataset_contract.get("source_runtime_commit")
        == source_metadata.get("source_runtime_commit")
        and dataset_contract.get("source_plan_sha256")
        == source_metadata.get("source_plan_sha256")
        and dataset_contract.get("source_corrective_profile_sha256")
        == source_metadata.get("source_corrective_profile_sha256")
        and dataset_contract.get("source_paired_final_status_sha256")
        == source_metadata.get("source_paired_final_status_sha256"),
        "proposal_stage_only": proposal.get(
            "conversion_execution_implemented"
        )
        is False,
        "authorization_state_closed": all(
            proposal.get(name) is False for name in closed
        ),
    }


def validate(
    contract_path: Path,
    proposal_path: Path,
    repo: Path,
    *,
    authorization_file: Path | None = None,
    authorization_sha256: str | None = None,
    repository_checks: Mapping[str, bool] | None = None,
    git_state: Mapping[str, str] | None = None,
    proposal_commit_is_ancestor: bool | None = None,
) -> dict[str, Any]:
    repo = repo.resolve()
    contract_path = contract_path.resolve()
    proposal_path = proposal_path.resolve()
    contract = _load_object(contract_path)
    proposal = _load_object(proposal_path)
    if (
        repository_checks is None
        or git_state is None
        or proposal_commit_is_ancestor is None
    ):
        (
            repository_checks,
            git_state,
            proposal_commit_is_ancestor,
        ) = _repository_checks(contract_path, proposal_path, repo)

    case_value = proposal.get("case")
    split_value = proposal.get("split")
    case = case_value if type(case_value) is int else -1
    split = split_value if isinstance(split_value, str) else ""
    namespace = namespace_for(case) if case > 0 else ""
    output_relative_path = (
        output_relative_path_for(case) if case > 0 else ""
    )
    contract_rows = _expected_identity_rows(
        repo,
        CODE_PATHS,
        contract.get("identities"),
    )
    contract_file_identity = _committed_identity(repo, contract_path)
    proposal_file_identity = _committed_identity(repo, proposal_path)
    proposal_rows = _proposal_identity_rows(
        repo,
        proposal.get("identities"),
    )

    source_checks: dict[str, bool] = {}
    source_metrics: dict[str, Any] = {}
    source_metadata: dict[str, Any] = {}
    source_error: str | None = None
    try:
        source_checks, source_metrics, source_metadata = _review_source(
            repo / str(proposal_rows["source_capture"]["path"]),
            repo / str(proposal_rows["source_final_status"]["path"]),
            case=case,
            split=split,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        source_error = str(exc)

    proposal_validation_checks = _proposal_checks(
        proposal,
        case=case,
        split=split,
        source_metrics=source_metrics,
        source_metadata=source_metadata,
        proposal_commit_is_ancestor=bool(proposal_commit_is_ancestor),
    )
    execution_contract = contract.get("execution_contract")
    contract_checks = {
        "schema": contract.get("schema") == SCHEMA,
        "canonical_contract": contract_path
        == (repo / CONTRACT_RELATIVE_PATH).resolve(),
        "contract_file_committed": contract_file_identity["passed"] is True,
        "proposal_file_committed": proposal_file_identity["passed"] is True,
        "reviewed_parent": contract.get("reviewed_parent_commit")
        == REVIEWED_PARENT,
        "allowed_splits": contract.get("allowed_splits")
        == list(ALLOWED_SPLITS),
        "identity_set": isinstance(contract.get("identities"), Mapping)
        and set(contract["identities"]) == set(CODE_PATHS)
        and set(contract_rows) == set(CODE_PATHS),
        "identity_hashes": bool(contract_rows)
        and all(row["passed"] is True for row in contract_rows.values()),
        "proposal_identity_set": isinstance(
            proposal.get("identities"), Mapping
        )
        and set(proposal["identities"]) == PROPOSAL_IDENTITY_NAMES
        and set(proposal_rows) == PROPOSAL_IDENTITY_NAMES,
        "proposal_identity_hashes": bool(proposal_rows)
        and all(row["passed"] is True for row in proposal_rows.values()),
        "case_split": case > 0 and split in ALLOWED_SPLITS,
        "source_ready": source_error is None
        and bool(source_checks)
        and all(source_checks.values()),
        "proposal_ready": all(proposal_validation_checks.values()),
        "execution_contract": execution_contract
        == {
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
        "fresh_namespace": bool(namespace)
        and not (repo / f"artifacts/two_wheel_riser/{namespace}").exists(),
        "authorization_state_closed": (
            contract.get("contract_ready") is True
            and contract.get("conversion_execution_implemented") is True
            and contract.get("conversion_authorized") is False
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
    cpu_ready = all(repository_checks.values()) and all(
        contract_checks.values()
    )
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
        "case": case,
        "split": split,
        "namespace": namespace,
        "git": dict(git_state),
        "repository_checks": dict(repository_checks),
        "contract_checks": contract_checks,
        "proposal_checks": proposal_validation_checks,
        "authorization_checks": authorization_checks,
        "contract_identities": contract_rows,
        "proposal_identities": proposal_rows,
        "contract": contract_file_identity,
        "proposal": proposal_file_identity,
        "source_checks": source_checks,
        "source_metrics": source_metrics,
        "source_error": source_error,
        "source_capture_relative_path": proposal_rows.get(
            "source_capture", {}
        ).get("path"),
        "source_final_status_relative_path": proposal_rows.get(
            "source_final_status", {}
        ).get("path"),
        "source_capture_sha256": source_metadata.get(
            "source_capture_sha256"
        ),
        "source_final_status_sha256": source_metadata.get(
            "source_final_status_sha256"
        ),
        "output_relative_path": output_relative_path,
        "dataset_name": dataset_name_for(case) if case > 0 else "",
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
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--authorization-file", type=Path)
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        args.contract,
        args.proposal,
        args.repo_root,
        authorization_file=args.authorization_file,
        authorization_sha256=args.authorization_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
