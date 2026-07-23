#!/usr/bin/env python3
"""Audit the sealed case-23 v4 capture for a later CPU-only conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_corrective_capture import (  # noqa: E402
    load_corrective_capture,
)
from rl_platform.tasks.two_wheel_balance.riser_model_based_corrective_dataset import (  # noqa: E402
    convert_admitted_capture,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    PREVIOUS_ACTION_INDICES,
)


SCHEMA = "cinebotrl_two_wheel_riser_case23_conversion_review_contract_v1"
RESULT_SCHEMA = "cinebotrl_two_wheel_riser_case23_conversion_readiness_v1"
CASE = 23
SPLIT = "train"
REVIEWED_PARENT = "46e121aaad944e3adc806a5a541b3a80a67c9655"
RUNTIME_COMMIT = "31bb9afbf3e9ce6c17e0fc1d2f06b5990e130d1c"
EXPECTED_SAMPLE_COUNT = 3273
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/"
    "model_based_corrective_case23_conversion_review_contract_v1.json"
)
OUTPUT_RELATIVE_PATH = (
    "artifacts/two_wheel_riser/"
    "20260723_model_based_corrective_case23_conversion_v1_cpu/"
    "case_0023_model_based_corrective_case_dataset_v1.npz"
)
EXPECTED_IDENTITY_PATHS = {
    "source_capture": (
        "docs/03_training/two_wheel_balance/"
        "evidence_20260723_case23_corrective_capture_v4/"
        "capture/case_0023_corrective_teacher_capture_v2.npz"
    ),
    "source_final_status": (
        "docs/03_training/two_wheel_balance/"
        "evidence_20260723_case23_corrective_capture_v4/final_status.json"
    ),
    "source_dynamic_gate": (
        "docs/03_training/two_wheel_balance/"
        "evidence_20260723_case23_corrective_capture_v4/case_0023.json"
    ),
    "source_admission": (
        "docs/03_training/two_wheel_balance/"
        "evidence_20260723_case23_corrective_capture_v4/admission.json"
    ),
    "source_capture_contract": (
        "docs/03_training/two_wheel_balance/"
        "evidence_20260723_case23_corrective_capture_v4/contract.json"
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
    "reviewer": (
        "scripts/two_wheel_balance/"
        "audit_model_based_corrective_case23_conversion_readiness.py"
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
    prefix = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(prefix + payload).hexdigest()


def _windows_path_to_wsl(value: str) -> str:
    if len(value) < 3 or value[1:3] not in (":\\", ":/"):
        raise ValueError(f"cannot map Windows repository path into WSL: {value}")
    drive = value[0].lower()
    suffix = value[3:].replace("\\", "/")
    return f"/mnt/{drive}/{suffix}"


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    if os.name == "nt":
        windows_root = os.environ.get("WINDIR", r"C:\Windows")
        executable = str(Path(windows_root) / "System32/wsl.exe")
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
        inside_repo = path.is_relative_to(repo)
        exists = inside_repo and path.is_file()
        actual_sha = _sha256(path) if exists else None
        actual_blob = _git_blob(path) if exists else None
        checks = {
            "path_exact": relative_path == expected_path,
            "inside_repository": inside_repo,
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
    canonical = (repo / CONTRACT_RELATIVE_PATH).resolve()
    contract_tracked = (
        _git(
            repo,
            "ls-files",
            "--error-unmatch",
            CONTRACT_RELATIVE_PATH,
            check=False,
        ).returncode
        == 0
    )
    committed_blob = _git(
        repo,
        "rev-parse",
        f"HEAD:{CONTRACT_RELATIVE_PATH}",
        check=False,
    ).stdout.strip()
    parent_is_ancestor = (
        _git(
            repo,
            "merge-base",
            "--is-ancestor",
            REVIEWED_PARENT,
            head,
            check=False,
        ).returncode
        == 0
    )
    checks = {
        "canonical_contract": contract_path.resolve() == canonical,
        "contract_tracked": contract_tracked,
        "contract_blob_matches_head": (
            contract_path.is_file()
            and _git_blob(contract_path) == committed_blob
        ),
        "head_matches_upstream": head == upstream,
        "tracked_worktree_clean": (
            _git(repo, "diff", "--quiet", check=False).returncode == 0
            and _git(repo, "diff", "--cached", "--quiet", check=False).returncode
            == 0
        ),
        "reviewed_parent_is_ancestor": parent_is_ancestor,
    }
    return checks, {"head": head, "upstream": upstream}


def _review_source(
    capture_path: Path, final_status_path: Path
) -> tuple[dict[str, bool], dict[str, object]]:
    metadata, converted = convert_admitted_capture(
        capture_path,
        final_status_path,
        expected_case=CASE,
        expected_split=SPLIT,
    )
    capture_metadata, capture = load_corrective_capture(
        capture_path,
        expected_case=CASE,
        expected_split=SPLIT,
    )
    observations = np.asarray(converted["observations"])
    source_observations = np.asarray(capture["observations"])
    actions = np.asarray(converted["actions"])
    requested_actions = np.asarray(converted["requested_actions_audit"])
    effective_source = np.asarray(
        capture["effective_corrective_normalized_actions"]
    )
    requested_source = np.asarray(
        capture["requested_corrective_normalized_actions"]
    )
    previous_delta = (
        observations[:, PREVIOUS_ACTION_INDICES]
        - source_observations[:, PREVIOUS_ACTION_INDICES]
    )
    non_previous = np.ones(observations.shape[1], dtype=bool)
    non_previous[list(PREVIOUS_ACTION_INDICES)] = False
    clipped = np.asarray(converted["command_clipped"], dtype=bool)
    checks = {
        "case_split": metadata["case"] == CASE
        and metadata["split"] == SPLIT
        and capture_metadata["case"] == CASE
        and capture_metadata["split"] == SPLIT,
        "sample_count": metadata["sample_count"] == EXPECTED_SAMPLE_COUNT
        and len(actions) == EXPECTED_SAMPLE_COUNT,
        "runtime_commit": metadata["source_runtime_commit"] == RUNTIME_COMMIT,
        "source_capture_hash": metadata["source_capture_sha256"]
        == _sha256(capture_path),
        "source_final_status_hash": metadata["source_final_status_sha256"]
        == _sha256(final_status_path),
        "effective_actions_exact": np.array_equal(actions, effective_source),
        "requested_actions_audit_exact": np.array_equal(
            requested_actions, requested_source
        ),
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
        "non_previous_observations_exact": np.array_equal(
            observations[:, non_previous],
            source_observations[:, non_previous],
        ),
        "elapsed_clock_exact": np.array_equal(
            converted["elapsed_time_s"], capture["elapsed_time_s"]
        ),
        "execution_clock_exact": np.array_equal(
            converted["execution_time_s"], capture["execution_time_s"]
        ),
        "source_clock_exact": np.array_equal(
            converted["source_time_s"], capture["source_time_s"]
        ),
        "case_ids_exact": np.array_equal(
            converted["case_ids"], capture["case_ids"]
        ),
        "effective_only_training_target": metadata[
            "requested_actions_used_as_training_targets"
        ]
        is False
        and metadata["effective_actions_used_as_training_targets"] is True,
        "learning_closed": metadata["merged_dataset_created"] is False
        and metadata["bc_authorized"] is False
        and metadata["ppo_authorized"] is False
        and metadata["training_started"] is False
        and metadata["valid_for_training"] is False,
    }
    metrics = {
        "sample_count": int(len(actions)),
        "observation_shape": list(observations.shape),
        "action_abs_max": np.max(np.abs(actions), axis=0).tolist(),
        "requested_action_abs_max": np.max(
            np.abs(requested_actions), axis=0
        ).tolist(),
        "clipped_rows": np.count_nonzero(clipped, axis=0).tolist(),
        "rebuilt_previous_rows_changed": int(
            np.count_nonzero(np.any(np.abs(previous_delta) > 0.0, axis=1))
        ),
        "rebuilt_previous_max_abs_delta": float(
            np.max(np.abs(previous_delta))
        ),
        "elapsed_clock_end_s": float(converted["elapsed_time_s"][-1]),
        "execution_clock_end_s": float(converted["execution_time_s"][-1]),
        "source_clock_end_s": float(converted["source_time_s"][-1]),
        "source_plan_sha256": metadata["source_plan_sha256"],
        "source_corrective_profile_sha256": metadata[
            "source_corrective_profile_sha256"
        ],
        "source_paired_final_status_sha256": metadata[
            "source_paired_final_status_sha256"
        ],
    }
    return checks, metrics


def audit_readiness(
    contract_path: Path,
    repo: Path,
    *,
    repository_checks: Mapping[str, bool] | None = None,
    git_state: Mapping[str, str] | None = None,
) -> dict[str, object]:
    repo = repo.resolve()
    contract_path = contract_path.resolve()
    contract = _load_object(contract_path)
    identities = contract.get("identities")
    rows = _identity_rows(contract, repo)
    if repository_checks is None or git_state is None:
        repository_checks, git_state = _repository_checks(contract_path, repo)
    output_path = (repo / OUTPUT_RELATIVE_PATH).resolve()
    conversion_contract = contract.get("conversion_contract")
    conversion_checks = {
        "schema": contract.get("schema") == SCHEMA,
        "reviewed_parent": contract.get("reviewed_parent_commit")
        == REVIEWED_PARENT,
        "case_split": contract.get("case") == CASE
        and contract.get("split") == SPLIT,
        "identity_set": isinstance(identities, Mapping)
        and set(identities) == set(EXPECTED_IDENTITY_PATHS)
        and set(rows) == set(EXPECTED_IDENTITY_PATHS),
        "identity_hashes": bool(rows)
        and all(row["passed"] is True for row in rows.values()),
        "conversion_contract": conversion_contract
        == {
            "expected_case": CASE,
            "expected_split": SPLIT,
            "output_relative_path": OUTPUT_RELATIVE_PATH,
            "execute_requested": False,
            "output_created": False,
            "conversion_authorized": False,
            "valid_for_case_merge_after_conversion_only": True,
            "merged_dataset_created": False,
            "bc_authorized": False,
            "ppo_authorized": False,
            "training_started": False,
            "valid_for_training": False,
        },
        "fresh_output": not output_path.exists(),
    }
    source_checks: dict[str, bool] = {}
    metrics: dict[str, object] = {}
    source_error: str | None = None
    if all(conversion_checks.values()) and all(repository_checks.values()):
        try:
            source_checks, metrics = _review_source(
                Path(rows["source_capture"]["path"]),
                Path(rows["source_final_status"]["path"]),
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            source_error = str(exc)
    passed = (
        all(repository_checks.values())
        and all(conversion_checks.values())
        and bool(source_checks)
        and all(source_checks.values())
        and source_error is None
        and not output_path.exists()
    )
    return {
        "schema": RESULT_SCHEMA,
        "case": CASE,
        "split": SPLIT,
        "reviewed_parent_commit": REVIEWED_PARENT,
        "runtime_commit": RUNTIME_COMMIT,
        "git": dict(git_state),
        "repository_checks": dict(repository_checks),
        "contract_checks": conversion_checks,
        "source_checks": source_checks,
        "source_error": source_error,
        "identities": rows,
        "prospective_dataset_metrics": metrics,
        "output": str(output_path),
        "output_created": False,
        "execute_requested": False,
        "conversion_authorized": False,
        "prospective_case_dataset_valid_for_merge": passed,
        "merged_dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
        "decision": (
            "ready_for_separate_case23_v4_cpu_conversion_authorization"
            if passed
            else "do_not_convert_case23_v4"
        ),
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_readiness(args.contract, args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
