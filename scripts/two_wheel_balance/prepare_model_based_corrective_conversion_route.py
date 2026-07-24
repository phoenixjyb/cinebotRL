#!/usr/bin/env python3
"""Prepare a generic, output-free conversion proposal for one admitted capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
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


SCHEMA = "cinebotrl_two_wheel_riser_corrective_conversion_proposal_v1"
EXPECTED_WSL_DISTRO = "Ubuntu"
CODE_PATHS = {
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
    "proposal_preparer": (
        "scripts/two_wheel_balance/"
        "prepare_model_based_corrective_conversion_route.py"
    ),
}


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


def _relative_file(repo: Path, path: Path) -> tuple[str | None, Path]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo.resolve()).as_posix()
    except ValueError:
        return None, resolved
    return relative, resolved


def _identity(repo: Path, path: Path) -> dict[str, Any]:
    relative, resolved = _relative_file(repo, path)
    exists = relative is not None and resolved.is_file()
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
    actual_blob = _git_blob(resolved) if exists else None
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
        "sha256": _sha256(resolved) if exists else None,
        "git_blob_sha1": actual_blob,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _repository_state(repo: Path) -> dict[str, Any]:
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    upstream = _git(repo, "rev-parse", "@{upstream}").stdout.strip()
    checks = {
        "head_matches_upstream": head == upstream,
        "tracked_worktree_clean": (
            _git(repo, "diff", "--quiet", check=False).returncode == 0
            and _git(repo, "diff", "--cached", "--quiet", check=False).returncode
            == 0
        ),
    }
    return {
        "head": head,
        "upstream": upstream,
        "checks": checks,
        "passed": all(checks.values()),
    }


def build_proposal(
    repo: Path,
    capture_path: Path,
    final_status_path: Path,
    *,
    case: int,
    split: str,
) -> dict[str, Any]:
    repo = repo.resolve()
    capture_path = capture_path.resolve()
    final_status_path = final_status_path.resolve()
    metadata, payload = convert_admitted_capture(
        capture_path,
        final_status_path,
        expected_case=case,
        expected_split=split,
    )
    identities = {
        "source_capture": _identity(repo, capture_path),
        "source_final_status": _identity(repo, final_status_path),
        **{
            name: _identity(repo, repo / relative)
            for name, relative in CODE_PATHS.items()
        },
    }
    repository = _repository_state(repo)
    observations = np.asarray(payload["observations"])
    actions = np.asarray(payload["actions"])
    requested_actions = np.asarray(payload["requested_actions_audit"])
    clipped = np.asarray(payload["command_clipped"], dtype=bool)
    elapsed = np.asarray(payload["elapsed_time_s"], dtype=np.float64)
    execution = np.asarray(payload["execution_time_s"], dtype=np.float64)
    source = np.asarray(payload["source_time_s"], dtype=np.float64)
    previous = observations[:, PREVIOUS_ACTION_INDICES]
    metrics = {
        "sample_count": int(metadata["sample_count"]),
        "observation_shape": list(observations.shape),
        "action_shape": list(actions.shape),
        "action_abs_max": np.max(np.abs(actions), axis=0).tolist(),
        "requested_action_abs_max": np.max(
            np.abs(requested_actions), axis=0
        ).tolist(),
        "clipped_rows": np.count_nonzero(clipped, axis=0).tolist(),
        "elapsed_clock_end_s": float(elapsed[-1]),
        "execution_clock_end_s": float(execution[-1]),
        "source_clock_end_s": float(source[-1]),
    }
    proposal_checks = {
        "case_split": metadata["case"] == case
        and metadata["split"] == split,
        "shape": observations.shape
        == (metadata["sample_count"], len(OBSERVATION_NAMES))
        and actions.shape
        == (metadata["sample_count"], len(ACTION_NAMES)),
        "effective_targets": metadata[
            "effective_actions_used_as_training_targets"
        ]
        is True
        and metadata["requested_actions_used_as_training_targets"] is False,
        "previous_action_rebuilt": metadata["previous_action_rebuilt"] is True
        and np.allclose(previous[0], 0.0, rtol=0.0, atol=1e-12)
        and np.array_equal(previous[1:], actions[:-1]),
        "clipping_audited": np.array_equal(
            clipped,
            np.abs(
                np.asarray(payload["requested_vs_effective_residual_delta"])
            )
            > 2e-7,
        ),
        "clocks_preserved": len(elapsed)
        == len(execution)
        == len(source)
        == metadata["sample_count"],
        "source_hashes": (
            metadata["source_capture_sha256"] == _sha256(capture_path)
            and metadata["source_final_status_sha256"]
            == _sha256(final_status_path)
        ),
        "learning_closed": metadata["merged_dataset_created"] is False
        and metadata["bc_authorized"] is False
        and metadata["ppo_authorized"] is False
        and metadata["training_started"] is False
        and metadata["valid_for_training"] is False,
    }
    proposal_ready = bool(
        repository["passed"]
        and all(row["passed"] for row in identities.values())
        and all(proposal_checks.values())
    )
    namespace = (
        f"model_based_corrective_case{case}_conversion_v1_cpu"
    )
    return {
        "schema": SCHEMA,
        "case": case,
        "split": split,
        "namespace": namespace,
        "repository": repository,
        "identities": identities,
        "proposal_checks": proposal_checks,
        "metrics": metrics,
        "dataset_contract": {
            "training_target_contract": TRAINING_TARGET_CONTRACT,
            "previous_action_contract": PREVIOUS_ACTION_CONTRACT,
            "observation_names": list(OBSERVATION_NAMES),
            "action_names": list(ACTION_NAMES),
            "expected_output_name": (
                f"case_{case:04d}_model_based_corrective_case_dataset_v1.npz"
            ),
            "source_capture_sha256": metadata["source_capture_sha256"],
            "source_final_status_sha256": metadata[
                "source_final_status_sha256"
            ],
            "source_runtime_commit": metadata["source_runtime_commit"],
            "source_plan_sha256": metadata["source_plan_sha256"],
            "source_corrective_profile_sha256": metadata[
                "source_corrective_profile_sha256"
            ],
            "source_paired_final_status_sha256": metadata[
                "source_paired_final_status_sha256"
            ],
        },
        "proposal_ready": proposal_ready,
        "conversion_execution_implemented": False,
        "authorization_token_issued": False,
        "conversion_authorized": False,
        "output_created": False,
        "merged_dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
        "passed": proposal_ready,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--final-status", type=Path, required=True)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_proposal(
        args.repo_root,
        args.capture,
        args.final_status,
        case=args.case,
        split=args.split,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite proposal: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
