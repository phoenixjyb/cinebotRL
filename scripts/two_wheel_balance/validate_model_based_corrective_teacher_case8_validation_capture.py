#!/usr/bin/env python3
"""Validate one out-of-band-authorized case-8 validation label capture."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
import re
import stat
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.two_wheel_balance.validate_model_based_corrective_teacher_case23_capture import (  # noqa: E402
    ADMISSION_SCHEMA,
    drive_profile_checks,
)
from scripts.two_wheel_balance.validate_model_based_corrective_teacher_case30_capture import (  # noqa: E402
    EXPECTED_CAPTURE,
    EXPECTED_HOLDOUT,
    EXPECTED_SCALES,
)
from scripts.two_wheel_balance.validate_model_based_corrective_teacher_case30_pair import (  # noqa: E402
    git,
    identity_row,
    sha256_file,
)


SCHEMA = "cinebotrl_two_wheel_riser_corrective_teacher_capture_contract_v2"
REVIEWED_PARENT = "07e48850c2e41140eb8751bc37d772a670d6de29"
NAMESPACE = (
    "20260728_model_based_corrective_teacher_"
    "case8_validation_capture_v1_coexistence"
)
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case8_validation_capture_contract_v1.json"
)
PAIR_SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_teacher_"
    "case8_validation_pair_final_v1"
)
EXPECTED_PROFILE_MAXIMUM_RESIDUALS = [
    0.015366143432421054,
    0.008147585946902451,
    0.0010025138153900227,
]
EXPECTED_EXECUTION = {
    "case": 8,
    "split": "validation",
    "rollout": "complete_model_based_planner_plus_corrective_teacher",
    "maximum_runtime_seconds": 600,
    "authorization_consumed_before_isaac": True,
    "fresh_namespace_required": True,
    "exclusive_isaac_runtime_required": True,
    "shared_windows_resource_admission_required": True,
    "resource_admission_before_token_consumption": True,
    "launch_minimum_windows_free_memory_gib": 5.0,
    "launch_minimum_gpu_free_memory_mib": 9_216,
    "cad_coexistence_allowed": True,
    "runtime_resource_monitor_required": True,
    "runtime_minimum_windows_free_memory_gib": 1.5,
    "runtime_minimum_gpu_free_memory_mib": 2_048,
    "dynamic_gate_required_before_save": True,
    "finalizer_reopens_archive": True,
    "capture_only": True,
}
REQUIRED_IDENTITIES = {
    "paired_final_status",
    "case8_plan",
    "perturbation_profile",
    "perturbation_runtime",
    "corrective_profile",
    "drive_profile_selection",
    "lqr_gains",
    "robot_build_audit",
    "robot_urdf",
    "robot_usd",
    "playback",
    "corrective_teacher_runtime",
    "corrective_capture_runtime",
    "capture_validator_runtime",
    "capture_finalizer_runtime",
    "resource_finalizer_runtime",
    "contract_validator",
    "preflight_wrapper",
    "shared_windows_resource_guard",
    "shared_windows_resource_monitor",
    "capture_finalizer",
}


def _load_bound_json(
    rows: dict[str, dict[str, object]],
    name: str,
) -> dict[str, object]:
    row = rows.get(name, {})
    if row.get("passed") is not True:
        return {}
    payload = json.loads(Path(str(row["path"])).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _authorization_checks(
    authorization_file: Path | None,
    authorization_sha256: str | None,
    repo: Path,
    contract_text: str,
) -> dict[str, bool]:
    present = authorization_file is not None and authorization_file.is_file()
    mode = (
        stat.S_IMODE(authorization_file.stat().st_mode)
        if present and authorization_file is not None
        else None
    )
    token_hash = (
        hashlib.sha256(authorization_file.read_bytes()).hexdigest()
        if present and authorization_file is not None
        else None
    )
    return {
        "authorization_file_present": present,
        "authorization_mode_0600": mode == 0o600,
        "authorization_not_symlink": bool(
            present
            and authorization_file is not None
            and not authorization_file.is_symlink()
        ),
        "authorization_file_outside_repository": bool(
            present
            and authorization_file is not None
            and not authorization_file.resolve().is_relative_to(repo)
        ),
        "authorization_hash_is_out_of_band": (
            isinstance(authorization_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", authorization_sha256) is not None
            and authorization_sha256 not in contract_text
        ),
        "authorization_hash_matches": (
            present
            and isinstance(authorization_sha256, str)
            and hmac.compare_digest(str(token_hash), authorization_sha256)
        ),
    }


def validate(
    contract_path: Path,
    repo: Path,
    *,
    namespace: str,
    authorization_file: Path | None = None,
    authorization_sha256: str | None = None,
    expected_case: int = 8,
    expected_namespace: str = NAMESPACE,
    contract_relative_path: str = CONTRACT_RELATIVE_PATH,
    reviewed_parent: str = REVIEWED_PARENT,
    plan_identity_name: str = "case8_plan",
    pair_schema: str = PAIR_SCHEMA,
    required_identities: set[str] = REQUIRED_IDENTITIES,
    expected_profile_maximum_residuals: list[float] = (
        EXPECTED_PROFILE_MAXIMUM_RESIDUALS
    ),
    expected_execution: dict[str, object] = EXPECTED_EXECUTION,
    validation_cases_opened: list[int] | None = None,
) -> dict[str, object]:
    if validation_cases_opened is None:
        validation_cases_opened = [8]
    repo = repo.resolve()
    contract_path = contract_path.resolve()
    canonical = (repo / contract_relative_path).resolve()
    contract_text = contract_path.read_text(encoding="utf-8")
    contract = json.loads(contract_text)
    identities = contract.get("identities", {})
    identities = identities if isinstance(identities, dict) else {}
    rows = {
        name: identity_row(repo, value)
        for name, value in identities.items()
        if isinstance(value, dict)
    }
    paired = _load_bound_json(rows, "paired_final_status")
    profile = _load_bound_json(rows, "corrective_profile")

    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    upstream = git(repo, "rev-parse", "@{u}").stdout.strip()
    tracked_clean = (
        git(repo, "diff", "--quiet", check=False).returncode == 0
        and git(repo, "diff", "--cached", "--quiet", check=False).returncode
        == 0
    )
    contract_tracked = (
        git(
            repo,
            "ls-files",
            "--error-unmatch",
            contract_relative_path,
            check=False,
        ).returncode
        == 0
    )
    contract_blob = git(
        repo, "hash-object", str(contract_path), check=False
    ).stdout.strip()
    committed_blob = git(
        repo,
        "rev-parse",
        f"HEAD:{contract_relative_path}",
        check=False,
    ).stdout.strip()
    reviewed_parent_is_ancestor = (
        git(
            repo,
            "merge-base",
            "--is-ancestor",
            reviewed_parent,
            head,
            check=False,
        ).returncode
        == 0
    )

    paired_metrics = paired.get("paired_admission", {})
    paired_metrics = (
        paired_metrics if isinstance(paired_metrics, dict) else {}
    )
    paired_checks = {
        "schema": paired.get("schema") == pair_schema,
        "case_split": paired.get("case") == expected_case
        and paired.get("split") == "validation",
        "passed": paired.get("passed") is True
        and paired.get("validation_pair_passed") is True,
        "measurable_improvement": (
            float(
                paired_metrics.get(
                    "position_p95_absolute_improvement_m",
                    0.0,
                )
            )
            >= 0.003
            and float(
                paired_metrics.get("position_p95_relative_improvement", 0.0)
            )
            >= 0.02
        ),
        "all_pair_checks": bool(paired_metrics.get("checks"))
        and all(paired_metrics.get("checks", {}).values()),
        "capture_closed": paired.get("label_capture_authorized") is False
        and paired.get("dataset_created") is False,
        "training_closed": paired.get("bc_authorized") is False
        and paired.get("ppo_authorized") is False
        and paired.get("training_started") is False,
    }
    profile_checks = {
        "schema": profile.get("schema")
        == "cinebotrl_two_wheel_riser_corrective_teacher_profile_v1",
        "case": profile.get("case") == expected_case,
        "limits": profile.get("maximum_residuals")
        == expected_profile_maximum_residuals,
    }
    authorization_requested = (
        authorization_file is not None or authorization_sha256 is not None
    )
    authorization_checks = _authorization_checks(
        authorization_file,
        authorization_sha256,
        repo,
        contract_text,
    )
    checks = {
        "schema": contract.get("schema") == SCHEMA,
        "case_split": contract.get("case") == expected_case
        and contract.get("split") == "validation",
        "reviewed_parent": contract.get("reviewed_parent_commit")
        == reviewed_parent
        and reviewed_parent_is_ancestor,
        "head_matches_upstream": head == upstream,
        "tracked_worktree_clean": tracked_clean,
        "canonical_contract": contract_path == canonical and contract_tracked,
        "contract_blob_matches_head": bool(contract_blob)
        and contract_blob == committed_blob,
        "fresh_namespace": contract.get("namespace")
        == namespace
        == expected_namespace
        and not (repo / "artifacts/two_wheel_riser" / namespace).exists(),
        "identity_set": set(identities) == required_identities
        and set(rows) == required_identities,
        "identity_hashes": bool(rows)
        and all(row.get("passed") for row in rows.values()),
        "tracked_blobs": all(
            isinstance(identities.get(name), dict)
            and bool(identities[name].get("git_blob_sha1"))
            for name in required_identities
        ),
        "paired_evidence": all(paired_checks.values()),
        "corrective_profile": all(profile_checks.values()),
        "plan_identity": identities.get(plan_identity_name, {}).get("sha256")
        == paired.get("candidate_metrics", {}).get("plan_sha256"),
        "residual_scales": contract.get("residual_action_scales")
        == EXPECTED_SCALES,
        "capture_contract": contract.get("capture_schema_contract")
        == EXPECTED_CAPTURE,
        "execution_contract": contract.get("execution_contract")
        == expected_execution,
        "holdout_closed": contract.get("holdout_cases") == EXPECTED_HOLDOUT
        and contract.get("holdout_opened") is False
        and contract.get("validation_cases_opened")
        == validation_cases_opened,
        "cpu_ready": contract.get("cpu_preflight_ready") is True,
        "tokenless_contract": (
            contract.get("runtime_authorized") is False
            and contract.get("gpu_launch_authorized") is False
            and contract.get("authorization_token_issued") is False
            and contract.get("runtime_authorization_token_sha256") == ""
            and contract.get("label_capture_authorized") is False
        ),
        "normalized_dataset_closed": contract.get(
            "dataset_creation_authorized"
        )
        is False,
        "training_closed": contract.get("bc_authorized") is False
        and contract.get("ppo_authorized") is False
        and contract.get("training_started") is False
        and contract.get("valid_for_training") is False,
        "authorization_state": (
            all(authorization_checks.values())
            if authorization_requested
            else authorization_file is None and authorization_sha256 is None
        ),
    }
    drive_checks = drive_profile_checks({"identities": rows})
    checks["active_drive_profile"] = all(drive_checks.values())
    cpu_passed = all(checks.values())
    runtime_authorized = bool(authorization_requested and cpu_passed)
    return {
        "schema": ADMISSION_SCHEMA,
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "contract_git_blob_sha1": contract_blob,
        "reviewed_parent_commit": reviewed_parent,
        "runtime_commit": head,
        "upstream_commit": upstream,
        "case": expected_case,
        "split": "validation",
        "namespace": namespace,
        "identities": rows,
        "paired_checks": paired_checks,
        "corrective_profile_checks": profile_checks,
        "drive_profile_checks": drive_checks,
        "checks": checks,
        "authorization_checks": authorization_checks,
        "authorization_file": (
            None
            if authorization_file is None
            else str(authorization_file.resolve())
        ),
        "authorization_consumed_before_isaac": runtime_authorized,
        "cpu_contract_ready": cpu_passed,
        "corrective_target_admission_passed": all(paired_checks.values()),
        "plan_sha256": identities.get(plan_identity_name, {}).get("sha256"),
        "corrective_profile_sha256": identities.get(
            "corrective_profile", {}
        ).get("sha256"),
        "paired_final_status_sha256": identities.get(
            "paired_final_status", {}
        ).get("sha256"),
        "runtime_authorized": runtime_authorized,
        "gpu_launch_authorized": runtime_authorized,
        "label_capture_authorized": runtime_authorized,
        "dataset_creation_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
        "passed": cpu_passed,
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
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
