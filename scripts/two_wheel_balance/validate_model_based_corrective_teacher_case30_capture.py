#!/usr/bin/env python3
"""Validate the no-token case-30 corrective-label capture preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.two_wheel_balance.validate_model_based_corrective_teacher_case30_pair import (
    git,
    identity_row,
    sha256_file,
    token_checks,
)


SCHEMA = "cinebotrl_two_wheel_riser_corrective_teacher_capture_contract_v2"
ADMISSION_SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_teacher_capture_admission_v2"
)
REVIEWED_PARENT = "f54db86768464c2d83feda9b2ec48c4ea2e732bf"
NAMESPACE = "20260722_model_based_corrective_teacher_case30_capture_v2_exclusive"
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case30_capture_contract_v2.json"
)
EXPECTED_HOLDOUT = [3, 5, 13, 19, 24]
EXPECTED_SCALES = [0.05, 0.05, 0.02]
EXPECTED_CAPTURE = {
    "observation_contract": (
        "current_physical_cam_link_pre_action_with_known_reference_lookahead_v1"
    ),
    "sample_alignment_contract": (
        "pre_action_observation_requested_and_effective_command_v2"
    ),
    "clock_contract": (
        "elapsed_execution_and_authoritative_source_time_separate_v1"
    ),
    "initialization_contract": "separate_and_excluded_from_capture_v1",
    "teacher_applied_to_commands": True,
    "safety_supervisor_contract": (
        "requested_teacher_intent_and_effective_applied_command_separate_v1"
    ),
    "training_target_contract": "effective_post_supervisor_residual_v1",
    "requested_and_effective_actions_recorded": True,
    "per_channel_command_clipping_recorded": True,
    "dynamic_quality_required_before_save": True,
    "maximum_normalized_action_exclusive": 0.95,
    "per_sample_plan_and_commit_identity": True,
    "amplitude_and_slew_flags": True,
    "perturbation_activity_flag": True,
    "normalized_training_dataset_created": False,
}
EXPECTED_EXECUTION = {
    "case": 30,
    "rollout": "complete_model_based_planner_plus_corrective_teacher",
    "maximum_runtime_seconds": 600,
    "authorization_consumed_before_isaac": True,
    "fresh_namespace_required": True,
    "exclusive_gpu_required": True,
    "dynamic_gate_required_before_save": True,
    "finalizer_reopens_archive": True,
    "capture_only": True,
}
REQUIRED_IDENTITIES = {
    "paired_final_status",
    "case30_plan",
    "perturbation_profile",
    "corrective_profile",
    "lqr_gains",
    "robot_build_audit",
    "robot_urdf",
    "robot_usd",
    "playback",
    "corrective_teacher_runtime",
    "corrective_capture_runtime",
    "contract_validator",
    "preflight_wrapper",
    "capture_finalizer",
}
TRACKED_IDENTITIES = REQUIRED_IDENTITIES - {
    "case30_plan",
    "perturbation_profile",
}


def validate(
    contract_path: Path,
    repo: Path,
    *,
    namespace: str,
    authorization_file: Path | None = None,
) -> dict[str, object]:
    repo = repo.resolve()
    contract_path = contract_path.resolve()
    canonical = (repo / CONTRACT_RELATIVE_PATH).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    identities = contract.get("identities", {})
    rows = {
        name: identity_row(repo, value)
        for name, value in identities.items()
        if isinstance(value, dict)
    }
    paired = {}
    paired_row = rows.get("paired_final_status", {})
    if paired_row.get("passed") is True:
        paired = json.loads(
            Path(str(paired_row["path"])).read_text(encoding="utf-8")
        )
    profile = {}
    profile_row = rows.get("corrective_profile", {})
    if profile_row.get("passed") is True:
        profile = json.loads(
            Path(str(profile_row["path"])).read_text(encoding="utf-8")
        )

    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    upstream = git(repo, "rev-parse", "@{u}").stdout.strip()
    tracked_clean = (
        git(repo, "diff", "--quiet", check=False).returncode == 0
        and git(repo, "diff", "--cached", "--quiet", check=False).returncode == 0
    )
    tracked = (
        git(repo, "ls-files", "--error-unmatch", CONTRACT_RELATIVE_PATH, check=False)
        .returncode
        == 0
    )
    blob = git(repo, "hash-object", str(contract_path), check=False).stdout.strip()
    committed_blob = git(
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
    paired_metrics = paired.get("paired_admission", {})
    paired_checks = {
        "schema": paired.get("schema")
        == "cinebotrl_two_wheel_riser_corrective_teacher_case30_pair_final_v1",
        "case_split": paired.get("case") == 30 and paired.get("split") == "train",
        "passed": paired.get("passed") is True
        and paired.get("corrective_target_admission_passed") is True,
        "measurable_improvement": (
            float(paired_metrics.get("position_p95_absolute_improvement_m", 0.0))
            >= 0.003
            and float(paired_metrics.get("position_p95_relative_improvement", 0.0))
            >= 0.02
        ),
        "capture_closed": paired.get("label_capture_authorized") is False
        and paired.get("dataset_created") is False,
        "training_closed": paired.get("bc_authorized") is False
        and paired.get("ppo_authorized") is False
        and paired.get("training_started") is False,
    }
    profile_checks = {
        "schema": profile.get("schema")
        == "cinebotrl_two_wheel_riser_corrective_teacher_profile_v1",
        "case": profile.get("case") == 30,
        "limits": profile.get("maximum_residuals") == [0.045, 0.045, 0.018],
    }
    checks = {
        "schema": contract.get("schema") == SCHEMA,
        "case_split": contract.get("case") == 30 and contract.get("split") == "train",
        "reviewed_parent": contract.get("reviewed_parent_commit") == REVIEWED_PARENT
        and parent_is_ancestor,
        "head_matches_upstream": head == upstream,
        "tracked_worktree_clean": tracked_clean,
        "canonical_contract": contract_path == canonical and tracked,
        "contract_blob_matches_head": bool(blob) and blob == committed_blob,
        "fresh_namespace": contract.get("namespace") == namespace == NAMESPACE
        and not (repo / "artifacts/two_wheel_riser" / namespace).exists(),
        "identity_set": set(identities) == REQUIRED_IDENTITIES
        and set(rows) == REQUIRED_IDENTITIES,
        "identity_hashes": bool(rows) and all(row.get("passed") for row in rows.values()),
        "tracked_blobs": all(
            isinstance(identities.get(name), dict)
            and bool(identities[name].get("git_blob_sha1"))
            for name in TRACKED_IDENTITIES
        ),
        "paired_evidence": all(paired_checks.values()),
        "corrective_profile": all(profile_checks.values()),
        "plan_identity": identities.get("case30_plan", {}).get("sha256")
        == paired.get("candidate_metrics", {}).get("plan_sha256"),
        "residual_scales": contract.get("residual_action_scales") == EXPECTED_SCALES,
        "capture_contract": contract.get("capture_schema_contract") == EXPECTED_CAPTURE,
        "execution_contract": contract.get("execution_contract")
        == EXPECTED_EXECUTION,
        "holdout_closed": contract.get("holdout_cases") == EXPECTED_HOLDOUT
        and contract.get("holdout_opened") is False
        and contract.get("validation_cases_opened") == [],
        "cpu_ready": contract.get("cpu_preflight_ready") is True,
        "authorization_state_consistent": (
            (
                contract.get("runtime_authorized") is False
                and contract.get("gpu_launch_authorized") is False
                and contract.get("authorization_token_issued") is False
                and contract.get("runtime_authorization_token_sha256") == ""
                and contract.get("label_capture_authorized") is False
            )
            or (
                contract.get("runtime_authorized") is True
                and contract.get("gpu_launch_authorized") is True
                and contract.get("authorization_token_issued") is True
                and isinstance(
                    contract.get("runtime_authorization_token_sha256"), str
                )
                and len(contract.get("runtime_authorization_token_sha256")) == 64
                and contract.get("label_capture_authorized") is True
            )
        ),
        "normalized_dataset_closed": contract.get("dataset_creation_authorized")
        is False,
        "training_closed": contract.get("bc_authorized") is False
        and contract.get("ppo_authorized") is False
        and contract.get("training_started") is False
        and contract.get("valid_for_training") is False,
    }
    cpu_passed = all(checks.values())
    authorization_issued = bool(contract.get("authorization_token_issued"))
    authorization_checks = token_checks(
        authorization_file, contract.get("runtime_authorization_token_sha256")
    )
    runtime_authorized = bool(
        cpu_passed
        and authorization_issued
        and authorization_file is not None
        and all(authorization_checks.values())
    )
    passed = cpu_passed and (
        authorization_file is None or runtime_authorized
    )
    return {
        "schema": ADMISSION_SCHEMA,
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "contract_git_blob_sha1": blob,
        "reviewed_parent_commit": REVIEWED_PARENT,
        "runtime_commit": head,
        "upstream_commit": upstream,
        "case": 30,
        "split": "train",
        "namespace": namespace,
        "identities": rows,
        "paired_checks": paired_checks,
        "corrective_profile_checks": profile_checks,
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
        "plan_sha256": identities.get("case30_plan", {}).get("sha256"),
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
        "passed": passed,
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
