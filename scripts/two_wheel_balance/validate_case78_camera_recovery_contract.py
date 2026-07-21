#!/usr/bin/env python3
"""Validate the CPU-only case-78 camera-recovery candidate contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .validate_riser_case78_dynamic_contract import (
        EXPECTED_CONTROLLER,
        EXPECTED_PLAN,
        EXPECTED_THRESHOLDS,
        git,
        identity_row,
        sha256_file,
    )
except ImportError:
    from validate_riser_case78_dynamic_contract import (
        EXPECTED_CONTROLLER,
        EXPECTED_PLAN,
        EXPECTED_THRESHOLDS,
        git,
        identity_row,
        sha256_file,
    )


SCHEMA = "cinebotrl_two_wheel_riser_case78_camera_recovery_cpu_contract_v1"
REVIEWED_PARENT = "ccb7875d4e056eb49d67de00cee44d7a588ff181"
NAMESPACE = "20260722_case78_camera_recovery_v1_exclusive"
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/case78_camera_recovery_cpu_contract_v1.json"
)
EXPECTED_IDENTITIES = {
    "base_cpu_contract",
    "case78_plan",
    "heartbeat_helper",
    "lqr_gains",
    "playback",
    "preflight_wrapper",
    "prior_gate",
    "prior_final_status",
    "recovery_audit",
    "robot_usd",
    "tracking",
    "validator",
}
EXPECTED_RECOVERY_CONTROLLER = {
    **EXPECTED_CONTROLLER,
    "enable_camera_error_recovery_governor": True,
    "camera_recovery_error_start_m": 0.13,
    "camera_recovery_error_full_m": 0.155,
    "minimum_camera_recovery_scale": 0.20,
}


def semantic_checks(
    contract: dict[str, object],
    prior_gate: dict[str, object],
    prior_final: dict[str, object],
    recovery_audit: dict[str, object],
) -> dict[str, bool]:
    results = prior_gate.get("results")
    prior_result = (
        results[0]
        if isinstance(results, list)
        and len(results) == 1
        and isinstance(results[0], dict)
        else {}
    )
    prior_checks = prior_result.get("checks", {})
    failed_checks = sorted(
        name for name, passed in prior_checks.items() if not passed
    )
    return {
        "schema_matches": contract.get("schema") == SCHEMA,
        "case_is_78_unused": contract.get("case") == 78
        and contract.get("current_split") == "unused",
        "namespace_matches": contract.get("namespace") == NAMESPACE,
        "reviewed_parent_matches": contract.get("reviewed_parent_commit")
        == REVIEWED_PARENT,
        "identity_set_exact": set(contract.get("identities", {}))
        == EXPECTED_IDENTITIES,
        "plan_contract_unchanged": contract.get("plan_contract") == EXPECTED_PLAN,
        "controller_change_exact": contract.get("controller_arguments")
        == EXPECTED_RECOVERY_CONTROLLER,
        "thresholds_unchanged": contract.get("dynamic_gate_thresholds")
        == EXPECTED_THRESHOLDS,
        "prior_result_is_narrow_failure": failed_checks
        == ["position_p95_bounded"]
        and prior_result.get("completed_phase_time_s")
        == prior_result.get("execution_duration_s")
        and prior_result.get("termination") is None
        and prior_result.get("position_error_p95_m") == 0.162649892749212
        and prior_result.get("position_error_max_m") == 0.22962387152256802,
        "prior_final_rejected": prior_final.get("dynamic_qualification_passed")
        is False
        and prior_final.get("case78_validation_admitted") is False
        and prior_final.get("split_changed") is False,
        "recovery_audit_supports_candidate": recovery_audit.get(
            "candidate_supported_for_bounded_canary"
        )
        is True
        and recovery_audit.get("offline_trace_estimate_is_physical_proof")
        is False
        and recovery_audit.get("projected_candidate_steps") == 87392
        and recovery_audit.get("maximum_steps") == 115381,
        "one_case_heartbeat_no_capture": contract.get("one_case_only") is True
        and contract.get("maximum_runtime_seconds") == 5400
        and contract.get("heartbeat_interval_policy_steps") == 2000
        and contract.get("dataset_creation_authorized") is False,
        "cpu_only": contract.get("cpu_preflight_ready") is True
        and contract.get("runtime_authorized") is False
        and contract.get("gpu_launch_authorized") is False
        and contract.get("dynamic_qualification_authorized") is False,
        "no_runtime_token": "runtime_authorization_token_sha256" not in contract,
        "learning_closed": contract.get("split_change_authorized") is False
        and contract.get("holdout_opened") is False
        and contract.get("dagger_authorized") is False
        and contract.get("bc_authorized") is False
        and contract.get("ppo_authorized") is False,
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
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    identities = contract.get("identities", {})
    rows = {
        name: identity_row(repo, payload)
        for name, payload in identities.items()
        if isinstance(payload, dict)
    }

    def read_identity(name: str) -> dict[str, object]:
        row = rows.get(name, {})
        if row.get("passed") is not True:
            return {}
        return json.loads(Path(str(row["path"])).read_text(encoding="utf-8"))

    checks = semantic_checks(
        contract,
        read_identity("prior_gate"),
        read_identity("prior_final_status"),
        read_identity("recovery_audit"),
    )
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    upstream = git(repo, "rev-parse", "@{u}").stdout.strip()
    contract_blob = git(
        repo, "hash-object", str(contract_path), check=False
    ).stdout.strip()
    committed_blob = git(
        repo, "rev-parse", f"HEAD:{CONTRACT_RELATIVE_PATH}", check=False
    ).stdout.strip()
    checks.update({
        "head_matches_upstream": head == upstream,
        "tracked_worktree_clean": (
            git(repo, "diff", "--quiet", check=False).returncode == 0
            and git(repo, "diff", "--cached", "--quiet", check=False).returncode
            == 0
        ),
        "reviewed_parent_is_ancestor": git(
            repo,
            "merge-base",
            "--is-ancestor",
            REVIEWED_PARENT,
            head,
            check=False,
        ).returncode
        == 0,
        "canonical_contract_path": contract_path == canonical_path,
        "contract_is_tracked": git(
            repo,
            "ls-files",
            "--error-unmatch",
            CONTRACT_RELATIVE_PATH,
            check=False,
        ).returncode
        == 0,
        "contract_blob_matches_head": bool(contract_blob)
        and contract_blob == committed_blob,
        "namespace_argument_matches": namespace == NAMESPACE,
        "namespace_is_fresh": not (
            repo / "artifacts/two_wheel_riser" / namespace
        ).exists(),
        "all_identity_hashes_match": bool(rows)
        and len(rows) == len(identities)
        and all(row["passed"] for row in rows.values()),
    })
    passed = all(checks.values())
    return {
        "schema": "cinebotrl_two_wheel_riser_case78_camera_recovery_cpu_admission_v1",
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "contract_git_blob_sha1": contract_blob,
        "runtime_commit": head,
        "upstream_commit": upstream,
        "case": 78,
        "current_split": "unused",
        "namespace": namespace,
        "identities": rows,
        "checks": checks,
        "cpu_contract_ready": passed,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "dynamic_qualification_authorized": False,
        "split_change_authorized": False,
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

