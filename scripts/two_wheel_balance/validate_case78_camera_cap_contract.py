#!/usr/bin/env python3
"""Validate the CPU-only case-78 camera-correction-cap candidate contract."""

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


SCHEMA = "cinebotrl_two_wheel_riser_case78_camera_cap_cpu_contract_v1"
REVIEWED_PARENT = "5751df8cd1b1825dc8ffba4828664fa03992d4a8"
NAMESPACE = "20260722_case78_camera_cap_v1_exclusive"
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/case78_camera_cap_cpu_contract_v1.json"
)
EXPECTED_IDENTITIES = {
    "base_cpu_contract",
    "baseline_final_status",
    "baseline_gate",
    "camera_cap_audit",
    "case78_plan",
    "heartbeat_helper",
    "lqr_gains",
    "playback",
    "preflight_wrapper",
    "recovery_gate",
    "recovery_outcome_audit",
    "robot_usd",
    "tracking",
    "validator",
}
EXPECTED_CAP_CONTROLLER = {
    **EXPECTED_CONTROLLER,
    "maximum_camera_lever_arm_correction_m": 0.10,
}


def semantic_checks(
    contract: dict[str, object],
    baseline_gate: dict[str, object],
    baseline_final: dict[str, object],
    recovery_outcome: dict[str, object],
    camera_cap_audit: dict[str, object],
) -> dict[str, bool]:
    results = baseline_gate.get("results")
    baseline = (
        results[0]
        if isinstance(results, list)
        and len(results) == 1
        and isinstance(results[0], dict)
        else {}
    )
    failed = sorted(
        name for name, passed in baseline.get("checks", {}).items() if not passed
    )
    controller = contract.get("controller_arguments")
    controller_delta = {
        name
        for name in set(EXPECTED_CONTROLLER) | set(controller or {})
        if EXPECTED_CONTROLLER.get(name) != (controller or {}).get(name)
    }
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
        "controller_change_exact": controller == EXPECTED_CAP_CONTROLLER
        and controller_delta == {"maximum_camera_lever_arm_correction_m"},
        "thresholds_unchanged": contract.get("dynamic_gate_thresholds")
        == EXPECTED_THRESHOLDS,
        "baseline_is_narrow_failure": failed == ["position_p95_bounded"]
        and baseline.get("completed_phase_time_s")
        == baseline.get("execution_duration_s")
        and baseline.get("termination") is None
        and baseline.get("position_error_p95_m") == 0.162649892749212
        and baseline.get("position_error_max_m") == 0.22962387152256802,
        "baseline_final_rejected": baseline_final.get(
            "dynamic_qualification_passed"
        )
        is False
        and baseline_final.get("case78_validation_admitted") is False
        and baseline_final.get("split_changed") is False,
        "recovery_candidate_rejected": recovery_outcome.get("audit_passed")
        is True
        and recovery_outcome.get("camera_recovery_candidate_rejected") is True
        and recovery_outcome.get("runtime_authorized") is False,
        "camera_cap_candidate_supported_cpu_only": camera_cap_audit.get(
            "audit_passed"
        )
        is True
        and camera_cap_audit.get("cpu_candidate_supported") is True
        and camera_cap_audit.get("current_cap_m") == 0.05
        and camera_cap_audit.get("candidate_cap_m") == 0.10
        and camera_cap_audit.get("dynamic_proof_obtained") is False
        and camera_cap_audit.get("gpu_launch_authorized") is False,
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
        read_identity("baseline_gate"),
        read_identity("baseline_final_status"),
        read_identity("recovery_outcome_audit"),
        read_identity("camera_cap_audit"),
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
        "schema": "cinebotrl_two_wheel_riser_case78_camera_cap_cpu_admission_v1",
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
    result = validate(args.contract, args.repo_root, namespace=args.namespace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
