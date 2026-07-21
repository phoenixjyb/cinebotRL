#!/usr/bin/env python3
"""Validate the heartbeat-enabled CPU-only case-78 qualification contract."""

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


SCHEMA = "cinebotrl_two_wheel_riser_case78_dynamic_cpu_contract_v2"
REVIEWED_PARENT = "ed207f41a04c492ce7a2be52bda80fc106eb8d92"
NAMESPACE = "20260721_case78_dynamic_qualification_v2_heartbeat_exclusive"
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/case78_dynamic_cpu_contract_v2.json"
)
EXPECTED_IDENTITIES = {
    "case78_plan",
    "fallback_proposal",
    "heartbeat_helper",
    "lqr_gains",
    "plan_manifest",
    "plan_summary",
    "playback",
    "preflight_wrapper_v2",
    "recovery_evidence",
    "riser_control",
    "riser_loader",
    "robot_build_audit",
    "robot_urdf",
    "robot_usd",
    "runtime_timeout_final",
    "timing_handoff",
    "tracking",
    "validator_v2",
    "wall_bound_audit",
}
EXPECTED_HEARTBEAT = {
    "schema": "cinebotrl_two_wheel_riser_runtime_heartbeat_v1",
    "relative_path": "runtime_heartbeat.json",
    "interval_policy_steps": 2000,
    "atomic_replace": True,
    "changes_commands": False,
    "creates_dataset": False,
}
EXPECTED_WALL_BOUND = {
    "reference_case": 30,
    "reference_completed_steps": 11494,
    "reference_virtual_step_duration_s": 57.47,
    "reference_host_filesystem_envelope_s": 440.0,
    "conservative_policy_step_rate_hz": 26.12272727272727,
    "target_maximum_steps": 115381,
    "estimated_maximum_loop_wall_s": 4416.881851400731,
    "startup_shutdown_and_diagnosis_margin_s": 900.0,
    "wall_timeout_s": 5400,
}


def semantic_checks(
    contract: dict[str, object],
    fallback: dict[str, object],
    plan_summary: dict[str, object],
    wall_audit: dict[str, object],
    timeout_final: dict[str, object],
) -> dict[str, bool]:
    case78 = next(
        (
            item
            for item in plan_summary.get("items", [])
            if item.get("case") == 78
        ),
        {},
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
        "fallback_pending_case78": fallback.get("decision")
        == "transparent_split_reset_pending_case78_dynamic_qualification"
        and fallback.get("split_changed") is False
        and fallback.get("case78_validation_admitted") is False,
        "plan_contract_matches": contract.get("plan_contract") == EXPECTED_PLAN,
        "summary_plan_identity": case78.get("plan_sha256")
        == EXPECTED_PLAN["plan_sha256"],
        "summary_plan_integrity": bool(case78)
        and all(case78.get("checks", {}).values())
        and all(case78.get("kinematic_checks", {}).values())
        and case78.get("timing_transition_kinematic_gate_passed") is True,
        "controller_unchanged": contract.get("controller_arguments")
        == EXPECTED_CONTROLLER,
        "thresholds_unchanged": contract.get("dynamic_gate_thresholds")
        == EXPECTED_THRESHOLDS,
        "heartbeat_contract_matches": contract.get("runtime_heartbeat")
        == EXPECTED_HEARTBEAT,
        "wall_bound_contract_matches": contract.get("wall_timeout_derivation")
        == EXPECTED_WALL_BOUND,
        "wall_audit_passes": wall_audit.get("audit_passed") is True
        and wall_audit.get("proposed_maximum_wall_duration_s") == 5400
        and wall_audit.get("runtime_retry_authorized") is False,
        "prior_timeout_rejected": timeout_final.get("playback_exit_code") == 124
        and timeout_final.get("dynamic_qualification_passed") is False
        and timeout_final.get("case78_validation_admitted") is False,
        "one_case_no_capture": contract.get("one_case_only") is True
        and contract.get("maximum_runtime_seconds") == 5400
        and contract.get("dataset_creation_authorized") is False,
        "cpu_only": contract.get("cpu_preflight_ready") is True
        and contract.get("runtime_authorized") is False
        and contract.get("gpu_launch_authorized") is False
        and contract.get("dynamic_qualification_authorized") is False,
        "no_runtime_token": "runtime_authorization_token_sha256" not in contract,
        "learning_closed": contract.get("dagger_authorized") is False
        and contract.get("bc_authorized") is False
        and contract.get("ppo_authorized") is False
        and contract.get("split_change_authorized") is False
        and contract.get("holdout_opened") is False,
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
        read_identity("fallback_proposal"),
        read_identity("plan_summary"),
        read_identity("wall_bound_audit"),
        read_identity("runtime_timeout_final"),
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
        "schema": "cinebotrl_two_wheel_riser_case78_dynamic_cpu_admission_v2",
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

