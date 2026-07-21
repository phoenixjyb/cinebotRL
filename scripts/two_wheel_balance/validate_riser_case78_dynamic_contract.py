#!/usr/bin/env python3
"""Validate the CPU-only case-78 dynamic qualification contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


SCHEMA = "cinebotrl_two_wheel_riser_case78_dynamic_cpu_contract_v1"
REVIEWED_PARENT = "f8ff44e7ccc52c6a320c190a5ce0fe670c340db1"
NAMESPACE = "20260721_case78_dynamic_qualification_v1_exclusive"
CONTRACT_RELATIVE_PATH = (
    "scripts/two_wheel_balance/case78_dynamic_cpu_contract_v1.json"
)
EXPECTED_IDENTITIES = {
    "case78_plan",
    "contract_validator",
    "coverage_audit",
    "fallback_proposal",
    "lqr_gains",
    "plan_manifest",
    "plan_summary",
    "playback",
    "recovery_evidence",
    "riser_control",
    "riser_loader",
    "robot_build_audit",
    "robot_urdf",
    "robot_usd",
    "timing_handoff",
    "tracking",
    "preflight_wrapper",
}
EXPECTED_CONTROLLER = {
    "trajectory_command_source": "deterministic_teacher",
    "controller_wz_kp": 1.05,
    "maximum_duration_scale": 3.0,
    "camera_lever_arm_compensation_enabled": True,
    "camera_lever_arm_compensation_gain": 1.0,
    "maximum_camera_lever_arm_correction_m": 0.05,
    "residual_policy": None,
    "zero_policy_action": False,
}
EXPECTED_THRESHOLDS = {
    "maximum_pitch_deg": 12.0,
    "maximum_position_p95_m": 0.15,
    "maximum_position_error_m": 0.25,
    "maximum_attitude_p95_deg": 5.0,
    "maximum_attitude_error_deg": 10.0,
    "maximum_riser_servo_error_m": 0.03,
    "maximum_proxy_servo_error_deg": 5.0,
    "maximum_internal_proxy_rate_deg_s": 360.0,
    "maximum_saturation_ratio": 0.2,
}
EXPECTED_PLAN = {
    "case": 78,
    "plan_sha256": (
        "28c69e20778e738d1ac4a0ae299160ed5764089094c2a0f9a018c49790860569"
    ),
    "source_pose_count": 6870,
    "execution_state_count": 6870,
    "source_duration_s": 135.487646,
    "execution_duration_s": 192.29956737098348,
}
EXPECTED_TIMING = {
    "case1_execution_duration_s": 77.833030,
    "case1_observed_wall_duration_s": 102.425,
    "case1_wall_to_execution_ratio": 1.3159580193653004,
    "case52_execution_duration_s": 292.740729,
    "case52_observed_wall_duration_s": 354.76,
    "case52_wall_to_execution_ratio": 1.2118573360524767,
    "maximum_observed_ratio": 1.3159580193653004,
    "case78_maximum_simulated_horizon_s": 576.8987021129504,
    "ratio_scaled_horizon_s": 759.1744734069707,
    "wall_timeout_s": 900,
    "startup_shutdown_margin_s": 140.82552659302928,
}


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


def identity_row(repo: Path, payload: dict[str, object]) -> dict[str, object]:
    raw_path = Path(str(payload.get("path", "")))
    path = raw_path if raw_path.is_absolute() else repo / raw_path
    exists = path.is_file()
    actual_sha = sha256_file(path) if exists else None
    expected_blob = payload.get("git_blob_sha1")
    actual_blob = None
    committed_blob = None
    if exists and expected_blob is not None:
        result = git(repo, "hash-object", str(path), check=False)
        actual_blob = result.stdout.strip() if result.returncode == 0 else None
        try:
            relative = path.resolve().relative_to(repo)
        except ValueError:
            relative = None
        if relative is not None:
            result = git(
                repo,
                "rev-parse",
                f"HEAD:{relative.as_posix()}",
                check=False,
            )
            committed_blob = (
                result.stdout.strip() if result.returncode == 0 else None
            )
    checks = {
        "file_exists": exists,
        "sha256_matches": actual_sha == payload.get("sha256"),
        "git_blob_matches": expected_blob is None or actual_blob == expected_blob,
        "committed_git_blob_matches": expected_blob is None
        or committed_blob == expected_blob,
    }
    return {
        "path": str(path.resolve()),
        "sha256": actual_sha,
        "git_blob_sha1": actual_blob,
        "committed_git_blob_sha1": committed_blob,
        "checks": checks,
        "passed": all(checks.values()),
    }


def semantic_checks(
    contract: dict[str, object],
    fallback: dict[str, object],
    plan_summary: dict[str, object],
) -> dict[str, bool]:
    case78 = next(
        (
            item
            for item in plan_summary.get("items", [])
            if item.get("case") == 78
        ),
        {},
    )
    plan_contract = contract.get("plan_contract", {})
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
        "plan_contract_matches": plan_contract == EXPECTED_PLAN,
        "summary_plan_identity": case78.get("plan_sha256")
        == EXPECTED_PLAN["plan_sha256"],
        "summary_plan_clocks": case78.get("source_pose_count") == 6870
        and case78.get("execution_state_count") == 6870
        and case78.get("source_duration_s") == 135.487646
        and case78.get("execution_duration_s") == 192.29956737098348,
        "summary_plan_integrity": bool(case78)
        and all(case78.get("checks", {}).values())
        and all(case78.get("kinematic_checks", {}).values())
        and case78.get("timing_transition_kinematic_gate_passed") is True,
        "controller_matches": contract.get("controller_arguments")
        == EXPECTED_CONTROLLER,
        "thresholds_unchanged": contract.get("dynamic_gate_thresholds")
        == EXPECTED_THRESHOLDS,
        "timing_contract_matches": contract.get("wall_timeout_derivation")
        == EXPECTED_TIMING,
        "one_case_no_capture": contract.get("one_case_only") is True
        and contract.get("maximum_runtime_seconds") == 900
        and contract.get("dataset_creation_authorized") is False,
        "cpu_only": contract.get("cpu_preflight_ready") is True
        and contract.get("runtime_authorized") is False
        and contract.get("gpu_launch_authorized") is False,
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
    fallback = {}
    summary = {}
    if rows.get("fallback_proposal", {}).get("passed"):
        fallback = json.loads(Path(rows["fallback_proposal"]["path"]).read_text())
    if rows.get("plan_summary", {}).get("passed"):
        summary = json.loads(Path(rows["plan_summary"]["path"]).read_text())
    checks = semantic_checks(contract, fallback, summary)
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
        "schema": "cinebotrl_two_wheel_riser_case78_dynamic_cpu_admission_v1",
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
