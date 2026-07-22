#!/usr/bin/env python3
"""Build the CPU-only teacher-41 validation tranche contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEWED_CASE78_RUNTIME_COMMIT = "0cd6548e48693ebfef368b09d9cd08b60e7e4201"
TRACKING_PROFILE = "riser_recovery_direction_v4_camera_lever_arm_v1"
POLICY_SHA256 = "0d796c600c6dca7dce176da555f4cd1f769163f41093d2b6313f4e6264888db7"
ACTION_SCALES = [0.35, 0.4, 0.1]

EXPECTED_CASES = {
    16: {
        "plan_sha256": "742d1f705d3559916c3e1d7d35caffd5ea9e7200b6e321d1f9f70c8e5a7dad16",
        "plan_report_sha256": "eebe41d0523a8ca270407167f78f7b87d732577c01df8e5ffb4a1eff0759656e",
        "teacher_gate_sha256": "8915d532633c52a7727fd24514141aa122d87cf593bc123d9a2776f2552a000a",
        "teacher_audit_sha256": "0093efcf72084b776b0fd78339a83f009f351c79e841738943fd0805a75e53ff",
        "source_pose_count": 896,
        "execution_state_count": 896,
        "source_duration_s": 17.548706,
        "execution_duration_s": 26.028629743189363,
    },
    22: {
        "plan_sha256": "8f1638cd771cfac32ca251906e2c095bd7091edb2561974f12ae09b0a65d4a79",
        "plan_report_sha256": "0eac1ed68000cc0877cf14fc023332a3fa2410cfa163864091264470a665ae99",
        "teacher_gate_sha256": "115623a6f1239b9e4fc78a7a60087a176b340f275f817f123c90f593e943892a",
        "teacher_audit_sha256": "ed2ad0c7a1b59b4a1f02414f3c047e2580423588af9ce66cefe8ba75680aafe0",
        "source_pose_count": 693,
        "execution_state_count": 693,
        "source_duration_s": 13.49578,
        "execution_duration_s": 20.23314121100052,
    },
    32: {
        "plan_sha256": "71b1986633613fdb13585ac4c12870addc553ad12e895b05cc424a83cf4e037f",
        "plan_report_sha256": "49e207862f606b514892103ba445d938086781ac1c85f04ec30f75d25098383f",
        "teacher_gate_sha256": "d2a7477254d6a80426370217d8f08db8fe2bdf65e5f4b892a33247f90cf1ce75",
        "teacher_audit_sha256": "2f7e36eddcaf96d85ace1690fe1268605835b56eaecdd293568f3f43093704ca",
        "source_pose_count": 1099,
        "execution_state_count": 1099,
        "source_duration_s": 21.648708,
        "execution_duration_s": 29.592866387237176,
    },
}

FIXED_HASHES = {
    "plan_manifest": "8351514a361d3be4e5fbf57f2dbb019a7d8d2f5b86e89cea2553a1cfda3c64a1",
    "plan_summary": "72bdb9b64f71bde125507944a0f43f244bfb766e445e996693fba67270cb104d",
    "teacher_capture_admission": "95de47d95c825bb6a65cddddf4866525c05e70049ae932ab29d1c24568d15df8",
    "teacher_selection": "e0f1d2b44061aabfe64ad2ffa3d23f57bf9b3e51015b2e3fa0703ba24316bb06",
    "policy_final": "d232ffac1f67d1a4510e2ad7e6670f82742b259433ed0ce6b15727c4e39db3d9",
    "policy_report": "b7915caddea9467847430a247924eae2e856ad486da06135e1b8f543c42b891a",
    "policy_torchscript": POLICY_SHA256,
    "lqr_gains": "2d955a8878b1086836cfffdaf89e2cd2ecf7c2c4ab2467c24bbfa43cbbd4d5e6",
    "robot_usd": "89f8e38f9290c4a0fcf206dd6966f067f543888f5422f978e566dbb655efa9d0",
    "playback": "ffe45cd5747f6e628caebafbc405d589f34df764df944ddc2b36a2efd0926b1d",
    "rollout_gate": "e3e327f2b7bc7f3bdc5f7c27ba36edcf2f3660e5ed4c8b4e4324325764927f5b",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def identity(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"missing validation-tranche input: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _single_result(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", [])
    return results[0] if isinstance(results, list) and len(results) == 1 else {}


def validate_case78_final(
    final: dict[str, Any], summary: dict[str, Any], summary_sha256: str
) -> dict[str, bool]:
    checks = final.get("checks", {})
    process_codes = final.get("process_exit_codes", {})
    evidence = final.get("evidence", {})
    return {
        "case78_runtime_exact": final.get("runtime_commit")
        == REVIEWED_CASE78_RUNTIME_COMMIT,
        "case78_final_passed": final.get("schema")
        == "cinebotrl_two_wheel_riser_initial_teacher41_validation_canary_final_v1"
        and final.get("case") == 78
        and final.get("split") == "validation"
        and final.get("passed") is True
        and final.get("dynamic_canary_passed") is True,
        "case78_all_checks_passed": bool(checks) and all(checks.values()),
        "case78_processes_passed": process_codes
        == {"learned": 0, "zero": 0, "comparison_gate": 0},
        "case78_summary_bound": evidence.get("comparison_summary", {}).get("sha256")
        == summary_sha256,
        "case78_summary_passed": summary.get("schema")
        == "cinebotrl_two_wheel_riser_residual_validation_canary_gate_v1"
        and summary.get("policy_sha256") == POLICY_SHA256
        and summary.get("cases") == [78]
        and summary.get("case_count") == 1
        and summary.get("maximum_regression_fraction") == 0.05
        and summary.get("minimum_zero_improvement_fraction") == 0.05
        and summary.get("expected_tracking_profile") == TRACKING_PROFILE
        and summary.get("passed") is True,
        "case78_expansion_still_closed": final.get(
            "remaining_validation_cases_authorized"
        )
        is False
        and final.get("broad_rollout_authorized") is False,
        "case78_learning_closed": final.get("dataset_created") is False
        and final.get("bc_authorized") is False
        and final.get("ppo_authorized") is False
        and final.get("ppo_started") is False
        and final.get("holdout_opened") is False
        and final.get("valid_for_training") is False,
    }


def validate_case_inputs(
    case: int,
    report: dict[str, Any],
    teacher_payload: dict[str, Any],
    teacher_audit: dict[str, Any],
) -> dict[str, bool]:
    expected = EXPECTED_CASES[case]
    teacher = _single_result(teacher_payload)
    plan_fields = {
        "case": case,
        "plan_sha256": expected["plan_sha256"],
        "source_pose_count": expected["source_pose_count"],
        "execution_state_count": expected["execution_state_count"],
        "source_duration_s": expected["source_duration_s"],
        "execution_duration_s": expected["execution_duration_s"],
    }
    return {
        f"case{case}_plan_exact": all(
            report.get(key) == value for key, value in plan_fields.items()
        )
        and report.get("passed") is True
        and report.get("timing_transition_kinematic_gate_passed") is True
        and bool(report.get("kinematic_checks"))
        and all(report["kinematic_checks"].values())
        and report.get("valid_for_training") is False,
        f"case{case}_teacher_contract_exact": teacher_payload.get("cases") == [case]
        and teacher_payload.get("trajectory_command_source")
        == "deterministic_teacher"
        and teacher_payload.get("tracking_profile") == TRACKING_PROFILE
        and teacher_payload.get("position_observation_link") == "physical_cam_link_fk"
        and teacher_payload.get("target_attitude_contract")
        == "semantic_dfr_to_physical_cam_v1"
        and teacher_payload.get("hardware_proxy_command_contract")
        == "semantic_attitude_position_only"
        and teacher_payload.get("controller_overrides") == {"wz_kp": 1.05}
        and teacher_payload.get("maximum_duration_scale") == 3.0
        and teacher_payload.get("camera_lever_arm_compensation_enabled") is True
        and teacher_payload.get("camera_lever_arm_compensation_gain") == 1.0
        and teacher_payload.get("maximum_camera_lever_arm_correction_m") == 0.05
        and teacher_payload.get("residual_policy") is None
        and teacher_payload.get("raw_teacher_capture_started") is True
        and teacher_payload.get("normalized_dataset_capture_started") is False
        and teacher_payload.get("passed") is True
        and teacher_payload.get("dynamic_quality_passed") is True,
        f"case{case}_teacher_result_exact": teacher.get("case") == case
        and teacher.get("source_duration_s") == expected["source_duration_s"]
        and teacher.get("execution_duration_s") == expected["execution_duration_s"]
        and teacher.get("completed_phase_time_s") == expected["execution_duration_s"]
        and teacher.get("position_error_p95_m", 1.0) <= 0.15
        and teacher.get("position_error_max_m", 1.0) <= 0.25
        and teacher.get("residual_action_abs_max") == [0.0, 0.0, 0.0]
        and teacher.get("executed_residual_dataset") is None
        and teacher.get("termination") is None
        and teacher.get("passed") is True
        and teacher.get("dynamic_quality_passed") is True,
        f"case{case}_teacher_audit_exact": teacher_audit.get("schema")
        == "cinebotrl_two_wheel_riser_raw_teacher_capture_audit_v1"
        and teacher_audit.get("case") == case
        and bool(teacher_audit.get("checks"))
        and all(teacher_audit["checks"].values())
        and teacher_audit.get("capture_admission_passed") is True
        and teacher_audit.get("gate_sha256") == expected["teacher_gate_sha256"]
        and teacher_audit.get("admission_sha256")
        == FIXED_HASHES["teacher_capture_admission"]
        and teacher_audit.get("selection_sha256") == FIXED_HASHES["teacher_selection"]
        and teacher_audit.get("source_duration_s") == expected["source_duration_s"]
        and teacher_audit.get("execution_duration_s") == expected["execution_duration_s"]
        and teacher_audit.get("raw_reconstruction_max_error", 1.0) <= 1e-6
        and teacher_audit.get("action_scale_frozen") is False
        and teacher_audit.get("valid_for_training") is False
        and teacher_audit.get("bc_authorized") is False
        and teacher_audit.get("ppo_authorized") is False
        and teacher_audit.get("training_started") is False
        and teacher_audit.get("passed") is True,
    }


def build_contract(
    *,
    case78_final: dict[str, Any],
    case78_summary: dict[str, Any],
    case78_summary_sha256: str,
    reports: dict[int, dict[str, Any]],
    teachers: dict[int, dict[str, Any]],
    teacher_audits: dict[int, dict[str, Any]],
    source_commit: str,
) -> dict[str, Any]:
    checks = validate_case78_final(
        case78_final, case78_summary, case78_summary_sha256
    )
    checks["source_commit_present"] = len(source_commit) == 40
    for case in EXPECTED_CASES:
        checks.update(
            validate_case_inputs(
                case, reports[case], teachers[case], teacher_audits[case]
            )
        )
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise ValueError(f"validation tranche CPU contract failed: {checks}")
    return {
        "schema": "cinebotrl_two_wheel_riser_initial_teacher41_validation_tranche_cpu_contract_v1",
        "source_commit": source_commit,
        "reviewed_case78_runtime_commit": REVIEWED_CASE78_RUNTIME_COMMIT,
        "cases": list(EXPECTED_CASES),
        "split": "validation",
        "namespace_proposed": (
            "20260722_initial_teacher41_masked_bc_validation_16_22_32_v1_exclusive"
        ),
        "case_contracts": {
            str(case): {
                **expected,
                "camera_lever_arm_cap_m": 0.05,
            }
            for case, expected in EXPECTED_CASES.items()
        },
        "controller_contract": {
            "controller_wz_kp": 1.05,
            "maximum_duration_scale": 3.0,
            "camera_lever_arm_compensation_enabled": True,
            "camera_lever_arm_compensation_gain": 1.0,
            "maximum_camera_lever_arm_correction_m": 0.05,
            "residual_action_scales": ACTION_SCALES,
            "tracking_profile": TRACKING_PROFILE,
            "position_observation_link": "physical_cam_link_fk",
            "target_attitude_contract": "semantic_dfr_to_physical_cam_v1",
        },
        "comparison_contract": {
            "fresh_zero_required_per_case": True,
            "fresh_learned_required_per_case": True,
            "maximum_teacher_regression_fraction": 0.05,
            "minimum_zero_improvement_fraction": 0.05,
            "fail_fast_on_learned_dynamic_failure": True,
        },
        "input_contract_checks": checks,
        "cpu_contract_ready": True,
        "runtime_authorization_token_issued": False,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "dynamic_canary_authorized": False,
        "runtime_namespace_created": False,
        "dataset_creation_authorized": False,
        "broad_rollout_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "holdout_opened": False,
        "valid_for_training": False,
    }


def _verify_final_evidence(final: dict[str, Any]) -> None:
    evidence = final.get("evidence", {})
    required = {
        "admission",
        "learned",
        "zero",
        "comparison_summary",
        "learned_heartbeat",
        "zero_heartbeat",
    }
    if set(evidence) != required:
        raise ValueError("case-78 final evidence set is incomplete")
    for name, row in evidence.items():
        path = Path(row.get("path", ""))
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            raise ValueError(f"case-78 sealed evidence mismatch: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case78-final", type=Path, required=True)
    parser.add_argument("--case78-summary", type=Path, required=True)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--plan-summary", type=Path, required=True)
    for case in EXPECTED_CASES:
        parser.add_argument(f"--case{case}-plan", type=Path, required=True)
        parser.add_argument(f"--case{case}-report", type=Path, required=True)
        parser.add_argument(f"--case{case}-teacher", type=Path, required=True)
        parser.add_argument(f"--case{case}-teacher-audit", type=Path, required=True)
    for name in FIXED_HASHES:
        if name not in {"plan_manifest", "plan_summary"}:
            parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise ValueError(f"refusing to overwrite tranche contract: {args.output}")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    if args.source_commit != head:
        raise ValueError("validation tranche source commit does not match HEAD")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", REVIEWED_CASE78_RUNTIME_COMMIT, head],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode != 0:
        raise ValueError("validation tranche HEAD does not descend from case-78 runtime")

    fixed_paths = {name: getattr(args, name) for name in FIXED_HASHES}
    fixed_identities = {name: identity(path) for name, path in fixed_paths.items()}
    if any(
        fixed_identities[name]["sha256"] != expected
        for name, expected in FIXED_HASHES.items()
    ):
        raise ValueError("validation tranche fixed input hash mismatch")

    reports: dict[int, dict[str, Any]] = {}
    teachers: dict[int, dict[str, Any]] = {}
    teacher_audits: dict[int, dict[str, Any]] = {}
    case_identities: dict[str, dict[str, str]] = {}
    for case, expected in EXPECTED_CASES.items():
        plan_path = getattr(args, f"case{case}_plan")
        report_path = getattr(args, f"case{case}_report")
        teacher_path = getattr(args, f"case{case}_teacher")
        teacher_audit_path = getattr(args, f"case{case}_teacher_audit")
        rows = {
            f"case{case}_plan": identity(plan_path),
            f"case{case}_report": identity(report_path),
            f"case{case}_teacher": identity(teacher_path),
            f"case{case}_teacher_audit": identity(teacher_audit_path),
        }
        if rows[f"case{case}_plan"]["sha256"] != expected["plan_sha256"]:
            raise ValueError(f"case {case} plan hash mismatch")
        if rows[f"case{case}_report"]["sha256"] != expected["plan_report_sha256"]:
            raise ValueError(f"case {case} report hash mismatch")
        if rows[f"case{case}_teacher"]["sha256"] != expected["teacher_gate_sha256"]:
            raise ValueError(f"case {case} teacher hash mismatch")
        if (
            rows[f"case{case}_teacher_audit"]["sha256"]
            != expected["teacher_audit_sha256"]
        ):
            raise ValueError(f"case {case} teacher audit hash mismatch")
        case_identities.update(rows)
        reports[case] = load_json(report_path)
        teachers[case] = load_json(teacher_path)
        teacher_audits[case] = load_json(teacher_audit_path)

    case78_final = load_json(args.case78_final)
    case78_summary = load_json(args.case78_summary)
    _verify_final_evidence(case78_final)
    case78_identities = {
        "case78_final": identity(args.case78_final),
        "case78_summary": identity(args.case78_summary),
    }
    result = build_contract(
        case78_final=case78_final,
        case78_summary=case78_summary,
        case78_summary_sha256=case78_identities["case78_summary"]["sha256"],
        reports=reports,
        teachers=teachers,
        teacher_audits=teacher_audits,
        source_commit=head,
    )
    result["inputs"] = {
        **case78_identities,
        **fixed_identities,
        **case_identities,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
