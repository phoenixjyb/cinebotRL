#!/usr/bin/env python3
"""Build the non-applying case-4 split-reset fallback proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REFERENCE_CASE = 4
REPLACEMENT_CASE = 78
EXPECTED_TRAIN = [
    2, 6, 7, 9, 10, 11, 12, 14, 15, 17, 18, 21, 23, 25, 26,
    28, 30, 31, 33, 34, 36, 37, 41, 52, 53, 66, 67, 68, 70, 74,
]
EXPECTED_VALIDATION = [4, 8, 16, 22, 32]
EXPECTED_HOLDOUT = [3, 5, 13, 19, 24]


def _identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing split-reset input: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build_proposal(
    architecture: dict[str, object],
    coverage: dict[str, object],
    unused: dict[str, object],
    plan_summary: dict[str, object],
) -> dict[str, object]:
    split = architecture["current_split_cases"]
    case78 = next(
        (
            item
            for item in plan_summary["items"]
            if item.get("case") == REPLACEMENT_CASE
        ),
        None,
    )
    best_unused = unused.get("best_unused_admitted", {})
    checks = {
        "architecture_prefers_perturbation_then_split_fallback": (
            architecture.get("decision") == "controlled_perturbation_contract_first"
            and architecture.get("options", {})
            .get("transparent_split_reset", {})
            .get("replacement_validation_candidate")
            == REPLACEMENT_CASE
        ),
        "frozen_train_split": split.get("train") == EXPECTED_TRAIN,
        "frozen_validation_split": split.get("validation")
        == EXPECTED_VALIDATION,
        "frozen_holdout_split": split.get("holdout") == EXPECTED_HOLDOUT,
        "perturbation_coverage_rejected": coverage.get(
            "coverage_admission_passed"
        )
        is False
        and coverage.get("state_coverage_materially_improved") is False
        and coverage.get("dagger_authorized") is False,
        "perturbation_labels_unadmitted": coverage.get("dataset_created") is False
        and coverage.get("valid_for_training") is False,
        "case78_is_best_unused_candidate": best_unused.get("case")
        == REPLACEMENT_CASE,
        "case78_plan_identity_matches": case78 is not None
        and best_unused.get("plan", {}).get("sha256")
        == case78.get("plan_sha256"),
        "case78_exact_source_integrity": case78 is not None
        and all(case78.get("checks", {}).values()),
        "case78_kinematic_gate": case78 is not None
        and all(case78.get("kinematic_checks", {}).values())
        and case78.get("timing_transition_kinematic_gate_passed") is True,
        "case78_not_yet_training_data": case78 is not None
        and case78.get("valid_for_training") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"split-reset proposal input contract failed: {checks}")
    proposed_train = sorted(EXPECTED_TRAIN + [REFERENCE_CASE])
    proposed_validation = sorted(
        [case for case in EXPECTED_VALIDATION if case != REFERENCE_CASE]
        + [REPLACEMENT_CASE]
    )
    return {
        "schema": "cinebotrl_two_wheel_riser_split_reset_fallback_proposal_v1",
        "decision": "transparent_split_reset_pending_case78_dynamic_qualification",
        "decision_status": "cpu_only_proposal_split_not_applied",
        "reason": (
            "the bounded case30 perturbation passed physics but improved case4 "
            "hotspot coverage by less than the frozen material threshold and "
            "produced no material shadow-label shift"
        ),
        "stronger_perturbation_authorized": False,
        "stronger_perturbation_rejection_basis": (
            "no measured evidence that force escalation closes the 3.65x "
            "reference-coverage gap"
        ),
        "current_split_cases": split,
        "proposed_split_cases_not_applied": {
            "train": proposed_train,
            "validation": proposed_validation,
            "holdout": EXPECTED_HOLDOUT,
        },
        "case4_contract": {
            "current_role": "validation_diagnostic_only",
            "proposed_role_after_all_gates": "train",
            "permanently_retired_from_validation_if_applied": True,
            "labels_currently_admitted": False,
        },
        "case78_contract": {
            "current_role": "unused_kinematically_admitted",
            "proposed_role_after_dynamic_pass": "validation",
            "plan_sha256": case78["plan_sha256"],
            "source_pose_count": case78["source_pose_count"],
            "execution_state_count": case78["execution_state_count"],
            "source_duration_s": case78["source_duration_s"],
            "execution_duration_s": case78["execution_duration_s"],
            "path_length_m": case78["path_metrics"]["source_path_length_m"],
            "kinematic_position_p95_m": case78["kinematic_metrics"][
                "position_error_p95_m"
            ],
            "kinematic_position_max_m": case78["kinematic_metrics"][
                "position_error_max_m"
            ],
            "dynamic_qualification_passed": False,
            "validation_admitted": False,
        },
        "exact_next_bounded_task": {
            "kind": "cpu_only_case78_dynamic_qualification_contract",
            "case": REPLACEMENT_CASE,
            "mode": "deterministic_reference_playback_no_policy_no_capture",
            "requirements": [
                "pin exact case78 plan and unchanged controller/gates",
                "benchmark a bounded wall timeout for the 192.30 s execution plan",
                "enforce exclusive WSL Windows and NVIDIA ownership",
                "write independent dynamic thermal and controller evidence",
                "stop after case78 and create no dataset",
            ],
            "runtime_or_gpu_authorized": False,
        },
        "input_contract_checks": checks,
        "split_changed": False,
        "case4_labels_admitted_for_training": False,
        "case78_validation_admitted": False,
        "holdout_opened": False,
        "runtime_authorized": False,
        "dataset_created": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "valid_for_training": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture-proposal", type=Path, required=True)
    parser.add_argument("--coverage-audit", type=Path, required=True)
    parser.add_argument("--unused-audit", type=Path, required=True)
    parser.add_argument("--plan-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "architecture_proposal": args.architecture_proposal,
        "coverage_audit": args.coverage_audit,
        "unused_audit": args.unused_audit,
        "plan_summary": args.plan_summary,
    }
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    report = build_proposal(
        payloads["architecture_proposal"],
        payloads["coverage_audit"],
        payloads["unused_audit"],
        payloads["plan_summary"],
    )
    report["inputs"] = {
        name: _identity(path) for name, path in paths.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
