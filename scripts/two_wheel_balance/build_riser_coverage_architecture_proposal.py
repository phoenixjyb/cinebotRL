#!/usr/bin/env python3
"""Build the fail-closed riser coverage-recovery architecture proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_TRAIN_CASES = [
    2,
    6,
    7,
    9,
    10,
    11,
    12,
    14,
    15,
    17,
    18,
    21,
    23,
    25,
    26,
    28,
    30,
    31,
    33,
    34,
    36,
    37,
    41,
    52,
    53,
    66,
    67,
    68,
    70,
    74,
]
EXPECTED_VALIDATION_CASES = [4, 8, 16, 22, 32]
EXPECTED_HOLDOUT_CASES = [3, 5, 13, 19, 24]
PERTURBATION_CASE = 30
SPLIT_RESET_REPLACEMENT_CASE = 78


def _identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing architecture input: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["metadata_json"].item()))


def _split_cases(metadata: dict[str, object]) -> dict[str, list[int]]:
    raw = metadata.get("split_cases")
    if not isinstance(raw, dict):
        raise ValueError("teacher dataset does not declare split_cases")
    return {
        split: [int(case) for case in raw[split]]
        for split in ("train", "validation", "holdout")
    }


def build_proposal(
    *,
    dataset_metadata: dict[str, object],
    case4_final: dict[str, object],
    case4_diagnosis: dict[str, object],
    case21_final: dict[str, object],
    case21_diagnosis: dict[str, object],
    localized_audit: dict[str, object],
    unused_audit: dict[str, object],
    input_identities: dict[str, dict[str, object]],
) -> dict[str, object]:
    splits = _split_cases(dataset_metadata)
    localized_best = localized_audit["ranked_training_cases"][0]
    unused_existing_best = unused_audit["best_existing_training"]
    unused_best = unused_audit["best_unused_admitted"]
    teacher_hash = input_identities["teacher_dataset"]["sha256"]

    checks = {
        "frozen_train_split": splits["train"] == EXPECTED_TRAIN_CASES,
        "frozen_validation_split": (
            splits["validation"] == EXPECTED_VALIDATION_CASES
        ),
        "frozen_holdout_split": splits["holdout"] == EXPECTED_HOLDOUT_CASES,
        "case4_is_validation": 4 in splits["validation"],
        "case4_dynamic_measurement_passed": case4_final.get("passed") is True,
        "case4_labels_remain_unadmitted": (
            case4_final.get("valid_for_training") is False
            and case4_final.get("dataset_created") is False
            and case4_final.get("dagger_authorized") is False
        ),
        "case4_has_material_on_policy_gap": (
            case4_diagnosis.get("dagger_dataset_proposal_supported") is True
            and case4_diagnosis.get("material_shadow_shift_by_channel")
            == [True, True, False]
        ),
        "case21_is_training_measurement": (
            case21_final.get("case") == 21
            and case21_final.get("split") == "train"
            and case21_final.get("passed") is True
        ),
        "case21_does_not_reproduce_gap": (
            case21_diagnosis.get("dagger_dataset_proposal_supported") is False
            and case21_diagnosis.get("material_shadow_shift_by_channel")
            == [False, False, False]
        ),
        "localized_nominal_coverage_closed": (
            localized_audit.get("coverage_admission_passed") is False
            and localized_audit.get("proposed_runtime_cases") == []
            and int(localized_best["case"]) == 18
        ),
        "unused_nominal_coverage_closed": (
            unused_audit.get("coverage_expansion_admission_passed") is False
            and unused_audit.get("proposed_shadow_measurement_cases") == []
            and int(unused_existing_best["case"]) == PERTURBATION_CASE
            and int(unused_best["case"]) == SPLIT_RESET_REPLACEMENT_CASE
        ),
        "unused_candidate_is_only_kinematically_admitted": (
            SPLIT_RESET_REPLACEMENT_CASE
            in unused_audit.get("unused_admitted_cases", [])
            and SPLIT_RESET_REPLACEMENT_CASE
            not in set().union(*map(set, splits.values()))
        ),
        "teacher_identity_consistent": all(
            evidence["sha256"] == teacher_hash
            for evidence in (
                case4_diagnosis["inputs"]["teacher_dataset"],
                case21_diagnosis["inputs"]["teacher_dataset"],
                localized_audit["inputs"]["teacher_dataset"],
                unused_audit["inputs"]["teacher_dataset"],
            )
        ),
        "localized_audit_identity_consistent": (
            unused_audit["inputs"]["localized_audit"]["sha256"]
            == input_identities["localized_audit"]["sha256"]
        ),
        "case4_shadow_identity_consistent": (
            case4_final["shadow_trace"]["sha256"]
            == case4_diagnosis["inputs"]["shadow_trace"]["sha256"]
            == localized_audit["inputs"]["shadow_trace"]["sha256"]
            == unused_audit["inputs"]["shadow_trace"]["sha256"]
        ),
        "case21_diagnosis_identity_consistent": (
            case21_final["diagnosis"]["sha256"]
            == input_identities["case21_diagnosis"]["sha256"]
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"coverage architecture input contract failed: {checks}")

    split_reset_train = sorted([*splits["train"], 4])
    split_reset_validation = sorted(
        [
            case
            for case in splits["validation"]
            if case != 4
        ]
        + [SPLIT_RESET_REPLACEMENT_CASE]
    )
    return {
        "schema": "cinebotrl_two_wheel_riser_coverage_architecture_proposal_v1",
        "decision": "controlled_perturbation_contract_first",
        "decision_status": "cpu_only_proposal_not_runtime_authorization",
        "problem": (
            "case4 has a material on-policy vx/wz teacher gap, but case4 is a "
            "frozen validation case and nominal train/unused plans do not cover "
            "the same policy-visited state region"
        ),
        "current_split_cases": splits,
        "evidence_summary": {
            "case4_material_shift_channels": ["vx", "wz"],
            "case21_material_shift_channels": [],
            "localized_best_train_case": int(localized_best["case"]),
            "localized_best_reference_score_ratio": float(
                localized_best["reference_score_ratio"]
            ),
            "nominal_command_best_train_case": int(
                unused_existing_best["case"]
            ),
            "nominal_command_best_unused_case": int(unused_best["case"]),
            "unused_to_existing_best_score_ratio": float(
                unused_audit["unused_to_existing_best_score_ratio"]
            ),
        },
        "options": {
            "controlled_perturbation": {
                "rank": 1,
                "recommendation": "preferred",
                "current_status": "runner_contract_not_implemented",
                "first_bounded_measurement_case": PERTURBATION_CASE,
                "case_split": "train",
                "selection_basis": (
                    "closest existing training plan in the nominal command and "
                    "lookahead signature audit; perturbation supplies the missing "
                    "policy-visited state deviation without opening validation"
                ),
                "perturbation_contract": {
                    "kind": "single_deterministic_horizontal_wrench_pulse",
                    "enabled_by_default": False,
                    "randomized": False,
                    "exact_plan_geometry_or_timing_changed": False,
                    "teacher_commands_changed": False,
                    "learned_actions_applied_to_commands": False,
                    "initial_profile_values": None,
                    "profile_values_require_separate_cpu_review": True,
                    "existing_dynamic_and_safety_gates_unchanged": True,
                    "measurement_trace_only": True,
                },
                "required_before_runtime": [
                    "disabled-by-default playback-runner perturbation plumbing",
                    "hash-bound perturbation profile and telemetry",
                    "negative tests for missing or conflicting profile fields",
                    "proof that zero/disabled perturbation is command-identical",
                    "fresh runtime namespace and explicit ownership authorization",
                ],
                "measurement_admission": [
                    "case30 passes every unchanged dynamic and safety gate",
                    "perturbation telemetry proves the requested pulse only",
                    "visited-state audit materially improves case4 hotspot coverage",
                    "shadow teacher gap is measured independently from physics",
                    "no labels are admitted and no dataset is written",
                ],
            },
            "transparent_split_reset": {
                "rank": 2,
                "recommendation": "fallback_only",
                "current_status": "not_admitted",
                "case4_permanently_retired_from_validation": True,
                "proposed_train_cases_not_applied": split_reset_train,
                "replacement_validation_candidate": SPLIT_RESET_REPLACEMENT_CASE,
                "proposed_validation_cases_not_applied": split_reset_validation,
                "holdout_cases_unchanged": splits["holdout"],
                "required_before_split_change": [
                    "case78 deterministic dynamic qualification under unchanged gates",
                    "a new immutable split manifest and dataset version",
                    "source-tagged case4 original and shadow rows",
                    "within-case weighting or bounded shadow-row sampling",
                    "all future reports stop describing case4 as validation",
                ],
                "reason_not_selected": (
                    "it sacrifices a validation case and requires a newly captured "
                    "replacement plus dataset-schema work; case78 currently has only "
                    "kinematic plan admission"
                ),
            },
        },
        "exact_next_bounded_task": {
            "kind": "cpu_only_runner_contract_change",
            "description": (
                "add disabled-by-default deterministic wrench-pulse plumbing and "
                "hash-bound measurement telemetry to riser reference playback"
            ),
            "runtime_or_gpu_launch": False,
            "case": PERTURBATION_CASE,
            "dataset_or_training": False,
        },
        "input_contract_checks": checks,
        "inputs": input_identities,
        "case4_labels_admitted_for_training": False,
        "case4_split_changed": False,
        "case78_validation_admitted": False,
        "holdout_opened": False,
        "runtime_authorized": False,
        "authorization_token_issued": False,
        "runtime_namespace_created": False,
        "dataset_created": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "valid_for_training": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dataset", type=Path, required=True)
    parser.add_argument("--case4-final", type=Path, required=True)
    parser.add_argument("--case4-diagnosis", type=Path, required=True)
    parser.add_argument("--case21-final", type=Path, required=True)
    parser.add_argument("--case21-diagnosis", type=Path, required=True)
    parser.add_argument("--localized-audit", type=Path, required=True)
    parser.add_argument("--unused-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "teacher_dataset": args.teacher_dataset,
        "case4_final": args.case4_final,
        "case4_diagnosis": args.case4_diagnosis,
        "case21_final": args.case21_final,
        "case21_diagnosis": args.case21_diagnosis,
        "localized_audit": args.localized_audit,
        "unused_audit": args.unused_audit,
    }
    proposal = build_proposal(
        dataset_metadata=_load_npz_metadata(args.teacher_dataset),
        case4_final=_load_json(args.case4_final),
        case4_diagnosis=_load_json(args.case4_diagnosis),
        case21_final=_load_json(args.case21_final),
        case21_diagnosis=_load_json(args.case21_diagnosis),
        localized_audit=_load_json(args.localized_audit),
        unused_audit=_load_json(args.unused_audit),
        input_identities={name: _identity(path) for name, path in paths.items()},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proposal, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(proposal, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
