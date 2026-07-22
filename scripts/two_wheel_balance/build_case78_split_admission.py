#!/usr/bin/env python3
"""Build the immutable post-qualification case-4/case-78 split admission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from .build_riser_split_reset_fallback_proposal import (
        EXPECTED_HOLDOUT,
        EXPECTED_TRAIN,
        EXPECTED_VALIDATION,
    )
else:
    from build_riser_split_reset_fallback_proposal import (
        EXPECTED_HOLDOUT,
        EXPECTED_TRAIN,
        EXPECTED_VALIDATION,
    )


CASE78_GATE_SHA256 = (
    "304fa9e1202d4099f976e6933e9ffc21a2833e7cc380ab9f95d7473bf2126c73"
)
CASE78_FINAL_SHA256 = (
    "e413b0df0b09c3c04ac49130ca2d38c7e495364504c04f3e89d72521e8e5a4f6"
)
PROPOSAL_SHA256 = (
    "975bfa46cede07daa70ae36f81c354ea707898a38addf2fe54ec73a75aaf8072"
)
DATASET_SUMMARY_SHA256 = (
    "815463ffa133addbaec4f09a453fd9dae8e63eb690b37f56fd0a5c1877879542"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing split-admission input: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def build_admission(
    proposal: dict[str, object],
    dataset: dict[str, object],
    final: dict[str, object],
    gate: dict[str, object],
) -> dict[str, object]:
    results = gate.get("results")
    case78 = (
        results[0]
        if isinstance(results, list)
        and len(results) == 1
        and isinstance(results[0], dict)
        else {}
    )
    proposed = proposal.get("proposed_split_cases_not_applied", {})
    expected_train = sorted(EXPECTED_TRAIN + [4])
    expected_validation = sorted(
        [case for case in EXPECTED_VALIDATION if case != 4] + [78]
    )
    checks = {
        "proposal_is_pending_exact_swap": proposal.get("decision")
        == "transparent_split_reset_pending_case78_dynamic_qualification"
        and proposal.get("split_changed") is False
        and proposed.get("train") == expected_train
        and proposed.get("validation") == expected_validation
        and proposed.get("holdout") == EXPECTED_HOLDOUT,
        "historical_dataset_split_is_frozen": dataset.get("split_cases")
        == {
            "train": EXPECTED_TRAIN,
            "validation": EXPECTED_VALIDATION,
            "holdout": EXPECTED_HOLDOUT,
        }
        and dataset.get("trajectory_leakage") is False,
        "historical_dataset_shape_is_unchanged": dataset.get("case_count") == 40
        and dataset.get("captured_case_count") == 41
        and dataset.get("row_count") == 403569,
        "case78_final_is_dynamic_pass": final.get("case") == 78
        and final.get("passed") is True
        and final.get("dynamic_qualification_passed") is True
        and final.get("physical_quality_passed") is True
        and final.get("dataset_created") is False
        and final.get("split_changed") is False,
        "case78_gate_is_complete_and_safe": gate.get("cases") == [78]
        and gate.get("passed") is True
        and case78.get("case") == 78
        and case78.get("completed_phase_time_s")
        == case78.get("execution_duration_s")
        and case78.get("dynamic_quality_passed") is True
        and case78.get("thermal_admission_passed") is True
        and case78.get("controller_evidence_passed") is True
        and case78.get("termination") is None,
        "case78_uses_admitted_controller": case78.get(
            "maximum_camera_lever_arm_correction_m"
        )
        == 0.10
        and case78.get("camera_recovery_governor_enabled") is False,
        "case78_meets_position_gates": case78.get("position_error_p95_m", 1.0)
        <= 0.15
        and case78.get("position_error_max_m", 1.0) <= 0.25,
        "label_admission_remains_closed": case78.get(
            "residual_label_envelope_passed"
        )
        is False
        and gate.get("normalized_dataset_capture_started") is False
        and gate.get("training_started") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"split admission contract failed: {checks}")
    return {
        "schema": "cinebotrl_two_wheel_riser_split_admission_v1",
        "decision": "case4_train_case78_validation_roles_admitted",
        "split_version": "initial_teacher_case4_train_case78_validation_v2",
        "previous_split_cases": {
            "train": EXPECTED_TRAIN,
            "validation": EXPECTED_VALIDATION,
            "holdout": EXPECTED_HOLDOUT,
        },
        "admitted_split_cases": {
            "train": expected_train,
            "validation": expected_validation,
            "holdout": EXPECTED_HOLDOUT,
        },
        "role_changes": {
            "case4": {"from": "validation", "to": "train"},
            "case78": {"from": "unused", "to": "validation"},
        },
        "case78_dynamic_evidence": {
            "position_error_p95_m": case78["position_error_p95_m"],
            "position_error_max_m": case78["position_error_max_m"],
            "pitch_max_deg": case78["pitch_max_deg"],
            "completed_steps": case78["completed_steps"],
            "source_duration_s": case78["source_duration_s"],
            "execution_duration_s": case78["execution_duration_s"],
        },
        "historical_dataset": {
            "preserved_immutable": True,
            "rewrite_performed": False,
            "case_count": dataset["case_count"],
            "row_count": dataset["row_count"],
        },
        "effective_for_next_dataset_build": True,
        "split_admitted": True,
        "case4_training_role_admitted": True,
        "case78_validation_role_admitted": True,
        "case78_labels_available": False,
        "holdout_opened": False,
        "dataset_creation_authorized": False,
        "label_capture_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "valid_for_training": False,
        "input_contract_checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--dataset-summary", type=Path, required=True)
    parser.add_argument("--case78-final", type=Path, required=True)
    parser.add_argument("--case78-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "proposal": args.proposal,
        "dataset_summary": args.dataset_summary,
        "case78_final": args.case78_final,
        "case78_gate": args.case78_gate,
    }
    expected_hashes = {
        "proposal": PROPOSAL_SHA256,
        "dataset_summary": DATASET_SUMMARY_SHA256,
        "case78_final": CASE78_FINAL_SHA256,
        "case78_gate": CASE78_GATE_SHA256,
    }
    identities = {name: identity(path) for name, path in paths.items()}
    if any(
        identities[name]["sha256"] != expected
        for name, expected in expected_hashes.items()
    ):
        raise ValueError("split-admission input hash mismatch")
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    result = build_admission(
        payloads["proposal"],
        payloads["dataset_summary"],
        payloads["case78_final"],
        payloads["case78_gate"],
    )
    result["inputs"] = identities
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
