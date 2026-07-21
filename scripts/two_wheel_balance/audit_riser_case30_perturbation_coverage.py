#!/usr/bin/env python3
"""Compare case-30 perturbation states with the frozen case-4 hotspot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.two_wheel_balance.rank_riser_dagger_localized_training_cases import (
    MATERIAL_SHIFT_P95,
    MAXIMUM_CANDIDATE_SAMPLES,
    MAXIMUM_REFERENCE_SCORE_RATIO,
    MAXIMUM_TARGET_SAMPLES,
    _directed_region_distance,
    _phase_aligned_actions,
    _subsample_indices,
)


REFERENCE_CASE = 4
CANDIDATE_CASE = 30
MAXIMUM_PERTURBED_TO_NOMINAL_SCORE_RATIO = 0.90


def _identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing coverage-audit input: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _load_npz(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        payload = {
            name: np.asarray(data[name])
            for name in data.files
            if name != "metadata_json"
        }
    return metadata, payload


def _select_case_rows(case_ids: np.ndarray, case: int) -> np.ndarray:
    rows = np.flatnonzero(case_ids == case)
    if len(rows) < 2:
        raise ValueError(f"teacher dataset has no usable case {case}")
    return rows


def _normalized_states(
    observations: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    effective: np.ndarray,
) -> np.ndarray:
    if observations.ndim != 2 or observations.shape[1] != len(mean):
        raise ValueError("observation normalization shape mismatch")
    if np.any(std <= 0.0):
        raise ValueError("observation standard deviation must be positive")
    return ((observations - mean) / std)[:, effective]


def audit_coverage(
    case4_shadow: dict[str, np.ndarray],
    case30_perturbed: dict[str, np.ndarray],
    teacher: dict[str, np.ndarray],
    teacher_metadata: dict[str, object],
    observation_mean: np.ndarray,
    observation_std: np.ndarray,
    observation_mask: np.ndarray,
) -> dict[str, object]:
    split_cases = teacher_metadata["split_cases"]
    if REFERENCE_CASE not in split_cases["validation"]:
        raise ValueError("case 4 must remain a validation diagnostic")
    if CANDIDATE_CASE not in split_cases["train"]:
        raise ValueError("case 30 must remain in the training split")
    effective = np.flatnonzero(np.asarray(observation_mask) != 0.0)
    if len(effective) < 1 or np.any(np.isin([23, 24, 25], effective)):
        raise ValueError("masked policy must exclude previous-action channels")

    case4_rows = _select_case_rows(teacher["case_ids"], REFERENCE_CASE)
    case30_rows = _select_case_rows(teacher["case_ids"], CANDIDATE_CASE)
    case4_phase_actions = _phase_aligned_actions(
        case4_shadow["phase_time_s"],
        teacher["phase_time_s"][case4_rows],
        teacher["actions"][case4_rows],
    )
    shadow_shift = (
        case4_shadow["shadow_teacher_normalized_residual_actions"]
        - case4_phase_actions
    )
    hotspot_mask = np.any(np.abs(shadow_shift) > MATERIAL_SHIFT_P95, axis=1)
    hotspot_rows = np.flatnonzero(hotspot_mask)
    if len(hotspot_rows) < 2:
        raise ValueError("case-4 shadow trace has no material-shift hotspot")
    target_rows = hotspot_rows[
        _subsample_indices(len(hotspot_rows), MAXIMUM_TARGET_SAMPLES)
    ]
    target_states = _normalized_states(
        case4_shadow["observations"][target_rows],
        observation_mean,
        observation_std,
        effective,
    )
    target_actions = case4_phase_actions[target_rows]

    reference_rows = case4_rows[
        _subsample_indices(len(case4_rows), MAXIMUM_CANDIDATE_SAMPLES)
    ]
    nominal_rows = case30_rows[
        _subsample_indices(len(case30_rows), MAXIMUM_CANDIDATE_SAMPLES)
    ]
    perturbed_rows = _subsample_indices(
        len(case30_perturbed["phase_time_s"]),
        MAXIMUM_CANDIDATE_SAMPLES,
    )
    perturbed_phase_actions = _phase_aligned_actions(
        case30_perturbed["phase_time_s"][perturbed_rows],
        teacher["phase_time_s"][case30_rows],
        teacher["actions"][case30_rows],
    )

    reference = _directed_region_distance(
        target_states,
        target_actions,
        _normalized_states(
            teacher["observations"][reference_rows],
            observation_mean,
            observation_std,
            effective,
        ),
        teacher["actions"][reference_rows],
    )
    nominal = _directed_region_distance(
        target_states,
        target_actions,
        _normalized_states(
            teacher["observations"][nominal_rows],
            observation_mean,
            observation_std,
            effective,
        ),
        teacher["actions"][nominal_rows],
    )
    perturbed = _directed_region_distance(
        target_states,
        target_actions,
        _normalized_states(
            case30_perturbed["observations"][perturbed_rows],
            observation_mean,
            observation_std,
            effective,
        ),
        perturbed_phase_actions,
    )
    reference_score = max(reference["score"], 1e-12)
    nominal_score = max(nominal["score"], 1e-12)
    score_ratio = perturbed["score"] / nominal_score
    improvement_fraction = 1.0 - score_ratio
    perturbed_reference_ratio = perturbed["score"] / reference_score
    materially_improved = bool(
        score_ratio <= MAXIMUM_PERTURBED_TO_NOMINAL_SCORE_RATIO
    )
    reference_calibrated = bool(
        perturbed_reference_ratio <= MAXIMUM_REFERENCE_SCORE_RATIO
    )
    coverage_passed = materially_improved and reference_calibrated
    return {
        "schema": (
            "cinebotrl_two_wheel_riser_case30_perturbation_coverage_audit_v1"
        ),
        "reference_case": REFERENCE_CASE,
        "reference_split": "validation_diagnostic_only",
        "candidate_case": CANDIDATE_CASE,
        "candidate_split": "train",
        "distance_contract": (
            "round129_directed_normalized_state_phase_action_distance_v1"
        ),
        "candidate_action_contract": (
            "phase_aligned_original_case30_teacher_actions_not_shadow_labels"
        ),
        "comparison_contract": (
            "perturbed_policy_visited_case30_vs_nominal_teacher_case30_v1"
        ),
        "causal_attribution_to_perturbation_proven": False,
        "material_shift_thresholds": MATERIAL_SHIFT_P95.tolist(),
        "maximum_perturbed_to_nominal_score_ratio": (
            MAXIMUM_PERTURBED_TO_NOMINAL_SCORE_RATIO
        ),
        "maximum_reference_score_ratio": MAXIMUM_REFERENCE_SCORE_RATIO,
        "effective_observation_count": len(effective),
        "previous_action_channels_effective": False,
        "hotspot_row_count": len(hotspot_rows),
        "hotspot_fraction": float(np.mean(hotspot_mask)),
        "hotspot_channel_counts": np.sum(
            np.abs(shadow_shift) > MATERIAL_SHIFT_P95, axis=0
        ).tolist(),
        "target_distance_sample_count": len(target_rows),
        "reference_calibration": reference,
        "nominal_case30": nominal,
        "perturbed_case30": perturbed,
        "nominal_case30_sample_count": len(nominal_rows),
        "perturbed_case30_sample_count": len(perturbed_rows),
        "perturbed_to_nominal_score_ratio": score_ratio,
        "perturbed_score_improvement_fraction": improvement_fraction,
        "perturbed_to_reference_score_ratio": perturbed_reference_ratio,
        "state_coverage_materially_improved": materially_improved,
        "reference_calibrated_coverage_passed": reference_calibrated,
        "coverage_admission_passed": coverage_passed,
        "classification": (
            "case30_perturbed_trace_materially_covers_case4_hotspot"
            if coverage_passed
            else "case30_perturbed_trace_does_not_cover_case4_hotspot"
        ),
        "case4_labels_admitted_for_training": False,
        "dataset_created": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "valid_for_training": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case4-shadow-trace", type=Path, required=True)
    parser.add_argument("--case30-perturbed-trace", type=Path, required=True)
    parser.add_argument("--teacher-dataset", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--case30-final", type=Path, required=True)
    parser.add_argument("--case30-diagnosis", type=Path, required=True)
    parser.add_argument("--architecture-proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    case4_metadata, case4_shadow = _load_npz(args.case4_shadow_trace)
    case30_metadata, case30_perturbed = _load_npz(
        args.case30_perturbed_trace
    )
    teacher_metadata, teacher = _load_npz(args.teacher_dataset)
    final = json.loads(args.case30_final.read_text(encoding="utf-8"))
    diagnosis = json.loads(args.case30_diagnosis.read_text(encoding="utf-8"))
    architecture = json.loads(
        args.architecture_proposal.read_text(encoding="utf-8")
    )
    input_checks = {
        "case4_trace_contract": case4_metadata.get("case") == 4
        and case4_metadata.get("trace_only") is True
        and case4_metadata.get("valid_for_training") is False,
        "case30_trace_contract": case30_metadata.get("case") == 30
        and case30_metadata.get("trace_only") is True
        and case30_metadata.get("valid_for_training") is False,
        "action_scales_match": case4_metadata.get("action_scales")
        == case30_metadata.get("action_scales"),
        "case30_measurement_passed": final.get("passed") is True
        and final.get("physical_quality_passed") is True
        and final.get("perturbation_contract_passed") is True,
        "case30_labels_unadmitted": final.get("dataset_created") is False
        and final.get("valid_for_training") is False
        and diagnosis.get("dataset_created") is False,
        "architecture_contract": architecture.get("decision")
        == "controlled_perturbation_contract_first"
        and architecture.get("case4_labels_admitted_for_training") is False,
    }
    if not all(input_checks.values()):
        raise ValueError(f"coverage-audit input contract failed: {input_checks}")
    model = torch.jit.load(str(args.policy), map_location="cpu")
    report = audit_coverage(
        case4_shadow,
        case30_perturbed,
        teacher,
        teacher_metadata,
        model.observation_mean.detach().cpu().numpy(),
        model.observation_std.detach().cpu().numpy(),
        model.observation_mask.detach().cpu().numpy(),
    )
    report["input_contract_checks"] = input_checks
    report["inputs"] = {
        "case4_shadow_trace": _identity(args.case4_shadow_trace),
        "case30_perturbed_trace": _identity(args.case30_perturbed_trace),
        "teacher_dataset": _identity(args.teacher_dataset),
        "policy": _identity(args.policy),
        "case30_final": _identity(args.case30_final),
        "case30_diagnosis": _identity(args.case30_diagnosis),
        "architecture_proposal": _identity(args.architecture_proposal),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
