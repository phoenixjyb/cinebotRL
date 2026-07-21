#!/usr/bin/env python3
"""Rank training cases against the localized case-4 shadow-shift region."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


MATERIAL_SHIFT_P95 = np.array([0.05, 0.05, 0.02], dtype=np.float64)
STATE_DISTANCE_WEIGHT = 0.75
ACTION_DISTANCE_WEIGHT = 0.25
MAXIMUM_REFERENCE_SCORE_RATIO = 1.50
MAXIMUM_TARGET_SAMPLES = 256
MAXIMUM_CANDIDATE_SAMPLES = 2048


def _identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing localized-ranking input: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _subsample_indices(count: int, maximum: int) -> np.ndarray:
    if count < 1:
        raise ValueError("cannot subsample an empty array")
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.rint(np.linspace(0, count - 1, maximum)).astype(np.int64)


def _phase_aligned_actions(
    target_phase: np.ndarray,
    source_phase: np.ndarray,
    source_actions: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(target_phase, source_phase, source_actions[:, index])
            for index in range(3)
        ]
    )


def _directed_region_distance(
    target_states: np.ndarray,
    target_actions: np.ndarray,
    candidate_states: np.ndarray,
    candidate_actions: np.ndarray,
) -> dict[str, float]:
    state_distance = (
        np.sum(np.square(target_states), axis=1)[:, None]
        + np.sum(np.square(candidate_states), axis=1)[None, :]
        - 2.0 * target_states @ candidate_states.T
    ) / target_states.shape[1]
    action_distance = (
        np.sum(np.square(target_actions), axis=1)[:, None]
        + np.sum(np.square(candidate_actions), axis=1)[None, :]
        - 2.0 * target_actions @ candidate_actions.T
    ) / target_actions.shape[1]
    nearest = np.sqrt(
        np.maximum(
            np.min(
                STATE_DISTANCE_WEIGHT * state_distance
                + ACTION_DISTANCE_WEIGHT * action_distance,
                axis=1,
            ),
            0.0,
        )
    )
    mean = float(np.mean(nearest))
    p50 = float(np.percentile(nearest, 50))
    p95 = float(np.percentile(nearest, 95))
    return {
        "nearest_distance_mean": mean,
        "nearest_distance_p50": p50,
        "nearest_distance_p95": p95,
        "nearest_distance_max": float(np.max(nearest)),
        "score": 0.5 * mean + 0.5 * p95,
    }


def rank_localized_cases(
    shadow: dict[str, np.ndarray],
    dataset: dict[str, np.ndarray],
    metadata: dict[str, object],
    observation_mean: np.ndarray,
    observation_std: np.ndarray,
    observation_mask: np.ndarray,
    *,
    reference_case: int = 4,
) -> dict[str, object]:
    train_cases = [int(case) for case in metadata["split_cases"]["train"]]
    validation_cases = [
        int(case) for case in metadata["split_cases"]["validation"]
    ]
    holdout_cases = [int(case) for case in metadata["split_cases"]["holdout"]]
    if reference_case not in validation_cases:
        raise ValueError("localized reference must remain in validation")
    if set(train_cases) & (set(validation_cases) | set(holdout_cases)):
        raise ValueError("dataset split cases overlap")
    effective = np.flatnonzero(np.asarray(observation_mask) != 0.0)
    if len(effective) < 1 or np.any(np.isin([23, 24, 25], effective)):
        raise ValueError("masked policy must exclude previous-action channels")

    reference_mask = dataset["case_ids"] == reference_case
    if int(np.count_nonzero(reference_mask)) < 2:
        raise ValueError("reference case is absent from the teacher dataset")
    phase_actions = _phase_aligned_actions(
        shadow["phase_time_s"],
        dataset["phase_time_s"][reference_mask],
        dataset["actions"][reference_mask],
    )
    shadow_shift = (
        shadow["shadow_teacher_normalized_residual_actions"] - phase_actions
    )
    hotspot_mask = np.any(
        np.abs(shadow_shift) > MATERIAL_SHIFT_P95,
        axis=1,
    )
    hotspot_indices = np.flatnonzero(hotspot_mask)
    if len(hotspot_indices) < 2:
        raise ValueError("case-4 shadow trace has no material shift region")
    target_selection = hotspot_indices[
        _subsample_indices(len(hotspot_indices), MAXIMUM_TARGET_SAMPLES)
    ]
    target_states = (
        (shadow["observations"][target_selection] - observation_mean)
        / observation_std
    )[:, effective]
    target_actions = phase_actions[target_selection]

    def case_distance(case: int) -> dict[str, object]:
        case_mask = dataset["case_ids"] == case
        case_rows = np.flatnonzero(case_mask)
        selection = case_rows[
            _subsample_indices(len(case_rows), MAXIMUM_CANDIDATE_SAMPLES)
        ]
        states = (
            (dataset["observations"][selection] - observation_mean)
            / observation_std
        )[:, effective]
        distance = _directed_region_distance(
            target_states,
            target_actions,
            states,
            dataset["actions"][selection],
        )
        return {
            "case": case,
            "split": (
                "validation" if case == reference_case else "train"
            ),
            "source_sample_count": len(case_rows),
            "distance_sample_count": len(selection),
            **distance,
        }

    reference = case_distance(reference_case)
    ranked = [case_distance(case) for case in train_cases]
    for item in ranked:
        item["reference_score_ratio"] = item["score"] / max(
            reference["score"], 1e-12
        )
    ranked.sort(key=lambda item: (item["score"], item["case"]))
    top = ranked[0]
    coverage_passed = bool(
        top["reference_score_ratio"] <= MAXIMUM_REFERENCE_SCORE_RATIO
    )
    proposed_cases = [top["case"]] if coverage_passed else []
    return {
        "schema": "cinebotrl_two_wheel_riser_dagger_localized_case_ranking_v1",
        "reference_case": reference_case,
        "reference_split": "validation_diagnostic_only",
        "localization_contract": (
            "material_shadow_minus_phase_label_rows_directed_state_action_distance_v1"
        ),
        "material_shift_thresholds": MATERIAL_SHIFT_P95.tolist(),
        "shadow_row_count": len(shadow["phase_time_s"]),
        "hotspot_row_count": len(hotspot_indices),
        "hotspot_fraction": float(np.mean(hotspot_mask)),
        "hotspot_channel_counts": np.sum(
            np.abs(shadow_shift) > MATERIAL_SHIFT_P95, axis=0
        ).tolist(),
        "target_distance_sample_count": len(target_selection),
        "maximum_candidate_distance_samples": MAXIMUM_CANDIDATE_SAMPLES,
        "effective_observation_count": len(effective),
        "previous_action_channels_effective": False,
        "distance_weights": {
            "policy_normalized_state": STATE_DISTANCE_WEIGHT,
            "normalized_phase_teacher_action": ACTION_DISTANCE_WEIGHT,
        },
        "score_contract": "half_nearest_mean_plus_half_nearest_p95_v1",
        "reference_calibration": reference,
        "maximum_reference_score_ratio": MAXIMUM_REFERENCE_SCORE_RATIO,
        "ranked_training_cases": ranked,
        "top_training_cases": [item["case"] for item in ranked[:5]],
        "coverage_admission_passed": coverage_passed,
        "proposed_runtime_cases": proposed_cases,
        "classification": (
            "one_training_case_covers_case4_shadow_shift_region"
            if coverage_passed
            else "no_training_case_covers_case4_shadow_shift_region"
        ),
        "validation_labels_used_for_training": False,
        "holdout_opened": False,
        "runtime_authorized": False,
        "authorization_token_issued": False,
        "dataset_created": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "valid_for_training": False,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-trace", type=Path, required=True)
    parser.add_argument("--teacher-dataset", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--case21-final", type=Path, required=True)
    parser.add_argument("--case21-diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shadow_metadata, shadow = _load_npz(args.shadow_trace)
    dataset_metadata, dataset = _load_npz(args.teacher_dataset)
    case21_final = json.loads(args.case21_final.read_text(encoding="utf-8"))
    case21_diagnosis = json.loads(
        args.case21_diagnosis.read_text(encoding="utf-8")
    )
    input_checks = {
        "shadow_schema": shadow_metadata.get("schema")
        == "cinebotrl_two_wheel_riser_shadow_teacher_trace_v1",
        "shadow_unapplied": shadow_metadata.get(
            "shadow_teacher_applied_to_commands"
        )
        is False,
        "shadow_not_trainable": shadow_metadata.get("valid_for_training") is False,
        "case21_physical_pass": case21_final.get("passed") is True,
        "case21_dagger_reject": case21_diagnosis.get(
            "dagger_dataset_proposal_supported"
        )
        is False,
    }
    if not all(input_checks.values()):
        raise ValueError(f"localized-ranking input contract failed: {input_checks}")
    model = torch.jit.load(str(args.policy), map_location="cpu")
    report = rank_localized_cases(
        shadow,
        dataset,
        dataset_metadata,
        model.observation_mean.detach().cpu().numpy(),
        model.observation_std.detach().cpu().numpy(),
        model.observation_mask.detach().cpu().numpy(),
    )
    report["input_contract_checks"] = input_checks
    report["inputs"] = {
        "shadow_trace": _identity(args.shadow_trace),
        "teacher_dataset": _identity(args.teacher_dataset),
        "policy": _identity(args.policy),
        "case21_final": _identity(args.case21_final),
        "case21_diagnosis": _identity(args.case21_diagnosis),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
