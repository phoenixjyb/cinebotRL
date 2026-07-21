#!/usr/bin/env python3
"""Diagnose case-aligned policy-input shift without creating teacher labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


GROUPS = {
    "balance": tuple(range(0, 6)),
    "base": tuple(range(6, 9)),
    "camera": tuple(range(9, 15)),
    "riser": tuple(range(15, 18)),
    "feedforward": tuple(range(18, 21)),
    "phase_progress": tuple(range(21, 23)),
    "previous_action": tuple(range(23, 26)),
    "lookahead_0p25_base": tuple(range(26, 29)),
    "lookahead_0p25_camera": tuple(range(29, 35)),
    "lookahead_0p25_riser": (35,),
    "lookahead_0p25_feedforward": tuple(range(36, 39)),
    "lookahead_0p50_base": tuple(range(39, 42)),
    "lookahead_0p50_camera": tuple(range(42, 48)),
    "lookahead_0p50_riser": (48,),
    "lookahead_0p50_feedforward": tuple(range(49, 52)),
    "lookahead_1p00_base": tuple(range(52, 55)),
    "lookahead_1p00_camera": tuple(range(55, 61)),
    "lookahead_1p00_riser": (61,),
    "lookahead_1p00_feedforward": tuple(range(62, 65)),
}
OUTCOME_COUPLED_GROUPS = {
    "camera",
    "lookahead_0p25_camera",
    "lookahead_0p50_camera",
    "lookahead_1p00_camera",
}


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _load_npz(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {
            name: np.asarray(data[name])
            for name in data.files
            if name != "metadata_json"
        }
    return metadata, arrays


def _interpolate_rows(
    target_time: np.ndarray, source_time: np.ndarray, values: np.ndarray
) -> np.ndarray:
    if values.ndim == 1:
        return np.interp(target_time, source_time, values)
    return np.column_stack(
        [np.interp(target_time, source_time, values[:, index]) for index in range(values.shape[1])]
    )


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def build_report(
    teacher: dict[str, np.ndarray],
    learned: dict[str, np.ndarray],
    teacher_dataset: dict[str, np.ndarray],
    observation_std: np.ndarray,
    observation_mask: np.ndarray,
    observation_names: list[str],
    *,
    case: int,
) -> dict[str, object]:
    phase = learned["phase_time_s"]
    teacher_phase = teacher["phase_time_s"]
    aligned_teacher_observation = _interpolate_rows(
        phase, teacher_phase, teacher["observations"]
    )
    standardized_delta = (
        (learned["observations"] - aligned_teacher_observation)
        / observation_std
        * observation_mask
    )
    aligned_teacher_error = _interpolate_rows(
        phase, teacher_phase, teacher["post_step_position_error_m"]
    )
    excess_error = learned["post_step_position_error_m"] - aligned_teacher_error

    group_metrics = {}
    for name, indices in GROUPS.items():
        norm = np.sqrt(np.mean(np.square(standardized_delta[:, indices]), axis=1))
        group_metrics[name] = {
            "effective_standardized_delta_p95": float(np.percentile(norm, 95)),
            "effective_standardized_delta_max": float(np.max(norm)),
            "correlation_with_excess_position_error": _correlation(norm, excess_error),
            "outcome_coupled_group": name in OUTCOME_COUPLED_GROUPS,
        }

    channel_metrics = []
    for index, name in enumerate(observation_names):
        magnitude = np.abs(standardized_delta[:, index])
        channel_metrics.append(
            {
                "index": index,
                "name": name,
                "effective": bool(observation_mask[index] != 0.0),
                "effective_standardized_delta_p95": float(
                    np.percentile(magnitude, 95)
                ),
                "correlation_with_excess_position_error": _correlation(
                    magnitude, excess_error
                ),
            }
        )

    case_mask = teacher_dataset["case_ids"] == case
    if int(np.count_nonzero(case_mask)) != len(teacher["observations"]):
        raise ValueError("teacher dataset and trace case row counts differ")
    dataset_phase = teacher_dataset["phase_time_s"][case_mask]
    aligned_teacher_actions = _interpolate_rows(
        phase, dataset_phase, teacher_dataset["actions"][case_mask]
    )
    action_error = learned["applied_residual_actions"] - aligned_teacher_actions
    action_metrics = {}
    for index, name in enumerate(("vx", "wz", "riser")):
        magnitude = np.abs(action_error[:, index])
        action_metrics[name] = {
            "normalized_error_p95": float(np.percentile(magnitude, 95)),
            "normalized_error_max": float(np.max(magnitude)),
            "signed_error_mean": float(np.mean(action_error[:, index])),
            "magnitude_correlation_with_excess_position_error": _correlation(
                magnitude, excess_error
            ),
        }

    non_output = {
        name: metrics
        for name, metrics in group_metrics.items()
        if name not in OUTCOME_COUPLED_GROUPS and name != "previous_action"
    }
    best_non_output = max(
        non_output,
        key=lambda name: non_output[name]["correlation_with_excess_position_error"],
    )
    strongest_outcome = max(
        OUTCOME_COUPLED_GROUPS,
        key=lambda name: group_metrics[name]["correlation_with_excess_position_error"],
    )
    precursor_proven = (
        non_output[best_non_output]["correlation_with_excess_position_error"] >= 0.30
    )
    return {
        "schema": "cinebotrl_two_wheel_riser_policy_rate_shift_diagnosis_v1",
        "case": case,
        "alignment_clock": "execution_phase_time_s",
        "teacher_rows": len(teacher["observations"]),
        "learned_rows": len(learned["observations"]),
        "observation_dimension": len(observation_names),
        "normalization_source": "masked_policy_observation_std_and_mask",
        "excess_position_error_m": {
            "median": float(np.median(excess_error)),
            "p95": float(np.percentile(excess_error, 95)),
            "max": float(np.max(excess_error)),
            "max_phase_time_s": float(phase[int(np.argmax(excess_error))]),
        },
        "group_metrics": group_metrics,
        "top_channels_by_effective_delta_p95": sorted(
            channel_metrics,
            key=lambda item: item["effective_standardized_delta_p95"],
            reverse=True,
        )[:15],
        "phase_aligned_teacher_action_error": action_metrics,
        "strongest_non_output_association": best_non_output,
        "strongest_outcome_coupled_association": strongest_outcome,
        "single_non_output_precursor_proven": precursor_proven,
        "classification": (
            "single_non_output_policy_input_precursor_candidate"
            if precursor_proven
            else "no_single_non_output_policy_input_precursor_proven"
        ),
        "interpretation": (
            "Magnitude-only shifts are not causal evidence. Camera error channels "
            "are outcome-coupled; phase-aligned teacher actions are not labels for "
            "policy-visited states."
        ),
        "next_measurement": (
            "bounded_teacher_relabel_on_existing_policy_visited_states_required"
        ),
        "teacher_relabel_capture_started": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "valid_for_training": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-trace", type=Path, required=True)
    parser.add_argument("--learned-trace", type=Path, required=True)
    parser.add_argument("--teacher-dataset", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--case", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    teacher_metadata, teacher = _load_npz(args.teacher_trace)
    learned_metadata, learned = _load_npz(args.learned_trace)
    _, teacher_dataset = _load_npz(args.teacher_dataset)
    if teacher_metadata != learned_metadata:
        raise ValueError("teacher and learned trace metadata differ")
    if teacher_metadata.get("valid_for_training") is not False:
        raise ValueError("input policy traces must remain non-trainable")
    model = torch.jit.load(str(args.policy), map_location="cpu")
    report = build_report(
        teacher,
        learned,
        teacher_dataset,
        model.observation_std.detach().cpu().numpy(),
        model.observation_mask.detach().cpu().numpy(),
        teacher_metadata["observation_names"],
        case=args.case,
    )
    report["inputs"] = {
        "teacher_trace": _identity(args.teacher_trace),
        "learned_trace": _identity(args.learned_trace),
        "teacher_dataset": _identity(args.teacher_dataset),
        "policy": _identity(args.policy),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
