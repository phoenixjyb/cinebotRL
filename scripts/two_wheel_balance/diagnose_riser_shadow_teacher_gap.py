#!/usr/bin/env python3
"""Compare unapplied on-policy shadow labels with policy and phase labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ACTION_NAMES = ("vx", "wz", "riser")
MATERIAL_SHIFT_P95 = np.array([0.05, 0.05, 0.02], dtype=np.float64)


def _identity(path: Path) -> dict[str, object]:
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


def _interpolate_actions(
    target_phase: np.ndarray, source_phase: np.ndarray, actions: np.ndarray
) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(target_phase, source_phase, actions[:, index])
            for index in range(3)
        ]
    )


def _metrics(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "abs_p95": np.percentile(np.abs(values), 95, axis=0).tolist(),
        "abs_max": np.max(np.abs(values), axis=0).tolist(),
        "rmse": np.sqrt(np.mean(np.square(values), axis=0)).tolist(),
        "signed_mean": np.mean(values, axis=0).tolist(),
    }


def build_report(
    shadow: dict[str, np.ndarray],
    teacher_dataset: dict[str, np.ndarray],
    *,
    case: int,
) -> dict[str, object]:
    case_mask = teacher_dataset["case_ids"] == case
    if int(np.count_nonzero(case_mask)) < 2:
        raise ValueError(f"teacher dataset has no usable case {case}")
    phase = np.asarray(shadow["phase_time_s"], dtype=np.float64)
    applied = np.asarray(shadow["applied_residual_actions"], dtype=np.float64)
    shadow_labels = np.asarray(
        shadow["shadow_teacher_normalized_residual_actions"], dtype=np.float64
    )
    phase_labels = _interpolate_actions(
        phase,
        teacher_dataset["phase_time_s"][case_mask],
        teacher_dataset["actions"][case_mask],
    )
    policy_to_shadow = applied - shadow_labels
    policy_to_phase = applied - phase_labels
    shadow_to_phase = shadow_labels - phase_labels
    policy_shadow_metrics = _metrics(policy_to_shadow)
    policy_phase_metrics = _metrics(policy_to_phase)
    shadow_shift_metrics = _metrics(shadow_to_phase)
    material_by_channel = (
        np.asarray(shadow_shift_metrics["abs_p95"]) > MATERIAL_SHIFT_P95
    )
    policy_shadow_rmse = float(np.sqrt(np.mean(np.square(policy_to_shadow))))
    policy_phase_rmse = float(np.sqrt(np.mean(np.square(policy_to_phase))))
    shadow_gap_ratio = policy_shadow_rmse / max(policy_phase_rmse, 1e-12)
    supports_proposal = bool(
        np.any(material_by_channel) and shadow_gap_ratio > 1.10
    )
    return {
        "schema": "cinebotrl_two_wheel_riser_shadow_teacher_gap_diagnosis_v1",
        "case": case,
        "alignment_clock": "execution_phase_time_s",
        "sample_count": len(phase),
        "action_names": list(ACTION_NAMES),
        "policy_to_on_policy_shadow_teacher": policy_shadow_metrics,
        "policy_to_original_phase_teacher": policy_phase_metrics,
        "on_policy_shadow_to_original_phase_teacher": shadow_shift_metrics,
        "material_shadow_shift_p95_threshold": MATERIAL_SHIFT_P95.tolist(),
        "material_shadow_shift_by_channel": material_by_channel.tolist(),
        "aggregate_policy_shadow_rmse": policy_shadow_rmse,
        "aggregate_policy_phase_rmse": policy_phase_rmse,
        "policy_shadow_to_phase_gap_ratio": shadow_gap_ratio,
        "dagger_dataset_proposal_supported": supports_proposal,
        "classification": (
            "on_policy_teacher_gap_supports_bounded_dagger_proposal"
            if supports_proposal
            else "on_policy_teacher_gap_does_not_yet_support_dagger_proposal"
        ),
        "shadow_teacher_applied_to_commands": False,
        "dataset_created": False,
        "training_started": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "valid_for_training": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-trace", type=Path, required=True)
    parser.add_argument("--teacher-dataset", type=Path, required=True)
    parser.add_argument("--case", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shadow_metadata, shadow = _load_npz(args.shadow_trace)
    _, teacher_dataset = _load_npz(args.teacher_dataset)
    contract_checks = {
        "schema": shadow_metadata.get("schema")
        == "cinebotrl_two_wheel_riser_shadow_teacher_trace_v1",
        "trace_only": shadow_metadata.get("trace_only") is True,
        "teacher_unapplied": shadow_metadata.get(
            "shadow_teacher_applied_to_commands"
        )
        is False,
        "labels_unadmitted": shadow_metadata.get(
            "shadow_teacher_labels_admitted_for_training"
        )
        is False,
        "not_trainable": shadow_metadata.get("valid_for_training") is False,
    }
    if not all(contract_checks.values()):
        raise ValueError(f"shadow trace contract failed: {contract_checks}")
    report = build_report(shadow, teacher_dataset, case=args.case)
    report["input_contract_checks"] = contract_checks
    report["inputs"] = {
        "shadow_trace": _identity(args.shadow_trace),
        "teacher_dataset": _identity(args.teacher_dataset),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
