#!/usr/bin/env python3
"""Diagnose teacher-forcing and rollout gaps for a riser BC policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ACTION_NAMES = ("residual_vx", "residual_wz", "residual_riser_target")
PREVIOUS_ACTION_SLICE = slice(23, 26)
SPLIT_CODES = {"train": 0, "validation": 1, "holdout": 2}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def infer_batched(
    model: torch.jit.ScriptModule, observations: np.ndarray, batch_size: int
) -> np.ndarray:
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(observations), batch_size):
            batch = torch.from_numpy(observations[start : start + batch_size])
            predictions.append(model(batch).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def infer_recursive_previous_action(
    model: torch.jit.ScriptModule, observations: np.ndarray
) -> np.ndarray:
    predictions = []
    previous = np.zeros(3, dtype=np.float32)
    with torch.inference_mode():
        for observation in observations:
            policy_input = observation.copy()
            policy_input[PREVIOUS_ACTION_SLICE] = previous
            previous = (
                model(torch.from_numpy(policy_input[None])).cpu().numpy()[0]
            )
            predictions.append(previous.copy())
    return np.asarray(predictions, dtype=np.float32)


def prediction_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - target
    mse = np.mean(np.square(error), axis=0)
    correlations = []
    for channel in range(target.shape[1]):
        if np.std(target[:, channel]) <= 1e-12 or np.std(prediction[:, channel]) <= 1e-12:
            correlations.append(0.0)
        else:
            correlations.append(
                float(np.corrcoef(target[:, channel], prediction[:, channel])[0, 1])
            )
    return {
        "mse_per_action": mse.tolist(),
        "mae_per_action": np.mean(np.abs(error), axis=0).tolist(),
        "p95_abs_error_per_action": np.quantile(
            np.abs(error), 0.95, axis=0
        ).tolist(),
        "correlation_per_action": correlations,
        "prediction_abs_max": np.max(np.abs(prediction), axis=0).tolist(),
        "aggregate_mse": float(np.mean(np.square(error))),
    }


def classify_diagnosis(
    teacher_previous: dict,
    recursive_previous: dict,
    zero_baseline: dict,
    learned_position_p95_m: float,
    teacher_position_p95_m: float,
) -> dict:
    teacher_ratio = teacher_previous["aggregate_mse"] / zero_baseline["aggregate_mse"]
    recursive_ratio = recursive_previous["aggregate_mse"] / zero_baseline["aggregate_mse"]
    fit_passed = teacher_ratio <= 0.10 and min(
        teacher_previous["correlation_per_action"]
    ) >= 0.90
    recursive_passed = recursive_ratio <= 0.10
    hard_tracking_passed = learned_position_p95_m <= 0.15
    teacher_budget_m = teacher_position_p95_m * 1.05
    comparison_tracking_passed = learned_position_p95_m <= teacher_budget_m
    if fit_passed and not recursive_passed and not comparison_tracking_passed:
        classification = "autoregressive_previous_action_exposure_bias"
    elif not fit_passed:
        classification = "teacher_state_policy_fit_failure"
    elif not comparison_tracking_passed:
        classification = "physical_state_covariate_shift_not_explained_by_action_history"
    else:
        classification = "no_case_level_bc_tracking_failure"
    return {
        "classification": classification,
        "teacher_state_fit_passed": fit_passed,
        "recursive_previous_action_stability_passed": recursive_passed,
        "hard_tracking_passed": hard_tracking_passed,
        "comparison_tracking_passed": comparison_tracking_passed,
        "teacher_state_mse_ratio_to_zero": teacher_ratio,
        "recursive_previous_mse_ratio_to_zero": recursive_ratio,
        "teacher_comparison_budget_m": teacher_budget_m,
    }


def phase_rows(
    phase_fraction: np.ndarray,
    target: np.ndarray,
    teacher_prediction: np.ndarray,
    recursive_prediction: np.ndarray,
    bin_count: int,
) -> list[dict]:
    rows = []
    for index in range(bin_count):
        low = index / bin_count
        high = (index + 1) / bin_count
        mask = (phase_fraction >= low) & (
            (phase_fraction < high) if index + 1 < bin_count else (phase_fraction <= high)
        )
        if not np.any(mask):
            continue
        row: dict[str, float | int] = {
            "phase_low": low,
            "phase_high": high,
            "row_count": int(np.sum(mask)),
        }
        for channel, name in enumerate(ACTION_NAMES):
            row[f"{name}_teacher_mean"] = float(np.mean(target[mask, channel]))
            row[f"{name}_teacher_previous_prediction_mean"] = float(
                np.mean(teacher_prediction[mask, channel])
            )
            row[f"{name}_recursive_previous_prediction_mean"] = float(
                np.mean(recursive_prediction[mask, channel])
            )
            row[f"{name}_teacher_previous_mae"] = float(
                np.mean(np.abs(teacher_prediction[mask, channel] - target[mask, channel]))
            )
            row[f"{name}_recursive_previous_mae"] = float(
                np.mean(np.abs(recursive_prediction[mask, channel] - target[mask, channel]))
            )
        rows.append(row)
    return rows


def write_phase_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_action_diagnosis(path: Path, rows: list[dict]) -> None:
    phase = np.asarray([(row["phase_low"] + row["phase_high"]) / 2 for row in rows])
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for axis, name in zip(axes, ACTION_NAMES, strict=True):
        axis.plot(phase, [row[f"{name}_teacher_mean"] for row in rows], label="Teacher")
        axis.plot(
            phase,
            [row[f"{name}_teacher_previous_prediction_mean"] for row in rows],
            label="BC with teacher previous action",
        )
        axis.plot(
            phase,
            [row[f"{name}_recursive_previous_prediction_mean"] for row in rows],
            label="BC with recursive previous action",
        )
        axis.set_ylabel(name)
        axis.grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Execution phase fraction")
    figure.suptitle("Case 4 BC teacher-forcing diagnosis")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def result_item(path: Path) -> tuple[dict, dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    results = document.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"expected exactly one result in {path}")
    return document, results[0]


def plot_rollout_diagnosis(
    path: Path, teacher: dict, zero: dict, learned: dict
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for label, item in (("Teacher", teacher), ("Zero residual", zero), ("BC residual", learned)):
        trace = item["trace"]
        phase = np.asarray([row["phase_time_s"] for row in trace]) / item[
            "execution_duration_s"
        ]
        axes[0].plot(phase, [row["position_error_m"] for row in trace], label=label)
    axes[0].axhline(0.15, color="red", linestyle="--", label="0.15 m hard gate")
    axes[0].set_ylabel("Camera position error (m)")
    axes[0].set_xlabel("Execution phase fraction")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")
    for label, item in (("Teacher", teacher), ("Zero residual", zero), ("BC residual", learned)):
        trace = item["trace"]
        phase = np.asarray([row["phase_time_s"] for row in trace]) / item[
            "execution_duration_s"
        ]
        axes[1].plot(
            phase,
            [float(row["camera_lever_arm_correction_saturated"]) for row in trace],
            label=label,
        )
    axes[1].set_ylabel("Lever correction saturated")
    axes[1].set_xlabel("Execution phase fraction")
    axes[1].set_yticks((0, 1))
    axes[1].grid(alpha=0.25)
    figure.suptitle("Case 4 dynamic rollout diagnosis (1 Hz evidence trace)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--teacher-gate", type=Path, required=True)
    parser.add_argument("--zero-rollout", type=Path, required=True)
    parser.add_argument("--learned-rollout", type=Path, required=True)
    parser.add_argument("--case", type=int, default=4)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--phase-bin-count", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if sha256(args.dataset) != args.expected_dataset_sha256:
        raise ValueError("dataset hash mismatch")
    if sha256(args.policy) != args.expected_policy_sha256:
        raise ValueError("policy hash mismatch")
    if args.phase_bin_count <= 0 or args.batch_size <= 0:
        raise ValueError("phase bin count and batch size must be positive")
    with np.load(args.dataset, allow_pickle=False) as dataset:
        metadata = json.loads(str(dataset["metadata_json"].item()))
        case_mask = dataset["case_ids"] == args.case
        if not np.any(case_mask):
            raise ValueError("case is absent from dataset")
        if args.case not in metadata["split_cases"]["validation"]:
            raise ValueError("diagnostic case must be in validation, not train or holdout")
        if not np.all(dataset["split_labels"][case_mask] == SPLIT_CODES["validation"]):
            raise ValueError("case split labels are not validation")
        observations = dataset["observations"][case_mask].astype(np.float32)
        target = dataset["actions"][case_mask].astype(np.float32)
        phase_time = dataset["phase_time_s"][case_mask].astype(np.float64)
    model = torch.jit.load(str(args.policy), map_location="cpu").eval()
    teacher_prediction = infer_batched(model, observations, args.batch_size)
    zero_previous_observations = observations.copy()
    zero_previous_observations[:, PREVIOUS_ACTION_SLICE] = 0.0
    zero_previous_prediction = infer_batched(
        model, zero_previous_observations, args.batch_size
    )
    recursive_prediction = infer_recursive_previous_action(model, observations)
    zero_prediction = np.zeros_like(target)
    predictions = {
        "teacher_previous_action": teacher_prediction,
        "zero_previous_action": zero_previous_prediction,
        "recursive_previous_action": recursive_prediction,
        "zero_policy_action": zero_prediction,
    }
    metrics = {name: prediction_metrics(target, value) for name, value in predictions.items()}
    teacher_document, teacher_item = result_item(args.teacher_gate)
    zero_document, zero_item = result_item(args.zero_rollout)
    learned_document, learned_item = result_item(args.learned_rollout)
    for name, document in (
        ("teacher", teacher_document),
        ("zero", zero_document),
        ("learned", learned_document),
    ):
        if document.get("cases") != [args.case]:
            raise ValueError(f"{name} rollout case mismatch")
    diagnosis = classify_diagnosis(
        metrics["teacher_previous_action"],
        metrics["recursive_previous_action"],
        metrics["zero_policy_action"],
        learned_item["position_error_p95_m"],
        teacher_item["position_error_p95_m"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    phase_fraction = phase_time / phase_time[-1]
    rows = phase_rows(
        phase_fraction,
        target,
        teacher_prediction,
        recursive_prediction,
        args.phase_bin_count,
    )
    phase_csv = args.output_dir / "case_0004_phase_action_diagnosis.csv"
    action_plot = args.output_dir / "case_0004_teacher_forcing_diagnosis.png"
    rollout_plot = args.output_dir / "case_0004_rollout_tracking_diagnosis.png"
    write_phase_csv(phase_csv, rows)
    plot_action_diagnosis(action_plot, rows)
    plot_rollout_diagnosis(rollout_plot, teacher_item, zero_item, learned_item)
    report = {
        "schema": "cinebotrl_two_wheel_riser_bc_rollout_diagnosis_v1",
        "case": args.case,
        "split": "validation",
        "inputs": {
            "dataset": identity(args.dataset),
            "policy": identity(args.policy),
            "teacher_gate": identity(args.teacher_gate),
            "zero_rollout": identity(args.zero_rollout),
            "learned_rollout": identity(args.learned_rollout),
        },
        "row_count": len(observations),
        "action_names": list(ACTION_NAMES),
        "previous_action_indices": [23, 24, 25],
        "metrics": metrics,
        "rollout_metrics": {
            "teacher_position_p95_m": teacher_item["position_error_p95_m"],
            "zero_position_p95_m": zero_item["position_error_p95_m"],
            "learned_position_p95_m": learned_item["position_error_p95_m"],
            "learned_completed_phase_time_s": learned_item["completed_phase_time_s"],
            "execution_duration_s": learned_item["execution_duration_s"],
            "learned_camera_lever_arm_correction_saturation_ratio": learned_item[
                "camera_lever_arm_correction_saturation_ratio"
            ],
        },
        "diagnosis": diagnosis,
        "recommended_next_change": (
            "Remove teacher-forced previous-action exposure from the first bounded "
            "candidate by masking indices 23:26 inside the policy, then retrain and "
            "require validation-only offline gates before one new case-4 canary."
        ),
        "alternative_if_masked_candidate_fails": (
            "Use sequence-aware scheduled sampling or DAgger-style deterministic "
            "teacher relabeling on policy-visited states; do not start PPO."
        ),
        "artifacts": {
            "phase_csv": identity(phase_csv),
            "teacher_forcing_plot": identity(action_plot),
            "rollout_tracking_plot": identity(rollout_plot),
        },
        "holdout_opened": False,
        "isaac_launched": False,
        "ppo_authorized": False,
        "ppo_started": False,
        "passed": diagnosis["classification"] == "autoregressive_previous_action_exposure_bias",
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
