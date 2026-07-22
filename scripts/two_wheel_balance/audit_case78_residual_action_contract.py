#!/usr/bin/env python3
"""Audit the teacher-40 residual scale against the passed case-78 canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


CASE78_GATE_SHA256 = (
    "304fa9e1202d4099f976e6933e9ffc21a2833e7cc380ab9f95d7473bf2126c73"
)
CORPUS_AUDIT_SHA256 = (
    "b5b19d4426185610ab66f2a4d755c8b076a0143f3642a1de9ab848a5b35f5308"
)
DATASET_SUMMARY_SHA256 = (
    "815463ffa133addbaec4f09a453fd9dae8e63eb690b37f56fd0a5c1877879542"
)
SPLIT_ADMISSION_SHA256 = (
    "eac2c8c5389b0a8e3590d5b6355eaa80b50019091d5eb906408a6599c19cb623"
)
EXPECTED_HOLDOUT = [3, 5, 13, 19, 24]
PERCENTILES = (50.0, 90.0, 95.0, 99.0, 99.9)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def vector3(value: Any, name: str, *, nonnegative: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite values")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def source_basename(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def compute_development_distribution(
    summary: dict[str, Any], raw_root: Path
) -> dict[str, Any]:
    chunks: list[np.ndarray] = []
    elapsed_chunks: list[np.ndarray] = []
    rows = []
    skipped_holdout = []
    for source in summary.get("source_rows", []):
        case = int(source["case"])
        if source.get("split") == "holdout":
            skipped_holdout.append(case)
            continue
        path = raw_root / source_basename(str(source["raw_case"]))
        if not path.is_file() or sha256_file(path) != source.get("raw_case_sha256"):
            raise ValueError(f"raw source identity mismatch for case {case}")
        with np.load(path, allow_pickle=False) as payload:
            raw = np.asarray(payload["raw_residual_commands"], dtype=np.float64)
            elapsed = np.asarray(payload["elapsed_time_s"], dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) != len(elapsed):
            raise ValueError(f"invalid raw residual shape for case {case}")
        if len(raw) != int(source["row_count"]) or not np.isfinite(raw).all():
            raise ValueError(f"raw residual row contract failed for case {case}")
        if len(elapsed) < 2 or not np.isfinite(elapsed).all():
            raise ValueError(f"invalid elapsed clock for case {case}")
        delta = np.diff(elapsed)
        if np.any(delta <= 0.0):
            raise ValueError(f"elapsed clock is not strictly increasing for case {case}")
        chunks.append(raw)
        elapsed_chunks.append(elapsed)
        rows.append(
            {
                "case": case,
                "split": source["split"],
                "row_count": len(raw),
                "raw_case_sha256": source["raw_case_sha256"],
            }
        )
    if not chunks:
        raise ValueError("no train/validation residual rows found")
    all_raw = np.concatenate(chunks, axis=0)
    absolute = np.abs(all_raw)
    return {
        "case_count": len(rows),
        "cases": [row["case"] for row in rows],
        "row_count": len(all_raw),
        "skipped_holdout_cases": sorted(skipped_holdout),
        "holdout_files_opened": False,
        "raw_residual_signed_min": np.min(all_raw, axis=0).tolist(),
        "raw_residual_signed_max": np.max(all_raw, axis=0).tolist(),
        "raw_residual_abs_max": np.max(absolute, axis=0).tolist(),
        "raw_residual_abs_percentiles": {
            str(percentile): np.percentile(absolute, percentile, axis=0).tolist()
            for percentile in PERCENTILES
        },
        "observed_duration_s": float(
            sum(np.sum(np.diff(elapsed)) for elapsed in elapsed_chunks)
        ),
        "rows": rows,
        "_raw_chunks": chunks,
        "_elapsed_chunks": elapsed_chunks,
    }


def overflow_statistics(
    raw_chunks: list[np.ndarray],
    elapsed_chunks: list[np.ndarray],
    scales: np.ndarray,
) -> dict[str, Any]:
    counts = np.zeros(3, dtype=np.int64)
    duration = np.zeros(3, dtype=np.float64)
    total_rows = 0
    total_duration = 0.0
    for raw, elapsed in zip(raw_chunks, elapsed_chunks, strict=True):
        overflow = np.abs(raw) >= scales
        delta = np.diff(elapsed)
        counts += np.sum(overflow, axis=0)
        duration += np.sum(overflow[:-1] * delta[:, None], axis=0)
        total_rows += len(raw)
        total_duration += float(np.sum(delta))
    return {
        "comparison": "absolute_raw_residual_greater_than_or_equal_to_scale",
        "duration_contract": "left_sample_forward_interval_v1",
        "overflow_sample_count": counts.tolist(),
        "overflow_sample_ratio": (counts / total_rows).tolist(),
        "overflow_duration_s": duration.tolist(),
        "overflow_duration_ratio": (
            duration / total_duration if total_duration > 0.0 else duration
        ).tolist(),
    }


def _zero3(value: Any) -> bool:
    try:
        return bool(np.allclose(vector3(value, "zero vector"), 0.0, atol=1e-12))
    except ValueError:
        return False


def build_report(
    gate: dict[str, Any],
    summary: dict[str, Any],
    corpus_audit: dict[str, Any],
    split: dict[str, Any],
    distribution: dict[str, Any],
) -> dict[str, Any]:
    results = gate.get("results", [])
    case78 = results[0] if len(results) == 1 and isinstance(results[0], dict) else {}
    runtime_scales = vector3(gate.get("residual_action_scales"), "runtime scales")
    frozen_scales = vector3(summary.get("action_scales"), "frozen scales")
    summary_action_max = vector3(summary.get("action_abs_max"), "action max")
    summary_clip_ratio = vector3(summary.get("action_clip_ratio"), "clip ratio")
    corpus_raw_max = vector3(
        corpus_audit.get("raw_residual_abs_max"), "corpus raw max", nonnegative=True
    )
    case78_raw_max = vector3(
        case78.get("raw_residual_command_abs_max"),
        "case78 raw max",
        nonnegative=True,
    )
    case78_normalized = vector3(
        case78.get("normalized_residual_label_abs_max"),
        "case78 normalized max",
        nonnegative=True,
    )
    admitted = split.get("admitted_split_cases", {})
    checks = {
        "case78_gate_is_passed_dynamic_evidence": gate.get("cases") == [78]
        and gate.get("passed") is True
        and gate.get("dynamic_quality_passed") is True
        and case78.get("case") == 78
        and case78.get("dynamic_quality_passed") is True
        and case78.get("thermal_admission_passed") is True
        and case78.get("controller_evidence_passed") is True
        and case78.get("termination") is None,
        "case78_completed_separate_clocks": case78.get("source_duration_s", 0.0) > 0.0
        and case78.get("execution_duration_s", 0.0)
        > case78.get("source_duration_s", 0.0)
        and np.isclose(
            case78.get("completed_phase_time_s", -1.0),
            case78.get("execution_duration_s", 0.0),
            atol=1e-9,
        ),
        "case78_label_was_observational_only": case78.get(
            "raw_residual_label_applied_to_commands"
        )
        is False
        and _zero3(case78.get("residual_action_abs_max"))
        and case78.get("executed_residual_dataset") is None,
        "case78_old_envelope_rejection_is_preserved": gate.get(
            "residual_label_envelope_passed"
        )
        is False
        and gate.get("residual_label_admission_passed") is False
        and case78.get("residual_label_envelope_passed") is False
        and case78.get("residual_label_admission_passed") is False,
        "case78_runtime_normalization_reconstructs": np.allclose(
            case78_raw_max / runtime_scales,
            case78_normalized,
            atol=2e-9,
            rtol=0.0,
        ),
        "teacher40_summary_is_valid_and_immutable": summary.get("passed") is True
        and summary.get("valid_for_bc_initialization") is True
        and summary.get("dataset_admission_passed") is True
        and summary.get("case_count") == 40
        and summary.get("captured_case_count") == 41
        and summary.get("row_count") == 403569
        and summary.get("trajectory_leakage") is False,
        "teacher40_actions_are_semantic_and_unclipped": summary.get(
            "physical_gimbal_labels_used_as_actions"
        )
        is False
        and summary.get("source_action_labels_used") is False
        and np.all(summary_clip_ratio == 0.0)
        and np.all(summary_action_max < 1.0),
        "corpus_audit_freezes_same_scale": corpus_audit.get("passed") is True
        and corpus_audit.get("valid_for_bc_initialization") is True
        and corpus_audit.get("action_scale_frozen") is True
        and corpus_audit.get("case_count") == 41
        and corpus_audit.get("row_count") == 406837
        and np.allclose(
            vector3(corpus_audit.get("frozen_action_scales"), "audit scales"),
            frozen_scales,
            atol=1e-12,
            rtol=0.0,
        ),
        "summary_normalization_reconstructs_corpus_max": np.allclose(
            summary_action_max * frozen_scales,
            corpus_raw_max,
            atol=2e-8,
            rtol=0.0,
        ),
        "development_distribution_preserves_holdout": distribution.get(
            "skipped_holdout_cases"
        )
        == EXPECTED_HOLDOUT
        and distribution.get("holdout_files_opened") is False
        and distribution.get("case_count") == 35,
        "split_roles_are_admitted_without_labels": split.get("split_admitted")
        is True
        and admitted.get("holdout") == EXPECTED_HOLDOUT
        and split.get("case78_labels_available") is False
        and split.get("dataset_creation_authorized") is False
        and split.get("label_capture_authorized") is False,
        "learning_paths_remain_closed": all(
            item is False
            for item in (
                gate.get("training_started"),
                gate.get("ppo_authorized"),
                summary.get("training_started"),
                summary.get("bc_authorized"),
                summary.get("ppo_authorized"),
                split.get("bc_authorized"),
                split.get("ppo_authorized"),
            )
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"residual action contract audit failed: {checks}")

    combined_max = np.maximum(corpus_raw_max, case78_raw_max)
    candidate_utilization = combined_max / frozen_scales
    runtime_overflow_channels = (case78_raw_max >= runtime_scales).tolist()
    candidate_max_compatible = bool(np.all(candidate_utilization < 1.0))
    public_distribution = {
        key: value for key, value in distribution.items() if not key.startswith("_")
    }
    return {
        "schema": "cinebotrl_two_wheel_riser_residual_action_contract_audit_v1",
        "decision": "retain_teacher40_scale_case78_series_measurement_required",
        "input_contract_checks": checks,
        "controller_boundary": {
            "primary_balance_owner": "deterministic_lqr_inner_loop",
            "trajectory_allocation_owner": "model_based_outer_controller",
            "learned_policy_role": "bounded_residual_only",
            "safety_supervisor_remains_final_authority": True,
            "raw_labels_applied_to_physical_commands": False,
        },
        "teacher40_label_distribution": public_distribution,
        "teacher40_candidate_scale": frozen_scales.tolist(),
        "teacher40_action_abs_max": summary_action_max.tolist(),
        "teacher40_action_clip_ratio": summary_clip_ratio.tolist(),
        "case78_observed_raw_abs_max": case78_raw_max.tolist(),
        "case78_runtime_scale": runtime_scales.tolist(),
        "case78_runtime_scale_normalized_abs_max": case78_normalized.tolist(),
        "case78_runtime_scale_overflow_channels": runtime_overflow_channels,
        "case78_runtime_scale_overflow_sample_count": None,
        "case78_runtime_scale_overflow_duration_s": None,
        "combined_observed_raw_abs_max": combined_max.tolist(),
        "candidate_scale_normalized_abs_max": candidate_utilization.tolist(),
        "candidate_scale_headroom": (frozen_scales - combined_max).tolist(),
        "candidate_scale_maximum_compatibility_passed": candidate_max_compatible,
        "teacher40_action_contract_retained": candidate_max_compatible,
        "action_scale_change_required": False,
        "action_clipping_permitted": False,
        "case78_policy_rate_label_series_available": False,
        "case78_quantiles_available": False,
        "case78_shadow_measurement_required_before_label_capture": True,
        "case78_label_capture_authorized": False,
        "case78_dataset_admission_passed": False,
        "historical_teacher40_dataset_rewrite_required": False,
        "next_dataset_split_manifest_available": True,
        "holdout_opened": False,
        "dataset_creation_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
        "next_bounded_task": (
            "define a no-application case78 shadow-label measurement contract that "
            "records policy-rate raw residual labels, timestamps, percentiles, and "
            "overflow duration without creating a dataset"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case78-gate", type=Path, required=True)
    parser.add_argument("--dataset-summary", type=Path, required=True)
    parser.add_argument("--corpus-audit", type=Path, required=True)
    parser.add_argument("--split-admission", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "case78_gate": args.case78_gate,
        "dataset_summary": args.dataset_summary,
        "corpus_audit": args.corpus_audit,
        "split_admission": args.split_admission,
    }
    expected = {
        "case78_gate": CASE78_GATE_SHA256,
        "dataset_summary": DATASET_SUMMARY_SHA256,
        "corpus_audit": CORPUS_AUDIT_SHA256,
        "split_admission": SPLIT_ADMISSION_SHA256,
    }
    identities = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"missing audit input: {path}")
        digest = sha256_file(path)
        if digest != expected[name]:
            raise ValueError(f"{name} hash mismatch")
        identities[name] = {"path": str(path.resolve()), "sha256": digest}
    summary = load_json(args.dataset_summary)
    if summary.get("corpus_audit_sha256") != CORPUS_AUDIT_SHA256:
        raise ValueError("dataset summary does not bind the corpus audit")
    distribution = compute_development_distribution(summary, args.raw_root)
    candidate_scales = vector3(summary.get("action_scales"), "candidate scales")
    distribution["candidate_scale_overflow"] = overflow_statistics(
        distribution["_raw_chunks"],
        distribution["_elapsed_chunks"],
        candidate_scales,
    )
    report = build_report(
        load_json(args.case78_gate),
        summary,
        load_json(args.corpus_audit),
        load_json(args.split_admission),
        distribution,
    )
    report["inputs"] = identities
    report["raw_root"] = str(args.raw_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
