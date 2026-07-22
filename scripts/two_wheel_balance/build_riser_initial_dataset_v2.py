#!/usr/bin/env python3
"""Append admitted case 78 to immutable teacher-40 using the sealed split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    OBSERVATION_INDEX,
    PREVIOUS_ACTION_INDICES,
    load_raw_teacher_case,
    normalize_raw_teacher_payload,
)


BASE_DATASET_SHA256 = (
    "53f3b679e227446c6008ba8bcd9191ae877b946dd86644388c43f89723bb9d44"
)
BASE_SUMMARY_SHA256 = (
    "815463ffa133addbaec4f09a453fd9dae8e63eb690b37f56fd0a5c1877879542"
)
SPLIT_ADMISSION_SHA256 = (
    "eac2c8c5389b0a8e3590d5b6355eaa80b50019091d5eb906408a6599c19cb623"
)
SPLIT_CODES = {"train": 0, "validation": 1, "holdout": 2}
ARRAY_NAMES = (
    "observations",
    "actions",
    "case_ids",
    "elapsed_time_s",
    "phase_time_s",
    "baseline_wheel_actions",
    "teacher_commands",
    "source_index",
    "split_labels",
    "action_valid_mask",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _identity(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"missing dataset-build input: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def build_dataset(
    *,
    base_metadata: dict[str, Any],
    base_arrays: dict[str, np.ndarray],
    base_summary: dict[str, Any],
    split_admission: dict[str, Any],
    raw_payload: dict[str, np.ndarray],
    raw_summary: dict[str, Any],
    label_admission: dict[str, Any],
    expected_base_rows: int = 403569,
    expected_total_rows: int = 486619,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    scales = np.asarray(base_metadata.get("action_scales"), dtype=np.float64)
    admitted = split_admission.get("admitted_split_cases", {})
    expected_split = {
        "train": [2, 4, 6, 7, 9, 10, 11, 12, 14, 15, 17, 18, 21, 23, 25, 26, 28, 30, 31, 33, 34, 36, 37, 41, 52, 53, 66, 67, 68, 70, 74],
        "validation": [8, 16, 22, 32, 78],
        "holdout": [3, 5, 13, 19, 24],
    }
    checks = {
        "base_schema": base_metadata.get("schema")
        == "cinebotrl_two_wheel_riser_residual_merged_v3",
        "base_admitted_and_closed": base_metadata.get("dataset_admission_passed")
        is True
        and base_metadata.get("valid_for_bc_initialization") is True
        and base_metadata.get("bc_authorized") is False
        and base_metadata.get("ppo_authorized") is False
        and base_metadata.get("training_started") is False,
        "base_shape": base_metadata.get("case_count") == 40
        and base_metadata.get("row_count") == expected_base_rows
        and len(base_arrays["observations"]) == expected_base_rows,
        "base_summary_matches": base_summary.get("dataset_sha256")
        == BASE_DATASET_SHA256
        and base_summary.get("split_cases") == base_metadata.get("split_cases"),
        "split_exact": split_admission.get("split_admitted") is True
        and admitted == expected_split,
        "case78_label_admitted": label_admission.get("label_admission_passed")
        is True
        and label_admission.get("offline_dataset_rebuild_authorized") is True
        and label_admission.get("case") == 78,
        "raw_conversion_admitted": raw_summary.get("raw_teacher_conversion_passed")
        is True
        and raw_summary.get("offline_dataset_rebuild_authorized") is True
        and raw_summary.get("case") == 78,
        "frozen_scale_exact": scales.shape == (3,)
        and np.allclose(
            scales, np.asarray([0.35, 0.4, 0.1]), atol=1e-15, rtol=0.0
        ),
        "learning_closed": raw_summary.get("bc_authorized") is False
        and raw_summary.get("ppo_authorized") is False
        and label_admission.get("bc_authorized") is False
        and label_admission.get("ppo_authorized") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"initial dataset-v2 admission failed: {checks}")
    normalized = normalize_raw_teacher_payload(raw_payload, scales)
    if not np.array_equal(np.unique(normalized["case_ids"]), [78]):
        raise ValueError("supplemental raw teacher is not case 78")
    merged = {
        name: np.asarray(base_arrays[name]).copy()
        for name in ARRAY_NAMES
        if name not in {"source_index", "split_labels", "action_valid_mask"}
    }
    for name in (
        "observations",
        "actions",
        "case_ids",
        "elapsed_time_s",
        "phase_time_s",
        "baseline_wheel_actions",
        "teacher_commands",
    ):
        merged[name] = np.concatenate((merged[name], normalized[name]), axis=0)
    old_sources = np.asarray(base_arrays["source_index"])
    new_source = int(np.max(old_sources)) + 1
    merged["source_index"] = np.concatenate(
        (old_sources, np.full(len(normalized["case_ids"]), new_source, dtype=old_sources.dtype))
    )
    split_by_case = {
        case: split for split, cases in expected_split.items() for case in cases
    }
    merged["split_labels"] = np.asarray(
        [SPLIT_CODES[split_by_case[int(case)]] for case in merged["case_ids"]],
        dtype=np.int8,
    )
    merged["action_valid_mask"] = np.ones_like(merged["actions"], dtype=np.float32)
    unique_cases = sorted(int(case) for case in np.unique(merged["case_ids"]))
    if unique_cases != sorted(sum(expected_split.values(), [])):
        raise ValueError("merged case set does not match admitted split")
    previous_ok = True
    for case in unique_cases:
        mask = merged["case_ids"] == case
        previous = merged["observations"][mask][:, PREVIOUS_ACTION_INDICES]
        actions = merged["actions"][mask]
        previous_ok &= bool(np.allclose(previous[0], 0.0, atol=1e-12))
        previous_ok &= bool(np.allclose(previous[1:], actions[:-1], atol=1e-7))
    reconstructed = np.column_stack(
        (
            merged["observations"][:, OBSERVATION_INDEX["feedforward_vx_m_s"]]
            + scales[0] * merged["actions"][:, 0],
            merged["observations"][:, OBSERVATION_INDEX["feedforward_wz_rad_s"]]
            + scales[1] * merged["actions"][:, 1],
            merged["observations"][:, OBSERVATION_INDEX["riser_position_m"]]
            + scales[2] * merged["actions"][:, 2],
        )
    )
    reconstruction_error = float(
        np.max(np.abs(reconstructed - merged["teacher_commands"]))
    )
    action_abs_max = np.max(np.abs(merged["actions"]), axis=0)
    passed = bool(
        previous_ok
        and reconstruction_error <= 2e-6
        and np.max(action_abs_max) < 1.0 - 1e-6
        and len(merged["observations"]) == expected_total_rows
    )
    if not passed:
        raise ValueError("merged dataset failed recurrence, reconstruction, or scale gate")
    source_rows = []
    for row in base_metadata.get("source_rows", []):
        copied = dict(row)
        copied["split"] = split_by_case[int(copied["case"])]
        source_rows.append(copied)
    source_rows.append(
        {
            "case": 78,
            "raw_case": raw_summary["raw_teacher"],
            "raw_case_sha256": raw_summary["raw_teacher_sha256"],
            "row_count": len(normalized["observations"]),
            "split": "validation",
        }
    )
    metadata = dict(base_metadata)
    base_corpus_audit = metadata.pop("corpus_audit", None)
    base_corpus_audit_sha256 = metadata.pop("corpus_audit_sha256", None)
    metadata.pop("seed", None)
    metadata.update(
        {
            "schema": "cinebotrl_two_wheel_riser_residual_merged_v3",
            "dataset_version": "initial_teacher41_case78_31_5_5_v2",
            "construction_contract": "immutable_teacher40_plus_admitted_case78_v1",
            "base_corpus_audit": base_corpus_audit,
            "base_corpus_audit_sha256": base_corpus_audit_sha256,
            "split_assignment_contract": "sealed_case4_train_case78_validation_v2",
            "split_assignment_randomized": False,
            "case_count": 41,
            "captured_case_count": 42,
            "row_count": len(merged["observations"]),
            "split_cases": expected_split,
            "split_case_counts": {name: len(cases) for name, cases in expected_split.items()},
            "coverage_only_cases": [77],
            "trajectory_leakage": False,
            "action_abs_max": action_abs_max.tolist(),
            "action_clip_ratio": np.mean(
                np.abs(merged["actions"]) >= 1.0 - 1e-6, axis=0
            ).tolist(),
            "teacher_command_reconstruction_max_error": reconstruction_error,
            "previous_action_rebuilt": previous_ok,
            "source_rows": source_rows,
            "base_dataset_preserved_immutable": True,
            "base_dataset_rewrite_performed": False,
            "holdout_source_files_opened": False,
            "holdout_policy_metrics_computed": False,
            "holdout_used_for_model_selection": False,
            "holdout_rows_copied_for_dataset_integrity": True,
            "input_contract_checks": checks,
            "dataset_admission_passed": True,
            "valid_for_bc_initialization": True,
            "bc_authorized": False,
            "ppo_authorized": False,
            "training_started": False,
        }
    )
    return metadata, merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument("--split-admission", type=Path, required=True)
    parser.add_argument("--label-admission", type=Path, required=True)
    parser.add_argument("--case78-raw", type=Path, required=True)
    parser.add_argument("--case78-raw-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.output.with_suffix(".summary.json")
    if args.output.exists() or summary_path.exists():
        raise ValueError("refusing to overwrite initial dataset-v2 output")
    identities = {
        "base_dataset": _identity(args.base_dataset),
        "base_summary": _identity(args.base_summary),
        "split_admission": _identity(args.split_admission),
        "label_admission": _identity(args.label_admission),
        "case78_raw": _identity(args.case78_raw),
        "case78_raw_summary": _identity(args.case78_raw_summary),
    }
    if identities["base_dataset"]["sha256"] != BASE_DATASET_SHA256:
        raise ValueError("base dataset identity mismatch")
    if identities["base_summary"]["sha256"] != BASE_SUMMARY_SHA256:
        raise ValueError("base summary identity mismatch")
    if identities["split_admission"]["sha256"] != SPLIT_ADMISSION_SHA256:
        raise ValueError("split admission identity mismatch")
    raw_summary = _load_json(args.case78_raw_summary)
    label_admission = _load_json(args.label_admission)
    if raw_summary.get("raw_teacher_sha256") != identities["case78_raw"]["sha256"]:
        raise ValueError("case-78 raw identity mismatch")
    if raw_summary.get("label_admission_sha256") != identities["label_admission"]["sha256"]:
        raise ValueError("case-78 label-admission identity mismatch")
    with np.load(args.base_dataset, allow_pickle=False) as data:
        base_metadata = json.loads(str(data["metadata_json"].item()))
        base_arrays = {name: np.asarray(data[name]) for name in ARRAY_NAMES}
    _, raw_payload = load_raw_teacher_case(args.case78_raw)
    metadata, merged = build_dataset(
        base_metadata=base_metadata,
        base_arrays=base_arrays,
        base_summary=_load_json(args.base_summary),
        split_admission=_load_json(args.split_admission),
        raw_payload=raw_payload,
        raw_summary=raw_summary,
        label_admission=label_admission,
    )
    metadata["inputs"] = identities
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **merged,
    )
    os.replace(temporary, args.output)
    summary = metadata | {
        "dataset": str(args.output.resolve()),
        "dataset_sha256": sha256_file(args.output),
        "dataset_size_bytes": args.output.stat().st_size,
        "passed": True,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
