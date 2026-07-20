#!/usr/bin/env python3
"""Build a case-disjoint BC initialization dataset from an admitted raw corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    ACTION_NAMES,
    LOOKAHEAD_HORIZONS_S,
    OBSERVATION_INDEX,
    OBSERVATION_NAMES,
    PREVIOUS_ACTION_INDICES,
    RAW_TEACHER_SCHEMA,
    load_raw_teacher_case,
    normalize_raw_teacher_payload,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-count", type=int, default=40)
    parser.add_argument("--train-count", type=int, default=30)
    parser.add_argument("--validation-count", type=int, default=5)
    parser.add_argument("--holdout-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite dataset: {args.output}")
    if sum((args.train_count, args.validation_count, args.holdout_count)) != (
        args.dataset_count
    ):
        raise ValueError("train/validation/holdout counts must equal dataset count")

    audit = json.loads(args.corpus_audit.read_text(encoding="utf-8"))
    if (
        audit.get("schema")
        != "cinebotrl_two_wheel_riser_raw_teacher_corpus_audit_v1"
        or audit.get("capture_admission_passed") is not True
        or audit.get("action_scale_frozen") is not True
        or audit.get("valid_for_bc_initialization") is not True
        or audit.get("bc_authorized") is not False
        or audit.get("ppo_authorized") is not False
        or audit.get("training_started") is not False
    ):
        raise ValueError("raw corpus is not admitted for offline dataset construction")
    scales = np.asarray(audit.get("frozen_action_scales"), dtype=np.float64)
    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("corpus audit has invalid frozen action scales")
    rows = {int(row["case"]): row for row in audit.get("rows", [])}
    cases = sorted(rows)
    if len(cases) != audit.get("case_count") or len(cases) < args.dataset_count:
        raise ValueError("corpus case count cannot satisfy requested dataset")

    shuffled = np.asarray(cases, dtype=np.int64)
    np.random.default_rng(args.seed).shuffle(shuffled)
    dataset_order = [int(case) for case in shuffled[: args.dataset_count]]
    split_cases = {
        "holdout": sorted(dataset_order[: args.holdout_count]),
        "validation": sorted(
            dataset_order[
                args.holdout_count : args.holdout_count + args.validation_count
            ]
        ),
        "train": sorted(dataset_order[args.holdout_count + args.validation_count :]),
    }
    coverage_only_cases = sorted(set(cases) - set(dataset_order))
    split_by_case = {
        case: split for split, members in split_cases.items() for case in members
    }
    split_codes = {"train": 0, "validation": 1, "holdout": 2}
    arrays = {
        name: []
        for name in (
            "observations",
            "actions",
            "case_ids",
            "elapsed_time_s",
            "phase_time_s",
            "baseline_wheel_actions",
            "teacher_commands",
        )
    }
    source_indices = []
    split_labels = []
    source_rows = []
    for source_index, case in enumerate(sorted(dataset_order)):
        row = rows[case]
        path = Path(row["raw_case"])
        if not path.is_file() or sha256(path) != row.get("raw_case_sha256"):
            raise ValueError(f"raw source identity mismatch for case {case}")
        metadata, raw = load_raw_teacher_case(path)
        if int(metadata["case"]) != case:
            raise ValueError(f"raw source case mismatch for case {case}")
        normalized = normalize_raw_teacher_payload(raw, scales)
        count = len(normalized["observations"])
        for name in arrays:
            arrays[name].append(normalized[name])
        source_indices.append(np.full(count, source_index, dtype=np.int16))
        split_labels.append(
            np.full(count, split_codes[split_by_case[case]], dtype=np.int8)
        )
        source_rows.append(
            {
                "case": case,
                "raw_case": str(path.resolve()),
                "raw_case_sha256": sha256(path),
                "row_count": count,
                "split": split_by_case[case],
            }
        )
    merged = {name: np.concatenate(parts, axis=0) for name, parts in arrays.items()}
    merged["source_index"] = np.concatenate(source_indices)
    merged["split_labels"] = np.concatenate(split_labels)
    merged["action_valid_mask"] = np.ones_like(merged["actions"], dtype=np.float32)

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
    previous_contract_passed = True
    for case in dataset_order:
        mask = merged["case_ids"] == case
        case_actions = merged["actions"][mask]
        previous = merged["observations"][mask][:, PREVIOUS_ACTION_INDICES]
        previous_contract_passed &= bool(np.allclose(previous[0], 0.0, atol=1e-12))
        previous_contract_passed &= bool(
            np.allclose(previous[1:], case_actions[:-1], atol=1e-7)
        )
    passed = bool(
        reconstruction_error <= 2e-6
        and np.max(action_abs_max) < 1.0 - 1e-6
        and previous_contract_passed
    )
    if not passed:
        raise ValueError("normalized corpus failed reconstruction or previous-action gate")
    metadata = {
        "schema": "cinebotrl_two_wheel_riser_residual_merged_v3",
        "source_case_schema": RAW_TEACHER_SCHEMA,
        "corpus_audit": str(args.corpus_audit.resolve()),
        "corpus_audit_sha256": sha256(args.corpus_audit),
        "observation_names": list(OBSERVATION_NAMES),
        "observation_contract": "executed_state_with_execution_time_lookahead_v2",
        "lookahead_horizons_s": list(LOOKAHEAD_HORIZONS_S),
        "lookahead_reference_clock": "execution_time_s",
        "action_names": list(ACTION_NAMES),
        "action_scales": scales.tolist(),
        "action_scale_contract": "corpus_raw_abs_max_margin_quantized_v1",
        "previous_action_contract": "previous_normalized_teacher_action_v1",
        "case_count": args.dataset_count,
        "captured_case_count": len(cases),
        "row_count": len(merged["observations"]),
        "seed": args.seed,
        "split_cases": split_cases,
        "split_case_counts": {
            name: len(members) for name, members in split_cases.items()
        },
        "coverage_only_cases": coverage_only_cases,
        "trajectory_leakage": False,
        "action_abs_max": action_abs_max.tolist(),
        "action_clip_ratio": np.mean(
            np.abs(merged["actions"]) >= 1.0 - 1e-6, axis=0
        ).tolist(),
        "teacher_command_reconstruction_max_error": reconstruction_error,
        "previous_action_rebuilt": previous_contract_passed,
        "source_rows": source_rows,
        "source_action_labels_used": False,
        "physical_gimbal_labels_used_as_actions": False,
        "dataset_admission_passed": passed,
        "valid_for_bc_initialization": passed,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        **merged,
    )
    summary = metadata | {
        "dataset": str(args.output.resolve()),
        "dataset_sha256": sha256(args.output),
        "dataset_size_bytes": args.output.stat().st_size,
        "passed": passed,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
