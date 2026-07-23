#!/usr/bin/env python3
"""Select diverse validation cases for model-based corrective-teacher canaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.two_wheel_balance.select_model_based_corrective_pair_tranche import (
    FEATURE_NAMES,
    HOLDOUT_CASES,
    _features,
    build_selection,
    load_json,
    sha256_file,
)


SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_corrective_validation_selection_v1"
)


def _normalize(features: np.ndarray) -> np.ndarray:
    minimum = features.min(axis=0)
    span = features.max(axis=0) - minimum
    return (features - minimum) / np.where(span > 1e-12, span, 1.0)


def _farthest_selection(
    cases: list[int], features: np.ndarray, *, count: int
) -> tuple[list[int], dict[int, float], np.ndarray]:
    if count < 2 or len(cases) < count:
        raise ValueError("validation tranche requires at least two eligible cases")
    normalized = _normalize(features)
    case_to_index = {case: index for index, case in enumerate(cases)}
    pair_scores = []
    for left_index, left_case in enumerate(cases):
        for right_case in cases[left_index + 1 :]:
            distance = float(
                np.linalg.norm(
                    normalized[case_to_index[left_case]]
                    - normalized[case_to_index[right_case]]
                )
            )
            pair_scores.append(
                (distance, -left_case, -right_case, left_case, right_case)
            )
    distance, _, _, left_case, right_case = max(pair_scores)
    selected = [left_case, right_case]
    distances = {left_case: distance, right_case: distance}
    while len(selected) < count:
        scored = []
        for case in cases:
            if case in selected:
                continue
            minimum_distance = min(
                float(
                    np.linalg.norm(
                        normalized[case_to_index[case]]
                        - normalized[case_to_index[chosen]]
                    )
                )
                for chosen in selected
            )
            scored.append((minimum_distance, -case, case))
        score, _, selected_case = max(scored)
        selected.append(selected_case)
        distances[selected_case] = score
    return selected, distances, normalized


def build_validation_selection(
    portfolio: Mapping[str, Any],
    dynamic_selection: Mapping[str, Any],
    split_admission: Mapping[str, Any],
    conversion_audit: Mapping[str, Any],
    *,
    portfolio_path: Path,
    dynamic_path: Path,
    split_path: Path,
    conversion_audit_path: Path,
    tranche_size: int,
) -> dict[str, Any]:
    # Reuse the train selector's complete source/provenance audit before
    # selecting against the separately admitted validation split.
    source_audit = build_selection(
        portfolio,
        dynamic_selection,
        split_admission,
        conversion_audit,
        portfolio_path=portfolio_path,
        dynamic_path=dynamic_path,
        split_path=split_path,
        conversion_audit_path=conversion_audit_path,
        tranche_size=3,
    )
    split_cases = split_admission["admitted_split_cases"]
    train = [int(case) for case in split_cases["train"]]
    validation = [int(case) for case in split_cases["validation"]]
    holdout = [int(case) for case in split_cases["holdout"]]
    if holdout != HOLDOUT_CASES:
        raise ValueError("validation selection holdout contract changed")

    portfolio_items = {
        int(item["case"]): item for item in portfolio.get("items", [])
    }
    dynamic_rows = {
        int(row["case"]): row for row in dynamic_selection.get("rows", [])
    }
    eligible_cases = sorted(
        set(validation) & set(portfolio_items) & set(dynamic_rows)
    )
    if len(eligible_cases) < tranche_size:
        raise ValueError("not enough dynamically qualified validation cases")

    eligibility_rows = []
    features = []
    for case in eligible_cases:
        item = portfolio_items[case]
        dynamic = dynamic_rows[case]
        plan_path = portfolio_path.parent / str(item.get("file", ""))
        checks = {
            "portfolio_passed": item.get("passed") is True
            and item.get("timing_transition_kinematic_gate_passed") is True,
            "plan_file_hash": plan_path.is_file()
            and sha256_file(plan_path) == item.get("plan_sha256"),
            "dynamic_plan_hash": dynamic.get("plan_sha256")
            == item.get("plan_sha256"),
            "dynamic_checks": isinstance(dynamic.get("checks"), Mapping)
            and bool(dynamic["checks"])
            and all(value is True for value in dynamic["checks"].values()),
            "individual_dynamic_quality": all(
                dynamic["checks"].get(name) is True
                for name in (
                    "row_dynamic_quality",
                    "gate_dynamic_quality",
                    "result_dynamic_quality",
                )
            ),
            "validation_case_only": case in validation
            and case not in train
            and case not in holdout,
        }
        if not all(checks.values()):
            raise ValueError(f"validation case {case} eligibility failed: {checks}")
        values = _features(item, dynamic)
        features.append(values)
        eligibility_rows.append(
            {
                "case": case,
                "plan_sha256": item["plan_sha256"],
                "dynamic_gate_sha256": dynamic["gate_sha256"],
                "features": dict(zip(FEATURE_NAMES, values.tolist(), strict=True)),
                "checks": checks,
            }
        )

    feature_matrix = np.vstack(features)
    selected, distances, normalized = _farthest_selection(
        eligible_cases, feature_matrix, count=tranche_size
    )
    normalized_by_case = {
        case: normalized[index] for index, case in enumerate(eligible_cases)
    }
    eligibility_by_case = {row["case"]: row for row in eligibility_rows}
    selected_rows = [
        {
            **eligibility_by_case[case],
            "selection_rank": rank,
            "selection_role": "same_seed_validation_paired_canary_required",
            "minimum_normalized_distance_at_selection": distances[case],
            "normalized_features": dict(
                zip(
                    FEATURE_NAMES,
                    normalized_by_case[case].tolist(),
                    strict=True,
                )
            ),
        }
        for rank, case in enumerate(selected, start=1)
    ]
    excluded_rows = [
        {
            "case": case,
            "reason": (
                "missing_portfolio_case"
                if case not in portfolio_items
                else "not_dynamically_quality_qualified"
            ),
        }
        for case in validation
        if case not in eligible_cases
    ]
    return {
        "schema": SCHEMA,
        "identities": source_audit["identities"],
        "checks": {
            **source_audit["checks"],
            "validation_split_exact": validation == [8, 16, 22, 32, 78],
            "validation_train_disjoint": not set(validation) & set(train),
            "validation_holdout_disjoint": not set(validation) & set(holdout),
            "minimum_validation_candidates_available": len(eligible_cases)
            >= tranche_size,
        },
        "feature_contract": FEATURE_NAMES,
        "source_validation_cases": validation,
        "eligible_validation_cases": eligible_cases,
        "excluded_validation_rows": excluded_rows,
        "eligible_case_count": len(eligible_cases),
        "tranche_size": tranche_size,
        "selected_cases": selected,
        "selected_rows": selected_rows,
        "minimum_validation_corpus_cases": 2,
        "same_seed_pair_required_before_capture": True,
        "case30_profile_reuse_authorized": False,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "label_capture_authorized": False,
        "dataset_conversion_authorized": False,
        "dataset_merge_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio", type=Path, required=True)
    parser.add_argument("--dynamic-selection", type=Path, required=True)
    parser.add_argument("--split-admission", type=Path, required=True)
    parser.add_argument("--case30-conversion-audit", type=Path, required=True)
    parser.add_argument("--tranche-size", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite selection: {args.output}")
    result = build_validation_selection(
        load_json(args.portfolio),
        load_json(args.dynamic_selection),
        load_json(args.split_admission),
        load_json(args.case30_conversion_audit),
        portfolio_path=args.portfolio,
        dynamic_path=args.dynamic_selection,
        split_path=args.split_admission,
        conversion_audit_path=args.case30_conversion_audit,
        tranche_size=args.tranche_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
