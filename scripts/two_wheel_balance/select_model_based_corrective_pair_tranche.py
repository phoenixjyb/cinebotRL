#!/usr/bin/env python3
"""Select a small diverse train-only tranche for corrective-teacher pair canaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "cinebotrl_two_wheel_riser_model_based_pair_tranche_selection_v1"
PORTFOLIO_SCHEMA = "cinebotrl_two_wheel_riser_smoothed_plan_export_v1"
DYNAMIC_SCHEMA = "cinebotrl_two_wheel_riser_initial_teacher_selection_v1"
SPLIT_SCHEMA = "cinebotrl_two_wheel_riser_split_admission_v1"
CONVERSION_AUDIT_SCHEMA = (
    "cinebotrl_two_wheel_riser_case30_effective_label_conversion_audit_v1"
)
HOLDOUT_CASES = [3, 5, 13, 19, 24]
ANCHOR_CASE = 30
FEATURE_NAMES = [
    "log1p_source_duration_s",
    "execution_source_duration_ratio",
    "log1p_source_path_length_m",
    "position_error_p95_m",
    "maximum_abs_base_linear_velocity_mps",
    "maximum_abs_base_yaw_rate_radps",
    "maximum_abs_riser_rate_mps",
    "maximum_abs_raw_proxy_target_rate_radps",
    "target_camera_height_span_m",
    "dynamic_raw_residual_vx_fraction",
    "dynamic_raw_residual_wz_fraction",
    "dynamic_raw_residual_riser_fraction",
]
LEGACY_DYNAMIC_SCALES = np.asarray([0.30, 0.40, 0.10], dtype=np.float64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _closed(payload: Mapping[str, object]) -> bool:
    return (
        payload.get("bc_authorized") is False
        and payload.get("ppo_authorized") is False
        and payload.get("training_started") is False
        and payload.get("valid_for_training") is False
    )


def _features(item: Mapping[str, Any], dynamic: Mapping[str, Any]) -> np.ndarray:
    path = item["path_metrics"]
    kinematic = item["kinematic_metrics"]
    raw = np.asarray(dynamic["raw_residual_command_abs_max"], dtype=np.float64)
    values = np.asarray(
        [
            np.log1p(float(item["source_duration_s"])),
            float(item["execution_source_duration_ratio"]),
            np.log1p(float(path["source_path_length_m"])),
            float(kinematic["position_error_p95_m"]),
            float(kinematic["maximum_abs_base_linear_velocity_mps"]),
            float(kinematic["maximum_abs_base_yaw_rate_radps"]),
            float(kinematic["maximum_abs_riser_rate_mps"]),
            float(kinematic["maximum_abs_raw_proxy_target_rate_radps"]),
            float(kinematic["maximum_target_camera_height_m"])
            - float(kinematic["minimum_target_camera_height_m"]),
            *(raw / LEGACY_DYNAMIC_SCALES),
        ],
        dtype=np.float64,
    )
    if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError(f"invalid diversity features for case {item.get('case')}")
    return values


def _farthest_point_selection(
    cases: list[int], feature_matrix: np.ndarray, *, count: int
) -> tuple[list[int], dict[int, float], np.ndarray]:
    minimum = feature_matrix.min(axis=0)
    span = feature_matrix.max(axis=0) - minimum
    normalized = (feature_matrix - minimum) / np.where(span > 1e-12, span, 1.0)
    case_to_index = {case: index for index, case in enumerate(cases)}
    selected = [ANCHOR_CASE]
    distances = {ANCHOR_CASE: 0.0}
    while len(selected) < count:
        remaining = [case for case in cases if case not in selected]
        scored = []
        for case in remaining:
            vector = normalized[case_to_index[case]]
            minimum_distance = min(
                float(np.linalg.norm(vector - normalized[case_to_index[chosen]]))
                for chosen in selected
            )
            scored.append((minimum_distance, -case, case))
        score, _, selected_case = max(scored)
        selected.append(selected_case)
        distances[selected_case] = score
    return selected, distances, normalized


def build_selection(
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
    if tranche_size < 3:
        raise ValueError("pair tranche must contain at least three cases")
    split_cases = split_admission.get("admitted_split_cases")
    if not isinstance(split_cases, Mapping):
        split_cases = {}
    train = [int(case) for case in split_cases.get("train", [])]
    validation = [int(case) for case in split_cases.get("validation", [])]
    holdout = [int(case) for case in split_cases.get("holdout", [])]
    disjoint = not (
        set(train) & set(validation)
        or set(train) & set(holdout)
        or set(validation) & set(holdout)
    )
    top_checks = {
        "portfolio_schema": portfolio.get("schema") == PORTFOLIO_SCHEMA,
        "portfolio_cpu_only": portfolio.get("isaac_started") is False
        and portfolio.get("residual_capture_started") is False
        and portfolio.get("bc_started") is False
        and portfolio.get("ppo_started") is False
        and portfolio.get("valid_for_training") is False,
        "dynamic_schema": dynamic_selection.get("schema") == DYNAMIC_SCHEMA,
        "dynamic_passed_closed": dynamic_selection.get("passed") is True
        and dynamic_selection.get("capture_gate_passed") is False
        and _closed(dynamic_selection),
        "dynamic_binds_portfolio": dynamic_selection.get(
            "portfolio_manifest_sha256"
        )
        == sha256_file(portfolio_path),
        "split_schema": split_admission.get("schema") == SPLIT_SCHEMA,
        "split_admitted_closed": split_admission.get("split_admitted") is True
        and split_admission.get("holdout_opened") is False
        and split_admission.get("label_capture_authorized") is False
        and split_admission.get("dataset_creation_authorized") is False
        and split_admission.get("bc_authorized") is False
        and split_admission.get("ppo_authorized") is False
        and split_admission.get("valid_for_training") is False,
        "case_splits_disjoint": disjoint,
        "holdout_exact": holdout == HOLDOUT_CASES,
        "conversion_schema": conversion_audit.get("schema")
        == CONVERSION_AUDIT_SCHEMA,
        "conversion_anchor_passed_closed": conversion_audit.get("case")
        == ANCHOR_CASE
        and conversion_audit.get("passed") is True
        and conversion_audit.get("valid_for_case_merge") is True
        and conversion_audit.get("merged_dataset_created") is False
        and _closed(conversion_audit),
        "anchor_is_train_only": ANCHOR_CASE in train
        and ANCHOR_CASE not in validation
        and ANCHOR_CASE not in holdout,
    }
    if not all(top_checks.values()):
        raise ValueError(f"pair tranche source contract failed: {top_checks}")

    portfolio_items = {
        int(item["case"]): item for item in portfolio.get("items", [])
    }
    dynamic_rows = {
        int(row["case"]): row for row in dynamic_selection.get("rows", [])
    }
    eligible_cases = sorted(set(train) & set(portfolio_items) & set(dynamic_rows))
    if len(eligible_cases) < tranche_size:
        raise ValueError("not enough dynamically qualified training cases")

    features = []
    eligibility_rows = []
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
            "training_case_only": case in train
            and case not in validation
            and case not in holdout,
        }
        if not all(checks.values()):
            raise ValueError(f"case {case} eligibility failed: {checks}")
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
    selected, distances, normalized = _farthest_point_selection(
        eligible_cases, feature_matrix, count=tranche_size
    )
    normalized_by_case = {
        case: normalized[index] for index, case in enumerate(eligible_cases)
    }
    eligibility_by_case = {row["case"]: row for row in eligibility_rows}
    selected_rows = []
    for rank, case in enumerate(selected, start=1):
        row = eligibility_by_case[case]
        selected_rows.append(
            {
                **row,
                "selection_rank": rank,
                "selection_role": (
                    "converted_pilot_anchor"
                    if case == ANCHOR_CASE
                    else "same_seed_paired_canary_required"
                ),
                "minimum_normalized_distance_at_selection": distances[case],
                "normalized_features": dict(
                    zip(
                        FEATURE_NAMES,
                        normalized_by_case[case].tolist(),
                        strict=True,
                    )
                ),
            }
        )

    return {
        "schema": SCHEMA,
        "identities": {
            "portfolio": {
                "path": str(portfolio_path.resolve()),
                "sha256": sha256_file(portfolio_path),
            },
            "dynamic_selection": {
                "path": str(dynamic_path.resolve()),
                "sha256": sha256_file(dynamic_path),
            },
            "split_admission": {
                "path": str(split_path.resolve()),
                "sha256": sha256_file(split_path),
            },
            "case30_conversion_audit": {
                "path": str(conversion_audit_path.resolve()),
                "sha256": sha256_file(conversion_audit_path),
            },
        },
        "checks": top_checks,
        "feature_contract": FEATURE_NAMES,
        "eligible_train_cases": eligible_cases,
        "eligible_case_count": len(eligible_cases),
        "tranche_size": tranche_size,
        "anchor_case": ANCHOR_CASE,
        "selected_cases": selected,
        "selected_rows": selected_rows,
        "validation_cases": validation,
        "holdout_cases": holdout,
        "same_seed_pair_required_before_capture": True,
        "case30_profile_reuse_authorized": False,
        "generic_profile_created": False,
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "label_capture_authorized": False,
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
    parser.add_argument("--tranche-size", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite selection: {args.output}")
    result = build_selection(
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
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
