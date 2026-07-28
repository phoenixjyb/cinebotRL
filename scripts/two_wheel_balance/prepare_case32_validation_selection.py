#!/usr/bin/env python3
"""Replace retired validation case 16 with case 32 without opening runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = PROJECT_ROOT / "docs/03_training/two_wheel_balance"
SCHEMA = (
    "cinebotrl_two_wheel_riser_model_based_corrective_validation_selection_v2"
)
DEFAULT_ORIGINAL_SELECTION = (
    DOC_ROOT
    / "evidence_20260723_model_based_corrective_validation_tranche_v1/"
    "selection.json"
)
DEFAULT_DISPOSITION = (
    DOC_ROOT
    / "evidence_20260728_case16_validation_disposition_cpu_v1/summary.json"
)
DEFAULT_PLAN = (
    DOC_ROOT
    / "evidence_20260728_case16_validation_disposition_cpu_v1/source/"
    "case_0032_smoothed_riser_plan_v1.npz"
)
DEFAULT_GATE = (
    DOC_ROOT
    / "evidence_20260728_case16_validation_disposition_cpu_v1/source/"
    "case_0032_historical_dynamic_gate.json"
)
EXPECTED_SHA256 = {
    "original_selection": (
        "5576c696e304eb9b9a173970e5fed06e887eccefe2d65a20678415148e22fa0b"
    ),
    "disposition": (
        "eb083d54bc528358311c7ff38acf73bcf5f31581cd02bf5dec1f3415189b3da1"
    ),
    "plan": "71b1986633613fdb13585ac4c12870addc553ad12e895b05cc424a83cf4e037f",
    "gate": "d2a7477254d6a80426370217d8f08db8fe2bdf65e5f4b892a33247f90cf1ce75",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _identity(name: str, path: Path) -> dict[str, object]:
    digest = _sha256(path)
    if digest != EXPECTED_SHA256[name]:
        raise ValueError(
            f"{name} sha256 mismatch: expected {EXPECTED_SHA256[name]}, "
            f"got {digest}"
        )
    return {
        "path": _display(path),
        "sha256": digest,
        "size_bytes": path.stat().st_size,
    }


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _single_case8_row(selection: Mapping[str, object]) -> dict[str, object]:
    rows = selection.get("selected_rows")
    if not isinstance(rows, list):
        raise TypeError("original validation selection rows are missing")
    matches = [
        row for row in rows if isinstance(row, dict) and row.get("case") == 8
    ]
    if len(matches) != 1:
        raise ValueError("original selection must contain one case-8 row")
    return matches[0]


def build_selection(
    original_selection_path: Path = DEFAULT_ORIGINAL_SELECTION,
    disposition_path: Path = DEFAULT_DISPOSITION,
    plan_path: Path = DEFAULT_PLAN,
    gate_path: Path = DEFAULT_GATE,
) -> dict[str, object]:
    identities = {
        "original_selection": _identity(
            "original_selection", original_selection_path
        ),
        "case16_disposition": _identity("disposition", disposition_path),
        "case32_plan": _identity("plan", plan_path),
        "case32_historical_dynamic_gate": _identity("gate", gate_path),
    }
    original = _load_object(original_selection_path)
    disposition = _load_object(disposition_path)
    case8_row = _single_case8_row(original)
    case32 = disposition.get("replacement_candidates", {}).get("32", {})
    plan = case32.get("plan", {})
    gate = case32.get("historical_dynamic_gate", {})
    selection_checks = case32.get("selection_checks", {})

    checks = {
        "original_schema": original.get("schema")
        == "cinebotrl_two_wheel_riser_model_based_corrective_validation_selection_v1",
        "original_passed": original.get("passed") is True
        and isinstance(original.get("checks"), Mapping)
        and all(value is True for value in original["checks"].values()),
        "original_selected_cases": original.get("selected_cases") == [8, 16],
        "case8_row_preserved": case8_row.get("case") == 8
        and case8_row.get("selection_role")
        == "same_seed_validation_paired_canary_required",
        "case16_retired": disposition.get("case16", {}).get("disposition")
        == "calibration_diagnostic_only_pair_rejection_preserved"
        and disposition.get("case16", {}).get(
            "further_case_specific_tuning_recommended"
        )
        is False,
        "case32_selected": disposition.get("selected_replacement_case") == 32,
        "case32_selection_checks": isinstance(selection_checks, Mapping)
        and bool(selection_checks)
        and all(value is True for value in selection_checks.values()),
        "case32_plan_identity": plan.get("case") == 32
        and plan.get("checks")
        and all(value is True for value in plan["checks"].values()),
        "case32_dynamic_identity": gate.get("case") == 32
        and gate.get("dynamic_quality_passed") is True
        and gate.get("action_saturation_ratio") == 0.0,
        "disposition_closed": disposition.get("runtime_authorized") is False
        and disposition.get("label_capture_authorized") is False
        and disposition.get("cpu_conversion_authorized") is False
        and disposition.get("bc_authorized") is False
        and disposition.get("ppo_authorized") is False
        and disposition.get("training_started") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"case-32 selection checks failed: {checks}")

    normalized_labels = [0.8831767412720528, 0.397463403512782, 0.13684031683247833]
    case32_row = {
        "case": 32,
        "plan_sha256": identities["case32_plan"]["sha256"],
        "dynamic_gate_sha256": identities[
            "case32_historical_dynamic_gate"
        ]["sha256"],
        "features": {
            "log1p_source_duration_s": math.log1p(plan["source_duration_s"]),
            "execution_source_duration_ratio": (
                plan["execution_duration_s"] / plan["source_duration_s"]
            ),
            "log1p_source_path_length_m": math.log1p(
                plan["source_path_length_m"]
            ),
            "position_error_p95_m": gate["position_error_p95_m"],
            "maximum_abs_base_linear_velocity_mps": plan[
                "maximum_abs_base_linear_velocity_mps"
            ],
            "maximum_abs_base_yaw_rate_radps": plan[
                "maximum_abs_base_yaw_rate_radps"
            ],
            "maximum_abs_riser_rate_mps": plan[
                "maximum_abs_riser_rate_mps"
            ],
            "target_camera_height_span_m": plan["camera_height_span_m"],
            "dynamic_raw_residual_vx_fraction": normalized_labels[0],
            "dynamic_raw_residual_wz_fraction": normalized_labels[1],
            "dynamic_raw_residual_riser_fraction": normalized_labels[2],
        },
        "checks": {
            "disposition_passed": disposition.get("passed") is True,
            "plan_file_hash": True,
            "dynamic_gate_hash": True,
            "exact_source_plan": selection_checks[
                "exact_source_plan_passed"
            ],
            "individual_dynamic_quality": selection_checks[
                "historical_dynamic_quality_passed"
            ],
            "zero_action_saturation": selection_checks[
                "zero_action_saturation"
            ],
            "validation_case_only": True,
        },
        "selection_rank": 2,
        "selection_role": "same_seed_validation_paired_canary_required",
        "replaces_case": 16,
        "replacement_reason": (
            "case16_ceiling_limited_and_case32_has_zero_saturation_with_"
            "corrective_headroom"
        ),
        "historical_evidence_is_selection_only": True,
        "fresh_pair_required": True,
    }
    if not all(case32_row["checks"].values()):
        raise ValueError("case-32 row checks failed")

    return {
        "schema": SCHEMA,
        "inputs": identities,
        "checks": checks,
        "source_validation_cases": original["source_validation_cases"],
        "eligible_validation_cases": original["eligible_validation_cases"],
        "minimum_validation_corpus_cases": 2,
        "same_seed_pair_required_before_capture": True,
        "selected_cases": [8, 32],
        "selected_rows": [case8_row, case32_row],
        "retired_validation_cases": [16],
        "case16_profile_reuse_authorized": False,
        "case8_profile_reuse_authorized": False,
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
        "next_bounded_action": (
            "audit_case32_validation_pair_readiness_cpu_only"
        ),
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-selection", type=Path, default=DEFAULT_ORIGINAL_SELECTION
    )
    parser.add_argument(
        "--case16-disposition", type=Path, default=DEFAULT_DISPOSITION
    )
    parser.add_argument("--case32-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--case32-gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_selection(
        args.original_selection,
        args.case16_disposition,
        args.case32_plan,
        args.case32_gate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(payload, indent=2) + "\n").encode())
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
