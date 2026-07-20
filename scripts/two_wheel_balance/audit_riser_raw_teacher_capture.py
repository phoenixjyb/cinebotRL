#!/usr/bin/env python3
"""Audit one scale-independent executed raw-teacher capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    OBSERVATION_INDEX,
    load_raw_teacher_case,
)
from scripts.two_wheel_balance.audit_riser_residual_capture import (  # noqa: E402
    recommended_scales,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def parse_scales(value: str) -> np.ndarray:
    values = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise argparse.ArgumentTypeError("expected three finite comma-separated values")
    return values


def canonical_cross_platform_path(value: str | Path) -> str:
    text = str(value).replace("\\", "/")
    match = re.fullmatch(r"([A-Za-z]):/(.*)", text)
    if match:
        return f"/mnt/{match.group(1).lower()}/{match.group(2)}"
    return str(Path(text).resolve())


def selected_row(selection: dict[str, Any], case: int) -> dict[str, Any]:
    rows = [row for row in selection.get("rows", []) if row.get("case") == case]
    if len(rows) != 1:
        raise ValueError(f"selection must contain exactly one row for case {case}")
    return rows[0]


def admitted_plan_sha256(admission: dict[str, Any], case: int) -> str | None:
    if admission.get("requested_cases") == [case]:
        return admission.get("plan_sha256")
    rows = [
        row
        for row in admission.get("selected_plans", [])
        if isinstance(row, dict) and row.get("case") == case
    ]
    return rows[0].get("plan_sha256") if len(rows) == 1 else None


def _zero_vector(value: Any) -> bool:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return vector.shape == (3,) and bool(np.allclose(vector, 0.0, atol=1e-12))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--raw-case", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument(
        "--minimum-scales", type=parse_scales, default=parse_scales("0.3,0.4,0.1")
    )
    parser.add_argument(
        "--scale-quantums", type=parse_scales, default=parse_scales("0.05,0.05,0.05")
    )
    parser.add_argument("--scale-margin", type=float, default=1.10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.case <= 0:
        raise ValueError("case must be positive")

    gate = load_json(args.gate)
    admission = load_json(args.admission)
    selection = load_json(args.selection)
    metadata, raw = load_raw_teacher_case(args.raw_case)
    results = gate.get("results", [])
    if len(results) != 1:
        raise ValueError("gate must contain exactly one result")
    result = results[0]
    row = selected_row(selection, args.case)

    observations = np.asarray(raw["observations"], dtype=np.float64)
    raw_commands = np.asarray(raw["raw_residual_commands"], dtype=np.float64)
    reconstructed = np.column_stack(
        (
            raw["teacher_commands"][:, 0]
            - observations[:, OBSERVATION_INDEX["feedforward_vx_m_s"]],
            raw["teacher_commands"][:, 1]
            - observations[:, OBSERVATION_INDEX["feedforward_wz_rad_s"]],
            raw["teacher_commands"][:, 2]
            - observations[:, OBSERVATION_INDEX["riser_position_m"]],
        )
    )
    reconstruction_error = float(np.max(np.abs(reconstructed - raw_commands)))
    raw_abs_max = np.max(np.abs(raw_commands), axis=0)
    provisional_scales = recommended_scales(
        raw_abs_max,
        args.minimum_scales,
        args.scale_quantums,
        args.scale_margin,
    )

    checks = {
        "selection_schema": selection.get("schema")
        == "cinebotrl_two_wheel_riser_initial_teacher_selection_v1",
        "selection_passed": selection.get("passed") is True,
        "selection_count_met": selection.get("selection_count_met") is True,
        "case_selected": args.case in selection.get("selected_cases", []),
        "selection_still_not_trainable": selection.get("valid_for_training") is False,
        "admission_passed": admission.get("passed") is True,
        "admission_case_pinned": args.case in admission.get("requested_cases", [])
        and len(admission.get("requested_cases", []))
        == len(set(admission.get("requested_cases", []))),
        "admission_selection_hash": admission.get("selection_sha256")
        == sha256(args.selection),
        "admission_source_hash": admission.get("source_manifest_sha256")
        == selection.get("source_manifest_sha256"),
        "admission_portfolio_hash": admission.get("portfolio_manifest_sha256")
        == selection.get("portfolio_manifest_sha256"),
        "admission_plan_hash": admitted_plan_sha256(admission, args.case)
        == row.get("plan_sha256"),
        "gate_case": result.get("case") == args.case,
        "gate_passed": gate.get("passed") is True and result.get("passed") is True,
        "dynamic_quality_passed": gate.get("dynamic_quality_passed") is True
        and result.get("dynamic_quality_passed") is True,
        "thermal_admission_passed": gate.get("thermal_admission_passed") is True
        and result.get("thermal_admission_passed") is True,
        "controller_evidence_passed": gate.get("controller_evidence_passed") is True
        and result.get("controller_evidence_passed") is True,
        "no_termination": result.get("termination") is None,
        "reference_completed": result.get("checks", {}).get("completed_reference")
        is True,
        "both_clocks_present": result.get("source_duration_s") is not None
        and result.get("execution_duration_s") is not None,
        "row_count_matches_steps": len(observations) == result.get("completed_steps"),
        "raw_capture_path_matches": canonical_cross_platform_path(
            result.get("executed_raw_teacher_capture", "")
        )
        == canonical_cross_platform_path(args.raw_case),
        "no_normalized_dataset": result.get("executed_residual_dataset") is None,
        "raw_labels_not_applied": result.get("raw_residual_label_applied_to_commands")
        is False,
        "zero_applied_residual": _zero_vector(result.get("residual_action_abs_max")),
        "raw_metadata_case": metadata.get("case") == args.case,
        "raw_metadata_not_trainable": metadata.get("valid_for_training") is False,
        "raw_metadata_scale_unfrozen": metadata.get("action_scale_frozen") is False,
        "raw_metadata_not_applied": metadata.get("raw_residual_applied_to_commands")
        is False,
        "raw_reconstruction_exact": reconstruction_error <= 2e-6,
        "gate_raw_max_matches": np.allclose(
            raw_abs_max,
            np.asarray(result.get("raw_residual_command_abs_max"), dtype=np.float64),
            atol=2e-6,
            rtol=0.0,
        ),
        "training_closed": gate.get("training_started") is False,
        "ppo_closed": gate.get("ppo_authorized") is False,
    }
    passed = all(checks.values())
    output = {
        "schema": "cinebotrl_two_wheel_riser_raw_teacher_capture_audit_v1",
        "case": args.case,
        "checks": checks,
        "capture_admission_passed": passed,
        "raw_case": str(args.raw_case.resolve()),
        "raw_case_sha256": sha256(args.raw_case),
        "gate": str(args.gate.resolve()),
        "gate_sha256": sha256(args.gate),
        "admission": str(args.admission.resolve()),
        "admission_sha256": sha256(args.admission),
        "selection": str(args.selection.resolve()),
        "selection_sha256": sha256(args.selection),
        "row_count": len(observations),
        "source_duration_s": result.get("source_duration_s"),
        "execution_duration_s": result.get("execution_duration_s"),
        "raw_residual_abs_max": raw_abs_max.tolist(),
        "raw_residual_signed_min": np.min(raw_commands, axis=0).tolist(),
        "raw_residual_signed_max": np.max(raw_commands, axis=0).tolist(),
        "raw_reconstruction_max_error": reconstruction_error,
        "provisional_recommended_action_scales": provisional_scales.tolist(),
        "action_scale_frozen": False,
        "scale_freeze_requires_full_selected_capture": True,
        "valid_for_training": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
