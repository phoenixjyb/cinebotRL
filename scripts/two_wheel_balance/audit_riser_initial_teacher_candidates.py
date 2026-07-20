#!/usr/bin/env python3
"""Select dynamically qualified riser plans for a fresh BC initialization capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_cases(value: str) -> list[int]:
    try:
        cases = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError("cases must be comma-separated integers") from error
    if not cases or len(cases) != len(set(cases)) or any(case <= 0 for case in cases):
        raise argparse.ArgumentTypeError("cases must be unique positive integers")
    return cases


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def selected_plan_sha256(admission: dict[str, Any], case: int) -> str | None:
    selected = admission.get("selected_plan")
    if isinstance(selected, dict) and selected.get("case") == case:
        return selected.get("plan_sha256")
    for item in admission.get("selected_plans", []):
        if isinstance(item, dict) and item.get("case") == case:
            return item.get("plan_sha256")
    identities = admission.get("runtime_identities", {})
    if isinstance(identities, dict):
        padded = f"case_{case:04d}"
        for name, identity in identities.items():
            if not isinstance(identity, dict):
                continue
            path = str(identity.get("path", "")).lower()
            if name in {"selected_plan", "case_plan"} or padded in path:
                return identity.get("sha256")
    return None


def _zero_residual(value: Any) -> bool:
    try:
        values = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return values.shape == (3,) and bool(np.allclose(values, 0.0, atol=1e-12))


def audit_candidate(
    *,
    summary_path: Path,
    row: dict[str, Any],
    case: int,
    expected_plan_sha256: str,
    expected_source_manifest_sha256: str,
) -> dict[str, Any] | None:
    summary = load_json(summary_path)
    admission_path = summary_path.parent / "admission.json"
    gate_path = Path(str(row.get("gate", "")))
    if not gate_path.is_absolute():
        gate_path = summary_path.parent / "gates" / f"case_{case:04d}.json"
    if not admission_path.is_file() or not gate_path.is_file():
        return None
    admission = load_json(admission_path)
    gate = load_json(gate_path)
    results = gate.get("results", [])
    if len(results) != 1 or results[0].get("case") != case:
        return None
    result = results[0]
    plan_sha = selected_plan_sha256(admission, case)
    checks = {
        "summary_timing_separated": summary.get("source_execution_timing_separated")
        is True,
        "row_passed": row.get("passed") is True,
        "row_dynamic_quality": row.get("dynamic_quality_passed") is True,
        "row_thermal_admission": row.get("thermal_admission_passed") is True,
        "row_runtime_contract": row.get("runtime_contract_passed") is True,
        "admission_passed": admission.get("passed") is True,
        "case_requested": case in admission.get("requested_cases", []),
        "source_manifest_hash": admission.get("source_manifest_sha256")
        == expected_source_manifest_sha256,
        "plan_hash": plan_sha == expected_plan_sha256,
        "admission_hash": summary.get("admission_sha256") == sha256(admission_path),
        "gate_hash": row.get("gate_sha256") == sha256(gate_path),
        "gate_dynamic_quality": gate.get("dynamic_quality_passed") is True,
        "gate_thermal_admission": gate.get("thermal_admission_passed") is True,
        "gate_controller_evidence_not_rejected": gate.get(
            "controller_evidence_passed", True
        )
        is True,
        "result_dynamic_quality": result.get("dynamic_quality_passed") is True,
        "result_thermal_admission": result.get("thermal_admission_passed") is True,
        "result_controller_evidence_not_rejected": result.get(
            "controller_evidence_passed", True
        )
        is True,
        "no_termination": result.get("termination") is None,
        "no_dataset": result.get("executed_residual_dataset") is None,
        "residual_not_applied": result.get("raw_residual_label_applied_to_commands")
        is False,
        "zero_residual_action": _zero_residual(result.get("residual_action_abs_max")),
        "summary_capture_closed": summary.get("residual_capture_started") is False,
        "summary_bc_closed": summary.get("bc_started") is False,
        "summary_ppo_closed": summary.get("ppo_started") is False,
        "thresholds_not_relaxed": summary.get("thresholds_relaxed") is False,
        "actions_not_clipped": summary.get("actions_clipped") is False,
    }
    if not all(checks.values()):
        return None
    raw_abs_max = result.get("raw_residual_command_abs_max")
    if (
        not isinstance(raw_abs_max, list)
        or len(raw_abs_max) != 3
        or not np.isfinite(np.asarray(raw_abs_max, dtype=np.float64)).all()
    ):
        return None
    return {
        "case": case,
        "summary": str(summary_path.resolve()),
        "summary_sha256": sha256(summary_path),
        "admission": str(admission_path.resolve()),
        "admission_sha256": sha256(admission_path),
        "gate": str(gate_path.resolve()),
        "gate_sha256": sha256(gate_path),
        "runtime_commit": admission.get("runtime_commit"),
        "batch_summary_dynamic_quality_passed": summary.get(
            "dynamic_quality_passed"
        ),
        "batch_summary_first_dynamic_reject": summary.get("first_dynamic_reject"),
        "controller_profile": row.get("controller_profile"),
        "tracking_profile": row.get("tracking_profile"),
        "legacy_controller_evidence_fields_observed": (
            "controller_evidence_passed" in gate
            and "controller_evidence_passed" in result
        ),
        "plan_sha256": plan_sha,
        "source_duration_s": row.get("source_duration_s"),
        "execution_duration_s": row.get("execution_duration_s"),
        "completed_steps": row.get("completed_steps"),
        "raw_residual_command_abs_max": raw_abs_max,
        "legacy_residual_label_envelope_passed": row.get(
            "residual_label_envelope_passed"
        )
        is True,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-manifest", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--cases", type=parse_cases, required=True)
    parser.add_argument("--minimum-teacher-cases", type=int, default=40)
    parser.add_argument("--minimum-train-cases", type=int, default=30)
    parser.add_argument("--minimum-validation-cases", type=int, default=5)
    parser.add_argument("--minimum-holdout-cases", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.minimum_teacher_cases <= 0:
        raise ValueError("minimum teacher count must be positive")
    split_minimum = (
        args.minimum_train_cases
        + args.minimum_validation_cases
        + args.minimum_holdout_cases
    )
    if split_minimum > args.minimum_teacher_cases:
        raise ValueError("split minimums exceed the teacher minimum")

    portfolio = load_json(args.portfolio_manifest)
    if portfolio.get("schema") != "cinebotrl_two_wheel_riser_smoothed_plan_export_v1":
        raise ValueError("wrong portfolio schema")
    source_manifest_sha = portfolio.get("source_manifest_sha256")
    if not isinstance(source_manifest_sha, str) or len(source_manifest_sha) != 64:
        raise ValueError("portfolio source manifest hash is missing")
    items = {int(item["case"]): item for item in portfolio.get("items", [])}
    missing_plans = sorted(set(args.cases) - set(items))
    if missing_plans:
        raise ValueError(f"portfolio is missing cases: {missing_plans}")
    rejected = [case for case in args.cases if items[case].get("passed") is not True]
    if rejected:
        raise ValueError(f"portfolio does not admit requested cases: {rejected}")
    for case in args.cases:
        item = items[case]
        plan_path = args.portfolio_manifest.parent / item["file"]
        if not plan_path.is_file() or sha256(plan_path) != item.get("plan_sha256"):
            raise ValueError(f"portfolio plan hash mismatch for case {case}")

    candidates: dict[int, list[dict[str, Any]]] = {case: [] for case in args.cases}
    for summary_path in sorted(args.evidence_root.glob("*/summary.json")):
        try:
            summary = load_json(summary_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        for row in summary.get("gate_rows", []):
            if not isinstance(row, dict) or row.get("case") not in candidates:
                continue
            case = int(row["case"])
            candidate = audit_candidate(
                summary_path=summary_path,
                row=row,
                case=case,
                expected_plan_sha256=items[case]["plan_sha256"],
                expected_source_manifest_sha256=source_manifest_sha,
            )
            if candidate is not None:
                candidates[case].append(candidate)

    selected = []
    for case in args.cases:
        rows = candidates[case]
        if rows:
            selected.append(max(rows, key=lambda row: row["summary"]))
    selected.sort(key=lambda row: row["case"])
    selected_cases = [row["case"] for row in selected]
    legacy_envelope_rejects = [
        row["case"]
        for row in selected
        if not row["legacy_residual_label_envelope_passed"]
    ]
    raw_max = (
        np.max(
            np.asarray(
                [row["raw_residual_command_abs_max"] for row in selected],
                dtype=np.float64,
            ),
            axis=0,
        ).tolist()
        if selected
        else None
    )
    selection_count_met = len(selected) >= args.minimum_teacher_cases
    output = {
        "schema": "cinebotrl_two_wheel_riser_initial_teacher_selection_v1",
        "portfolio_manifest": str(args.portfolio_manifest.resolve()),
        "portfolio_manifest_sha256": sha256(args.portfolio_manifest),
        "source_manifest_sha256": source_manifest_sha,
        "requested_cases": args.cases,
        "selected_cases": selected_cases,
        "missing_cases": sorted(set(args.cases) - set(selected_cases)),
        "selected_case_count": len(selected),
        "minimum_teacher_cases": args.minimum_teacher_cases,
        "minimum_split_cases": {
            "train": args.minimum_train_cases,
            "validation": args.minimum_validation_cases,
            "holdout": args.minimum_holdout_cases,
        },
        "selection_count_met": selection_count_met,
        "legacy_action_scales": [0.30, 0.40, 0.10],
        "legacy_residual_label_envelope_passed_count": (
            len(selected) - len(legacy_envelope_rejects)
        ),
        "legacy_residual_label_envelope_rejected_cases": legacy_envelope_rejects,
        "raw_residual_command_abs_max": raw_max,
        "teacher_role": "fresh_capture_selection_only",
        "fresh_homogeneous_capture_required": True,
        "fresh_capture_minimum_case_count": args.minimum_teacher_cases,
        "action_scale_reaudit_required": True,
        "action_clipping_permitted": False,
        "physical_gimbal_labels_permitted": False,
        "semantic_camera_attitude_required": True,
        "deterministic_lqr_and_safety_supervisor_unchanged": True,
        "coverage_target_70_is_initialization_gate": False,
        "coverage_target_79_remains_final_evaluation_goal": True,
        "capture_gate_passed": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
        "rows": selected,
        "passed": selection_count_met,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0 if output["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
