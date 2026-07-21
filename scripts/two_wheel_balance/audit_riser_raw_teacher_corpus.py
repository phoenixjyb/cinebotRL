#!/usr/bin/env python3
"""Audit a fresh multi-case raw-teacher corpus and freeze action scales."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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


PARENT_ADMISSION_SCHEMA = (
    "cinebotrl_two_wheel_riser_smoothed_representative_admission_v1"
)
SUBSET_ADMISSION_SCHEMA = (
    "cinebotrl_two_wheel_riser_raw_teacher_subset_admission_v1"
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
        raise ValueError(f"expected JSON object: {path}")
    return payload


def parse_scales(value: str) -> np.ndarray:
    scales = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    if scales.shape != (3,) or not np.isfinite(scales).all():
        raise argparse.ArgumentTypeError("expected three finite scale values")
    return scales


def _zero(value: Any) -> bool:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return vector.shape == (3,) and bool(np.allclose(vector, 0.0, atol=1e-12))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument(
        "--minimum-scales", type=parse_scales, default=parse_scales("0.3,0.4,0.1")
    )
    parser.add_argument(
        "--scale-quantums", type=parse_scales, default=parse_scales("0.05,0.05,0.05")
    )
    parser.add_argument("--scale-margin", type=float, default=1.10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_count < 40:
        raise ValueError("initial teacher corpus requires at least 40 cases")

    selection = load_json(args.selection)
    admission = load_json(args.admission)
    requested = sorted(int(case) for case in admission.get("requested_cases", []))
    if len(requested) != args.expected_count or len(requested) != len(set(requested)):
        raise ValueError("admission requested cases do not match expected count")
    gate_paths = sorted(args.gate_dir.glob("case_*.json"))
    raw_paths = sorted(args.raw_dir.glob("case_*_executed_raw_teacher_v1.npz"))
    if len(raw_paths) != args.expected_count or len(gate_paths) < args.expected_count:
        raise ValueError(
            f"expected at least {args.expected_count} gates and exactly that many "
            "raw cases, found "
            f"{len(gate_paths)}/{len(raw_paths)}"
        )

    selected_rows = {int(row["case"]): row for row in selection.get("rows", [])}
    admitted_plans = {
        int(row["case"]): row.get("plan_sha256")
        for row in admission.get("selected_plans", [])
    }
    gates: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in gate_paths:
        gate = load_json(path)
        results = gate.get("results", [])
        if len(results) != 1:
            raise ValueError(f"gate must contain one result: {path}")
        case = int(results[0]["case"])
        if case in gates:
            raise ValueError(f"duplicate gate case {case}")
        gates[case] = (path, gate)

    rows = []
    raw_chunks = []
    for path in raw_paths:
        metadata, payload = load_raw_teacher_case(path)
        case = int(metadata["case"])
        if case not in gates or case not in selected_rows:
            raise ValueError(f"raw case {case} lacks gate or selection")
        gate_path, gate = gates[case]
        result = gate["results"][0]
        observations = np.asarray(payload["observations"], dtype=np.float64)
        raw_commands = np.asarray(payload["raw_residual_commands"], dtype=np.float64)
        reconstructed = np.column_stack(
            (
                payload["teacher_commands"][:, 0]
                - observations[:, OBSERVATION_INDEX["feedforward_vx_m_s"]],
                payload["teacher_commands"][:, 1]
                - observations[:, OBSERVATION_INDEX["feedforward_wz_rad_s"]],
                payload["teacher_commands"][:, 2]
                - observations[:, OBSERVATION_INDEX["riser_position_m"]],
            )
        )
        reconstruction_error = float(np.max(np.abs(reconstructed - raw_commands)))
        raw_max = np.max(np.abs(raw_commands), axis=0)
        checks = {
            "plan_hash_bound": admitted_plans.get(case)
            == selected_rows[case].get("plan_sha256"),
            "gate_passed": gate.get("passed") is True and result.get("passed") is True,
            "dynamic_passed": gate.get("dynamic_quality_passed") is True
            and result.get("dynamic_quality_passed") is True,
            "thermal_passed": gate.get("thermal_admission_passed") is True
            and result.get("thermal_admission_passed") is True,
            "controller_evidence_passed": gate.get("controller_evidence_passed")
            is True
            and result.get("controller_evidence_passed") is True,
            "reference_completed": result.get("checks", {}).get(
                "completed_reference"
            )
            is True,
            "no_termination": result.get("termination") is None,
            "row_count_matches": len(observations) == result.get("completed_steps"),
            "no_normalized_dataset": result.get("executed_residual_dataset") is None,
            "raw_not_applied": result.get("raw_residual_label_applied_to_commands")
            is False,
            "zero_applied_residual": _zero(result.get("residual_action_abs_max")),
            "reconstruction_exact": reconstruction_error <= 2e-6,
            "gate_raw_max_matches": np.allclose(
                raw_max,
                np.asarray(result.get("raw_residual_command_abs_max"), dtype=np.float64),
                atol=2e-6,
                rtol=0.0,
            ),
        }
        if not all(checks.values()):
            raise ValueError(f"raw corpus case {case} failed: {checks}")
        raw_chunks.append(raw_commands)
        rows.append(
            {
                "case": case,
                "plan_sha256": selected_rows[case]["plan_sha256"],
                "gate": str(gate_path.resolve()),
                "gate_sha256": sha256(gate_path),
                "raw_case": str(path.resolve()),
                "raw_case_sha256": sha256(path),
                "row_count": len(observations),
                "source_duration_s": result.get("source_duration_s"),
                "execution_duration_s": result.get("execution_duration_s"),
                "raw_residual_abs_max": raw_max.tolist(),
                "reconstruction_max_error": reconstruction_error,
                "checks": checks,
            }
        )
    rows.sort(key=lambda row: row["case"])
    cases = [row["case"] for row in rows]
    requested_gates = {case: item for case, item in gates.items() if case in requested}
    extra_gates = {case: item for case, item in gates.items() if case not in requested}
    if cases != requested or set(cases) != set(requested_gates):
        raise ValueError("admission, gate, and raw case sets do not match")
    if not set(cases).issubset(selection.get("selected_cases", [])):
        raise ValueError("capture includes a case outside the sealed selection")

    raw_all = np.concatenate(raw_chunks, axis=0)
    raw_abs_max = np.max(np.abs(raw_all), axis=0)
    scales = recommended_scales(
        raw_abs_max, args.minimum_scales, args.scale_quantums, args.scale_margin
    )
    rows_by_case = {int(row["case"]): row for row in rows}
    admission_schema = admission.get("schema")
    subset_checks: dict[str, bool] = {}
    if admission_schema == SUBSET_ADMISSION_SCHEMA:
        parent_identity = admission.get("parent_admission", {})
        progress_identity = admission.get("progress_status", {})
        parent_path = Path(parent_identity.get("path", ""))
        progress_path = Path(progress_identity.get("path", ""))
        excluded_rows = {
            int(row["case"]): row
            for row in admission.get("excluded_case_evidence", [])
        }
        retained_rows = {
            int(row["case"]): row
            for row in admission.get("retained_case_evidence", [])
        }
        subset_checks = {
            "corpus_audit_authorized": admission.get("corpus_audit_authorized")
            is True,
            "runtime_closed": admission.get("runtime_authorized") is False,
            "new_capture_closed": admission.get("new_raw_teacher_capture_authorized")
            is False,
            "parent_identity": parent_path.is_file()
            and parent_identity.get("sha256") == sha256(parent_path),
            "progress_identity": progress_path.is_file()
            and progress_identity.get("sha256") == sha256(progress_path),
            "retained_evidence_complete": set(retained_rows) == set(requested),
            "retained_gate_hashes": all(
                retained_rows.get(case, {}).get("gate", {}).get("sha256")
                == sha256(requested_gates[case][0])
                for case in requested
            ),
            "retained_raw_hashes": all(
                retained_rows.get(case, {}).get("raw_case", {}).get("sha256")
                == sha256(Path(rows_by_case[case]["raw_case"]))
                for case in requested
            ),
            "excluded_gate_set": set(excluded_rows) == set(extra_gates),
            "excluded_gate_hashes": all(
                excluded_rows.get(case, {}).get("gate", {}).get("sha256")
                == sha256(extra_gates[case][0])
                for case in extra_gates
            ),
            "excluded_are_dynamic_rejects": all(
                extra_gates[case][1].get("dynamic_quality_passed") is False
                and extra_gates[case][1].get("passed") is False
                for case in extra_gates
            ),
        }
    top_checks = {
        "admission_schema": admission_schema
        in {PARENT_ADMISSION_SCHEMA, SUBSET_ADMISSION_SCHEMA},
        "selection_schema": selection.get("schema")
        == "cinebotrl_two_wheel_riser_initial_teacher_selection_v1",
        "selection_passed": selection.get("passed") is True,
        "selection_hash_bound": admission.get("selection_sha256")
        == sha256(args.selection),
        "admission_passed": admission.get("passed") is True,
        "raw_capture_authorized": admission.get("raw_teacher_capture_authorized")
        is True,
        "normalized_capture_closed": admission.get(
            "normalized_dataset_capture_authorized"
        )
        is False,
        "residual_application_closed": admission.get(
            "residual_action_application_authorized"
        )
        is False,
        "training_closed": admission.get("training_started") is False,
        "bc_closed": admission.get("bc_authorized") is False,
        "ppo_closed": admission.get("ppo_authorized") is False,
        "count_met": len(rows) == args.expected_count,
        "extra_gates_bound": not extra_gates
        or admission_schema == SUBSET_ADMISSION_SCHEMA,
        "subset_provenance": all(subset_checks.values()),
    }
    passed = all(top_checks.values())
    output = {
        "schema": "cinebotrl_two_wheel_riser_raw_teacher_corpus_audit_v1",
        "selection": str(args.selection.resolve()),
        "selection_sha256": sha256(args.selection),
        "admission": str(args.admission.resolve()),
        "admission_sha256": sha256(args.admission),
        "case_count": len(rows),
        "cases": cases,
        "row_count": sum(row["row_count"] for row in rows),
        "raw_residual_abs_max": raw_abs_max.tolist(),
        "frozen_action_scales": scales.tolist(),
        "scale_margin": args.scale_margin,
        "scale_quantums": args.scale_quantums.tolist(),
        "action_scale_frozen": passed,
        "top_checks": top_checks,
        "subset_checks": subset_checks,
        "rows": rows,
        "capture_admission_passed": passed,
        "valid_for_bc_initialization": passed,
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
