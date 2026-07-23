#!/usr/bin/env python3
"""Audit validation case 16 before structural paired-profile design."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOC_ROOT = PROJECT_ROOT / "docs/03_training/two_wheel_balance"
FORMULA_ENGINE_PATH = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/"
    "audit_model_based_corrective_case8_validation_pair_readiness.py"
)
SCHEMA = "cinebotrl_two_wheel_riser_case16_validation_pair_readiness_cpu_v1"
CASE = 16
DEFAULT_SELECTION = (
    DOC_ROOT
    / "evidence_20260723_model_based_corrective_validation_tranche_v1/"
    "selection.json"
)
DEFAULT_PLAN = (
    DOC_ROOT
    / "evidence_20260724_case16_validation_pair_readiness_cpu_v1/source/"
    "case_0016_smoothed_riser_plan_v1.npz"
)
DEFAULT_GATE = (
    DOC_ROOT
    / "evidence_20260724_case16_validation_pair_readiness_cpu_v1/source/"
    "case_0016_dynamic_gate.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _load_engine():
    name = "_cinebotrl_case16_validation_readiness_engine"
    spec = importlib.util.spec_from_file_location(name, FORMULA_ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load readiness engine: {FORMULA_ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.SCHEMA = SCHEMA
    module.CASE = CASE
    module.DEFAULT_SELECTION = DEFAULT_SELECTION
    module.DEFAULT_PLAN = DEFAULT_PLAN
    module.DEFAULT_GATE = DEFAULT_GATE
    return module


def audit_readiness(
    selection_path: Path,
    plan_path: Path,
    gate_path: Path,
) -> dict[str, object]:
    engine = _load_engine()
    result = engine.audit_readiness(selection_path, plan_path, gate_path)
    selection_checks = result["selection_checks"]
    selection_checks["case16_role"] = selection_checks.pop("case8_role")
    selection_checks["case16_checks"] = selection_checks.pop("case8_checks")
    result["inputs"]["readiness_formula_engine"] = {
        "path": _display(FORMULA_ENGINE_PATH),
        "sha256": _sha256(FORMULA_ENGINE_PATH),
        "size_bytes": FORMULA_ENGINE_PATH.stat().st_size,
    }
    result["next_bounded_action"] = (
        "design_case16_validation_structural_natural_error_profile_cpu_only"
    )
    if result["case"] != CASE or result["split"] != "validation":
        raise ValueError("case-16 readiness identity drift")
    if not all(selection_checks.values()):
        raise ValueError("case-16 selection checks did not pass")
    if result["case_specific_profile_required"] is not True:
        raise ValueError("case-16 must retain a case-specific profile")
    if result["profile_window_contract"]["bounded_window_found"] is not False:
        raise ValueError("case-16 unexpectedly has a low-motion pulse window")
    if result["safe_window_absent_requires_structural_profile"] is not True:
        raise ValueError("case-16 structural profile requirement is missing")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--dynamic-gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_readiness(args.selection, args.plan, args.dynamic_gate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
