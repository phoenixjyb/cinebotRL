#!/usr/bin/env python3
"""Build closed, case-specific validation profiles for case 8."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMULA_ENGINE_PATH = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/"
    "build_model_based_corrective_case7_pair_profiles.py"
)
SCHEMA = (
    "cinebotrl_two_wheel_riser_case8_validation_pair_profile_proposal_cpu_v1"
)
READINESS_SCHEMA = (
    "cinebotrl_two_wheel_riser_case8_validation_pair_readiness_cpu_v1"
)
CASE = 8
SPLIT = "validation"
RAW_ENVELOPE_RETENTION = 0.40
SLEW_HORIZON_S = 0.40
PULSE_FORCE_BODY_X_N = 18.0
CASE7_RAW_ENVELOPE_RETENTION = 0.50
CANONICAL_CORRECTIVE_PROFILE = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case8_validation_profile_v1.json"
)
CANONICAL_WRENCH_PROFILE = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case8_validation_wrench_profile_v1.json"
)
CASE7_CORRECTIVE_PROFILE = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case7_profile_v1.json"
)
CASE7_WRENCH_PROFILE = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case7_wrench_profile_v1.json"
)


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, indent=2) + "\n").encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _load_formula_engine():
    module_name = "_cinebotrl_case8_validation_profile_formula_engine"
    spec = importlib.util.spec_from_file_location(module_name, FORMULA_ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load formula engine: {FORMULA_ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.SCHEMA = SCHEMA
    module.READINESS_SCHEMA = READINESS_SCHEMA
    module.CASE = CASE
    module.RAW_ENVELOPE_RETENTION = RAW_ENVELOPE_RETENTION
    module.SLEW_HORIZON_S = SLEW_HORIZON_S
    module.PULSE_FORCE_BODY_X_N = PULSE_FORCE_BODY_X_N
    module.CANONICAL_CORRECTIVE_PROFILE = CANONICAL_CORRECTIVE_PROFILE
    module.CANONICAL_WRENCH_PROFILE = CANONICAL_WRENCH_PROFILE
    return module


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_plan(path: Path):
    return _load_formula_engine()._load_plan(path)


def build_profiles(
    *,
    readiness: Mapping[str, object],
    readiness_path: Path,
    plan_metadata: Mapping[str, object],
    plan_arrays: Mapping[str, object],
    plan_path: Path,
    plant: Mapping[str, object],
    plant_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    selection_checks = readiness.get("selection_checks", {})
    validation_checks = {
        "readiness_split_validation": readiness.get("split") == SPLIT,
        "selection_case8_role": isinstance(selection_checks, Mapping)
        and selection_checks.get("case8_role") is True,
        "selection_forbids_train_profile_reuse": isinstance(
            selection_checks, Mapping
        )
        and selection_checks.get("train_profile_reuse_forbidden") is True,
        "readiness_forbids_case30_reuse": readiness.get(
            "case30_profile_reuse_authorized"
        )
        is False,
        "readiness_forbids_case7_reuse": readiness.get(
            "case7_profile_reuse_authorized"
        )
        is False,
    }
    validation_checks = {
        name: bool(value) for name, value in validation_checks.items()
    }
    if not all(validation_checks.values()):
        raise ValueError(
            f"case-8 validation profile checks failed: {validation_checks}"
        )

    engine = _load_formula_engine()
    corrective, wrench, proposal = engine.build_profiles(
        readiness=readiness,
        readiness_path=readiness_path,
        plan_metadata=plan_metadata,
        plan_arrays=plan_arrays,
        plan_path=plan_path,
        plant=plant,
        plant_path=plant_path,
    )
    corrective_bytes = _json_bytes(corrective)
    wrench_bytes = _json_bytes(wrench)
    non_reuse_checks = {
        "case7_corrective_profile_not_reused": (
            _sha256_bytes(corrective_bytes) != _sha256(CASE7_CORRECTIVE_PROFILE)
        ),
        "case7_wrench_profile_not_reused": (
            _sha256_bytes(wrench_bytes) != _sha256(CASE7_WRENCH_PROFILE)
        ),
        "validation_retention_is_more_conservative_than_case7": (
            RAW_ENVELOPE_RETENTION < CASE7_RAW_ENVELOPE_RETENTION
        ),
        "validation_pulse_is_smaller_than_case7": PULSE_FORCE_BODY_X_N < 20.0,
    }
    non_reuse_checks = {
        name: bool(value) for name, value in non_reuse_checks.items()
    }
    if not all(non_reuse_checks.values()):
        raise ValueError(f"case-8 profile reuse checks failed: {non_reuse_checks}")

    proposal["split"] = SPLIT
    proposal["input_checks"].update(validation_checks)
    proposal["validation_profile_checks"] = non_reuse_checks
    proposal["identities"]["formula_engine"] = {
        "path": _display(FORMULA_ENGINE_PATH),
        "sha256": _sha256(FORMULA_ENGINE_PATH),
    }
    proposal["identities"]["case7_corrective_profile_comparison"] = {
        "path": _display(CASE7_CORRECTIVE_PROFILE),
        "sha256": _sha256(CASE7_CORRECTIVE_PROFILE),
    }
    proposal["identities"]["case7_wrench_profile_comparison"] = {
        "path": _display(CASE7_WRENCH_PROFILE),
        "sha256": _sha256(CASE7_WRENCH_PROFILE),
    }
    proposal["case30_profile_reuse_authorized"] = False
    proposal["case7_profile_reuse_authorized"] = False
    proposal["train_profile_reuse_authorized"] = False
    proposal["validation_runtime_opened"] = False
    proposal["holdout_opened"] = False
    proposal["next_bounded_action"] = (
        "implement_case8_validation_pair_runtime_contract_cpu_only_"
        "without_authorization"
    )
    proposal["passed"] = (
        proposal.get("passed") is True
        and all(proposal["input_checks"].values())
        and all(non_reuse_checks.values())
    )
    if proposal["passed"] is not True:
        raise ValueError("case-8 validation profile proposal did not pass")
    return corrective, wrench, proposal


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite profile output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plant-prior", type=Path, required=True)
    parser.add_argument("--corrective-profile-output", type=Path, required=True)
    parser.add_argument("--wrench-profile-output", type=Path, required=True)
    parser.add_argument("--proposal-output", type=Path, required=True)
    args = parser.parse_args()
    outputs = (
        args.corrective_profile_output,
        args.wrench_profile_output,
        args.proposal_output,
    )
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite profile outputs: {existing}")
    engine = _load_formula_engine()
    plan_metadata, plan_arrays = engine._load_plan(args.plan)
    corrective, wrench, proposal = build_profiles(
        readiness=_load_object(args.readiness),
        readiness_path=args.readiness,
        plan_metadata=plan_metadata,
        plan_arrays=plan_arrays,
        plan_path=args.plan,
        plant=_load_object(args.plant_prior),
        plant_path=args.plant_prior,
    )
    for path, payload in (
        (args.corrective_profile_output, _json_bytes(corrective)),
        (args.wrench_profile_output, _json_bytes(wrench)),
        (args.proposal_output, _json_bytes(proposal)),
    ):
        _write_new(path, payload)
    print(json.dumps(proposal, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
