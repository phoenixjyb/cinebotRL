#!/usr/bin/env python3
"""Build a closed structural natural-error validation profile for case 32."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMULA_ENGINE_PATH = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/"
    "build_model_based_corrective_case2_natural_error_profile.py"
)
SCHEMA = (
    "cinebotrl_two_wheel_riser_case32_validation_natural_error_"
    "profile_proposal_cpu_v1"
)
READINESS_SCHEMA = (
    "cinebotrl_two_wheel_riser_case32_validation_pair_readiness_cpu_v1"
)
CASE = 32
POSITION_P95_GATE_M = 0.15
MAXIMUM_VALIDATION_RETENTION = 0.40
SLEW_HORIZON_S = 0.40
CANONICAL_PROFILE = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case32_validation_natural_error_"
    "profile_v1.json"
)
CASE8_PROFILE = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case8_validation_profile_v1.json"
)
CASE16_PROFILE = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case16_validation_natural_error_"
    "profile_v1.json"
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


def _load_engine(*, retention: float):
    name = "_cinebotrl_case32_validation_natural_error_engine"
    spec = importlib.util.spec_from_file_location(name, FORMULA_ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load formula engine: {FORMULA_ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.SCHEMA = SCHEMA
    module.READINESS_SCHEMA = READINESS_SCHEMA
    module.CASE = CASE
    module.RAW_ENVELOPE_RETENTION = retention
    module.SLEW_HORIZON_S = SLEW_HORIZON_S
    module.CANONICAL_PROFILE = CANONICAL_PROFILE
    return module


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _load_plan(path: Path):
    return _load_engine(retention=0.25)._load_plan(path)


def _retention_fraction(readiness: Mapping[str, object]) -> float:
    margin = float(
        readiness.get("zero_residual_dynamic_gate", {})
        .get("dynamic_margins", {})
        .get("position_error_p95_m", float("nan"))
    )
    retention = min(
        MAXIMUM_VALIDATION_RETENTION,
        margin / POSITION_P95_GATE_M,
    )
    if not 0.0 < retention <= MAXIMUM_VALIDATION_RETENTION:
        raise ValueError(
            "case-32 p95 margin cannot support a validation residual profile"
        )
    return retention


def build_profile(
    *,
    readiness: Mapping[str, object],
    readiness_path: Path,
    plan_metadata: Mapping[str, object],
    plan_arrays: Mapping[str, object],
    plan_path: Path,
    gate: Mapping[str, object],
    gate_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    selection_checks = readiness.get("selection_checks")
    validation_checks = {
        "readiness_split_validation": readiness.get("split") == "validation",
        "selection_case32_role": isinstance(selection_checks, Mapping)
        and selection_checks.get("case32_role") is True,
        "selection_case32_checks": isinstance(selection_checks, Mapping)
        and selection_checks.get("case32_checks") is True,
        "train_profile_reuse_forbidden": isinstance(selection_checks, Mapping)
        and selection_checks.get("train_profile_reuse_forbidden") is True,
        "case_specific_profile_required": readiness.get(
            "case_specific_profile_required"
        )
        is True,
        "structural_profile_required": readiness.get(
            "safe_window_absent_requires_structural_profile"
        )
        is True,
        "no_low_motion_window": readiness.get(
            "profile_window_contract", {}
        ).get("windows")
        == [],
        "case8_profile_reuse_forbidden": readiness.get(
            "case8_profile_reuse_authorized"
        )
        is False,
        "case16_profile_reuse_forbidden": readiness.get(
            "case16_profile_reuse_authorized"
        )
        is False,
    }
    validation_checks = {
        name: bool(value) for name, value in validation_checks.items()
    }
    if not all(validation_checks.values()):
        raise ValueError(
            f"case-32 validation profile checks failed: {validation_checks}"
        )

    retention = _retention_fraction(readiness)
    engine = _load_engine(retention=retention)
    profile, proposal = engine.build_profile(
        readiness=readiness,
        readiness_path=readiness_path,
        plan_metadata=plan_metadata,
        plan_arrays=plan_arrays,
        plan_path=plan_path,
        gate=gate,
        gate_path=gate_path,
    )
    profile_bytes = _json_bytes(profile)
    directions = proposal["projection_envelope"]["directions"]
    negative_clipped = directions["negative"][
        "command_clipped_transition_count"
    ]
    positive_clipped = directions["positive"][
        "command_clipped_transition_count"
    ]
    margin = readiness["zero_residual_dynamic_gate"]["dynamic_margins"][
        "position_error_p95_m"
    ]
    profile_checks = {
        "dedicated_case32_profile": _sha256_bytes(profile_bytes)
        not in {_sha256(CASE8_PROFILE), _sha256(CASE16_PROFILE)},
        "retention_derived_from_dynamic_margin": abs(
            retention
            - min(
                MAXIMUM_VALIDATION_RETENTION,
                margin / POSITION_P95_GATE_M,
            )
        )
        <= 1e-12,
        "retention_within_validation_cap": 0.0
        < retention
        <= MAXIMUM_VALIDATION_RETENTION,
        "both_linear_directions_projected": negative_clipped[0] > 0
        and positive_clipped[0] > 0,
        "both_yaw_directions_projected": negative_clipped[1] > 0
        and positive_clipped[1] > 0,
        "riser_not_projected": negative_clipped[2] == 0
        and positive_clipped[2] == 0,
        "projection_contractive": proposal["projection_envelope"][
            "all_projections_contractive"
        ]
        is True,
    }
    if not all(profile_checks.values()):
        raise ValueError(f"case-32 profile checks failed: {profile_checks}")

    proposal["split"] = "validation"
    proposal["input_checks"].update(validation_checks)
    proposal["validation_profile_checks"] = profile_checks
    proposal["identities"]["formula_engine"] = {
        "path": _display(FORMULA_ENGINE_PATH),
        "sha256": _sha256(FORMULA_ENGINE_PATH),
    }
    proposal["identities"]["case8_profile_comparison"] = {
        "path": _display(CASE8_PROFILE),
        "sha256": _sha256(CASE8_PROFILE),
    }
    proposal["identities"]["case16_profile_comparison"] = {
        "path": _display(CASE16_PROFILE),
        "sha256": _sha256(CASE16_PROFILE),
    }
    proposal["profile_formula"].update(
        {
            "retention_fraction": retention,
            "retention_cap": MAXIMUM_VALIDATION_RETENTION,
            "position_p95_gate_m": POSITION_P95_GATE_M,
            "position_p95_margin_m": margin,
            "retention_rationale": (
                "case32 natural-error envelope bounded by its own p95 "
                "dynamic margin and the validation retention cap"
            ),
        }
    )
    proposal["natural_error_contract"].update(
        {
            "validation_only": True,
            "external_perturbation_forbidden": True,
            "requested_residual_is_not_a_training_label": True,
        }
    )
    proposal["held_out_validation_contract"] = {
        "teacher_admission_authorized": False,
        "label_capture_authorized": False,
        "dataset_creation_authorized": False,
        "dataset_merge_authorized": False,
        "candidate_may_report_validation_improvement_only": True,
        "effective_projected_residual_must_be_assessed": True,
    }
    proposal["case8_profile_reuse_authorized"] = False
    proposal["case16_profile_reuse_authorized"] = False
    proposal["train_profile_reuse_authorized"] = False
    proposal["validation_pair_profile_cpu_ready"] = True
    proposal["runtime_route_implemented"] = False
    proposal["validation_runtime_opened"] = False
    proposal["holdout_opened"] = False
    proposal["teacher_admission_authorized"] = False
    proposal["dataset_creation_authorized"] = False
    proposal["next_bounded_action"] = (
        "implement_case32_validation_natural_error_pair_contract_cpu_only_"
        "without_authorization"
    )
    proposal["passed"] = (
        proposal.get("passed") is True
        and all(proposal["input_checks"].values())
        and all(profile_checks.values())
        and all(
            value is False
            for key, value in proposal["held_out_validation_contract"].items()
            if key.endswith("_authorized")
        )
    )
    if proposal["passed"] is not True:
        raise ValueError("case-32 validation profile proposal did not pass")
    return profile, proposal


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite profile output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dynamic-gate", type=Path, required=True)
    parser.add_argument("--corrective-profile-output", type=Path, required=True)
    parser.add_argument("--proposal-output", type=Path, required=True)
    args = parser.parse_args()
    outputs = (args.corrective_profile_output, args.proposal_output)
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"refusing to overwrite profile outputs: {outputs}")
    plan_metadata, plan_arrays = _load_plan(args.plan)
    profile, proposal = build_profile(
        readiness=_load_object(args.readiness),
        readiness_path=args.readiness,
        plan_metadata=plan_metadata,
        plan_arrays=plan_arrays,
        plan_path=args.plan,
        gate=_load_object(args.dynamic_gate),
        gate_path=args.dynamic_gate,
    )
    _write_new(args.corrective_profile_output, _json_bytes(profile))
    _write_new(args.proposal_output, _json_bytes(proposal))
    print(json.dumps(proposal, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
