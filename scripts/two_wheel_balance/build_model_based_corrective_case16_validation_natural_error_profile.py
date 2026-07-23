#!/usr/bin/env python3
"""Build a closed structural natural-error validation profile for case 16."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMULA_ENGINE_PATH = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/"
    "build_model_based_corrective_case2_natural_error_profile.py"
)
SCHEMA = (
    "cinebotrl_two_wheel_riser_case16_validation_natural_error_"
    "profile_proposal_cpu_v1"
)
READINESS_SCHEMA = (
    "cinebotrl_two_wheel_riser_case16_validation_pair_readiness_cpu_v1"
)
CASE = 16
SPLIT = "validation"
POSITION_P95_GATE_M = 0.15
MAXIMUM_VALIDATION_RETENTION = 0.40
SLEW_HORIZON_S = 0.40
CANONICAL_PROFILE = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case16_validation_natural_error_"
    "profile_v1.json"
)
CASE2_PROFILE = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case2_natural_error_profile_v1.json"
)
CASE8_PROFILE = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case8_validation_profile_v1.json"
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


def _load_formula_engine(*, retention: float = MAXIMUM_VALIDATION_RETENTION):
    module_name = "_cinebotrl_case16_validation_natural_error_formula_engine"
    spec = importlib.util.spec_from_file_location(module_name, FORMULA_ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load formula engine: {FORMULA_ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
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
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_plan(path: Path):
    return _load_formula_engine()._load_plan(path)


def _retention_fraction(readiness: Mapping[str, object]) -> float:
    zero_gate = readiness.get("zero_residual_dynamic_gate")
    if not isinstance(zero_gate, Mapping):
        raise ValueError("case-16 readiness is missing zero-residual gate evidence")
    margins = zero_gate.get("dynamic_margins")
    if not isinstance(margins, Mapping):
        raise ValueError("case-16 readiness is missing dynamic margins")
    margin = float(margins.get("position_error_p95_m", float("nan")))
    retention = min(
        MAXIMUM_VALIDATION_RETENTION,
        margin / POSITION_P95_GATE_M,
    )
    if not 0.0 < retention <= MAXIMUM_VALIDATION_RETENTION:
        raise ValueError(
            "case-16 p95 margin cannot support a validation residual profile"
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
    plan_headroom = readiness.get("plan", {}).get("headroom", {})
    validation_checks = {
        "readiness_split_validation": readiness.get("split") == SPLIT,
        "selection_case16_role": isinstance(selection_checks, Mapping)
        and selection_checks.get("case16_role") is True,
        "selection_forbids_train_profile_reuse": isinstance(
            selection_checks, Mapping
        )
        and selection_checks.get("train_profile_reuse_forbidden") is True,
        "readiness_requires_case_specific_profile": readiness.get(
            "case_specific_profile_required"
        )
        is True,
        "readiness_requires_structural_profile": readiness.get(
            "safe_window_absent_requires_structural_profile"
        )
        is True,
        "readiness_has_no_pulse_window": readiness.get(
            "profile_window_contract", {}
        ).get("windows")
        == [],
        "base_linear_headroom_exhausted": float(
            plan_headroom.get("base_linear_velocity_mps", float("nan"))
        )
        == 0.0,
        "base_yaw_headroom_exhausted": float(
            plan_headroom.get("base_yaw_rate_radps", float("nan"))
        )
        == 0.0,
        "proxy_headroom_exhausted": float(
            plan_headroom.get("proxy_rate_radps", float("nan"))
        )
        == 0.0,
        "readiness_forbids_case2_reuse": readiness.get(
            "case2_profile_reuse_authorized"
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
            f"case-16 validation profile checks failed: {validation_checks}"
        )

    retention = _retention_fraction(readiness)
    engine = _load_formula_engine(retention=retention)
    readiness_inputs = readiness.get("inputs", {})
    normalized_envelope = np.asarray(
        readiness.get("zero_residual_dynamic_gate", {}).get(
            "normalized_residual_label_abs_max"
        ),
        dtype=np.float64,
    )
    plan_sha = _sha256(plan_path)
    gate_sha = _sha256(gate_path)
    input_checks = {
        "readiness_schema": readiness.get("schema") == READINESS_SCHEMA,
        "readiness_case": readiness.get("case") == CASE,
        "readiness_passed": readiness.get("passed") is True,
        "readiness_closed": engine._closed(readiness),
        "readiness_plan_hash": readiness_inputs.get("plan", {}).get("sha256")
        == plan_sha,
        "readiness_gate_hash": readiness_inputs.get(
            "dynamic_gate", {}
        ).get("sha256")
        == gate_sha,
        "plan_schema_case": plan_metadata.get("schema") == engine.PLAN_SCHEMA
        and plan_metadata.get("case") == CASE,
        "plan_integrity": plan_metadata.get("trajectory_integrity_passed")
        is True,
        "plan_kinematic_gate": plan_metadata.get(
            "timing_transition_kinematic_gate_passed"
        )
        is True,
        "normalized_envelope": normalized_envelope.shape == (3,)
        and np.isfinite(normalized_envelope).all()
        and np.all(normalized_envelope > 0.0)
        and np.all(normalized_envelope < 0.95),
    }
    input_checks.update(validation_checks)
    input_checks = {name: bool(value) for name, value in input_checks.items()}
    if not all(input_checks.values()):
        raise ValueError(
            f"case-16 profile input checks failed: {input_checks}"
        )

    time_s = np.asarray(plan_arrays["time_s"], dtype=np.float64)
    execution_time_s = np.asarray(
        plan_arrays["execution_time_s"], dtype=np.float64
    )
    source_time_s = np.asarray(plan_arrays["source_time_s"], dtype=np.float64)
    anchor_map = np.asarray(
        plan_arrays["source_anchor_execution_index"], dtype=np.int64
    )
    riser_q = np.asarray(plan_arrays["riser_q"], dtype=np.float64)
    feedforward_v_wz = np.asarray(
        plan_arrays["feedforward_v_wz"], dtype=np.float64
    )
    count = len(time_s)
    shape_checks = {
        "state_arrays": count > 1
        and execution_time_s.shape == (count,)
        and source_time_s.shape == (count,)
        and anchor_map.shape == (count,)
        and riser_q.shape == (count,),
        "transition_arrays": feedforward_v_wz.shape == (count - 1, 2),
        "finite": all(
            np.isfinite(value).all()
            for value in (
                time_s,
                execution_time_s,
                source_time_s,
                riser_q,
                feedforward_v_wz,
            )
        ),
        "separate_clocks": np.array_equal(time_s, execution_time_s)
        and np.all(np.diff(time_s) > 0.0)
        and np.all(np.diff(source_time_s) > 0.0)
        and float(source_time_s[-1]) < float(time_s[-1]),
        "source_anchor_map": np.array_equal(
            anchor_map, np.arange(count, dtype=np.int64)
        ),
    }
    shape_checks = {name: bool(value) for name, value in shape_checks.items()}
    if not all(shape_checks.values()):
        raise ValueError(
            f"case-16 profile shape checks failed: {shape_checks}"
        )

    gate_result = engine._single_gate_result(gate)
    trace = gate_result.get("trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError("case-16 gate trace is missing")
    trace_phase = np.asarray(
        [sample.get("phase_time_s") for sample in trace], dtype=np.float64
    )
    trace_position_error = np.asarray(
        [sample.get("position_error_m") for sample in trace],
        dtype=np.float64,
    )
    natural_error_count = int(
        np.count_nonzero(
            trace_position_error > engine.NATURAL_ERROR_THRESHOLD_M
        )
    )
    gate_checks = {
        "schema_case": gate.get("schema") == engine.GATE_SCHEMA
        and gate.get("cases") == [CASE]
        and gate_result.get("case") == CASE,
        "dynamic_pass": gate.get("dynamic_quality_passed") is True
        and gate_result.get("dynamic_quality_passed") is True,
        "thermal_pass": gate.get("thermal_admission_passed") is True
        and gate_result.get("thermal_admission_passed") is True,
        "controller_pass": gate.get("controller_evidence_passed") is True
        and gate_result.get("controller_evidence_passed") is True,
        "zero_residual": np.allclose(
            gate_result.get("residual_action_abs_max"),
            np.zeros(3),
            rtol=0.0,
            atol=1e-12,
        ),
        "trace_finite": np.isfinite(trace_phase).all()
        and np.isfinite(trace_position_error).all(),
        "trace_clock_coverage": abs(float(trace_phase[0])) <= 1e-12
        and np.isclose(
            float(trace_phase[-1]),
            float(time_s[-1]),
            rtol=0.0,
            atol=1e-9,
        ),
        "natural_error_excitation": natural_error_count
        >= engine.MINIMUM_NATURAL_ERROR_TRACE_SAMPLES,
    }
    gate_checks = {name: bool(value) for name, value in gate_checks.items()}
    if not all(gate_checks.values()):
        raise ValueError(
            f"case-16 profile gate checks failed: {gate_checks}"
        )

    raw_envelope = normalized_envelope * engine.POLICY_RESIDUAL_SCALES
    maximum_residuals = raw_envelope * retention
    maximum_slew_rates = maximum_residuals / SLEW_HORIZON_S
    config = engine.CorrectiveTeacherConfig(
        longitudinal_gain_s_inv=0.20,
        lateral_to_yaw_gain_rad_s_m=0.30,
        vertical_gain=0.30,
        deadbands_m=(0.01, 0.01, 0.005),
        maximum_residuals=tuple(maximum_residuals.tolist()),
        maximum_slew_rates=tuple(maximum_slew_rates.tolist()),
    )
    config.validate()
    profile = {
        "schema": engine.CORRECTIVE_TEACHER_PROFILE_SCHEMA,
        "case": CASE,
        "longitudinal_gain_s_inv": config.longitudinal_gain_s_inv,
        "lateral_to_yaw_gain_rad_s_m": (
            config.lateral_to_yaw_gain_rad_s_m
        ),
        "vertical_gain": config.vertical_gain,
        "deadbands_m": list(config.deadbands_m),
        "maximum_residuals": list(config.maximum_residuals),
        "maximum_slew_rates": list(config.maximum_slew_rates),
    }

    model_commands = np.column_stack((feedforward_v_wz, riser_q[:-1]))
    projection = engine._projection_envelope(
        model_commands, maximum_residuals
    )
    negative_clipped = projection["directions"]["negative"][
        "command_clipped_transition_count"
    ]
    positive_clipped = projection["directions"]["positive"][
        "command_clipped_transition_count"
    ]
    p95_margin = float(
        readiness["zero_residual_dynamic_gate"]["dynamic_margins"][
            "position_error_p95_m"
        ]
    )
    formula_checks = {
        "retained_raw_envelope": np.allclose(
            maximum_residuals,
            normalized_envelope
            * engine.POLICY_RESIDUAL_SCALES
            * retention,
            rtol=0.0,
            atol=1e-12,
        ),
        "reserved_policy_margin": np.all(maximum_residuals > 0.0)
        and np.all(maximum_residuals < engine.POLICY_RESIDUAL_SCALES),
        "slew_horizon": np.allclose(
            maximum_slew_rates * SLEW_HORIZON_S,
            maximum_residuals,
            rtol=0.0,
            atol=1e-12,
        ),
        "positive_dynamic_margin": p95_margin > 0.0,
        "projection_command_limits": projection[
            "all_command_limits_passed"
        ],
        "projection_contractive": projection[
            "all_projections_contractive"
        ],
        "negative_linear_projection_not_required": negative_clipped[0] == 0,
        "outward_positive_linear_projection_required": (
            positive_clipped[0] > 0
        ),
        "both_yaw_directions_require_projection": (
            negative_clipped[1] > 0 and positive_clipped[1] > 0
        ),
        "riser_projection_not_required": (
            negative_clipped[2] == 0 and positive_clipped[2] == 0
        ),
        "natural_error_replaces_external_wrench": natural_error_count
        >= engine.MINIMUM_NATURAL_ERROR_TRACE_SAMPLES,
    }
    formula_checks = {
        name: bool(value) for name, value in formula_checks.items()
    }
    if not all(formula_checks.values()):
        raise ValueError(
            f"case-16 profile formula checks failed: {formula_checks}"
        )

    profile_bytes = _json_bytes(profile)
    proposal = {
        "schema": SCHEMA,
        "case": CASE,
        "split": SPLIT,
        "input_checks": input_checks,
        "shape_checks": shape_checks,
        "gate_checks": gate_checks,
        "formula_checks": formula_checks,
        "identities": {
            "readiness": engine._identity(readiness_path),
            "plan": engine._identity(plan_path),
            "dynamic_gate": engine._identity(gate_path),
            "safety_projection_source": engine._identity(
                engine.SAFETY_PROJECTION_SOURCE
            ),
            "corrective_profile": {
                "path": _display(CANONICAL_PROFILE),
                "sha256": _sha256_bytes(profile_bytes),
                "size_bytes": len(profile_bytes),
            },
        },
        "profile_formula": {
            "policy_residual_scales": (
                engine.POLICY_RESIDUAL_SCALES.tolist()
            ),
            "observed_normalized_raw_envelope": normalized_envelope.tolist(),
            "observed_raw_residual_envelope": raw_envelope.tolist(),
            "retention_fraction": retention,
            "maximum_residuals": maximum_residuals.tolist(),
            "slew_horizon_s": SLEW_HORIZON_S,
            "maximum_slew_rates": maximum_slew_rates.tolist(),
        },
        "natural_error_contract": {
            "external_wrench_required": False,
            "external_wrench_profile_created": False,
            "trace_sample_count": len(trace),
            "position_error_threshold_m": engine.NATURAL_ERROR_THRESHOLD_M,
            "samples_above_threshold": natural_error_count,
            "position_error_trace_max_m": float(
                np.max(trace_position_error)
            ),
            "projection_contract": (
                "model_based_residual_safety_projection_before_actuation"
            ),
            "capture_labels_must_use_effective_projected_residual": True,
        },
        "projection_envelope": projection,
        "case23_profile_reuse_authorized": False,
        "case6_profile_reuse_authorized": False,
        "pair_profile_cpu_ready": True,
        "natural_error_pair_profile_cpu_ready": True,
        "runtime_route_implemented": False,
        "authorization_token_issued": False,
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

    retention_checks = {
        "retention_uses_p95_margin_formula": abs(
            retention
            - min(
                MAXIMUM_VALIDATION_RETENTION,
                p95_margin / POSITION_P95_GATE_M,
            )
        )
        <= 1e-12,
        "retention_is_validation_bounded": (
            0.0 < retention <= MAXIMUM_VALIDATION_RETENTION
        ),
        "effective_normalized_envelope_below_0p15": max(
            residual / scale
            for residual, scale in zip(
                profile["maximum_residuals"],
                proposal["profile_formula"]["policy_residual_scales"],
                strict=True,
            )
        )
        < 0.15,
        "case2_profile_not_reused": (
            _sha256_bytes(profile_bytes) != _sha256(CASE2_PROFILE)
        ),
        "case8_profile_not_reused": (
            _sha256_bytes(profile_bytes) != _sha256(CASE8_PROFILE)
        ),
        "no_external_wrench": (
            proposal["natural_error_contract"]["external_wrench_required"]
            is False
            and proposal["natural_error_contract"][
                "external_wrench_profile_created"
            ]
            is False
        ),
    }
    retention_checks = {
        name: bool(value) for name, value in retention_checks.items()
    }
    if not all(retention_checks.values()):
        raise ValueError(
            f"case-16 retention checks failed: {retention_checks}"
        )

    proposal["split"] = SPLIT
    proposal["input_checks"].update(validation_checks)
    proposal["validation_profile_checks"] = retention_checks
    proposal["identities"]["formula_engine"] = {
        "path": _display(FORMULA_ENGINE_PATH),
        "sha256": _sha256(FORMULA_ENGINE_PATH),
    }
    proposal["identities"]["case2_profile_comparison"] = {
        "path": _display(CASE2_PROFILE),
        "sha256": _sha256(CASE2_PROFILE),
    }
    proposal["identities"]["case8_profile_comparison"] = {
        "path": _display(CASE8_PROFILE),
        "sha256": _sha256(CASE8_PROFILE),
    }
    proposal["profile_formula"]["retention_fraction"] = retention
    proposal["profile_formula"]["retention_cap"] = (
        MAXIMUM_VALIDATION_RETENTION
    )
    proposal["profile_formula"]["position_p95_gate_m"] = POSITION_P95_GATE_M
    proposal["profile_formula"]["position_p95_margin_m"] = p95_margin
    proposal["profile_formula"]["retention_rationale"] = (
        "case-16 natural-error envelope bounded by its own p95 dynamic "
        "margin and the held-out validation cap"
    )
    proposal["natural_error_contract"]["validation_only"] = True
    proposal["natural_error_contract"]["external_perturbation_forbidden"] = True
    proposal["natural_error_contract"][
        "requested_residual_is_not_a_training_label"
    ] = True
    proposal["held_out_validation_contract"] = {
        "teacher_admission_authorized": False,
        "label_capture_authorized": False,
        "dataset_creation_authorized": False,
        "dataset_merge_authorized": False,
        "candidate_may_report_validation_improvement_only": True,
        "effective_projected_residual_must_be_assessed": True,
    }
    proposal["case2_profile_reuse_authorized"] = False
    proposal["case7_profile_reuse_authorized"] = False
    proposal["case8_profile_reuse_authorized"] = False
    proposal["train_profile_reuse_authorized"] = False
    proposal["validation_pair_profile_cpu_ready"] = True
    proposal["runtime_route_implemented"] = False
    proposal["validation_runtime_opened"] = False
    proposal["holdout_opened"] = False
    proposal["teacher_admission_authorized"] = False
    proposal["dataset_creation_authorized"] = False
    proposal["next_bounded_action"] = (
        "implement_case16_validation_natural_error_pair_contract_cpu_only_"
        "without_authorization"
    )
    proposal["passed"] = (
        proposal.get("passed") is True
        and all(proposal["input_checks"].values())
        and all(retention_checks.values())
        and all(
            value is False
            for key, value in proposal["held_out_validation_contract"].items()
            if key.endswith("_authorized")
        )
    )
    if proposal["passed"] is not True:
        raise ValueError("case-16 validation profile proposal did not pass")
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
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite profile outputs: {existing}")
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
