#!/usr/bin/env python3
"""Build a closed, projection-aware natural-error profile for case 2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.riser_corrective_teacher import (  # noqa: E402
    CORRECTIVE_TEACHER_PROFILE_SCHEMA,
    CorrectiveTeacherConfig,
)


SCHEMA = (
    "cinebotrl_two_wheel_riser_case2_natural_error_profile_proposal_cpu_v1"
)
READINESS_SCHEMA = "cinebotrl_two_wheel_riser_case2_pair_readiness_cpu_v1"
PLAN_SCHEMA = "cinebotrl_two_wheel_riser_smoothed_plan_v1"
GATE_SCHEMA = "recomo_two_wheel_riser_reference_playback_v1"
CASE = 2
POLICY_RESIDUAL_SCALES = np.array([0.05, 0.05, 0.02], dtype=np.float64)
RAW_ENVELOPE_RETENTION = 0.25
SLEW_HORIZON_S = 0.40
COMMAND_LOWER_BOUNDS = np.array([-0.4, -0.4, 0.0], dtype=np.float64)
COMMAND_UPPER_BOUNDS = np.array([0.4, 0.4, 1.2], dtype=np.float64)
NATURAL_ERROR_THRESHOLD_M = 0.03
MINIMUM_NATURAL_ERROR_TRACE_SAMPLES = 10
CANONICAL_PROFILE = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case2_natural_error_profile_v1.json"
)
SAFETY_PROJECTION_SOURCE = (
    PROJECT_ROOT
    / "src/rl_platform/tasks/two_wheel_balance/riser_residual_policy.py"
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


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": _display(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_plan(
    path: Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {
            name: np.asarray(data[name])
            for name in (
                "time_s",
                "execution_time_s",
                "source_time_s",
                "source_anchor_execution_index",
                "riser_q",
                "feedforward_v_wz",
            )
        }
    if not isinstance(metadata, dict):
        raise ValueError("case-2 plan metadata must be an object")
    return metadata, arrays


def _closed(payload: Mapping[str, object]) -> bool:
    return all(
        payload.get(field) is False
        for field in (
            "runtime_authorized",
            "gpu_launch_authorized",
            "label_capture_authorized",
            "dataset_conversion_authorized",
            "dataset_merge_authorized",
            "bc_authorized",
            "ppo_authorized",
            "training_started",
            "valid_for_training",
        )
    )


def _single_gate_result(gate: Mapping[str, object]) -> dict[str, object]:
    results = gate.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("case-2 gate must contain exactly one result")
    result = results[0]
    if not isinstance(result, dict):
        raise ValueError("case-2 gate result must be an object")
    return result


def _projection_envelope(
    model_commands: np.ndarray,
    maximum_residuals: np.ndarray,
) -> dict[str, object]:
    directions: dict[str, object] = {}
    all_limits_passed = True
    all_contractive = True
    for name, sign in (("negative", -1.0), ("positive", 1.0)):
        requested_residual = np.broadcast_to(
            sign * maximum_residuals, model_commands.shape
        )
        requested_commands = model_commands + requested_residual
        effective_commands = np.clip(
            requested_commands,
            COMMAND_LOWER_BOUNDS,
            COMMAND_UPPER_BOUNDS,
        )
        effective_residual = effective_commands - model_commands
        clipped = np.abs(effective_commands - requested_commands) > 1e-12
        limits_passed = bool(
            np.all(effective_commands >= COMMAND_LOWER_BOUNDS - 1e-12)
            and np.all(effective_commands <= COMMAND_UPPER_BOUNDS + 1e-12)
        )
        contractive = bool(
            np.all(
                np.abs(effective_residual)
                <= np.abs(requested_residual) + 1e-12
            )
        )
        all_limits_passed &= limits_passed
        all_contractive &= contractive
        directions[name] = {
            "requested_residual": (sign * maximum_residuals).tolist(),
            "command_clipped_transition_count": np.sum(
                clipped, axis=0
            ).tolist(),
            "effective_residual_abs_max": np.max(
                np.abs(effective_residual), axis=0
            ).tolist(),
            "requested_vs_effective_delta_abs_max": np.max(
                np.abs(effective_residual - requested_residual), axis=0
            ).tolist(),
            "command_limits_passed": limits_passed,
            "projection_is_contractive": contractive,
        }
    return {
        "command_lower_bounds": COMMAND_LOWER_BOUNDS.tolist(),
        "command_upper_bounds": COMMAND_UPPER_BOUNDS.tolist(),
        "directions": directions,
        "all_command_limits_passed": all_limits_passed,
        "all_projections_contractive": all_contractive,
    }


def build_profile(
    *,
    readiness: Mapping[str, object],
    readiness_path: Path,
    plan_metadata: Mapping[str, object],
    plan_arrays: Mapping[str, np.ndarray],
    plan_path: Path,
    gate: Mapping[str, object],
    gate_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    readiness_inputs = readiness.get("inputs", {})
    readiness_plan = readiness_inputs.get("plan", {})
    readiness_gate = readiness_inputs.get("dynamic_gate", {})
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
        "readiness_requires_structural_profile": readiness.get(
            "safe_window_absent_requires_structural_profile"
        )
        is True,
        "readiness_forbids_case23_reuse": readiness.get(
            "case23_profile_reuse_authorized"
        )
        is False,
        "readiness_forbids_case6_reuse": readiness.get(
            "case6_profile_reuse_authorized"
        )
        is False,
        "readiness_closed": _closed(readiness),
        "readiness_plan_hash": isinstance(readiness_plan, Mapping)
        and readiness_plan.get("sha256") == plan_sha,
        "readiness_gate_hash": isinstance(readiness_gate, Mapping)
        and readiness_gate.get("sha256") == gate_sha,
        "plan_schema_case": plan_metadata.get("schema") == PLAN_SCHEMA
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
    input_checks = {name: bool(value) for name, value in input_checks.items()}
    if not all(input_checks.values()):
        raise ValueError(f"case-2 profile input checks failed: {input_checks}")

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
        "finite": bool(
            all(
                np.isfinite(value).all()
                for value in (
                    time_s,
                    execution_time_s,
                    source_time_s,
                    riser_q,
                    feedforward_v_wz,
                )
            )
        ),
        "separate_clocks": bool(
            np.array_equal(time_s, execution_time_s)
            and np.all(np.diff(time_s) > 0.0)
            and np.all(np.diff(source_time_s) > 0.0)
            and float(source_time_s[-1]) < float(time_s[-1])
        ),
        "source_anchor_map": bool(
            np.array_equal(anchor_map, np.arange(count, dtype=np.int64))
        ),
    }
    shape_checks = {name: bool(value) for name, value in shape_checks.items()}
    if not all(shape_checks.values()):
        raise ValueError(f"case-2 profile shape checks failed: {shape_checks}")

    gate_result = _single_gate_result(gate)
    trace = gate_result.get("trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError("case-2 gate trace is missing")
    trace_phase = np.asarray(
        [sample.get("phase_time_s") for sample in trace], dtype=np.float64
    )
    trace_position_error = np.asarray(
        [sample.get("position_error_m") for sample in trace], dtype=np.float64
    )
    natural_error_count = int(
        np.count_nonzero(trace_position_error > NATURAL_ERROR_THRESHOLD_M)
    )
    gate_checks = {
        "schema_case": gate.get("schema") == GATE_SCHEMA
        and gate.get("cases") == [CASE]
        and gate_result.get("case") == CASE,
        "dynamic_pass": gate.get("dynamic_quality_passed") is True
        and gate_result.get("dynamic_quality_passed") is True,
        "thermal_pass": gate.get("thermal_admission_passed") is True
        and gate_result.get("thermal_admission_passed") is True,
        "controller_pass": gate.get("controller_evidence_passed") is True
        and gate_result.get("controller_evidence_passed") is True,
        "zero_residual": bool(
            np.allclose(
                gate_result.get("residual_action_abs_max"),
                np.zeros(3),
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "trace_finite": bool(
            np.isfinite(trace_phase).all()
            and np.isfinite(trace_position_error).all()
        ),
        "trace_clock_coverage": bool(
            abs(float(trace_phase[0])) <= 1e-12
            and np.isclose(
                float(trace_phase[-1]),
                float(time_s[-1]),
                rtol=0.0,
                atol=1e-9,
            )
        ),
        "natural_error_excitation": (
            natural_error_count >= MINIMUM_NATURAL_ERROR_TRACE_SAMPLES
        ),
    }
    gate_checks = {name: bool(value) for name, value in gate_checks.items()}
    if not all(gate_checks.values()):
        raise ValueError(f"case-2 profile gate checks failed: {gate_checks}")

    raw_envelope = normalized_envelope * POLICY_RESIDUAL_SCALES
    maximum_residuals = raw_envelope * RAW_ENVELOPE_RETENTION
    maximum_slew_rates = maximum_residuals / SLEW_HORIZON_S
    config = CorrectiveTeacherConfig(
        longitudinal_gain_s_inv=0.20,
        lateral_to_yaw_gain_rad_s_m=0.30,
        vertical_gain=0.30,
        deadbands_m=(0.01, 0.01, 0.005),
        maximum_residuals=tuple(maximum_residuals.tolist()),
        maximum_slew_rates=tuple(maximum_slew_rates.tolist()),
    )
    config.validate()
    profile = {
        "schema": CORRECTIVE_TEACHER_PROFILE_SCHEMA,
        "case": CASE,
        "longitudinal_gain_s_inv": config.longitudinal_gain_s_inv,
        "lateral_to_yaw_gain_rad_s_m": config.lateral_to_yaw_gain_rad_s_m,
        "vertical_gain": config.vertical_gain,
        "deadbands_m": list(config.deadbands_m),
        "maximum_residuals": list(config.maximum_residuals),
        "maximum_slew_rates": list(config.maximum_slew_rates),
    }

    model_commands = np.column_stack(
        (feedforward_v_wz, riser_q[:-1])
    )
    projection = _projection_envelope(model_commands, maximum_residuals)
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
        "retained_raw_envelope": bool(
            np.allclose(
                maximum_residuals,
                normalized_envelope
                * POLICY_RESIDUAL_SCALES
                * RAW_ENVELOPE_RETENTION,
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "reserved_policy_margin": bool(
            np.all(maximum_residuals > 0.0)
            and np.all(maximum_residuals < POLICY_RESIDUAL_SCALES)
        ),
        "slew_horizon": bool(
            np.allclose(
                maximum_slew_rates * SLEW_HORIZON_S,
                maximum_residuals,
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "positive_dynamic_margin": p95_margin > 0.0,
        "projection_command_limits": projection[
            "all_command_limits_passed"
        ],
        "projection_contractive": projection[
            "all_projections_contractive"
        ],
        "outward_linear_projection_required": negative_clipped[0] > 0,
        "outward_yaw_projection_required": positive_clipped[1] > 0,
        "riser_projection_not_required": (
            negative_clipped[2] == 0 and positive_clipped[2] == 0
        ),
        "natural_error_replaces_external_wrench": natural_error_count
        >= MINIMUM_NATURAL_ERROR_TRACE_SAMPLES,
    }
    formula_checks = {
        name: bool(value) for name, value in formula_checks.items()
    }
    if not all(formula_checks.values()):
        raise ValueError(
            f"case-2 profile formula checks failed: {formula_checks}"
        )

    profile_bytes = _json_bytes(profile)
    proposal = {
        "schema": SCHEMA,
        "case": CASE,
        "split": "train",
        "input_checks": input_checks,
        "shape_checks": shape_checks,
        "gate_checks": gate_checks,
        "formula_checks": formula_checks,
        "identities": {
            "readiness": _identity(readiness_path),
            "plan": _identity(plan_path),
            "dynamic_gate": _identity(gate_path),
            "safety_projection_source": _identity(SAFETY_PROJECTION_SOURCE),
            "corrective_profile": {
                "path": _display(CANONICAL_PROFILE),
                "sha256": _sha256_bytes(profile_bytes),
                "size_bytes": len(profile_bytes),
            },
        },
        "profile_formula": {
            "policy_residual_scales": POLICY_RESIDUAL_SCALES.tolist(),
            "observed_normalized_raw_envelope": normalized_envelope.tolist(),
            "observed_raw_residual_envelope": raw_envelope.tolist(),
            "retention_fraction": RAW_ENVELOPE_RETENTION,
            "retention_rationale": (
                "conservative natural-error bootstrap because no shared "
                "low-motion perturbation window exists"
            ),
            "maximum_residuals": maximum_residuals.tolist(),
            "slew_horizon_s": SLEW_HORIZON_S,
            "maximum_slew_rates": maximum_slew_rates.tolist(),
        },
        "natural_error_contract": {
            "external_wrench_required": False,
            "external_wrench_profile_created": False,
            "trace_sample_count": len(trace),
            "position_error_threshold_m": NATURAL_ERROR_THRESHOLD_M,
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
        "next_bounded_action": (
            "implement_case2_natural_error_pair_contract_cpu_only"
        ),
        "passed": True,
    }
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
