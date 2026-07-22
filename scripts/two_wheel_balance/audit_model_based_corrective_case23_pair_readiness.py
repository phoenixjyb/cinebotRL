#!/usr/bin/env python3
"""Audit whether the sealed case-23 pair is observable and physically bounded."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "cinebotrl_two_wheel_riser_case23_pair_readiness_audit_v1"
CASE = 23
POLICY_HZ = 200.0
MOTION_LIMITS = {
    "base_linear_velocity_mps": 0.4,
    "base_yaw_rate_radps": 0.4,
    "riser_rate_mps": 1.0,
    "proxy_rate_radps": 0.41887902047863906,
}
DYNAMIC_THRESHOLDS = {
    "position_error_p95_m": 0.15,
    "position_error_max_m": 0.25,
    "pitch_max_deg": 12.0,
    "attitude_error_max_deg": 10.0,
    "riser_servo_error_max_m": 0.03,
    "action_saturation_ratio": 0.2,
}
MINIMUM_P95_IMPROVEMENT_M = 0.003
MINIMUM_RELATIVE_IMPROVEMENT = 0.02


def _identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing case23 readiness input: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _single_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    results = payload.get("results", [])
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("expected one rollout result")
    result = results[0]
    if not isinstance(result, dict):
        raise ValueError("rollout result must be an object")
    return result


def _selection_row(selection: Mapping[str, Any]) -> dict[str, Any]:
    rows = selection.get("selected_rows", [])
    match = [row for row in rows if isinstance(row, dict) and row.get("case") == CASE]
    if len(match) != 1:
        raise ValueError("selection must contain exactly one case23 row")
    return match[0]


def _plan_payload(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        arrays = {
            name: np.asarray(data[name])
            for name in (
                "time_s",
                "feedforward_v_wz",
                "feedforward_riser_velocity",
                "feedforward_proxy_velocity",
            )
        }
    return metadata, arrays


def audit_readiness(
    *,
    proposal: Mapping[str, Any],
    selection: Mapping[str, Any],
    plan_metadata: Mapping[str, Any],
    plan_arrays: Mapping[str, np.ndarray],
    plan_sha256: str,
    dynamic_gate: Mapping[str, Any],
    dynamic_gate_sha256: str,
    plant: Mapping[str, Any],
    case30_final: Mapping[str, Any],
    case30_baseline: Mapping[str, Any],
    case30_candidate: Mapping[str, Any],
    corrective_profile: Mapping[str, Any],
    wrench_profile: Mapping[str, Any],
) -> dict[str, Any]:
    selection_row = _selection_row(selection)
    gate_result = _single_result(dynamic_gate)
    baseline_result = _single_result(case30_baseline)
    candidate_result = _single_result(case30_candidate)
    baseline_pulse = baseline_result.get("deterministic_wrench_perturbation", {}).get(
        "profile", {}
    )
    candidate_pulse = candidate_result.get("deterministic_wrench_perturbation", {}).get(
        "profile", {}
    )
    time_s = np.asarray(plan_arrays["time_s"], dtype=np.float64)
    feedforward_v_wz = np.asarray(plan_arrays["feedforward_v_wz"], dtype=np.float64)
    feedforward_riser = np.asarray(
        plan_arrays["feedforward_riser_velocity"], dtype=np.float64
    )
    feedforward_proxy = np.asarray(
        plan_arrays["feedforward_proxy_velocity"], dtype=np.float64
    )
    start_phase = float(wrench_profile["start_phase_time_s"])
    midpoint_index = int(np.searchsorted(time_s, start_phase, side="left"))
    transition_index = min(midpoint_index, len(feedforward_v_wz) - 1)
    local_slice = slice(
        max(0, transition_index - 5),
        min(len(feedforward_v_wz), transition_index + 6),
    )
    local_max = {
        "base_linear_velocity_mps": float(
            np.max(np.abs(feedforward_v_wz[local_slice, 0]))
        ),
        "base_yaw_rate_radps": float(
            np.max(np.abs(feedforward_v_wz[local_slice, 1]))
        ),
        "riser_rate_mps": float(np.max(np.abs(feedforward_riser[local_slice]))),
        "proxy_rate_radps": float(np.max(np.abs(feedforward_proxy[local_slice]))),
    }
    local_fraction = {
        name: value / MOTION_LIMITS[name] for name, value in local_max.items()
    }
    duration_s = float(wrench_profile["duration_steps"]) / POLICY_HZ
    impulse_ns = abs(float(wrench_profile["force_body_x_n"])) * duration_s
    total_mass_kg = float(plant["nominal"]["total_mass_kg"])
    ideal_delta_velocity_mps = impulse_ns / total_mass_kg
    ideal_displacement_during_pulse_m = (
        0.5
        * abs(float(wrench_profile["force_body_x_n"]))
        / total_mass_kg
        * duration_s**2
    )
    accepted_impulses = plant["provisional_operating_envelope"][
        "accepted_signed_push_impulse_ns"
    ]
    dynamic_margins = {
        name: DYNAMIC_THRESHOLDS[name] - float(gate_result[name])
        for name in DYNAMIC_THRESHOLDS
    }
    case30_pair = case30_final.get("paired_admission", {})
    profile_residuals = np.asarray(
        corrective_profile["maximum_residuals"], dtype=np.float64
    )
    profile_scales = np.array([0.05, 0.05, 0.02], dtype=np.float64)
    expected_longitudinal_correction_from_historical_p95_mps = min(
        float(corrective_profile["maximum_residuals"][0]),
        float(corrective_profile["longitudinal_gain_s_inv"])
        * float(gate_result["position_error_p95_m"]),
    )
    checks = {
        "proposal_passed_closed": proposal.get("case") == CASE
        and proposal.get("passed") is True
        and proposal.get("runtime_authorized") is False
        and proposal.get("gpu_launch_authorized") is False
        and proposal.get("label_capture_authorized") is False
        and proposal.get("dataset_created") is False
        and proposal.get("bc_authorized") is False
        and proposal.get("ppo_authorized") is False,
        "selection_case23_pair": selection.get("passed") is True
        and selection_row.get("selection_role") == "same_seed_paired_canary_required"
        and selection_row.get("plan_sha256") == plan_sha256
        and selection_row.get("dynamic_gate_sha256") == dynamic_gate_sha256,
        "plan_integrity_and_kinematics": plan_metadata.get("case") == CASE
        and plan_metadata.get("trajectory_integrity_passed") is True
        and plan_metadata.get("timing_transition_kinematic_gate_passed") is True,
        "plan_hash_bound": proposal.get("identities", {})
        .get("plan", {})
        .get("sha256")
        == plan_sha256,
        "midpoint_in_plan": 0 < midpoint_index < len(time_s) - 1
        and abs(float(time_s[midpoint_index]) - start_phase)
        <= float(np.max(np.diff(time_s))) + 1e-12,
        "midpoint_motion_headroom": all(value <= 0.8 for value in local_fraction.values()),
        "dynamic_gate_passed": dynamic_gate.get("passed") is True
        and gate_result.get("case") == CASE
        and gate_result.get("dynamic_quality_passed") is True
        and gate_result.get("thermal_admission_passed") is True
        and all(value > 0.0 for value in dynamic_margins.values()),
        "pulse_profile_exact": wrench_profile.get("case") == CASE
        and wrench_profile.get("duration_steps") == 20
        and wrench_profile.get("force_body_x_n") == 20.0
        and wrench_profile.get("application_height_m") == 0.5
        and proposal.get("proposed_perturbation") == wrench_profile,
        "pulse_within_provisional_impulse_envelope": impulse_ns
        <= max(abs(float(value)) for value in accepted_impulses) + 1e-12,
        "pulse_observable_by_free_body_lower_model": (
            ideal_displacement_during_pulse_m >= MINIMUM_P95_IMPROVEMENT_M
        ),
        "case30_same_pulse_precedent": case30_final.get("passed") is True
        and case30_pair.get("corrective_target_admission_passed") is True
        and float(case30_pair.get("position_p95_absolute_improvement_m", 0.0))
        >= MINIMUM_P95_IMPROVEMENT_M
        and float(case30_pair.get("position_p95_relative_improvement", 0.0))
        >= MINIMUM_RELATIVE_IMPROVEMENT
        and baseline_pulse == candidate_pulse
        and baseline_pulse.get("duration_steps") == 20
        and baseline_pulse.get("force_body_x_n") == 20.0
        and baseline_pulse.get("application_height_m") == 0.5,
        "corrective_profile_case_and_margin": bool(
            corrective_profile.get("case") == CASE
            and np.all(profile_residuals > 0.0)
            and np.all(profile_residuals < profile_scales)
        ),
        "historical_error_exceeds_deadband": float(
            gate_result["position_error_p95_m"]
        )
        > float(corrective_profile["deadbands_m"][0]),
        "learning_paths_closed": case30_final.get("dataset_created") is False
        and case30_final.get("bc_authorized") is False
        and case30_final.get("ppo_authorized") is False
        and case30_final.get("training_started") is False,
    }
    passed = all(checks.values())
    return {
        "schema": SCHEMA,
        "case": CASE,
        "checks": checks,
        "plan_sha256": plan_sha256,
        "dynamic_gate_sha256": dynamic_gate_sha256,
        "pulse": {
            "policy_hz": POLICY_HZ,
            "duration_s": duration_s,
            "impulse_ns": impulse_ns,
            "application_moment_impulse_nms": impulse_ns
            * float(wrench_profile["application_height_m"]),
            "ideal_free_body_delta_velocity_mps": ideal_delta_velocity_mps,
            "ideal_free_body_displacement_during_pulse_m": (
                ideal_displacement_during_pulse_m
            ),
            "model_limitation": (
                "free-body estimate is an observability screen, not a prediction "
                "of the closed-loop Isaac response"
            ),
        },
        "midpoint": {
            "requested_phase_time_s": start_phase,
            "plan_index": midpoint_index,
            "plan_time_s": float(time_s[midpoint_index]),
            "local_transition_max_abs": local_max,
            "local_limit_fraction": local_fraction,
        },
        "historical_case23_dynamic_metrics": {
            name: float(gate_result[name]) for name in DYNAMIC_THRESHOLDS
        },
        "historical_case23_dynamic_margins": dynamic_margins,
        "expected_longitudinal_correction_from_historical_p95_mps": (
            expected_longitudinal_correction_from_historical_p95_mps
        ),
        "expected_longitudinal_normalized_fraction": (
            expected_longitudinal_correction_from_historical_p95_mps / 0.05
        ),
        "case30_precedent": {
            "position_p95_absolute_improvement_m": case30_pair.get(
                "position_p95_absolute_improvement_m"
            ),
            "position_p95_relative_improvement": case30_pair.get(
                "position_p95_relative_improvement"
            ),
            "corrective_target_admission_passed": case30_pair.get(
                "corrective_target_admission_passed"
            ),
        },
        "decision": (
            "recommend_exactly_one_bounded_case23_pair_canary"
            if passed
            else "do_not_authorize_case23_pair_canary"
        ),
        "decision_limit": (
            "CPU evidence supports one paired measurement only; dynamic benefit "
            "and safety remain unproven until the baseline and candidate pass "
            "the unchanged Isaac gates"
        ),
        "runtime_authorized": False,
        "gpu_launch_authorized": False,
        "label_capture_authorized": False,
        "dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dynamic-gate", type=Path, required=True)
    parser.add_argument("--plant-prior", type=Path, required=True)
    parser.add_argument("--case30-final", type=Path, required=True)
    parser.add_argument("--case30-baseline", type=Path, required=True)
    parser.add_argument("--case30-candidate", type=Path, required=True)
    parser.add_argument("--corrective-profile", type=Path, required=True)
    parser.add_argument("--wrench-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "proposal": args.proposal,
        "selection": args.selection,
        "plan": args.plan,
        "dynamic_gate": args.dynamic_gate,
        "plant_prior": args.plant_prior,
        "case30_final": args.case30_final,
        "case30_baseline": args.case30_baseline,
        "case30_candidate": args.case30_candidate,
        "corrective_profile": args.corrective_profile,
        "wrench_profile": args.wrench_profile,
    }
    payloads = {
        name: _load_json(path)
        for name, path in paths.items()
        if name != "plan"
    }
    plan_metadata, plan_arrays = _plan_payload(args.plan)
    identities = {name: _identity(path) for name, path in paths.items()}
    report = audit_readiness(
        proposal=payloads["proposal"],
        selection=payloads["selection"],
        plan_metadata=plan_metadata,
        plan_arrays=plan_arrays,
        plan_sha256=str(identities["plan"]["sha256"]),
        dynamic_gate=payloads["dynamic_gate"],
        dynamic_gate_sha256=str(identities["dynamic_gate"]["sha256"]),
        plant=payloads["plant_prior"],
        case30_final=payloads["case30_final"],
        case30_baseline=payloads["case30_baseline"],
        case30_candidate=payloads["case30_candidate"],
        corrective_profile=payloads["corrective_profile"],
        wrench_profile=payloads["wrench_profile"],
    )
    report["inputs"] = identities
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite readiness audit: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
