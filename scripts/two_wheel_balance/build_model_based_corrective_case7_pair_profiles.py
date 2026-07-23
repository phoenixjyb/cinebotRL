#!/usr/bin/env python3
"""Build closed, case-specific corrective profiles for case 7."""

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
from rl_platform.tasks.two_wheel_balance.riser_perturbation import (  # noqa: E402
    DeterministicWrenchPulse,
)


SCHEMA = "cinebotrl_two_wheel_riser_case7_pair_profile_proposal_cpu_v1"
READINESS_SCHEMA = "cinebotrl_two_wheel_riser_case7_pair_readiness_cpu_v1"
PLAN_SCHEMA = "cinebotrl_two_wheel_riser_smoothed_plan_v1"
CASE = 7
POLICY_HZ = 200.0
POLICY_RESIDUAL_SCALES = np.array([0.05, 0.05, 0.02], dtype=np.float64)
RAW_ENVELOPE_RETENTION = 0.50
SLEW_HORIZON_S = 0.35
PULSE_DURATION_STEPS = 20
PULSE_FORCE_BODY_X_N = 20.0
PULSE_APPLICATION_HEIGHT_M = 0.5
MINIMUM_RECOVERY_TAIL_S = 1.0
MINIMUM_SOURCE_SAMPLES_DURING_PULSE = 4
MINIMUM_FREE_BODY_DISPLACEMENT_M = 0.003
RISER_POSITION_BOUNDS_M = (0.0, 1.2)
COMMAND_LOWER_BOUNDS = np.array([-0.4, -0.4, 0.0], dtype=np.float64)
COMMAND_UPPER_BOUNDS = np.array([0.4, 0.4, 1.2], dtype=np.float64)
MOTION_LIMITS = np.array(
    [0.4, 0.4, 1.0, 0.41887902047863906], dtype=np.float64
)
LOW_MOTION_LIMITS = np.array([0.3, 0.3, 0.25, 0.3], dtype=np.float64)
CANONICAL_CORRECTIVE_PROFILE = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case7_profile_v1.json"
)
CANONICAL_WRENCH_PROFILE = (
    PROJECT_ROOT
    / "scripts/two_wheel_balance/"
    "model_based_corrective_teacher_case7_wrench_profile_v1.json"
)
SAFETY_PROJECTION_SOURCE = (
    PROJECT_ROOT
    / "src/rl_platform/tasks/two_wheel_balance/"
    "riser_residual_dataset.py"
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
                "feedforward_riser_velocity",
                "feedforward_proxy_velocity",
            )
        }
    if not isinstance(metadata, dict):
        raise ValueError("case-7 plan metadata must be an object")
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


def _low_motion_windows(
    feedforward_v_wz: np.ndarray,
    feedforward_riser: np.ndarray,
    feedforward_proxy: np.ndarray,
) -> list[tuple[int, int]]:
    utilization = np.column_stack(
        (
            np.abs(feedforward_v_wz[:, 0]),
            np.abs(feedforward_v_wz[:, 1]),
            np.abs(feedforward_riser),
            np.max(np.abs(feedforward_proxy), axis=1),
        )
    )
    mask = np.all(utilization <= LOW_MOTION_LIMITS + 1e-12, axis=1)
    windows: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        if start is None:
            continue
        if active and index != len(mask) - 1:
            continue
        windows.append((start, index if active else index - 1))
        start = None
    return windows


def _select_pulse_window(
    time_s: np.ndarray,
    feedforward_v_wz: np.ndarray,
    feedforward_riser: np.ndarray,
    feedforward_proxy: np.ndarray,
) -> dict[str, object]:
    transition_time_s = time_s[:-1]
    pulse_duration_s = PULSE_DURATION_STEPS / POLICY_HZ
    candidates: list[dict[str, object]] = []
    for window_index, (window_start, window_end) in enumerate(
        _low_motion_windows(
            feedforward_v_wz,
            feedforward_riser,
            feedforward_proxy,
        )
    ):
        for start_index in range(window_start, window_end + 1):
            start_s = float(transition_time_s[start_index])
            end_s = start_s + pulse_duration_s
            if end_s > float(transition_time_s[window_end]) + 1e-12:
                continue
            recovery_tail_s = float(time_s[-1] - end_s)
            if recovery_tail_s < MINIMUM_RECOVERY_TAIL_S - 1e-12:
                continue
            mask = (transition_time_s >= start_s - 1e-12) & (
                transition_time_s < end_s - 1e-12
            )
            sample_count = int(np.count_nonzero(mask))
            if sample_count < MINIMUM_SOURCE_SAMPLES_DURING_PULSE:
                continue
            local_maximums = np.array(
                [
                    np.max(np.abs(feedforward_v_wz[mask, 0])),
                    np.max(np.abs(feedforward_v_wz[mask, 1])),
                    np.max(np.abs(feedforward_riser[mask])),
                    np.max(np.abs(feedforward_proxy[mask])),
                ],
                dtype=np.float64,
            )
            normalized = local_maximums / MOTION_LIMITS
            candidates.append(
                {
                    "window_index": window_index,
                    "window_start_index": window_start,
                    "window_end_index": window_end,
                    "pulse_start_index": start_index,
                    "pulse_start_phase_time_s": start_s,
                    "pulse_nominal_end_phase_time_s": end_s,
                    "pulse_source_sample_count": sample_count,
                    "recovery_tail_s": recovery_tail_s,
                    "pulse_mask": mask,
                    "local_maximums": local_maximums,
                    "selection_key": (
                        round(float(np.max(normalized)), 9),
                        float(np.sum(normalized)),
                        start_s,
                    ),
                }
            )
    if not candidates:
        raise ValueError("case-7 profile has no eligible low-motion pulse")
    return min(candidates, key=lambda item: item["selection_key"])


def _projection_envelope(
    model_commands: np.ndarray,
    maximum_residuals: np.ndarray,
) -> dict[str, object]:
    directions: dict[str, object] = {}
    all_unclipped = True
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
        contractive = bool(
            np.all(
                np.abs(effective_residual)
                <= np.abs(requested_residual) + 1e-12
            )
        )
        all_unclipped &= not bool(np.any(clipped))
        all_contractive &= contractive
        directions[name] = {
            "requested_residual": (sign * maximum_residuals).tolist(),
            "command_clipped_transition_count": np.sum(
                clipped, axis=0
            ).tolist(),
            "effective_residual_abs_max": np.max(
                np.abs(effective_residual), axis=0
            ).tolist(),
            "projection_is_contractive": contractive,
        }
    return {
        "command_lower_bounds": COMMAND_LOWER_BOUNDS.tolist(),
        "command_upper_bounds": COMMAND_UPPER_BOUNDS.tolist(),
        "directions": directions,
        "all_requested_commands_unclipped": all_unclipped,
        "all_projections_contractive": all_contractive,
    }


def build_profiles(
    *,
    readiness: Mapping[str, object],
    readiness_path: Path,
    plan_metadata: Mapping[str, object],
    plan_arrays: Mapping[str, np.ndarray],
    plan_path: Path,
    plant: Mapping[str, object],
    plant_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    readiness_inputs = readiness.get("inputs", {})
    readiness_plan = readiness_inputs.get("plan", {})
    normalized_envelope = np.asarray(
        readiness.get("zero_residual_dynamic_gate", {}).get(
            "normalized_residual_label_abs_max"
        ),
        dtype=np.float64,
    )
    plan_sha = _sha256(plan_path)
    input_checks = {
        "readiness_schema": readiness.get("schema") == READINESS_SCHEMA,
        "readiness_case": readiness.get("case") == CASE,
        "readiness_passed": readiness.get("passed") is True,
        "readiness_requires_case_profile": readiness.get(
            "case_specific_profile_required"
        )
        is True,
        "readiness_has_low_motion_window": readiness.get(
            "profile_window_contract", {}
        ).get("bounded_window_found")
        is True,
        "readiness_forbids_case23_reuse": readiness.get(
            "case23_profile_reuse_authorized"
        )
        is False,
        "readiness_forbids_case6_reuse": readiness.get(
            "case6_profile_reuse_authorized"
        )
        is False,
        "readiness_forbids_case2_reuse": readiness.get(
            "case2_profile_reuse_authorized"
        )
        is False,
        "readiness_not_profile_ready": readiness.get(
            "pair_profile_cpu_ready"
        )
        is False,
        "readiness_closed": _closed(readiness),
        "readiness_plan_hash": isinstance(readiness_plan, Mapping)
        and readiness_plan.get("sha256") == plan_sha,
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
        raise ValueError(f"case-7 profile input checks failed: {input_checks}")

    time_s = np.asarray(plan_arrays["time_s"], dtype=np.float64)
    execution_time_s = np.asarray(
        plan_arrays["execution_time_s"], dtype=np.float64
    )
    source_time_s = np.asarray(
        plan_arrays["source_time_s"], dtype=np.float64
    )
    anchor_map = np.asarray(
        plan_arrays["source_anchor_execution_index"], dtype=np.int64
    )
    riser_q = np.asarray(plan_arrays["riser_q"], dtype=np.float64)
    feedforward_v_wz = np.asarray(
        plan_arrays["feedforward_v_wz"], dtype=np.float64
    )
    feedforward_riser = np.asarray(
        plan_arrays["feedforward_riser_velocity"], dtype=np.float64
    )
    feedforward_proxy = np.asarray(
        plan_arrays["feedforward_proxy_velocity"], dtype=np.float64
    )
    count = len(time_s)
    transition_count = count - 1
    shape_checks = {
        "state_arrays": count > 1
        and execution_time_s.shape == (count,)
        and source_time_s.shape == (count,)
        and anchor_map.shape == (count,)
        and riser_q.shape == (count,),
        "transition_arrays": feedforward_v_wz.shape == (transition_count, 2)
        and feedforward_riser.shape == (transition_count,)
        and feedforward_proxy.shape == (transition_count, 3),
        "finite": all(
            np.isfinite(value).all()
            for value in (
                time_s,
                execution_time_s,
                source_time_s,
                riser_q,
                feedforward_v_wz,
                feedforward_riser,
                feedforward_proxy,
            )
        ),
        "clocks": bool(
            np.array_equal(time_s, execution_time_s)
            and np.all(np.diff(time_s) > 0.0)
            and np.all(np.diff(source_time_s) > 0.0)
        ),
        "source_anchor_map": bool(
            np.array_equal(anchor_map, np.arange(count, dtype=np.int64))
        ),
    }
    shape_checks = {name: bool(value) for name, value in shape_checks.items()}
    if not all(shape_checks.values()):
        raise ValueError(f"case-7 plan array checks failed: {shape_checks}")

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
    corrective_profile = {
        "schema": CORRECTIVE_TEACHER_PROFILE_SCHEMA,
        "case": CASE,
        "longitudinal_gain_s_inv": config.longitudinal_gain_s_inv,
        "lateral_to_yaw_gain_rad_s_m": config.lateral_to_yaw_gain_rad_s_m,
        "vertical_gain": config.vertical_gain,
        "deadbands_m": list(config.deadbands_m),
        "maximum_residuals": list(config.maximum_residuals),
        "maximum_slew_rates": list(config.maximum_slew_rates),
    }

    selected = _select_pulse_window(
        time_s,
        feedforward_v_wz,
        feedforward_riser,
        feedforward_proxy,
    )
    pulse_mask = np.asarray(selected.pop("pulse_mask"), dtype=bool)
    selection_key = list(selected.pop("selection_key"))
    pulse_indices = np.flatnonzero(pulse_mask)
    pulse_start_s = float(selected["pulse_start_phase_time_s"])
    pulse_duration_s = PULSE_DURATION_STEPS / POLICY_HZ
    pulse = DeterministicWrenchPulse(
        case=CASE,
        start_phase_time_s=pulse_start_s,
        duration_steps=PULSE_DURATION_STEPS,
        force_body_x_n=PULSE_FORCE_BODY_X_N,
        application_height_m=PULSE_APPLICATION_HEIGHT_M,
    )
    pulse.validate(expected_case=CASE)
    wrench_profile = pulse.as_dict()

    local_maximums = np.asarray(selected["local_maximums"], dtype=np.float64)
    local_headroom = MOTION_LIMITS - local_maximums
    local_riser_min = float(np.min(riser_q[pulse_indices]))
    local_riser_max = float(np.max(riser_q[pulse_indices]))
    riser_target_headroom = min(
        local_riser_min - RISER_POSITION_BOUNDS_M[0],
        RISER_POSITION_BOUNDS_M[1] - local_riser_max,
    )
    model_commands = np.column_stack(
        (feedforward_v_wz, riser_q[:-1])
    )
    projection = _projection_envelope(model_commands, maximum_residuals)
    pulse_projection = _projection_envelope(
        model_commands[pulse_mask], maximum_residuals
    )
    negative_clips = projection["directions"]["negative"][
        "command_clipped_transition_count"
    ]
    positive_clips = projection["directions"]["positive"][
        "command_clipped_transition_count"
    ]
    effective_normalized_max = max(
        np.max(
            np.asarray(direction["effective_residual_abs_max"])
            / POLICY_RESIDUAL_SCALES
        )
        for direction in projection["directions"].values()
    )

    total_mass_kg = float(plant["nominal"]["total_mass_kg"])
    impulse_ns = abs(PULSE_FORCE_BODY_X_N) * pulse_duration_s
    displacement_m = (
        0.5
        * abs(PULSE_FORCE_BODY_X_N)
        / total_mass_kg
        * pulse_duration_s**2
    )
    accepted_impulses = [
        abs(float(value))
        for value in plant["provisional_operating_envelope"][
            "accepted_signed_push_impulse_ns"
        ]
    ]
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
        "normalized_policy_margin": bool(
            np.max(maximum_residuals / POLICY_RESIDUAL_SCALES) < 0.95
        ),
        "slew_horizon": bool(
            np.allclose(
                maximum_slew_rates * SLEW_HORIZON_S,
                maximum_residuals,
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "pulse_source_samples": int(
            selected["pulse_source_sample_count"]
        )
        >= MINIMUM_SOURCE_SAMPLES_DURING_PULSE,
        "recovery_tail": float(selected["recovery_tail_s"])
        >= MINIMUM_RECOVERY_TAIL_S,
        "base_residual_headroom": maximum_residuals[0]
        < local_headroom[0],
        "yaw_residual_headroom": maximum_residuals[1]
        < local_headroom[1],
        "riser_target_headroom": maximum_residuals[2]
        < riser_target_headroom,
        "proxy_motion_headroom": local_headroom[3] > 0.0,
        "full_plan_base_yaw_unclipped": negative_clips[:2] == [0, 0]
        and positive_clips[:2] == [0, 0],
        "full_plan_riser_clipping_bounded": negative_clips[2] <= 4
        and positive_clips[2] == 0,
        "pulse_window_projection_unclipped": pulse_projection[
            "all_requested_commands_unclipped"
        ]
        is True,
        "projection_contractive": projection[
            "all_projections_contractive"
        ]
        is True,
        "effective_normalized_margin": effective_normalized_max < 0.95,
        "plant_mass": total_mass_kg == 28.0,
        "pulse_impulse_envelope": impulse_ns
        <= max(accepted_impulses) + 1e-12,
        "pulse_observable_lower_bound": displacement_m
        >= MINIMUM_FREE_BODY_DISPLACEMENT_M,
    }
    formula_checks = {
        name: bool(value) for name, value in formula_checks.items()
    }
    if not all(formula_checks.values()):
        raise ValueError(f"case-7 profile formula checks failed: {formula_checks}")

    selected["local_maximums"] = local_maximums.tolist()
    selected["local_headroom"] = local_headroom.tolist()
    selected["local_riser_target_range_m"] = [
        local_riser_min,
        local_riser_max,
    ]
    selected["local_riser_target_headroom_m"] = riser_target_headroom
    selected["selection_key"] = selection_key
    selected["pulse_duration_steps"] = PULSE_DURATION_STEPS
    selected["pulse_duration_s_at_policy_rate"] = pulse_duration_s

    corrective_bytes = _json_bytes(corrective_profile)
    wrench_bytes = _json_bytes(wrench_profile)
    proposal = {
        "schema": SCHEMA,
        "case": CASE,
        "split": "train",
        "input_checks": input_checks,
        "shape_checks": shape_checks,
        "formula_checks": formula_checks,
        "identities": {
            "readiness": {
                "path": _display(readiness_path),
                "sha256": _sha256(readiness_path),
            },
            "plan": {
                "path": _display(plan_path),
                "sha256": plan_sha,
            },
            "plant_prior": {
                "path": _display(plant_path),
                "sha256": _sha256(plant_path),
            },
            "safety_projection_source": {
                "path": _display(SAFETY_PROJECTION_SOURCE),
                "sha256": _sha256(SAFETY_PROJECTION_SOURCE),
            },
            "corrective_profile": {
                "path": _display(CANONICAL_CORRECTIVE_PROFILE),
                "sha256": _sha256_bytes(corrective_bytes),
            },
            "wrench_profile": {
                "path": _display(CANONICAL_WRENCH_PROFILE),
                "sha256": _sha256_bytes(wrench_bytes),
            },
        },
        "source_execution_clock_contract": {
            "source_state_count": count,
            "transition_count": transition_count,
            "source_duration_s": float(source_time_s[-1]),
            "execution_duration_s": float(time_s[-1]),
            "source_anchor_map_exact": True,
        },
        "profile_formula": {
            "policy_residual_scales": POLICY_RESIDUAL_SCALES.tolist(),
            "observed_normalized_raw_envelope": normalized_envelope.tolist(),
            "observed_raw_residual_envelope": raw_envelope.tolist(),
            "retention_fraction": RAW_ENVELOPE_RETENTION,
            "maximum_residuals": maximum_residuals.tolist(),
            "slew_horizon_s": SLEW_HORIZON_S,
            "maximum_slew_rates": maximum_slew_rates.tolist(),
        },
        "pulse_window": selected,
        "projection_envelope": {
            "full_plan": projection,
            "pulse_window": pulse_projection,
            "effective_normalized_action_abs_max": (
                effective_normalized_max
            ),
            "label_contract": "effective_post_supervisor_residual_only",
        },
        "pulse_lower_model": {
            "total_mass_kg": total_mass_kg,
            "impulse_ns": impulse_ns,
            "application_moment_impulse_nms": (
                impulse_ns * PULSE_APPLICATION_HEIGHT_M
            ),
            "ideal_free_body_delta_velocity_mps": impulse_ns / total_mass_kg,
            "ideal_free_body_displacement_during_pulse_m": displacement_m,
            "limitation": (
                "observability and envelope screen only; not a prediction of "
                "closed-loop Isaac response"
            ),
        },
        "case23_profile_reuse_authorized": False,
        "case6_profile_reuse_authorized": False,
        "case2_profile_reuse_authorized": False,
        "pair_profile_cpu_ready": True,
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
            "implement_case7_pair_runtime_contract_cpu_only_without_authorization"
        ),
        "passed": True,
    }
    return corrective_profile, wrench_profile, proposal


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
    plan_metadata, plan_arrays = _load_plan(args.plan)
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
