#!/usr/bin/env python3
"""Design one non-authorizing case-30 perturbation measurement profile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from rl_platform.tasks.two_wheel_balance.riser_perturbation import (  # noqa: E402
    PERTURBATION_SCHEMA,
    load_deterministic_wrench_profile,
)
from rl_platform.tasks.two_wheel_balance.riser_playback import (  # noqa: E402
    RiserPlaybackPlan,
    load_riser_playback_plan,
)
from scripts.two_wheel_balance.rank_riser_unused_plan_coverage import (  # noqa: E402
    MATERIAL_SHIFT_THRESHOLDS,
    MAXIMUM_NORMALIZATION_SAMPLES_PER_PLAN,
    MAXIMUM_TARGET_SAMPLES,
    MINIMUM_FEATURE_SCALE,
    _phase_aligned_actions,
    _subsample,
    build_plan_command_signature,
)


TARGET_CASE = 30
REFERENCE_CASE = 4
LOCAL_SUPPORT_COUNT = 16
START_MARGIN_S = 2.0
TERMINAL_RECOVERY_MARGIN_S = 5.0
FORCE_BODY_X_N = 20.0
DURATION_STEPS = 20
APPLICATION_HEIGHT_M = 0.5


def _identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing perturbation-design input: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        payload = {
            name: np.asarray(data[name])
            for name in data.files
            if name != "metadata_json"
        }
    return metadata, payload


def select_localized_phase(
    target_features: np.ndarray,
    candidate_features: np.ndarray,
    candidate_times_s: np.ndarray,
    *,
    execution_duration_s: float,
    local_support_count: int = LOCAL_SUPPORT_COUNT,
    start_margin_s: float = START_MARGIN_S,
    terminal_recovery_margin_s: float = TERMINAL_RECOVERY_MARGIN_S,
) -> dict[str, object]:
    target = np.asarray(target_features, dtype=np.float64)
    candidate = np.asarray(candidate_features, dtype=np.float64)
    times = np.asarray(candidate_times_s, dtype=np.float64)
    if (
        target.ndim != 2
        or candidate.ndim != 2
        or target.shape[1] != candidate.shape[1]
        or len(candidate) != len(times)
        or not np.isfinite(target).all()
        or not np.isfinite(candidate).all()
        or not np.isfinite(times).all()
    ):
        raise ValueError("invalid localized phase feature arrays")
    if not 1 <= local_support_count <= len(target):
        raise ValueError("invalid localized support count")
    eligible = (
        (times >= start_margin_s)
        & (times <= execution_duration_s - terminal_recovery_margin_s)
    )
    if not np.any(eligible):
        raise ValueError("no case phase satisfies perturbation recovery margins")
    eligible_indices = np.flatnonzero(eligible)
    candidates = candidate[eligible]
    squared_distance = (
        np.sum(np.square(candidates), axis=1)[:, None]
        + np.sum(np.square(target), axis=1)[None, :]
        - 2.0 * candidates @ target.T
    ) / target.shape[1]
    distance = np.sqrt(np.maximum(squared_distance, 0.0))
    local_distance = np.partition(
        distance, local_support_count - 1, axis=1
    )[:, :local_support_count]
    score = np.mean(local_distance, axis=1)
    local_index = int(np.argmin(score))
    selected_index = int(eligible_indices[local_index])
    selected_distances = distance[local_index]
    nearest_target_indices = np.argsort(selected_distances)[:local_support_count]
    return {
        "candidate_index": selected_index,
        "start_phase_time_s": float(times[selected_index]),
        "local_support_count": local_support_count,
        "local_support_distance_mean": float(np.mean(selected_distances[nearest_target_indices])),
        "local_support_distance_p95": float(
            np.percentile(selected_distances[nearest_target_indices], 95)
        ),
        "nearest_target_indices": nearest_target_indices.tolist(),
        "start_margin_s": start_margin_s,
        "terminal_recovery_margin_s": terminal_recovery_margin_s,
    }


def validate_lqr_envelope(results: list[dict[str, object]]) -> dict[str, object]:
    heights = sorted(
        float(item["summary"]["riser_plant"]["riser_position_target_m"])
        for item in results
    )
    checks = {
        "three_height_results": len(results) == 3,
        "all_passed": all(item.get("passed") is True for item in results),
        "riser_height_coverage": heights == [0.0, 0.6, 1.2],
        "force_magnitude": all(
            item["push"]["forces_x_n"] == [-20.0, 20.0] for item in results
        ),
        "duration_steps": all(
            item["push"]["duration_steps"] == DURATION_STEPS for item in results
        ),
        "application_height": all(
            item["push"]["application_height_above_base_com_m"]
            == APPLICATION_HEIGHT_M
            for item in results
        ),
        "complete_success": all(
            item["summary"]["scenarios"] == 56
            and item["summary"]["success_rate"] == 1.0
            for item in results
        ),
        "zero_action_saturation": all(
            item["summary"]["action_saturation_ratio"] == 0.0
            for item in results
        ),
        "reference_frame_is_global": all(
            item["push"]["application"].startswith("global_x_force")
            for item in results
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"LQR disturbance envelope contract failed: {checks}")
    return {
        "checks": checks,
        "riser_positions_m": heights,
        "scenario_count_total": sum(
            item["summary"]["scenarios"] for item in results
        ),
        "peak_pitch_deg_max": max(
            item["summary"]["peak_pitch_deg_max"] for item in results
        ),
        "action_saturation_ratio_max": max(
            item["summary"]["action_saturation_ratio"] for item in results
        ),
        "validated_force_frame": "global_x",
        "proposed_force_frame": "body_x",
        "frame_transfer_dynamically_validated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--plan-summary", type=Path, required=True)
    parser.add_argument("--teacher-dataset", type=Path, required=True)
    parser.add_argument("--shadow-trace", type=Path, required=True)
    parser.add_argument("--case4-diagnosis", type=Path, required=True)
    parser.add_argument("--localized-audit", type=Path, required=True)
    parser.add_argument("--unused-audit", type=Path, required=True)
    parser.add_argument("--architecture-proposal", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument(
        "--lqr-envelope-result", type=Path, action="append", required=True
    )
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("--proposal-output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "plan_summary": args.plan_summary,
        "teacher_dataset": args.teacher_dataset,
        "shadow_trace": args.shadow_trace,
        "case4_diagnosis": args.case4_diagnosis,
        "localized_audit": args.localized_audit,
        "unused_audit": args.unused_audit,
        "architecture_proposal": args.architecture_proposal,
        "policy": args.policy,
    }
    identities = {name: _identity(path) for name, path in paths.items()}
    lqr_identities = [_identity(path) for path in args.lqr_envelope_result]
    summary = _load_json(args.plan_summary)
    architecture = _load_json(args.architecture_proposal)
    localized = _load_json(args.localized_audit)
    unused = _load_json(args.unused_audit)
    diagnosis = _load_json(args.case4_diagnosis)
    dataset_metadata, dataset = _load_npz(args.teacher_dataset)
    shadow_metadata, shadow = _load_npz(args.shadow_trace)
    lqr_results = [_load_json(path) for path in args.lqr_envelope_result]
    lqr_envelope = validate_lqr_envelope(lqr_results)

    input_checks = {
        "architecture_decision": architecture.get("decision")
        == "controlled_perturbation_contract_first",
        "architecture_runtime_closed": (
            architecture.get("runtime_authorized") is False
        ),
        "architecture_identity": (
            architecture["inputs"]["localized_audit"]["sha256"]
            == identities["localized_audit"]["sha256"]
            and architecture["inputs"]["unused_audit"]["sha256"]
            == identities["unused_audit"]["sha256"]
        ),
        "teacher_identity": all(
            item == identities["teacher_dataset"]["sha256"]
            for item in (
                diagnosis["inputs"]["teacher_dataset"]["sha256"],
                localized["inputs"]["teacher_dataset"]["sha256"],
                unused["inputs"]["teacher_dataset"]["sha256"],
            )
        ),
        "shadow_identity": all(
            item == identities["shadow_trace"]["sha256"]
            for item in (
                diagnosis["inputs"]["shadow_trace"]["sha256"],
                localized["inputs"]["shadow_trace"]["sha256"],
                unused["inputs"]["shadow_trace"]["sha256"],
            )
        ),
        "policy_identity": localized["inputs"]["policy"]["sha256"]
        == identities["policy"]["sha256"],
        "case4_material_gap": diagnosis.get("material_shadow_shift_by_channel")
        == [True, True, False],
        "case4_trace_only": shadow_metadata.get("trace_only") is True
        and shadow_metadata.get("valid_for_training") is False,
        "case30_is_train": TARGET_CASE in dataset_metadata["split_cases"]["train"],
        "holdout_unchanged": dataset_metadata["split_cases"]["holdout"]
        == [3, 5, 13, 19, 24],
        "case30_nominal_command_best": (
            unused["best_existing_training"]["case"] == TARGET_CASE
        ),
        "all70_plan_gate": len(summary["passed_cases"]) == 70,
    }
    if not all(input_checks.values()):
        raise ValueError(f"perturbation design input contract failed: {input_checks}")

    plan_paths = {
        int(path.name.split("_")[1]): path
        for path in args.plan_dir.glob("case_*_smoothed_riser_plan_v1.npz")
    }
    plans: dict[int, RiserPlaybackPlan] = {
        case: load_riser_playback_plan(plan_paths[case])
        for case in summary["passed_cases"]
    }
    case30_plan_identity = _identity(plan_paths[TARGET_CASE])
    if (
        case30_plan_identity["sha256"]
        != unused["best_existing_training"]["plan"]["sha256"]
    ):
        raise ValueError("case30 plan identity drift")

    normalization = np.concatenate(
        [
            build_plan_command_signature(
                plans[case],
                _subsample(
                    plans[case].time_s,
                    MAXIMUM_NORMALIZATION_SAMPLES_PER_PLAN,
                ),
            )
            for case in summary["passed_cases"]
        ]
    )
    feature_mean = np.mean(normalization, axis=0)
    feature_scale = np.maximum(
        np.std(normalization, axis=0), MINIMUM_FEATURE_SCALE
    )

    reference_mask = dataset["case_ids"] == REFERENCE_CASE
    phase_actions = _phase_aligned_actions(
        shadow["phase_time_s"],
        dataset["phase_time_s"][reference_mask],
        dataset["actions"][reference_mask],
    )
    shift = shadow["shadow_teacher_normalized_residual_actions"] - phase_actions
    hotspot_mask = np.any(np.abs(shift) > MATERIAL_SHIFT_THRESHOLDS, axis=1)
    hotspot_times = _subsample(
        shadow["phase_time_s"][hotspot_mask], MAXIMUM_TARGET_SAMPLES
    )
    target = (
        build_plan_command_signature(plans[REFERENCE_CASE], hotspot_times)
        - feature_mean
    ) / feature_scale
    candidate = (
        build_plan_command_signature(plans[TARGET_CASE], plans[TARGET_CASE].time_s)
        - feature_mean
    ) / feature_scale
    selection = select_localized_phase(
        target,
        candidate,
        plans[TARGET_CASE].time_s,
        execution_duration_s=float(plans[TARGET_CASE].time_s[-1]),
    )
    selected_index = int(selection["candidate_index"])
    selected_time = float(selection["start_phase_time_s"])
    selected_plan = plans[TARGET_CASE]
    selected_plan_context = {
        "plan_index": selected_index,
        "phase_time_s": selected_time,
        "execution_duration_s": float(selected_plan.time_s[-1]),
        "feedforward_v_mps": float(
            selected_plan.feedforward_v_wz[selected_index, 0]
        ),
        "feedforward_wz_rad_s": float(
            selected_plan.feedforward_v_wz[selected_index, 1]
        ),
        "feedforward_riser_velocity_mps": float(
            selected_plan.feedforward_riser_velocity[selected_index]
        ),
        "riser_position_m": float(selected_plan.riser_q[selected_index]),
    }

    profile = {
        "schema": PERTURBATION_SCHEMA,
        "case": TARGET_CASE,
        "start_phase_time_s": selected_time,
        "duration_steps": DURATION_STEPS,
        "force_body_x_n": FORCE_BODY_X_N,
        "application_height_m": APPLICATION_HEIGHT_M,
    }
    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    args.profile_output.write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8"
    )
    loaded_profile, profile_identity = load_deterministic_wrench_profile(
        args.profile_output
    )
    if loaded_profile.as_dict() != profile:
        raise ValueError("generated perturbation profile failed round-trip validation")

    proposal = {
        "schema": "cinebotrl_two_wheel_riser_case30_perturbation_proposal_v1",
        "decision": "one_case30_body_forward_pulse_measurement_candidate",
        "decision_status": "cpu_only_profile_not_runtime_authorization",
        "case": TARGET_CASE,
        "split": "train",
        "phase_selection_contract": (
            "all70_normalized_nominal_command_local16_case4_hotspot_v1"
        ),
        "phase_selection": selection,
        "selected_plan_context": selected_plan_context,
        "profile": profile_identity,
        "profile_payload": profile,
        "force_direction_hypothesis": {
            "case4_shadow_minus_original_vx_signed_mean": diagnosis[
                "on_policy_shadow_to_original_phase_teacher"
            ]["signed_mean"][0],
            "reason": (
                "body-forward impulse is expected to advance the base and induce "
                "the negative vx teacher correction observed in case4"
            ),
            "wz_coverage_expected_without_measurement": False,
            "hypothesis_dynamically_validated": False,
        },
        "lqr_disturbance_prior": lqr_envelope,
        "lqr_envelope_results": lqr_identities,
        "case30_plan": case30_plan_identity,
        "input_contract_checks": input_checks,
        "inputs": identities,
        "canary_admission_requirements": [
            "fresh hash-bound runtime namespace and single-use authorization",
            "exclusive WSL Windows and NVIDIA ownership preflight",
            "unchanged dynamic thermal controller and residual-label gates",
            "exact 20-step pulse trigger and release telemetry",
            "no planner or policy command modification by the perturbation",
            "no dataset or training artifact",
        ],
        "post_canary_measurement_requirements": [
            "physical admission passes independently",
            "visited-state coverage is compared to the case4 hotspot",
            "shadow-teacher vx and wz materiality is reported independently",
            "stop after case30 regardless of outcome",
        ],
        "case4_split_changed": False,
        "case4_labels_admitted_for_training": False,
        "holdout_opened": False,
        "runtime_authorized": False,
        "authorization_token_issued": False,
        "runtime_namespace_created": False,
        "dataset_created": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "valid_for_training": False,
    }
    args.proposal_output.parent.mkdir(parents=True, exist_ok=True)
    args.proposal_output.write_text(
        json.dumps(proposal, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(proposal, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
