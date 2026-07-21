#!/usr/bin/env python3
"""Rank unused admitted plans against the case-4 shadow-shift command region."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl_platform.tasks.two_wheel_balance.camera_attitude import (  # noqa: E402
    quaternion_matrix_wxyz,
    rotation_error_vector,
)
from rl_platform.tasks.two_wheel_balance.riser_playback import (  # noqa: E402
    RiserPlaybackPlan,
    interpolate_riser_playback_plan,
    load_riser_playback_plan,
)
from rl_platform.tasks.two_wheel_balance.riser_residual_dataset import (  # noqa: E402
    LOOKAHEAD_HORIZONS_S,
)


MATERIAL_SHIFT_THRESHOLDS = np.array([0.05, 0.05, 0.02], dtype=np.float64)
MAXIMUM_TARGET_SAMPLES = 256
MAXIMUM_CANDIDATE_SAMPLES = 2048
MAXIMUM_NORMALIZATION_SAMPLES_PER_PLAN = 512
MAXIMUM_UNUSED_TO_EXISTING_SCORE_RATIO = 0.80
MINIMUM_FEATURE_SCALE = 1e-4


def _identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing plan-coverage input: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _load_npz(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        payload = {
            name: np.asarray(data[name])
            for name in data.files
            if name != "metadata_json"
        }
    return metadata, payload


def _subsample(values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) < 1:
        raise ValueError("cannot subsample an empty array")
    if len(values) <= maximum:
        return values
    indices = np.rint(np.linspace(0, len(values) - 1, maximum)).astype(np.int64)
    return values[indices]


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _world_xy_to_base(vector_xy: np.ndarray, yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array(
        [c * vector_xy[0] + s * vector_xy[1], -s * vector_xy[0] + c * vector_xy[1]],
        dtype=np.float64,
    )


def _signature_feature_names() -> list[str]:
    names = ["feedforward_vx", "feedforward_wz", "feedforward_riser_velocity"]
    channels = (
        "base_longitudinal_delta",
        "base_lateral_delta",
        "base_yaw_delta",
        "camera_longitudinal_delta",
        "camera_lateral_delta",
        "camera_vertical_delta",
        "camera_attitude_x_delta",
        "camera_attitude_y_delta",
        "camera_attitude_z_delta",
        "riser_delta",
        "feedforward_vx",
        "feedforward_wz",
        "feedforward_riser_velocity",
    )
    for horizon in LOOKAHEAD_HORIZONS_S:
        prefix = f"lookahead_{horizon:.2f}s".replace(".", "p")
        names.extend(f"{prefix}_{channel}" for channel in channels)
    return names


def build_plan_command_signature(
    plan: RiserPlaybackPlan,
    sample_times: np.ndarray,
) -> np.ndarray:
    """Build nominal command/lookahead features without simulated state claims."""

    rows = []
    for elapsed_s in np.asarray(sample_times, dtype=np.float64):
        current = interpolate_riser_playback_plan(plan, float(elapsed_s))
        current_attitude = quaternion_matrix_wxyz(
            current.target_semantic_dfr_quat_wxyz
        )
        row = [
            current.feedforward_v_mps,
            current.feedforward_wz_rad_s,
            current.feedforward_riser_velocity_mps,
        ]
        for horizon_s in LOOKAHEAD_HORIZONS_S:
            future = interpolate_riser_playback_plan(
                plan,
                min(float(plan.time_s[-1]), float(elapsed_s) + horizon_s),
            )
            base_delta = _world_xy_to_base(
                future.base_xy_yaw[:2] - current.base_xy_yaw[:2],
                current.base_xy_yaw[2],
            )
            camera_delta = _world_xy_to_base(
                future.target_position_world_m[:2]
                - current.target_position_world_m[:2],
                current.base_xy_yaw[2],
            )
            attitude_delta = rotation_error_vector(
                current_attitude,
                quaternion_matrix_wxyz(future.target_semantic_dfr_quat_wxyz),
            )
            row.extend(
                (
                    *base_delta,
                    _wrap_angle(
                        future.base_xy_yaw[2] - current.base_xy_yaw[2]
                    ),
                    *camera_delta,
                    future.target_position_world_m[2]
                    - current.target_position_world_m[2],
                    *attitude_delta,
                    future.riser_q - current.riser_q,
                    future.feedforward_v_mps,
                    future.feedforward_wz_rad_s,
                    future.feedforward_riser_velocity_mps,
                )
            )
        rows.append(row)
    signature = np.asarray(rows, dtype=np.float64)
    if signature.shape != (len(sample_times), len(_signature_feature_names())):
        raise ValueError("invalid plan command signature shape")
    if not np.isfinite(signature).all():
        raise ValueError("plan command signature is non-finite")
    return signature


def _directed_distance(
    target: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    distance = (
        np.sum(np.square(target), axis=1)[:, None]
        + np.sum(np.square(candidate), axis=1)[None, :]
        - 2.0 * target @ candidate.T
    ) / target.shape[1]
    nearest = np.sqrt(np.maximum(np.min(distance, axis=1), 0.0))
    mean = float(np.mean(nearest))
    p95 = float(np.percentile(nearest, 95))
    return {
        "nearest_distance_mean": mean,
        "nearest_distance_p50": float(np.percentile(nearest, 50)),
        "nearest_distance_p95": p95,
        "nearest_distance_max": float(np.max(nearest)),
        "score": 0.5 * mean + 0.5 * p95,
    }


def select_unused_candidate(
    existing_ranked: list[dict[str, object]],
    unused_ranked: list[dict[str, object]],
    *,
    maximum_score_ratio: float = MAXIMUM_UNUSED_TO_EXISTING_SCORE_RATIO,
) -> tuple[list[int], float]:
    if not existing_ranked or not unused_ranked:
        raise ValueError("candidate selection requires both comparison pools")
    ratio = float(unused_ranked[0]["score"]) / max(
        float(existing_ranked[0]["score"]), 1e-12
    )
    selected = [int(unused_ranked[0]["case"])] if ratio <= maximum_score_ratio else []
    return selected, ratio


def _phase_aligned_actions(
    target_phase: np.ndarray,
    source_phase: np.ndarray,
    source_actions: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(target_phase, source_phase, source_actions[:, index])
            for index in range(3)
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--plan-summary", type=Path, required=True)
    parser.add_argument("--shadow-trace", type=Path, required=True)
    parser.add_argument("--teacher-dataset", type=Path, required=True)
    parser.add_argument("--localized-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.plan_summary.read_text(encoding="utf-8"))
    manifest = json.loads(args.plan_manifest.read_text(encoding="utf-8"))
    shadow_metadata, shadow = _load_npz(args.shadow_trace)
    dataset_metadata, dataset = _load_npz(args.teacher_dataset)
    localized_audit = json.loads(args.localized_audit.read_text(encoding="utf-8"))
    plan_paths = sorted(args.plan_dir.glob("case_*_smoothed_riser_plan_v1.npz"))
    all_case_ids = [int(path.name.split("_")[1]) for path in plan_paths]
    passed_cases = [int(case) for case in summary["passed_cases"]]
    rejected_cases = [int(case) for case in summary["rejected_cases"]]
    current_split_cases = sorted(
        int(case)
        for split in ("train", "validation", "holdout")
        for case in dataset_metadata["split_cases"][split]
    )
    existing_training_cases = [
        int(case) for case in dataset_metadata["split_cases"]["train"]
    ]
    unused_admitted_cases = sorted(set(passed_cases) - set(current_split_cases))
    input_checks = {
        "all_79_plans_present": all_case_ids == list(range(1, 80)),
        "portfolio_gate": summary.get("portfolio_gate_passed") is True,
        "portfolio_pass_count": len(passed_cases) == 70,
        "portfolio_partition": sorted(passed_cases + rejected_cases)
        == list(range(1, 80)),
        "current_dataset_case_count": len(current_split_cases) == 40,
        "existing_training_case_count": len(existing_training_cases) == 30,
        "unused_admitted_case_count": len(unused_admitted_cases) == 30,
        "reference_remains_validation": 4
        in dataset_metadata["split_cases"]["validation"],
        "unused_excludes_all_current_splits": not (
            set(unused_admitted_cases) & set(current_split_cases)
        ),
        "shadow_measurement_only": shadow_metadata.get("trace_only") is True
        and shadow_metadata.get("shadow_teacher_applied_to_commands") is False
        and shadow_metadata.get("valid_for_training") is False,
        "localized_audit_closed": localized_audit.get(
            "coverage_admission_passed"
        )
        is False
        and localized_audit.get("proposed_runtime_cases") == [],
        "manifest_schema": isinstance(manifest.get("items"), list),
    }
    if not all(input_checks.values()):
        raise ValueError(f"plan coverage input contract failed: {input_checks}")

    plans: dict[int, RiserPlaybackPlan] = {}
    for path in plan_paths:
        case = int(path.name.split("_")[1])
        metadata, _ = _load_npz(path)
        if case in passed_cases:
            plan_checks = {
                "case": metadata.get("case") == case,
                "integrity": metadata.get("trajectory_integrity_passed") is True,
                "timing_transition_kinematic": metadata.get(
                    "timing_transition_kinematic_gate_passed"
                )
                is True,
            }
            if not all(plan_checks.values()):
                raise ValueError(f"admitted plan {case} failed: {plan_checks}")
            plans[case] = load_riser_playback_plan(path)

    normalization_samples = np.concatenate(
        [
            build_plan_command_signature(
                plans[case],
                _subsample(
                    plans[case].time_s,
                    MAXIMUM_NORMALIZATION_SAMPLES_PER_PLAN,
                ),
            )
            for case in passed_cases
        ],
        axis=0,
    )
    feature_mean = np.mean(normalization_samples, axis=0)
    feature_scale = np.maximum(
        np.std(normalization_samples, axis=0), MINIMUM_FEATURE_SCALE
    )

    reference_mask = dataset["case_ids"] == 4
    phase_actions = _phase_aligned_actions(
        shadow["phase_time_s"],
        dataset["phase_time_s"][reference_mask],
        dataset["actions"][reference_mask],
    )
    shift = shadow["shadow_teacher_normalized_residual_actions"] - phase_actions
    hotspot_mask = np.any(
        np.abs(shift) > MATERIAL_SHIFT_THRESHOLDS,
        axis=1,
    )
    hotspot_times = _subsample(
        shadow["phase_time_s"][hotspot_mask], MAXIMUM_TARGET_SAMPLES
    )
    target = (
        build_plan_command_signature(plans[4], hotspot_times) - feature_mean
    ) / feature_scale

    def rank_case(case: int, pool: str) -> dict[str, object]:
        sample_times = _subsample(
            plans[case].time_s, MAXIMUM_CANDIDATE_SAMPLES
        )
        candidate = (
            build_plan_command_signature(plans[case], sample_times) - feature_mean
        ) / feature_scale
        return {
            "case": case,
            "pool": pool,
            "source_sample_count": len(plans[case].time_s),
            "distance_sample_count": len(sample_times),
            "plan": _identity(
                args.plan_dir / f"case_{case:04d}_smoothed_riser_plan_v1.npz"
            ),
            **_directed_distance(target, candidate),
        }

    existing_ranked = sorted(
        [rank_case(case, "existing_train") for case in existing_training_cases],
        key=lambda item: (item["score"], item["case"]),
    )
    unused_ranked = sorted(
        [rank_case(case, "unused_admitted") for case in unused_admitted_cases],
        key=lambda item: (item["score"], item["case"]),
    )
    proposed_cases, score_ratio = select_unused_candidate(
        existing_ranked, unused_ranked
    )
    coverage_passed = bool(proposed_cases)
    report = {
        "schema": "cinebotrl_two_wheel_riser_unused_plan_coverage_audit_v1",
        "reference_case": 4,
        "reference_contract": "validation_diagnostic_hotspot_times_only",
        "feature_contract": (
            "nominal_plan_current_plus_0p25_0p50_1p00_lookahead_command_v1"
        ),
        "feature_names": _signature_feature_names(),
        "feature_count": len(_signature_feature_names()),
        "feature_normalization": "all_70_admitted_plan_sample_mean_std_v1",
        "feature_scale_floor": MINIMUM_FEATURE_SCALE,
        "target_sample_count": len(hotspot_times),
        "hotspot_row_count": int(np.count_nonzero(hotspot_mask)),
        "current_split_cases": current_split_cases,
        "existing_training_cases": existing_training_cases,
        "unused_admitted_cases": unused_admitted_cases,
        "kinematically_rejected_cases": rejected_cases,
        "best_existing_training": existing_ranked[0],
        "best_unused_admitted": unused_ranked[0],
        "top_existing_training": existing_ranked[:5],
        "top_unused_admitted": unused_ranked[:10],
        "unused_to_existing_best_score_ratio": score_ratio,
        "maximum_unused_to_existing_score_ratio": (
            MAXIMUM_UNUSED_TO_EXISTING_SCORE_RATIO
        ),
        "coverage_expansion_admission_passed": coverage_passed,
        "proposed_shadow_measurement_cases": proposed_cases,
        "classification": (
            "one_unused_admitted_plan_materially_improves_command_coverage"
            if coverage_passed
            else "no_unused_admitted_plan_materially_improves_command_coverage"
        ),
        "input_contract_checks": input_checks,
        "inputs": {
            "plan_manifest": _identity(args.plan_manifest),
            "plan_summary": _identity(args.plan_summary),
            "shadow_trace": _identity(args.shadow_trace),
            "teacher_dataset": _identity(args.teacher_dataset),
            "localized_audit": _identity(args.localized_audit),
        },
        "case4_labels_admitted_for_training": False,
        "holdout_opened": False,
        "runtime_authorized": False,
        "authorization_token_issued": False,
        "dataset_created": False,
        "dagger_authorized": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "valid_for_training": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
