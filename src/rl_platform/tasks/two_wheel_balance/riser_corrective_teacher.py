"""Bounded corrective-teacher contract above the complete model planner."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from .riser_residual_dataset import MODEL_BASED_POLICY_RESIDUAL_SCALES


CORRECTIVE_TEACHER_CONTRACT = (
    "model_based_planner_plus_camera_error_corrective_teacher_v1"
)
CORRECTIVE_TARGET_ADMISSION_CONTRACT = (
    "same_seed_paired_dynamic_improvement_before_label_capture_v1"
)


@dataclass(frozen=True)
class CorrectiveTeacherConfig:
    """Conservative camera-position feedback used only as a teacher candidate."""

    longitudinal_gain_s_inv: float = 0.20
    lateral_to_yaw_gain_rad_s_m: float = 0.30
    vertical_gain: float = 0.30
    deadbands_m: tuple[float, float, float] = (0.01, 0.01, 0.005)
    maximum_residuals: tuple[float, float, float] = (0.045, 0.045, 0.018)
    maximum_slew_rates: tuple[float, float, float] = (0.10, 0.10, 0.04)

    def validate(self) -> None:
        gains = np.asarray(
            [
                self.longitudinal_gain_s_inv,
                self.lateral_to_yaw_gain_rad_s_m,
                self.vertical_gain,
            ],
            dtype=np.float64,
        )
        deadbands = np.asarray(self.deadbands_m, dtype=np.float64)
        limits = np.asarray(self.maximum_residuals, dtype=np.float64)
        slew = np.asarray(self.maximum_slew_rates, dtype=np.float64)
        checks = {
            "gains": gains.shape == (3,) and np.isfinite(gains).all() and np.all(gains > 0),
            "deadbands": (
                deadbands.shape == (3,)
                and np.isfinite(deadbands).all()
                and np.all(deadbands >= 0)
            ),
            "limits": (
                limits.shape == (3,)
                and np.isfinite(limits).all()
                and np.all(limits > 0)
                and np.all(limits < MODEL_BASED_POLICY_RESIDUAL_SCALES)
            ),
            "slew": slew.shape == (3,) and np.isfinite(slew).all() and np.all(slew > 0),
        }
        if not all(checks.values()):
            raise ValueError(f"invalid corrective teacher configuration: {checks}")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "contract": CORRECTIVE_TEACHER_CONTRACT,
            "longitudinal_gain_s_inv": self.longitudinal_gain_s_inv,
            "lateral_to_yaw_gain_rad_s_m": self.lateral_to_yaw_gain_rad_s_m,
            "vertical_gain": self.vertical_gain,
            "deadbands_m": list(self.deadbands_m),
            "maximum_residuals": list(self.maximum_residuals),
            "maximum_slew_rates": list(self.maximum_slew_rates),
            "policy_residual_scales": MODEL_BASED_POLICY_RESIDUAL_SCALES.tolist(),
        }


@dataclass(frozen=True)
class CorrectiveTeacherOutput:
    unbounded_residual: np.ndarray
    bounded_residual: np.ndarray
    applied_residual: np.ndarray
    normalized_action: np.ndarray
    amplitude_limited: np.ndarray
    slew_limited: np.ndarray


def _remove_deadband(value: np.ndarray, deadband: np.ndarray) -> np.ndarray:
    return np.sign(value) * np.maximum(np.abs(value) - deadband, 0.0)


def build_corrective_teacher_action(
    camera_position_error_body_m: np.ndarray,
    previous_residual: np.ndarray,
    *,
    dt_s: float,
    config: CorrectiveTeacherConfig = CorrectiveTeacherConfig(),
) -> CorrectiveTeacherOutput:
    """Build one causal, bounded correction from target-minus-actual camera error."""

    config.validate()
    error = np.asarray(camera_position_error_body_m, dtype=np.float64)
    previous = np.asarray(previous_residual, dtype=np.float64)
    if error.shape != (3,) or not np.isfinite(error).all():
        raise ValueError("invalid camera-position error")
    if previous.shape != (3,) or not np.isfinite(previous).all():
        raise ValueError("invalid previous corrective residual")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("corrective teacher timestep must be positive")

    deadbanded = _remove_deadband(error, np.asarray(config.deadbands_m))
    gains = np.asarray(
        [
            config.longitudinal_gain_s_inv,
            config.lateral_to_yaw_gain_rad_s_m,
            config.vertical_gain,
        ],
        dtype=np.float64,
    )
    unbounded = gains * deadbanded
    limits = np.asarray(config.maximum_residuals, dtype=np.float64)
    bounded = np.clip(unbounded, -limits, limits)
    maximum_delta = np.asarray(config.maximum_slew_rates, dtype=np.float64) * dt_s
    applied = previous + np.clip(bounded - previous, -maximum_delta, maximum_delta)
    normalized = applied / MODEL_BASED_POLICY_RESIDUAL_SCALES
    if np.max(np.abs(normalized)) >= 1.0 - 1e-6:
        raise ValueError("corrective teacher violated the reserved policy margin")
    return CorrectiveTeacherOutput(
        unbounded_residual=unbounded,
        bounded_residual=bounded,
        applied_residual=applied,
        normalized_action=normalized,
        amplitude_limited=np.abs(unbounded - bounded) > 1e-12,
        slew_limited=np.abs(bounded - applied) > 1e-12,
    )


def assess_paired_corrective_rollouts(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    """Admit a target only after a same-seed candidate beats the zero residual."""

    identity_fields = (
        "case",
        "split",
        "plan_sha256",
        "physics_seed",
        "source_duration_s",
        "execution_duration_s",
    )
    required_numeric = (
        "position_error_p95_m",
        "position_error_max_m",
        "attitude_error_max_deg",
        "pitch_max_deg",
        "riser_error_max_m",
        "action_saturation_ratio",
    )
    for name, payload in (("baseline", baseline), ("candidate", candidate)):
        missing = [field for field in identity_fields + required_numeric if field not in payload]
        if missing:
            raise ValueError(f"{name} paired evidence is missing fields: {missing}")
        values = np.asarray([payload[field] for field in required_numeric], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} paired evidence contains non-finite metrics")

    baseline_p95 = float(baseline["position_error_p95_m"])
    candidate_p95 = float(candidate["position_error_p95_m"])
    absolute_improvement = baseline_p95 - candidate_p95
    relative_improvement = absolute_improvement / max(baseline_p95, 1e-12)
    candidate_action_max = np.asarray(
        candidate.get("normalized_residual_action_abs_max"), dtype=np.float64
    )
    if candidate_action_max.shape != (3,) or not np.isfinite(candidate_action_max).all():
        raise ValueError("candidate residual-action evidence is missing or invalid")

    checks = {
        "same_case_plan_seed_and_clocks": all(
            baseline[field] == candidate[field] for field in identity_fields
        ),
        "training_split_only": baseline["split"] == candidate["split"] == "train",
        "baseline_dynamic_quality": baseline.get("dynamic_quality_passed") is True,
        "candidate_dynamic_quality": candidate.get("dynamic_quality_passed") is True,
        "baseline_zero_residual": bool(
            np.allclose(
                baseline.get("normalized_residual_action_abs_max"),
                np.zeros(3),
                atol=1e-12,
            )
        ),
        "candidate_nonzero_bounded_residual": bool(
            np.any(candidate_action_max > 1e-6) and np.max(candidate_action_max) < 0.95
        ),
        "minimum_position_p95_improvement": (
            absolute_improvement >= 0.003 and relative_improvement >= 0.02
        ),
        "position_max_not_regressed": float(candidate["position_error_max_m"])
        <= float(baseline["position_error_max_m"]) + 0.005,
        "attitude_max_not_regressed": float(candidate["attitude_error_max_deg"])
        <= float(baseline["attitude_error_max_deg"]) + 0.10,
        "pitch_not_regressed": float(candidate["pitch_max_deg"])
        <= float(baseline["pitch_max_deg"]) + 0.50,
        "riser_not_regressed": float(candidate["riser_error_max_m"])
        <= float(baseline["riser_error_max_m"]) + 0.002,
        "saturation_not_regressed": float(candidate["action_saturation_ratio"])
        <= float(baseline["action_saturation_ratio"]),
        "no_dataset_or_training": all(
            payload.get("dataset_created") is False
            and payload.get("training_started") is False
            and payload.get("ppo_started") is False
            for payload in (baseline, candidate)
        ),
    }
    passed = all(checks.values())
    return {
        "schema": "cinebotrl_two_wheel_riser_corrective_target_pair_admission_v1",
        "contract": CORRECTIVE_TARGET_ADMISSION_CONTRACT,
        "case": baseline["case"],
        "split": baseline["split"],
        "position_p95_absolute_improvement_m": absolute_improvement,
        "position_p95_relative_improvement": relative_improvement,
        "checks": checks,
        "corrective_target_admission_passed": passed,
        "label_capture_authorized": False,
        "dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
    }
