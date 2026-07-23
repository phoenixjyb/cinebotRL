"""Held-out paired validation for model-based corrective residuals."""

from __future__ import annotations

from typing import Mapping

import numpy as np


VALIDATION_PAIR_CONTRACT = (
    "same_seed_held_out_validation_improvement_without_teacher_admission_v1"
)


def assess_paired_corrective_validation(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    """Compare a validation pair without admitting labels or training data."""

    identity_fields = (
        "case",
        "split",
        "plan_sha256",
        "physics_seed",
        "source_duration_s",
        "execution_duration_s",
    )
    numeric_fields = (
        "position_error_p95_m",
        "position_error_max_m",
        "attitude_error_max_deg",
        "pitch_max_deg",
        "riser_error_max_m",
        "action_saturation_ratio",
    )
    for name, payload in (("baseline", baseline), ("candidate", candidate)):
        missing = [
            field for field in identity_fields + numeric_fields if field not in payload
        ]
        if missing:
            raise ValueError(
                f"{name} validation-pair evidence is missing fields: {missing}"
            )
        values = np.asarray(
            [payload[field] for field in numeric_fields], dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise ValueError(
                f"{name} validation-pair evidence contains non-finite metrics"
            )

    baseline_p95 = float(baseline["position_error_p95_m"])
    candidate_p95 = float(candidate["position_error_p95_m"])
    absolute_improvement = baseline_p95 - candidate_p95
    relative_improvement = absolute_improvement / max(baseline_p95, 1e-12)
    candidate_action_max = np.asarray(
        candidate.get("normalized_residual_action_abs_max"), dtype=np.float64
    )
    if (
        candidate_action_max.shape != (3,)
        or not np.isfinite(candidate_action_max).all()
    ):
        raise ValueError("candidate residual-action evidence is missing or invalid")

    checks = {
        "same_case_plan_seed_and_clocks": all(
            baseline[field] == candidate[field] for field in identity_fields
        ),
        "validation_split_only": baseline["split"]
        == candidate["split"]
        == "validation",
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
            np.any(candidate_action_max > 1e-6)
            and np.max(candidate_action_max) < 0.95
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
        "schema": (
            "cinebotrl_two_wheel_riser_corrective_validation_pair_assessment_v1"
        ),
        "contract": VALIDATION_PAIR_CONTRACT,
        "case": baseline["case"],
        "split": baseline["split"],
        "position_p95_absolute_improvement_m": absolute_improvement,
        "position_p95_relative_improvement": relative_improvement,
        "checks": checks,
        "validation_pair_passed": passed,
        "teacher_admission_opened": False,
        "label_capture_authorized": False,
        "dataset_created": False,
        "bc_authorized": False,
        "ppo_authorized": False,
        "training_started": False,
        "valid_for_training": False,
    }
