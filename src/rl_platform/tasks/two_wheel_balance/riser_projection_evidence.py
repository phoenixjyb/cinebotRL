"""Fail-closed projection evidence derived from atomic playback results."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


PROJECTION_EVIDENCE_SCHEMA = (
    "cinebotrl_two_wheel_riser_corrective_projection_evidence_v2"
)
PROJECTION_EVIDENCE_SOURCE = (
    "runtime_requested_effective_post_supervisor_aggregates_v1"
)
_VECTOR_FIELDS = (
    "requested_policy_residual_action_abs_max",
    "effective_policy_residual_action_abs_max",
    "policy_residual_projection_delta_abs_max",
)


def _finite_vector(value: object) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vector.shape != (3,) or not np.isfinite(vector).all():
        return None
    return vector


def _nonnegative_integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    if integer < 0 or integer != value:
        return None
    return integer


def audit_runtime_projection_evidence(
    playback: Mapping[str, object],
    result: Mapping[str, object],
    *,
    enabled: bool,
) -> dict[str, object]:
    """Audit projection evidence without intercepting or changing commands."""

    scales = _finite_vector(playback.get("residual_action_scales"))
    vectors = {
        name: _finite_vector(result.get(name)) for name in _VECTOR_FIELDS
    }
    completed_steps = _nonnegative_integer(result.get("completed_steps"))
    affected_samples = _nonnegative_integer(
        result.get("policy_residual_projection_sample_count")
    )
    checks = {
        "action_scales_valid": scales is not None
        and bool(np.all(scales > 0.0)),
        "completed_steps_valid": completed_steps is not None
        and completed_steps > 0,
        "requested_action_aggregate_valid": (
            vectors[_VECTOR_FIELDS[0]] is not None
        ),
        "effective_action_aggregate_valid": (
            vectors[_VECTOR_FIELDS[1]] is not None
        ),
        "projection_delta_aggregate_valid": (
            vectors[_VECTOR_FIELDS[2]] is not None
        ),
        "affected_sample_count_valid": affected_samples is not None
        and completed_steps is not None
        and affected_samples <= completed_steps,
    }
    vectors_valid = all(
        vectors[name] is not None for name in _VECTOR_FIELDS
    )
    if vectors_valid:
        requested = vectors[_VECTOR_FIELDS[0]]
        effective = vectors[_VECTOR_FIELDS[1]]
        delta = vectors[_VECTOR_FIELDS[2]]
        assert requested is not None
        assert effective is not None
        assert delta is not None
        checks["normalized_actions_bounded"] = bool(
            np.max(requested) <= 1.0 + 1e-6
            and np.max(effective) <= 1.0 + 1e-6
            and np.max(delta) <= 2.0 + 2e-6
            and np.min(requested) >= 0.0
            and np.min(effective) >= 0.0
            and np.min(delta) >= 0.0
        )
        checks["disabled_route_is_exact_zero"] = enabled or bool(
            np.max(requested) == 0.0
            and np.max(effective) == 0.0
            and np.max(delta) == 0.0
            and affected_samples == 0
        )
    else:
        requested = effective = delta = None
        checks["normalized_actions_bounded"] = False
        checks["disabled_route_is_exact_zero"] = False

    ready = all(checks.values())
    physical_ready = ready and scales is not None
    return {
        "schema": PROJECTION_EVIDENCE_SCHEMA,
        "evidence_source": PROJECTION_EVIDENCE_SOURCE,
        "enabled": enabled,
        "sample_count": completed_steps if enabled and ready else 0,
        "requested_residual_abs_max": (
            (requested * scales).tolist() if physical_ready else None
        ),
        "effective_residual_abs_max": (
            (effective * scales).tolist() if physical_ready else None
        ),
        "effective_normalized_action_abs_max": (
            effective.tolist() if ready else None
        ),
        "requested_effective_delta_abs_max": (
            (delta * scales).tolist() if physical_ready else None
        ),
        "projection_affected_sample_count": affected_samples,
        "runtime_aggregate_fields_complete": vectors_valid,
        "observer_modified_commands": False,
        "applied_to_commands": False,
        "labels_captured": False,
        "dataset_created": False,
        "training_started": False,
        "valid_for_training": False,
        "checks": checks,
        "passed": ready,
    }
