import numpy as np

from rl_platform.tasks.two_wheel_balance.riser_projection_evidence import (
    PROJECTION_EVIDENCE_SCHEMA,
    audit_runtime_projection_evidence,
)


def _playback() -> dict[str, object]:
    return {"residual_action_scales": [0.05, 0.05, 0.02]}


def _result(*, candidate: bool) -> dict[str, object]:
    return {
        "completed_steps": 100,
        "requested_policy_residual_action_abs_max": (
            [0.20, 0.08, 0.03] if candidate else [0.0, 0.0, 0.0]
        ),
        "effective_policy_residual_action_abs_max": (
            [0.16, 0.06, 0.03] if candidate else [0.0, 0.0, 0.0]
        ),
        "policy_residual_projection_delta_abs_max": (
            [0.04, 0.02, 0.0] if candidate else [0.0, 0.0, 0.0]
        ),
        "policy_residual_projection_sample_count": 15 if candidate else 0,
    }


def test_candidate_projection_evidence_uses_runtime_aggregates() -> None:
    evidence = audit_runtime_projection_evidence(
        _playback(), _result(candidate=True), enabled=True
    )
    assert evidence["schema"] == PROJECTION_EVIDENCE_SCHEMA
    assert evidence["sample_count"] == 100
    assert evidence["projection_affected_sample_count"] == 15
    np.testing.assert_allclose(
        evidence["requested_residual_abs_max"],
        [0.010, 0.004, 0.0006],
    )
    np.testing.assert_allclose(
        evidence["effective_residual_abs_max"],
        [0.008, 0.003, 0.0006],
    )
    assert evidence["observer_modified_commands"] is False
    assert evidence["dataset_created"] is False
    assert evidence["training_started"] is False
    assert evidence["passed"] is True


def test_disabled_projection_requires_exact_zero_aggregates() -> None:
    evidence = audit_runtime_projection_evidence(
        _playback(), _result(candidate=False), enabled=False
    )
    assert evidence["sample_count"] == 0
    assert evidence["passed"] is True
    result = _result(candidate=False)
    result["effective_policy_residual_action_abs_max"] = [0.01, 0.0, 0.0]
    evidence = audit_runtime_projection_evidence(
        _playback(), result, enabled=False
    )
    assert evidence["checks"]["disabled_route_is_exact_zero"] is False
    assert evidence["passed"] is False


def test_projection_evidence_rejects_missing_or_invalid_runtime_fields() -> None:
    for name, value in (
        ("requested_policy_residual_action_abs_max", None),
        ("effective_policy_residual_action_abs_max", [np.nan, 0.0, 0.0]),
        ("policy_residual_projection_delta_abs_max", [2.1, 0.0, 0.0]),
        ("policy_residual_projection_sample_count", 101),
    ):
        result = _result(candidate=True)
        result[name] = value
        evidence = audit_runtime_projection_evidence(
            _playback(), result, enabled=True
        )
        assert evidence["passed"] is False
        assert evidence["valid_for_training"] is False
