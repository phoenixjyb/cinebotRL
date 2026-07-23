import json
from pathlib import Path

import pytest

from scripts.two_wheel_balance.gate_riser_residual_rollouts import (
    REGRESSION_METRICS,
    gate_rollouts,
)


def _write_rollout(
    root: Path,
    case: int,
    source: str,
    position_p95_m: float,
    *,
    pitch_max_deg: float = 4.0,
    tracking_profile: str = "riser_phase_consistent_v2",
    policy_command_base: str = "phase_feedforward",
    residual_action_scales: list[float] | None = None,
) -> None:
    metrics = {name: 1.0 for name in REGRESSION_METRICS}
    metrics.update(
        {
            "position_error_p95_m": position_p95_m,
            "position_error_max_m": position_p95_m + 0.01,
            "pitch_p95_deg": pitch_max_deg - 0.2,
            "pitch_max_deg": pitch_max_deg,
        }
    )
    result = {
        "case": case,
        "passed": True,
        "residual_action_abs_max": [0.2, 0.3, 0.1],
        **metrics,
    }
    payload = {
        "cases": [case],
        "passed": True,
        "trajectory_command_source": source,
        "tracking_profile": tracking_profile,
        "phase_feedforward_contract": "derivatives_scaled_by_progress_v1",
        "policy_command_base": policy_command_base,
        "residual_action_scales": residual_action_scales or [0.3, 0.4, 0.1],
        "results": [result],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / f"case_{case:04d}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_holdout_gate_beats_zero_without_teacher_regression(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher"
    zero = tmp_path / "zero"
    learned = tmp_path / "learned"
    policy = tmp_path / "policy.pt"
    policy.write_bytes(b"policy")
    for case in (1, 2, 3):
        _write_rollout(teacher, case, "deterministic_teacher", 0.10)
        _write_rollout(zero, case, "zero_policy_action_baseline", 0.15)
        _write_rollout(learned, case, "torchscript_residual_policy", 0.102)
    summary = gate_rollouts(
        teacher_dir=teacher,
        zero_dir=zero,
        learned_dir=learned,
        cases=[1, 2, 3],
        policy=policy,
        mode="holdout",
        maximum_regression_fraction=0.05,
    )
    assert summary["passed"]
    assert summary["aggregate_checks"]["learned_beats_zero_by_required_mean"]


def test_holdout_gate_rejects_balance_regression(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher"
    zero = tmp_path / "zero"
    learned = tmp_path / "learned"
    policy = tmp_path / "policy.pt"
    policy.write_bytes(b"policy")
    _write_rollout(teacher, 1, "deterministic_teacher", 0.10, pitch_max_deg=4.0)
    _write_rollout(zero, 1, "zero_policy_action_baseline", 0.20, pitch_max_deg=4.0)
    _write_rollout(
        learned, 1, "torchscript_residual_policy", 0.10, pitch_max_deg=4.3
    )
    summary = gate_rollouts(
        teacher_dir=teacher,
        zero_dir=zero,
        learned_dir=learned,
        cases=[1],
        policy=policy,
        mode="holdout",
        maximum_regression_fraction=0.05,
    )
    assert not summary["passed"]
    assert not summary["rows"][0]["checks"]["regression_pitch_max_deg"]


def test_all79_mode_requires_the_complete_case_set(tmp_path: Path) -> None:
    policy = tmp_path / "policy.pt"
    policy.write_bytes(b"policy")
    with pytest.raises(ValueError, match="cases 1 through 79"):
        gate_rollouts(
            teacher_dir=tmp_path,
            learned_dir=tmp_path,
            cases=[1],
            policy=policy,
            mode="all79",
            maximum_regression_fraction=0.05,
        )


def test_validation_canary_accepts_explicit_current_tracking_profile(
    tmp_path: Path,
) -> None:
    teacher = tmp_path / "teacher"
    zero = tmp_path / "zero"
    learned = tmp_path / "learned"
    policy = tmp_path / "policy.pt"
    policy.write_bytes(b"policy")
    profile = "riser_recovery_direction_v4_camera_lever_arm_v1"
    _write_rollout(
        teacher, 4, "deterministic_teacher", 0.10, tracking_profile=profile
    )
    _write_rollout(
        zero, 4, "zero_policy_action_baseline", 0.15, tracking_profile=profile
    )
    _write_rollout(
        learned, 4, "torchscript_residual_policy", 0.102, tracking_profile=profile
    )
    summary = gate_rollouts(
        teacher_dir=teacher,
        zero_dir=zero,
        learned_dir=learned,
        cases=[4],
        policy=policy,
        mode="validation_canary",
        maximum_regression_fraction=0.05,
        expected_tracking_profile=profile,
    )
    assert summary["passed"]
    assert summary["expected_tracking_profile"] == profile


def test_model_based_all79_contract_uses_planner_sources_and_scales(
    tmp_path: Path,
) -> None:
    teacher = tmp_path / "teacher"
    learned = tmp_path / "learned"
    policy = tmp_path / "policy.pt"
    policy.write_bytes(b"policy")
    scales = [0.05, 0.05, 0.02]
    profile = "riser_recovery_direction_v4_camera_lever_arm_v1"
    provenance = {}
    for name in ("admission", "preflight", "plan_manifest"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        provenance[name] = path
    for case in range(1, 80):
        _write_rollout(
            teacher,
            case,
            "model_based_planner_plus_zero_policy_residual",
            0.10,
            tracking_profile=profile,
            policy_command_base="model_based_planner",
            residual_action_scales=scales,
        )
        _write_rollout(
            learned,
            case,
            "model_based_planner_plus_torchscript_residual",
            0.09,
            tracking_profile=profile,
            policy_command_base="model_based_planner",
            residual_action_scales=scales,
        )
    summary = gate_rollouts(
        teacher_dir=teacher,
        learned_dir=learned,
        cases=list(range(1, 80)),
        policy=policy,
        mode="all79",
        maximum_regression_fraction=0.05,
        policy_command_contract=(
            "model_based_planner_plus_bounded_policy_residual_v1"
        ),
        expected_tracking_profile=profile,
        rollout_admission=provenance["admission"],
        preflight_receipt=provenance["preflight"],
        plan_manifest=provenance["plan_manifest"],
        execution_commit="a" * 40,
    )
    assert summary["passed"]
    assert summary["residual_action_scales"] == scales


def test_model_based_validation_requires_provenance_and_uses_baseline(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    learned = tmp_path / "learned"
    policy = tmp_path / "policy.pt"
    policy.write_bytes(b"policy")
    scales = [0.05, 0.05, 0.02]
    profile = "riser_recovery_direction_v4_camera_lever_arm_v1"
    _write_rollout(
        baseline,
        8,
        "model_based_planner_plus_zero_policy_residual",
        0.15,
        tracking_profile=profile,
        policy_command_base="model_based_planner",
        residual_action_scales=scales,
    )
    _write_rollout(
        learned,
        8,
        "model_based_planner_plus_torchscript_residual",
        0.10,
        tracking_profile=profile,
        policy_command_base="model_based_planner",
        residual_action_scales=scales,
    )
    kwargs = {
        "teacher_dir": baseline,
        "zero_dir": baseline,
        "learned_dir": learned,
        "cases": [8],
        "policy": policy,
        "mode": "validation_canary",
        "maximum_regression_fraction": 0.05,
        "policy_command_contract": (
            "model_based_planner_plus_bounded_policy_residual_v1"
        ),
        "expected_tracking_profile": profile,
    }
    with pytest.raises(ValueError, match="bound runtime provenance"):
        gate_rollouts(**kwargs)
    provenance = {}
    for name in ("admission", "preflight", "plan_manifest"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        provenance[name] = path
    summary = gate_rollouts(
        **kwargs,
        rollout_admission=provenance["admission"],
        preflight_receipt=provenance["preflight"],
        plan_manifest=provenance["plan_manifest"],
        execution_commit="a" * 40,
    )
    assert summary["passed"] is True
    assert summary["schema"].endswith("validation_canary_gate_v1")
    assert summary["rows"][0]["teacher_rollout"] == (
        summary["rows"][0]["zero_rollout"]
    )
