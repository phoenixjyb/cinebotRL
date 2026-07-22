import json
from pathlib import Path

from scripts.two_wheel_balance.summarize_initial_teacher41_validation_canary import (
    LEARNED_SOURCE,
    ZERO_SOURCE,
    finalize,
)


CASE = 78
SOURCE_DURATION = 135.487646
EXECUTION_DURATION = 192.29956737098348
PROFILE = "riser_recovery_direction_v4_camera_lever_arm_v1"


def _write_rollout(path: Path, source: str, residual_max: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cases": [CASE],
                "trajectory_command_source": source,
                "tracking_profile": PROFILE,
                "phase_feedforward_contract": "derivatives_scaled_by_progress_v1",
                "raw_teacher_capture_started": False,
                "normalized_dataset_capture_started": False,
                "dynamic_quality_passed": True,
                "passed": True,
                "results": [
                    {
                        "case": CASE,
                        "source_duration_s": SOURCE_DURATION,
                        "execution_duration_s": EXECUTION_DURATION,
                        "executed_residual_dataset": None,
                        "residual_action_abs_max": residual_max,
                        "dynamic_quality_passed": True,
                        "passed": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "case78"
    root.mkdir()
    (root / "admission.json").write_text(
        json.dumps({"passed": True, "runtime_commit": "a" * 40, "case": CASE, "split": "validation"}),
        encoding="utf-8",
    )
    _write_rollout(root / "learned/case_0078.json", LEARNED_SOURCE, [0.3, 0.2, 0.1])
    _write_rollout(root / "zero/case_0078.json", ZERO_SOURCE, [0.0, 0.0, 0.0])
    (root / "summary.json").write_text(
        json.dumps(
            {
                "passed": True,
                "cases": [CASE],
                "case_count": 1,
                "expected_tracking_profile": PROFILE,
                "maximum_regression_fraction": 0.05,
                "minimum_zero_improvement_fraction": 0.05,
            }
        ),
        encoding="utf-8",
    )
    heartbeat = {
        "schema": "cinebotrl_two_wheel_riser_runtime_heartbeat_v1",
        "case": CASE,
        "dataset_created": False,
        "valid_for_training": False,
    }
    for mode in ("learned", "zero"):
        (root / mode / "runtime_heartbeat.json").write_text(
            json.dumps(heartbeat), encoding="utf-8"
        )
    return root


def _finalize(root: Path, **overrides):
    values = {"learned_exit_code": 0, "zero_exit_code": 0, "gate_exit_code": 0}
    values.update(overrides)
    return finalize(
        root,
        case=CASE,
        source_duration_s=SOURCE_DURATION,
        execution_duration_s=EXECUTION_DURATION,
        tracking_profile=PROFILE,
        runtime_commit="a" * 40,
        **values,
    )


def test_healthy_validation_canary_passes_but_does_not_expand(
    tmp_path: Path,
) -> None:
    result = _finalize(_root(tmp_path))
    assert result["passed"]
    assert not result["remaining_validation_cases_authorized"]
    assert not result["holdout_opened"]
    assert not result["ppo_authorized"]


def test_rejects_missing_zero_or_learned_failure(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "zero/case_0078.json").unlink()
    result = _finalize(root, learned_exit_code=6)
    assert not result["passed"]
    assert not result["checks"]["learned_process_exit_zero"]
    assert not result["checks"]["zero_baseline_recorded"]


def test_rejects_dataset_side_effect(tmp_path: Path) -> None:
    root = _root(tmp_path)
    heartbeat_path = root / "learned/runtime_heartbeat.json"
    heartbeat = json.loads(heartbeat_path.read_text())
    heartbeat["dataset_created"] = True
    heartbeat_path.write_text(json.dumps(heartbeat), encoding="utf-8")
    result = _finalize(root)
    assert not result["checks"]["no_dataset_created"]
    assert not result["passed"]
