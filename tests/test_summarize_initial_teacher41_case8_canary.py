import json
from pathlib import Path

from scripts.two_wheel_balance.summarize_initial_teacher41_case8_canary import (
    EXECUTION_DURATION_S,
    LEARNED_SOURCE,
    SOURCE_DURATION_S,
    TRACKING_PROFILE,
    ZERO_SOURCE,
    finalize,
)


def _write_rollout(root: Path, source: str, *, learned: bool) -> None:
    result = {
        "case": 8,
        "source_duration_s": SOURCE_DURATION_S,
        "execution_duration_s": EXECUTION_DURATION_S,
        "executed_residual_dataset": None,
        "residual_action_abs_max": [0.2, 0.3, 0.1] if learned else [0.0, 0.0, 0.0],
        "dynamic_quality_passed": True,
        "passed": True,
    }
    payload = {
        "cases": [8],
        "trajectory_command_source": source,
        "tracking_profile": TRACKING_PROFILE,
        "phase_feedforward_contract": "derivatives_scaled_by_progress_v1",
        "raw_teacher_capture_started": False,
        "normalized_dataset_capture_started": False,
        "dynamic_quality_passed": True,
        "passed": True,
        "results": [result],
    }
    root.parent.mkdir(parents=True, exist_ok=True)
    root.write_text(json.dumps(payload), encoding="utf-8")


def _healthy_root(tmp_path: Path) -> Path:
    root = tmp_path / "canary"
    root.mkdir()
    (root / "admission.json").write_text(
        json.dumps({"passed": True, "runtime_commit": "a" * 40, "case": 8, "split": "validation"}),
        encoding="utf-8",
    )
    _write_rollout(root / "zero/case_0008.json", ZERO_SOURCE, learned=False)
    _write_rollout(root / "learned/case_0008.json", LEARNED_SOURCE, learned=True)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "passed": True,
                "cases": [8],
                "case_count": 1,
                "expected_tracking_profile": TRACKING_PROFILE,
                "maximum_regression_fraction": 0.05,
                "minimum_zero_improvement_fraction": 0.05,
            }
        ),
        encoding="utf-8",
    )
    heartbeat = {
        "schema": "cinebotrl_two_wheel_riser_runtime_heartbeat_v1",
        "case": 8,
        "dataset_created": False,
        "valid_for_training": False,
    }
    for mode in ("zero", "learned"):
        (root / mode / "runtime_heartbeat.json").write_text(
            json.dumps(heartbeat), encoding="utf-8"
        )
    return root


def test_healthy_canary_passes_without_authorizing_next_case(tmp_path: Path) -> None:
    result = finalize(
        _healthy_root(tmp_path),
        runtime_commit="a" * 40,
        zero_exit_code=6,
        learned_exit_code=0,
        gate_exit_code=0,
    )
    assert result["passed"]
    assert not result["case78_authorized"]
    assert not result["ppo_authorized"]
    assert not result["valid_for_training"]


def test_rejects_missing_learned_evidence(tmp_path: Path) -> None:
    root = _healthy_root(tmp_path)
    (root / "learned/case_0008.json").unlink()
    result = finalize(
        root,
        runtime_commit="a" * 40,
        zero_exit_code=0,
        learned_exit_code=0,
        gate_exit_code=0,
    )
    assert not result["passed"]
    assert not result["checks"]["learned_rollout_contract"]


def test_rejects_comparison_failure_or_dataset_side_effect(tmp_path: Path) -> None:
    root = _healthy_root(tmp_path)
    heartbeat_path = root / "learned/runtime_heartbeat.json"
    heartbeat = json.loads(heartbeat_path.read_text())
    heartbeat["dataset_created"] = True
    heartbeat_path.write_text(json.dumps(heartbeat), encoding="utf-8")
    result = finalize(
        root,
        runtime_commit="a" * 40,
        zero_exit_code=0,
        learned_exit_code=0,
        gate_exit_code=6,
    )
    assert not result["passed"]
    assert not result["checks"]["comparison_gate_exit_zero"]
    assert not result["checks"]["no_dataset_created"]
